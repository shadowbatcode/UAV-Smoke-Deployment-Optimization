%% ========================================================================
%                     2025年全国大学生数学建模竞赛 A题
%               问题二：单一文件完整解决方案 (遍历多情况)
%
%   功能: 1. 支持多个导弹与无人机初始坐标的遍历。
%         2. 使用遗传算法(GA)寻找每个场景的最优遮蔽策略。
%         3. 输出每种情况的最大遮蔽时长，并可选进行可视化。
% =========================================================================

%% 1. 主程序脚本 (Main Script)
% =========================================================================
clear; clc; close all;

fprintf('问题二：多场景遍历求解程序已启动...\n');

% --- 定义优化问题 ---
n_vars = 4; % 决策变量个数: [v_F, theta_F, t_launch, t_fuse]
lb = [ 70,   0,  0.1,  0.1]; % 下界
ub = [140, 2*pi, 40.0, 19.0]; % 上界

options = optimoptions('ga', ...
    'PopulationSize', 400, ...
    'MaxGenerations', 30, ...
    'Display', 'none', ...
    'UseParallel', false);

% --- 定义导弹与无人机初始坐标 (可扩展) ---
missile_positions = [
    20000, 0, 2000;
    21000, -500, 2200
];
uav_positions = [
    17800, 0, 1800;
    18500, -200, 1700
];

% --- 遍历所有组合 ---
results = [];
case_id = 1;

for iM = 1:size(missile_positions,1)
    for iF = 1:size(uav_positions,1)

        fprintf('\n===== 正在计算场景 #%d (M%d-F%d) =====\n', case_id, iM, iF);

        % 把初始坐标打包传递
        context.P_M1_0 = missile_positions(iM,:);
        context.P_F1_0 = uav_positions(iF,:);

        % 定义带场景的适应度函数
        fitness_fcn = @(X) -fitness_function(X, context);

        % 遗传算法优化
        [best_X, max_shielding_time_neg] = ga(fitness_fcn, n_vars, [], [], [], [], lb, ub, [], options);
        max_shielding_time = -max_shielding_time_neg;

        % 保存结果
        results(case_id).case_id = case_id;
        results(case_id).missile_init = context.P_M1_0;
        results(case_id).uav_init = context.P_F1_0;
        results(case_id).best_X = best_X;
        results(case_id).max_shielding_time = max_shielding_time;

        % 输出结果
        fprintf('最优策略: v_F=%.2f, θ=%.2f°, t_launch=%.2f, t_fuse=%.2f\n', ...
            best_X(1), rad2deg(best_X(2)), best_X(3), best_X(4));
        fprintf('最大遮蔽时长: %.3f 秒\n', max_shielding_time);

        % 可视化（只画第一个场景以节省时间）
        if case_id == 1
            visualize_optimal_strategy(best_X, max_shielding_time, context);
        end

        case_id = case_id + 1;
    end
end

fprintf('\n==== 所有场景计算完成 ====\n');
disp(struct2table(results));


%% ========================================================================
%                      本地函数定义区域 (Local Functions)
% =========================================================================

%% 函数 1: 适应度函数
function total_shielding_time = fitness_function(X, context)
    % --- 解码策略向量 ---
    v_F = X(1); theta_F = X(2); t_launch = X(3); t_fuse = X(4);

    % --- 定义常量 ---
    g = 9.8; P_target = [0, 0, 0]; R_target = 7; H_target = 10;
    P_T_bottom_center = [0, 200, 0];
    v_M1 = 300;
    R_c = 10; T_smoke = 20; v_sink = 3; dt = 0.02;

    % --- 导弹轨迹 ---
    d_M1 = P_target - context.P_M1_0;
    u_M1 = d_M1 / norm(d_M1);
    P_M1 = @(t) context.P_M1_0 + v_M1 * (t' * u_M1);

    % --- 无人机轨迹 ---
    u_F1_h = [cos(theta_F), sin(theta_F), 0];
    P_F1 = @(t) context.P_F1_0 + v_F * (t' * u_F1_h);

    % --- 烟幕参数 ---
    P_launch = P_F1(t_launch); V_launch = v_F * u_F1_h;
    t_exp = t_launch + t_fuse;
    P_exp = P_launch + V_launch * t_fuse + [0,0,-0.5*g*t_fuse^2];
    P_c = @(t) P_exp + (t' - t_exp) * [0,0,-v_sink];

    % --- 目标点集 ---
    target_points = get_target_points(P_T_bottom_center, R_target, H_target);

    % --- 仿真时间 ---
    t_start = t_exp; t_end = t_exp + T_smoke;
    if t_start >= t_end, total_shielding_time = 0; return; end
    time_vector = t_start : dt : t_end;

    % --- 遮蔽判定 ---
    is_shielded = false(size(time_vector));
    for i = 1:length(time_vector)
        is_shielded(i) = isTargetFullyShielded(P_M1(time_vector(i)), P_c(time_vector(i)), target_points, R_c);
    end
    total_shielding_time = sum(is_shielded) * dt;
end


%% 函数 2: 可视化函数
function visualize_optimal_strategy(optimal_X, max_shielding_time, context)
    % 使用 run_detailed_simulation 获取数据
    [history, models] = run_detailed_simulation(optimal_X, context);

    % 解包
    time_vector = history.time_vector;
    is_shielded_flag = history.is_shielded_flag;
    P_M1 = models.P_M1;
    P_c = models.P_c;
    P_T_bottom_center = models.P_T_bottom_center;
    R_target = models.R_target;
    H_target = models.H_target;
    R_c = models.R_c;

    % 绘制全局三维场景
    figure('Name','全局场景');
    hold on; grid on; axis equal;
    t_vis = time_vector(1):0.2:time_vector(end);
    pos_M1 = P_M1(t_vis); pos_C = P_c(t_vis);
    plot3(pos_M1(:,1),pos_M1(:,2),pos_M1(:,3),'r-','LineWidth',2);
    plot3(pos_C(:,1),pos_C(:,2),pos_C(:,3),'g--','LineWidth',2);
    [Xc,Yc,Zc] = createCylinder(P_T_bottom_center,R_target,H_target);
    surf(Xc,Yc,Zc,'FaceColor',[0.2 0.5 0.8],'EdgeColor','none','FaceAlpha',0.6);
    [Xs,Ys,Zs] = createSphere(P_c(time_vector(round(end/2))),R_c);
    surf(Xs,Ys,Zs,'FaceColor',[0.5 0.8 0.5],'EdgeAlpha','none','FaceAlpha',0.3);
    xlabel('X'); ylabel('Y'); zlabel('Z');
    title(sprintf('最优策略全局场景 (最大遮蔽 %.2fs)',max_shielding_time));
end


%% 函数 3: 详细仿真
function [history, models] = run_detailed_simulation(X, context)
    models = get_models_and_constants(X, context);
    P_M1 = models.P_M1;
    P_c = models.P_c;
    target_points = models.target_points;
    R_c = models.R_c; dt = models.dt;

    t_exp = X(3) + X(4);
    t_start = t_exp; t_end = t_exp + models.T_smoke;
    time_vector = t_start:dt:t_end;

    num_steps = length(time_vector);
    is_shielded_flag = false(1,num_steps);
    for i=1:num_steps
        t = time_vector(i);
        is_shielded_flag(i) = isTargetFullyShielded(P_M1(t),P_c(t),target_points,R_c);
    end

    history.time_vector = time_vector;
    history.is_shielded_flag = is_shielded_flag;
end


%% ========================================================================
% 辅助函数
% =========================================================================

function models = get_models_and_constants(X, context)
    v_F = X(1); theta_F = X(2); t_launch = X(3); t_fuse = X(4);

    models.g = 9.8; models.P_target=[0,0,0]; models.R_target=7; models.H_target=10;
    models.P_T_bottom_center=[0,200,0];
    models.P_T_center=models.P_T_bottom_center+[0,0,models.H_target/2];
    models.v_M1=300;
    models.P_M1_0=context.P_M1_0;
    models.P_F1_0=context.P_F1_0;
    models.R_c=10; models.T_smoke=20; models.v_sink=3; models.dt=0.02;

    d_M1=models.P_target-models.P_M1_0; u_M1=d_M1/norm(d_M1);
    models.P_M1=@(t) models.P_M1_0+models.v_M1*(t'*u_M1);

    u_F1_h=[cos(theta_F),sin(theta_F),0];
    P_F1=@(t) models.P_F1_0+v_F*(t'*u_F1_h);
    P_launch=P_F1(t_launch); V_launch=v_F*u_F1_h;
    t_exp=t_launch+t_fuse;
    P_exp=P_launch+V_launch*t_fuse+[0,0,-0.5*models.g*t_fuse^2];
    models.P_c=@(t) P_exp+(t'-t_exp)*[0,0,-models.v_sink];

    models.target_points=get_target_points(models.P_T_bottom_center,models.R_target,models.H_target);
end

function target_points=get_target_points(P_T_bottom_center,R_target,H_target)
    num_edge_points=8;
    theta=linspace(0,2*pi,num_edge_points+1); theta(end)=[];
    edge_x=R_target*cos(theta)+P_T_bottom_center(1);
    edge_y=R_target*sin(theta)+P_T_bottom_center(2);
    target_points=[
        P_T_bottom_center; P_T_bottom_center+[0,0,H_target];
        [edge_x',edge_y',repmat(P_T_bottom_center(3),num_edge_points,1)];
        [edge_x',edge_y',repmat(P_T_bottom_center(3)+H_target,num_edge_points,1)];
    ];
end

function fully_shielded=isTargetFullyShielded(pos_M1,pos_Cloud,all_target_points,R_c)
    fully_shielded=true;
    for i=1:size(all_target_points,1)
        if getDistanceToLOS(pos_M1,pos_Cloud,all_target_points(i,:))>R_c
            fully_shielded=false; return;
        end
    end
end

function dist=getDistanceToLOS(pos_M1,pos_Cloud,pos_Point)
    vec_AP=pos_Point-pos_M1; vec_AC=pos_Cloud-pos_M1;
    if norm(vec_AP)<1e-6, dist=norm(vec_AC); return; end
    proj_len=dot(vec_AC,vec_AP)/norm(vec_AP);
    if proj_len<0 || proj_len>norm(vec_AP)
        dist=min(norm(vec_AC),norm(pos_Cloud-pos_Point));
    else
        dist=norm(cross(vec_AP,vec_AC))/norm(vec_AP);
    end
end

function [X,Y,Z]=createCylinder(center_bottom,R,H)
    [Xc,Yc,Zc]=cylinder(R,30);
    X=Xc+center_bottom(1); Y=Yc+center_bottom(2); Z=Zc*H+center_bottom(3);
end

function [X,Y,Z]=createSphere(center,R)
    [Xs,Ys,Zs]=sphere(20);
    X=Xs*R+center(1); Y=Ys*R+center(2); Z=Zs*R+center(3);
end
