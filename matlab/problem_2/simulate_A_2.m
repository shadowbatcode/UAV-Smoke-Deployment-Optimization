%% ========================================================================
%                     2025年全国大学生数学建模竞赛 A题
%               问题二：单一文件完整解决方案 (优化与可视化)
%
%   功能: 1. 将问题二的所有功能（优化、计算、可视化）整合到此文件中。
%         2. 使用遗传算法(GA)寻找最优策略。
%         3. 对找到的最优策略进行深度分析与多图可视化展示。
% =========================================================================

%% 1. 主程序脚本 (Main Script)
% =========================================================================
clear; clc; close all;

fprintf('问题二：最优策略求解程序已启动...\n');

% --- 定义优化问题 ---
n_vars = 4; % 决策变量个数: [v_F, theta_F, t_launch, t_fuse]
lb = [ 70,   0,  0.1,  0.1]; % 下界
ub = [140, 2*pi, 40.0, 19.0]; % 上界

% --- 配置并运行遗传算法 ---
fitness_fcn = @(X) -fitness_function(X); % GA求最小值，故对目标函数取负
options = optimoptions('ga', ...
    'PopulationSize', 1200, ...
    'MaxGenerations', 50, ...
    'Display', 'iter', ...
    'PlotFcn', {@gaplotbestf, @gaplotstopping}, ...
    'UseParallel', false); % 开启并行计算以加速

[best_X, max_shielding_time_neg] = ga(fitness_fcn, n_vars, [], [], [], [], lb, ub, [], options);
max_shielding_time = -max_shielding_time_neg;

% --- 输出并可视化最优结果 ---
fprintf('\n==================== 问题二 最优策略求解结果 ====================\n');
fprintf('遗传算法已收敛，找到的最优策略如下:\n');
fprintf('  - 无人机飞行速度 (v_F):      %.2f m/s\n', best_X(1));
fprintf('  - 无人机飞行方向 (theta_F):  %.2f 度\n', rad2deg(best_X(2)));
fprintf('  - 烟幕弹投放时刻 (t_launch): %.2f s\n', best_X(3));
fprintf('  - 烟幕弹引信时长 (t_fuse):   %.2f s\n', best_X(4));
fprintf('------------------------------------------------------------------\n');
fprintf('  在此最优策略下，可实现的最大有效遮蔽时长为: %.4f s\n', max_shielding_time);
fprintf('==================================================================\n\n');

fprintf('正在为最优策略生成高质量可视化结果...\n');
visualize_optimal_strategy(best_X, max_shielding_time);

fprintf('程序运行完毕。\n');


%% ========================================================================
%                      本地函数定义区域 (Local Functions)
% =========================================================================

%% 函数 1: 适应度函数 (Fitness Function for GA)
function total_shielding_time = fitness_function(X)
% 计算给定策略X下的有效遮蔽时长 (为GA优化提供快速计算)
    
    % --- 解码策略向量 ---
    v_F = X(1); theta_F = X(2); t_launch = X(3); t_fuse = X(4);

    % --- 定义常量 ---
    g = 9.8; P_target = [0, 0, 0]; R_target = 7; H_target = 10;
    P_T_bottom_center = [0, 200, 0];
    P_M1_0 = [20000, 0, 2000]; v_M1 = 300;
    P_F1_0 = [17800, 0, 1800];
    R_c = 10; T_smoke = 20; v_sink = 3; dt = 0.01;

    % --- 建立模型 ---
    d_M1 = P_target - P_M1_0; u_M1 = d_M1 / norm(d_M1);
    P_M1 = @(t) P_M1_0 + v_M1 * (t' * u_M1);
    u_F1_h = [cos(theta_F), sin(theta_F), 0];
    P_F1 = @(t) P_F1_0 + v_F * (t' * u_F1_h);
    P_launch = P_F1(t_launch); V_launch = v_F * u_F1_h;
    t_exp = t_launch + t_fuse;
    P_exp = P_launch + V_launch * t_fuse + [0, 0, -0.5 * g * t_fuse^2];
    P_c = @(t) P_exp + (t' - t_exp) * [0, 0, -v_sink];

    % --- 建立目标轮廓 ---
    target_points = get_target_points(P_T_bottom_center, R_target, H_target);
    
    % --- 仿真计算 ---
    t_start_sim = t_exp; t_end_sim = t_exp + T_smoke;
    if t_start_sim >= t_end_sim, total_shielding_time = 0; return; end
    time_vector = t_start_sim : dt : t_end_sim;
    
    is_shielded_flag = false(1, length(time_vector));
    for i = 1:length(time_vector)
        is_shielded_flag(i) = isTargetFullyShielded(P_M1(time_vector(i)), P_c(time_vector(i)), target_points, R_c);
    end
    
    total_shielding_time = sum(is_shielded_flag) * dt;
end


%% 函数 2: 可视化函数
function visualize_optimal_strategy(optimal_X, max_shielding_time)
% 对找到的最优策略进行深度仿真与多图可视化
    
    % --- 重新进行一次完整的、带数据记录的仿真 ---
    [history] = run_detailed_simulation(optimal_X);

    % --- 解包历史数据 ---
    time_vector = history.time_vector;
    is_shielded_flag = history.is_shielded_flag;
    dist_center_hist = history.dist_center_hist;
    dist_top_hist = history.dist_top_hist;
    dist_bottom_edge_hist = history.dist_bottom_edge_hist;
    t_min_dist = history.t_min_dist;
    
    % --- 获取模型句柄和常量 ---
    models = get_models_and_constants(optimal_X);
    P_M1 = models.P_M1;
    P_c = models.P_c;
    P_T_bottom_center = models.P_T_bottom_center;
    R_target = models.R_target;
    H_target = models.H_target;
    R_c = models.R_c;
    P_T_center = models.P_T_center;
    
    % --- 图1: 全局对抗场景图 ---
    figure('Name', '最优策略图1：全局对抗场景', 'NumberTitle', 'off', 'Position', [50, 200, 800, 600]);
    ax1 = axes; hold(ax1, 'on');
    t_vis = time_vector(1):0.2:time_vector(end); 
    pos_M1_vis = P_M1(t_vis); pos_cloud_vis = P_c(t_vis);
    plot3(ax1, pos_M1_vis(:,1), pos_M1_vis(:,2), pos_M1_vis(:,3), 'r-', 'LineWidth', 3, 'DisplayName', 'M1 轨迹');
    plot3(ax1, pos_cloud_vis(:,1), pos_cloud_vis(:,2), pos_cloud_vis(:,3), 'g--', 'LineWidth', 2, 'DisplayName', '烟幕中心轨迹');
    [Xc, Yc, Zc] = createCylinder(P_T_bottom_center, R_target, H_target);
    surf(ax1, Xc, Yc, Zc, 'FaceColor', [0.2 0.5 0.8], 'EdgeColor', 'none', 'DisplayName', '真目标');
    [Xs, Ys, Zs] = createSphere(P_c(t_min_dist), R_c);
    surf(ax1, Xs, Ys, Zs, 'FaceColor', [0.5 0.8 0.5], 'EdgeColor', 'none', 'FaceAlpha', 0.4, 'DisplayName', '烟幕云团');
    axis(ax1, 'equal'); grid(ax1, 'on'); view(ax1, -70, 25); camproj(ax1, 'perspective');
    title(ax1, sprintf('图1：最优策略全局场景 (最大遮蔽: %.2fs)', max_shielding_time), 'FontSize', 16);
    xlabel(ax1,'X (m)'); ylabel(ax1,'Y (m)'); zlabel(ax1,'Z (m)');
    legend(ax1, 'Location', 'northwest'); lightangle(ax1, -45, 30); lighting(ax1, 'gouraud');

    % --- 图2: 目标区域交互特写图 ---
    figure('Name', '最优策略图2：目标区域交互特写', 'NumberTitle', 'off', 'Position', [900, 200, 800, 600]);
    ax2 = axes; hold(ax2, 'on');
    h_target2 = surf(ax2, Xc, Yc, Zc, 'FaceColor', [0.2 0.5 0.8], 'EdgeColor', 'none', 'DisplayName', '真目标');
    plot3(ax2, pos_cloud_vis(:,1), pos_cloud_vis(:,2), pos_cloud_vis(:,3), 'g--', 'LineWidth', 2, 'DisplayName', '烟幕中心轨迹');
    P_M1_min = P_M1(t_min_dist); P_c_min = P_c(t_min_dist);
    [Xs2, Ys2, Zs2] = createSphere(P_c_min, R_c);
    h_smoke2 = surf(ax2, Xs2, Ys2, Zs2, 'FaceColor', [0.5 0.8 0.5], 'EdgeColor', 'none', 'FaceAlpha', 0.3, 'DisplayName', '烟幕云团');
    point_top = P_T_bottom_center + [0,0,H_target];
    point_bottom_edge = P_T_bottom_center + [R_target, 0, 0];
    plot3(ax2, [P_M1_min(1) P_T_center(1)], [P_M1_min(2) P_T_center(2)], [P_M1_min(3) P_T_center(3)], 'k:', 'DisplayName', 'LOS (中心)');
    plot3(ax2, [P_M1_min(1) point_top(1)], [P_M1_min(2) point_top(2)], [P_M1_min(3) point_top(3)], 'm:', 'DisplayName', 'LOS (顶部)');
    plot3(ax2, [P_M1_min(1) point_bottom_edge(1)], [P_M1_min(2) point_bottom_edge(2)], [P_M1_min(3) point_bottom_edge(3)], 'c:', 'DisplayName', 'LOS (底边)');
    axis(ax2, 'equal'); grid(ax2, 'on');
    xlim(ax2, [P_T_center(1)-50, P_T_center(1)+50]); ylim(ax2, [P_T_center(2)-50, P_T_center(2)+50]); zlim(ax2, [P_T_center(3)-30, P_T_center(3)+30]);
    view(ax2, -110, 15); camproj(ax2, 'perspective');
    title(ax2, '图2：最优策略目标区特写', 'FontSize', 16);
    xlabel(ax2,'X (m)'); ylabel(ax2,'Y (m)'); zlabel(ax2,'Z (m)');
    legend(ax2, 'Location', 'northeast'); lightangle(ax2, -45, 30); lighting(ax2, 'gouraud');
    material(h_target2, 'shiny'); material(h_smoke2, 'dull');

    % --- 图3: 多点距离分析图 ---
    figure('Name', '最优策略图3：多点距离分析', 'NumberTitle', 'off', 'Position', [450, 50, 900, 500]);
    ax3 = axes; hold(ax3, 'on');
    plot(ax3, time_vector, dist_center_hist, 'k-', 'LineWidth', 1.5, 'DisplayName', '距LOS(中心)');
    plot(ax3, time_vector, dist_top_hist, 'm--', 'LineWidth', 1.5, 'DisplayName', '距LOS(顶部)');
    plot(ax3, time_vector, dist_bottom_edge_hist, 'c-.', 'LineWidth', 1.5, 'DisplayName', '距LOS(底边)');
    yline(ax3, R_c, 'r-', 'LineWidth', 2, 'DisplayName', '有效遮蔽半径(10m)');
    shielded_indices = find(is_shielded_flag);
    if ~isempty(shielded_indices)
        t_shielded = time_vector(shielded_indices);
        fill_area_y = R_c * ones(size(t_shielded));
        fill(ax3, [t_shielded, fliplr(t_shielded)], [fill_area_y*0, fliplr(fill_area_y)], [0.2 0.8 0.2], 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'DisplayName', '完全遮蔽区间');
    end
    grid(ax3, 'on'); box(ax3, 'on');
    title(ax3, sprintf('图3：最优策略距离分析 (总遮蔽: %.2fs)', max_shielding_time), 'FontSize', 16);
    xlabel(ax3, '时间 (s)', 'FontSize', 12); ylabel(ax3, '距离 (m)', 'FontSize', 12);
    legend(ax3, 'Location', 'best'); ylim(ax3, [0, max(dist_top_hist)*1.1]);
end


%% 函数 3: 详细仿真函数 (供可视化调用)
function [history] = run_detailed_simulation(X)
% 运行一次完整的仿真并记录详细历史数据
    
    % --- 获取模型和常量 ---
    models = get_models_and_constants(X);
    P_M1 = models.P_M1;
    P_c = models.P_c;
    target_points = models.target_points;
    R_c = models.R_c;
    dt = models.dt;
    
    % --- 选取几个有代表性的点用于数据分析图 ---
    point_center = models.P_T_center;
    point_top = models.P_T_bottom_center + [0,0,models.H_target];
    point_bottom_edge = models.P_T_bottom_center + [models.R_target, 0, 0];
    
    % --- 仿真时间 ---
    t_exp = X(3) + X(4);
    t_start_sim = t_exp;
    t_end_sim = t_exp + models.T_smoke;
    time_vector = t_start_sim : dt : t_end_sim;
    
    % --- 初始化历史记录 ---
    num_steps = length(time_vector);
    is_shielded_flag = false(1, num_steps);
    dist_center_hist = zeros(1, num_steps);
    dist_top_hist = zeros(1, num_steps);
    dist_bottom_edge_hist = zeros(1, num_steps);
    min_dist_overall = inf; t_min_dist = t_start_sim;
    
    % --- 仿真循环 ---
    for i = 1:num_steps
        t = time_vector(i);
        pos_M1 = P_M1(t); pos_Cloud = P_c(t);
        is_shielded_flag(i) = isTargetFullyShielded(pos_M1, pos_Cloud, target_points, R_c);
        dist_center_hist(i) = getDistanceToLOS(pos_M1, pos_Cloud, point_center);
        dist_top_hist(i) = getDistanceToLOS(pos_M1, pos_Cloud, point_top);
        dist_bottom_edge_hist(i) = getDistanceToLOS(pos_M1, pos_Cloud, point_bottom_edge);
        if dist_center_hist(i) < min_dist_overall
            min_dist_overall = dist_center_hist(i); t_min_dist = t;
        end
    end
    
    % --- 打包历史数据 ---
    history.time_vector = time_vector;
    history.is_shielded_flag = is_shielded_flag;
    history.dist_center_hist = dist_center_hist;
    history.dist_top_hist = dist_top_hist;
    history.dist_bottom_edge_hist = dist_bottom_edge_hist;
    history.t_min_dist = t_min_dist;
end


%% ========================================================================
%                      通用辅助函数区域 (Helper Functions)
% =========================================================================

function models = get_models_and_constants(X)
% 一个集中的地方来生成所有模型和常量，避免重复
    v_F = X(1); theta_F = X(2); t_launch = X(3); t_fuse = X(4);
    
    models.g = 9.8; models.P_target = [0, 0, 0]; models.R_target = 7; models.H_target = 10;
    models.P_T_bottom_center = [0, 200, 0];
    models.P_T_center = models.P_T_bottom_center + [0, 0, models.H_target/2];
    models.P_M1_0 = [20000, 0, 2000]; models.v_M1 = 300;
    models.P_F1_0 = [17800, 0, 1800];
    models.R_c = 10; models.T_smoke = 20; models.v_sink = 3; models.dt = 0.01;

    d_M1 = models.P_target - models.P_M1_0; u_M1 = d_M1 / norm(d_M1);
    models.P_M1 = @(t) models.P_M1_0 + models.v_M1 * (t' * u_M1);
    
    u_F1_h = [cos(theta_F), sin(theta_F), 0];
    P_F1 = @(t) models.P_F1_0 + v_F * (t' * u_F1_h);
    P_launch = P_F1(t_launch); V_launch = v_F * u_F1_h;
    t_exp = t_launch + t_fuse;
    P_exp = P_launch + V_launch * t_fuse + [0, 0, -0.5 * models.g * t_fuse^2];
    models.P_c = @(t) P_exp + (t' - t_exp) * [0, 0, -models.v_sink];
    
    models.target_points = get_target_points(models.P_T_bottom_center, models.R_target, models.H_target);
end

function target_points = get_target_points(P_T_bottom_center, R_target, H_target)
    num_edge_points = 8;
    theta = linspace(0, 2*pi, num_edge_points + 1); theta(end) = [];
    edge_x = R_target * cos(theta) + P_T_bottom_center(1);
    edge_y = R_target * sin(theta) + P_T_bottom_center(2);
    target_points = [
        P_T_bottom_center; P_T_bottom_center + [0,0,H_target];
        [edge_x', edge_y', repmat(P_T_bottom_center(3), num_edge_points, 1)];
        [edge_x', edge_y', repmat(P_T_bottom_center(3)+H_target, num_edge_points, 1)];
    ];
end

function dist = getDistanceToLOS(pos_M1, pos_Cloud, pos_Point)
    vec_AP = pos_Point - pos_M1; vec_AC = pos_Cloud - pos_M1;
    if norm(vec_AP) < 1e-6, dist = norm(vec_AC); return; end % Avoid division by zero
    proj_len = dot(vec_AC, vec_AP) / norm(vec_AP);
    if proj_len < 0 || proj_len > norm(vec_AP)
        dist = min(norm(vec_AC), norm(pos_Cloud - pos_Point));
    else
        dist = norm(cross(vec_AP, vec_AC)) / norm(vec_AP);
    end
end

function shielded = isPointShielded(pos_M1, pos_Cloud, pos_Point, R_c)
    shielded = getDistanceToLOS(pos_M1, pos_Cloud, pos_Point) <= R_c;
end

function fully_shielded = isTargetFullyShielded(pos_M1, pos_Cloud, all_target_points, R_c)
    fully_shielded = true;
    for i = 1:size(all_target_points, 1)
        if ~isPointShielded(pos_M1, pos_Cloud, all_target_points(i, :), R_c)
            fully_shielded = false; return;
        end
    end
end

function [X, Y, Z] = createCylinder(center_bottom, R, H)
    [Xc, Yc, Zc] = cylinder(R, 30);
    X = Xc + center_bottom(1); Y = Yc + center_bottom(2); Z = Zc * H + center_bottom(3);
end

function [X, Y, Z] = createSphere(center, R)
    [Xs, Ys, Zs] = sphere(20);
    X = Xs * R + center(1); Y = Ys * R + center(2); Z = Zs * R + center(3);
end