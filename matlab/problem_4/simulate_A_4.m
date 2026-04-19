%% ========================================================================
%                     2025年全国大学生数学建模竞赛 A题
%               问题四：终极强化版 - 多起点混合算法求解
%
%   功能: 1. 采用多起点(Multi-Start)策略，多次独立运行混合优化器。
%         2. 第一阶段：使用超强参数的PSO进行全局探索。
%         3. 第二阶段：使用增强配置的fmincon进行局部精细优化。
%         4. 从多次运行结果中选取历史最优解，确保结果的稳定性和质量。
% =========================================================================

%% 1. 主程序脚本 (Main Script for Q4 - Ultimate Enhanced Version)
% =========================================================================
clear; clc; close all;

fprintf('问题四：多起点混合算法求解程序已启动...\n');

% --- 定义共享的优化问题参数 ---
n_vars = 12;
P_M1_0 = [20000, 0, 2000]; v_M1 = 300;
M1_TTI = norm(P_M1_0) / v_M1;
% 稍微放宽时间上限，给算法更多空间
t_launch_ub = max(15.0, M1_TTI - 10.0); 
fprintf('导弹M1预计到达时间: %.2fs, 投放时间上限设为: %.2fs\n', M1_TTI, t_launch_ub);

lb = [ 70,   0,  0.1,  0.1,  70,   0,  0.1,  0.1,  70,   0,  0.1,  0.1];
ub = [140, 2*pi, t_launch_ub, 20.0, 140, 2*pi, t_launch_ub, 20.0, 140, 2*pi, t_launch_ub, 20.0];
fitness_fcn = @(X) -fitness_function_q4(X);

% --- 配置终极强化版优化器 ---

% 1. PSO 全局搜索器 (更激进的参数)
options_pso = optimoptions('particleswarm', ...
    'SwarmSize', 1000, ...         % 极大增加粒子数
    'MaxIterations', 400, ...      % 极大增加迭代次数
    'Display', 'iter', ...
    'PlotFcn', @pswplotbestf, ...
    'UseParallel', true, ...
    'SocialAdjustmentWeight', 2.0, ... % 进一步强调全局最优
    'SelfAdjustmentWeight', 1.0, ...  % 进一步降低自我认知
    'InertiaRange', [0.1, 1.1]);      % 使用动态惯性权重，前期探索，后期收敛

% 2. fmincon 局部精调器 (更充分的评估)
options_local = optimoptions('fmincon', ...
    'Display', 'iter', ...
    'Algorithm', 'sqp', ...
    'MaxFunctionEvaluations', 30000, ... % 给予极高的函数评估次数预算
    'StepTolerance', 1e-8, ...         % 更高的精度要求
    'OptimalityTolerance', 1e-8);

% --- 实施多起点 (Multi-Start) 策略 ---
num_starts = 50; % 定义独立运行的总次数，可以设为3, 5或更高
historical_best_X = [];
historical_best_fitness = -Inf; % 记录历史最优适应度 (此处为真实时长)

fprintf('\n==================================================================\n');
fprintf('即将开始 %d 次独立的多起点优化运行...\n', num_starts);
fprintf('==================================================================\n');

for i = 1:num_starts
    fprintf('\n----------- 开始第 %d / %d 次优化运行 -----------\n', i, num_starts);
    
    % --- 阶段一: PSO 全局搜索 ---
    fprintf('--> 阶段 1: PSO 全局搜索 (运行 %d)...\n', i);
    [best_X_pso, fval_pso_neg] = particleswarm(fitness_fcn, n_vars, lb, ub, options_pso);
    fprintf('--> PSO 阶段完成，找到的初步最优时长: %.4f s\n', -fval_pso_neg);

    % --- 阶段二: fmincon 局部精调 ---
    fprintf('\n--> 阶段 2: fmincon 局部精调 (运行 %d)...\n', i);
    [run_best_X, fval_final_neg] = fmincon(fitness_fcn, best_X_pso, [], [], [], [], lb, ub, [], options_local);
    run_best_fitness = -fval_final_neg;
    
    fprintf('--> 第 %d 次运行完成，最终时长: %.4f s\n', i, run_best_fitness);
    
    % --- 更新历史最优解 ---
    if run_best_fitness > historical_best_fitness
        fprintf('!!! 发现新的全局最优解！ 时长从 %.4f s 提升至 %.4f s !!!\n', ...
            historical_best_fitness, run_best_fitness);
        historical_best_fitness = run_best_fitness;
        historical_best_X = run_best_X;
    end
end

% --- 使用最终的历史最优解 ---
best_X = historical_best_X;
max_shielding_time = historical_best_fitness;

fprintf('\n==================== 问题四 最终全局最优策略 ====================\n');
fprintf('在 %d 次独立运行后，找到的最佳策略如下:\n', num_starts);
fprintf('  最大有效遮蔽总时长为: %.4f s\n', max_shielding_time);
fprintf('------------------------------------------------------------------\n');
drone_names = {'FY1', 'FY2', 'FY3'};
for i = 1:3
    base_idx = (i-1)*4;
    fprintf('无人机 %s 策略:\n', drone_names{i});
    fprintf('  - 飞行速度:      %.2f m/s\n', best_X(base_idx + 1));
    fprintf('  - 飞行方向:      %.2f 度\n', rad2deg(best_X(base_idx + 2)));
    fprintf('  - 投放时刻:      %.2f s\n', best_X(base_idx + 3));
    fprintf('  - 引信时长:        %.2f s\n', best_X(base_idx + 4));
end
fprintf('==================================================================\n\n');

% --- 保存和可视化最终的最优结果 ---
save_results_to_excel_q4(best_X, max_shielding_time, 'result2.xlsx');
visualize_optimal_strategy_q4(best_X, max_shielding_time);

fprintf('程序运行完毕。\n');

% [注意：下面的本地函数部分 (fitness_function_q4, save_results_to_excel_q4 等) 保持不变]

%% ========================================================================
%                      本地函数定义区域 (Local Functions)
% =========================================================================

%% 函数 1: 适应度函数 (Fitness Function for Q4)
function [total_union_time, intervals] = fitness_function_q4(X)
% 计算给定三机协同策略X下的有效遮蔽时长并集
    
    % --- 定义常量与基础模型 ---
    g = 9.8; P_target = [0, 0, 0]; R_target = 7; H_target = 10;
    P_T_bottom_center = [0, 200, 0];
    P_M1_0 = [20000, 0, 2000]; v_M1 = 300;
    R_c = 10; T_smoke = 20; v_sink = 3; 

    % 三架无人机的初始位置
    P_F_0s = [17800,    0, 1800;   % FY1
              12000, 1400, 1400;   % FY2
               6000,-3000,  700];  % FY3
    
    dt = (nargout <= 1) * 0.05 + (nargout > 1) * 0.01; % 优化时步长大，可视化时步长小
    
    d_M1 = P_target - P_M1_0; u_M1 = d_M1 / norm(d_M1);
    P_M1 = @(t) P_M1_0 + v_M1 * (t' * u_M1);
    target_points = get_target_points(P_T_bottom_center, R_target, H_target);
    
    intervals = zeros(3, 2); % 初始化存储每枚弹的 [t_start, t_end]

    % --- 对每架无人机及其弹药进行仿真计算 ---
    for i = 1:3
        % --- 解码当前无人机的策略 ---
        base_idx = (i-1)*4;
        v_F      = X(base_idx + 1);
        theta_F  = X(base_idx + 2);
        t_launch = X(base_idx + 3);
        t_fuse   = X(base_idx + 4);

        % --- 建立该无人机的轨迹和速度模型 ---
        P_F_0 = P_F_0s(i, :);
        u_F_h = [cos(theta_F), sin(theta_F), 0];
        P_F = @(t) P_F_0 + v_F * (t' * u_F_h);
        V_F = v_F * u_F_h;

        % --- 计算投放点、起爆点和烟幕云团轨迹 ---
        P_launch = P_F(t_launch);
        t_exp = t_launch + t_fuse;
        P_exp = P_launch + V_F * t_fuse + [0, 0, -0.5 * g * t_fuse^2];
        P_c = @(t) P_exp + (t' - t_exp) * [0, 0, -v_sink];

        % --- 仿真遮蔽效果 ---
        t_start_sim = t_exp; t_end_sim = t_exp + T_smoke;
        time_vector = t_start_sim : dt : t_end_sim;
        
        if isempty(time_vector), continue; end
        
        is_shielded_flag = false(1, length(time_vector));
        for j = 1:length(time_vector)
            t = time_vector(j);
            is_shielded_flag(j) = isTargetFullyShielded(P_M1(t), P_c(t), target_points, R_c);
        end
        
        shielded_indices = find(is_shielded_flag);
        if ~isempty(shielded_indices)
            intervals(i, 1) = time_vector(shielded_indices(1));
            intervals(i, 2) = time_vector(shielded_indices(end));
        end
    end
    
    % --- 计算时间区间的并集总长度 ---
    total_union_time = calculate_union_of_intervals(intervals);
end

%% 函数 2: 结果保存函数 (Save Results for Q4)
function save_results_to_excel_q4(best_X, max_shielding_time, filename)
    fprintf('正在将最优结果保存到 %s ...\n', filename);

    g = 9.8;
    P_F_0s = [17800,    0, 1800;   % FY1
              12000, 1400, 1400;   % FY2
               6000,-3000,  700];  % FY3
    
    % 初始化存储结果的数组
    directions_deg = zeros(3, 1);
    speeds = zeros(3, 1);
    launch_coords = zeros(3, 3);
    exp_coords = zeros(3, 3);

    % 循环计算每架无人机的物理坐标
    for i = 1:3
        % 解码当前无人机的最优策略
        base_idx = (i-1)*4;
        v_F      = best_X(base_idx + 1);
        theta_F  = best_X(base_idx + 2);
        t_launch = best_X(base_idx + 3);
        t_fuse   = best_X(base_idx + 4);

        speeds(i) = v_F;
        directions_deg(i) = rad2deg(theta_F);

        % 计算轨迹和坐标
        P_F_0 = P_F_0s(i, :);
        u_F_h = [cos(theta_F), sin(theta_F), 0];
        P_F = @(t) P_F_0 + v_F * (t' * u_F_h);
        V_F = v_F * u_F_h;

        launch_coords(i, :) = P_F(t_launch);
        exp_coords(i, :) = launch_coords(i, :) + V_F * t_fuse + [0, 0, -0.5 * g * t_fuse^2];
    end

    % --- 按照题目格式构建最终的输出表格 ---
    UAV_ID = {'FY1'; 'FY2'; 'FY3'};
    Total_Time = [max_shielding_time; NaN; NaN]; % 总时长只在第一行显示

    T = table(UAV_ID, directions_deg, speeds, ...
              launch_coords(:,1), launch_coords(:,2), launch_coords(:,3), ...
              exp_coords(:,1), exp_coords(:,2), exp_coords(:,3), ...
              Total_Time);
              
    T.Properties.VariableNames = { ...
        '无人机编号', '无人机运动方向', '无人机运动速度(m/s)', ...
        '烟幕干扰弹投放点的x坐标(m)', '烟幕干扰弹投放点的y坐标(m)', '烟幕干扰弹投放点的z坐标(m)', ...
        '烟幕干扰弹起爆点的x坐标(m)', '烟幕干扰弹起爆点的y坐标(m)', '烟幕干扰弹起爆点的z坐标(m)', ...
        '有效干扰时长(s)'};

    % --- 写入文件 ---
    try
        writetable(T, filename, 'Sheet', 1);
        fprintf('结果已成功保存到 %s，格式符合题目要求。\n', filename);
    catch ME
        fprintf('错误：无法写入Excel文件。请确保没有打开同名文件。\n');
        disp(ME.message);
    end
end

%% 函数 3: 可视化函数 (Visualize for Q4)
function visualize_optimal_strategy_q4(optimal_X, max_shielding_time)
% 对找到的最优三机协同策略进行深度仿真与可视化
    
    % 调用 fitness_function_q4 并请求两个输出以获取详细区间
    [~, intervals] = fitness_function_q4(optimal_X);

    figure('Name', '问题四最优策略：时间轴遮蔽效果', 'NumberTitle', 'off', 'Position', [400, 400, 900, 400]);
    ax1 = axes; hold(ax1, 'on');
    
    colors = [0.8 0.2 0.2; 0.2 0.8 0.2; 0.2 0.2 0.8];
    drone_names = {'FY1', 'FY2', 'FY3'};

    for i = 1:3
        if intervals(i,2) > intervals(i,1)
            t_start = intervals(i,1);
            t_end = intervals(i,2);
            plot(ax1, [t_start, t_end], [i, i], 'LineWidth', 20, 'Color', colors(i,:));
        end
    end
    
    merged_intervals = calculate_union_of_intervals_verbose(intervals);
    if ~isempty(merged_intervals)
        for i = 1:size(merged_intervals, 1)
            t_start = merged_intervals(i,1);
            t_end = merged_intervals(i,2);
            fill(ax1, [t_start, t_end, t_end, t_start], [0.4, 0.4, 4, 4], [0.5 0.8 0.5], ...
                'FaceAlpha', 0.2, 'EdgeColor', 'none', 'DisplayName', '总有效遮蔽');
        end
    end

    ylim(ax1, [0.5, 3.5]);
    yticks(ax1, 1:3);
    yticklabels(ax1, drone_names);
    grid(ax1, 'on'); box(ax1, 'on');
    title(ax1, sprintf('图1：最优协同策略时间轴分析 (总遮蔽: %.2fs)', max_shielding_time), 'FontSize', 16);
    xlabel(ax1, '时间 (s)', 'FontSize', 12);
    ylabel(ax1, '无人机编号', 'FontSize', 12);
    legend(ax1, 'Location', 'bestoutside');
end

%% ========================================================================
%                      通用辅助函数区域 (Helper Functions)
%        (这些函数与问题三版本完全相同，无需修改)
% =========================================================================

function union_len = calculate_union_of_intervals(intervals)
    intervals = intervals(intervals(:,1) > 0 & intervals(:,2) > intervals(:,1), :);
    if isempty(intervals), union_len = 0; return; end
    intervals = sortrows(intervals, 1);
    merged = [];
    if ~isempty(intervals)
        current_merge = intervals(1,:);
        for i = 2:size(intervals, 1)
            next_interval = intervals(i,:);
            if next_interval(1) <= current_merge(2)
                current_merge(2) = max(current_merge(2), next_interval(2));
            else
                merged = [merged; current_merge];
                current_merge = next_interval;
            end
        end
        merged = [merged; current_merge];
    end
    union_len = sum(merged(:,2) - merged(:,1));
end

function merged = calculate_union_of_intervals_verbose(intervals)
    intervals = intervals(intervals(:,1) > 0 & intervals(:,2) > intervals(:,1), :);
    if isempty(intervals), merged = []; return; end
    intervals = sortrows(intervals, 1);
    merged = [];
    if ~isempty(intervals)
        current_merge = intervals(1,:);
        for i = 2:size(intervals, 1)
            next_interval = intervals(i,:);
            if next_interval(1) <= current_merge(2)
                current_merge(2) = max(current_merge(2), next_interval(2));
            else
                merged = [merged; current_merge];
                current_merge = next_interval;
            end
        end
        merged = [merged; current_merge];
    end
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

function fully_shielded = isTargetFullyShielded(pos_M1, pos_Cloud, all_target_points, R_c)
    fully_shielded = true;
    for i = 1:size(all_target_points, 1)
        if getDistanceToLOS(pos_M1, pos_Cloud, all_target_points(i, :)) > R_c
            fully_shielded = false; return;
        end
    end
end

function dist = getDistanceToLOS(pos_M1, pos_Cloud, pos_Point)
    vec_AP = pos_Point - pos_M1; vec_AC = pos_Cloud - pos_M1;
    norm_AP = norm(vec_AP);
    if norm_AP < 1e-6, dist = norm(vec_AC); return; end
    proj_len = dot(vec_AC, vec_AP) / norm_AP;
    if proj_len < 0, dist = norm(vec_AC);
    elseif proj_len > norm_AP, dist = norm(pos_Cloud - pos_Point);
    else
        dist = norm(cross(vec_AP, vec_AC)) / norm_AP;
    end
end