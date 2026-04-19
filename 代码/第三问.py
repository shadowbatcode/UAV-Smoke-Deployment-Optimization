import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize
import time
import warnings
warnings.filterwarnings('ignore')

# ========================================================================
#                     2025年全国大学生数学建模竞赛 A题
#               问题三：单一文件完整解决方案 (优化与结果输出)
#
#   功能: 1. 使用遗传算法(GA)寻找针对M1的最优三弹投放策略。
#         2. 目标函数为最大化三个遮蔽时间区间的并集总长度。
#         3. 包含发射间隔 >= 1s 的约束。
#         4. 将最优结果保存到 result1.xlsx。
#         5. 对找到的最优策略进行深度分析与多图可视化展示。
# =========================================================================

# 定义常量
g = 9.8
P_target = np.array([0, 0, 0])
R_target = 7
H_target = 10
P_T_bottom_center = np.array([0, 200, 0])
P_M1_0 = np.array([20000, 0, 2000])
v_M1 = 300
P_F1_0 = np.array([17800, 0, 1800])
R_c = 10
T_smoke = 20
v_sink = 3

# 辅助函数
def get_target_points(P_T_bottom_center, R_target, H_target):
    num_edge_points = 8
    theta = np.linspace(0, 2*np.pi, num_edge_points + 1)[:-1]
    edge_x = R_target * np.cos(theta) + P_T_bottom_center[0]
    edge_y = R_target * np.sin(theta) + P_T_bottom_center[1]
    target_points = np.vstack([
        P_T_bottom_center,
        P_T_bottom_center + [0, 0, H_target],
        np.column_stack([edge_x, edge_y, np.full(num_edge_points, P_T_bottom_center[2])]),
        np.column_stack([edge_x, edge_y, np.full(num_edge_points, P_T_bottom_center[2] + H_target)])
    ])
    return target_points

def getDistanceToLOS(pos_M1, pos_Cloud, pos_Point):
    vec_AP = pos_Point - pos_M1
    vec_AC = pos_Cloud - pos_M1
    if np.linalg.norm(vec_AP) < 1e-6:
        return np.linalg.norm(vec_AC)
    proj_len = np.dot(vec_AC, vec_AP) / np.linalg.norm(vec_AP)
    if proj_len < 0 or proj_len > np.linalg.norm(vec_AP):
        return min(np.linalg.norm(vec_AC), np.linalg.norm(pos_Cloud - pos_Point))
    else:
        return np.linalg.norm(np.cross(vec_AP, vec_AC)) / np.linalg.norm(vec_AP)

def isPointShielded(pos_M1, pos_Cloud, pos_Point, R_c):
    return getDistanceToLOS(pos_M1, pos_Cloud, pos_Point) <= R_c

def isTargetFullyShielded(pos_M1, pos_Cloud, all_target_points, R_c):
    for i in range(len(all_target_points)):
        if not isPointShielded(pos_M1, pos_Cloud, all_target_points[i], R_c):
            return False
    return True

def calculate_union_of_intervals(intervals):
    intervals = intervals[intervals[:, 0] > 0]
    if len(intervals) == 0:
        return 0
    
    intervals = intervals[np.argsort(intervals[:, 0])]
    merged = []
    current_merge = intervals[0]
    
    for i in range(1, len(intervals)):
        next_interval = intervals[i]
        if next_interval[0] <= current_merge[1]:
            current_merge[1] = max(current_merge[1], next_interval[1])
        else:
            merged.append(current_merge)
            current_merge = next_interval
    
    merged.append(current_merge)
    merged = np.array(merged)
    return np.sum(merged[:, 1] - merged[:, 0])

def calculate_union_of_intervals_verbose(intervals):
    intervals = intervals[intervals[:, 0] > 0]
    if len(intervals) == 0:
        return np.array([])
    
    intervals = intervals[np.argsort(intervals[:, 0])]
    merged = []
    current_merge = intervals[0]
    
    for i in range(1, len(intervals)):
        next_interval = intervals[i]
        if next_interval[0] <= current_merge[1]:
            current_merge[1] = max(current_merge[1], next_interval[1])
        else:
            merged.append(current_merge)
            current_merge = next_interval
    
    merged.append(current_merge)
    return np.array(merged)

# 适应度函数
def fitness_function_q3(X, return_intervals=False):
    v_F, theta_F = X[0], X[1]
    t_launch_vec = np.array([X[2], X[4], X[6]])
    t_fuse_vec = np.array([X[3], X[5], X[7]])
    
    # 约束检查
    if t_launch_vec[1] < t_launch_vec[0] + 1 or t_launch_vec[2] < t_launch_vec[1] + 1:
        if return_intervals:
            return 0, np.zeros((3, 2))
        return 0
    
    # 定义模型
    d_M1 = P_target - P_M1_0
    u_M1 = d_M1 / np.linalg.norm(d_M1)
    P_M1 = lambda t: P_M1_0 + v_M1 * np.outer(t, u_M1)
    
    u_F1_h = np.array([np.cos(theta_F), np.sin(theta_F), 0])
    P_F1 = lambda t: P_F1_0 + v_F * np.outer(t, u_F1_h)
    V_F = v_F * u_F1_h
    
    target_points = get_target_points(P_T_bottom_center, R_target, H_target)
    
    intervals = np.zeros((3, 2))
    dt = 0.01 if return_intervals else 0.05
    
    for i in range(3):
        t_launch = t_launch_vec[i]
        t_fuse = t_fuse_vec[i]
        
        P_launch = P_F1(np.array([t_launch]))[0]
        t_exp = t_launch + t_fuse
        P_exp = P_launch + V_F * t_fuse + np.array([0, 0, -0.5 * g * t_fuse**2])
        P_c = lambda t: P_exp + (t - t_exp) * np.array([0, 0, -v_sink])
        
        t_start_sim = t_exp
        t_end_sim = t_exp + T_smoke
        
        if t_end_sim <= t_start_sim:
            continue
            
        time_vector = np.arange(t_start_sim, t_end_sim + dt, dt)
        if len(time_vector) == 0:
            continue
            
        is_shielded_flag = np.zeros(len(time_vector), dtype=bool)
        
        for j, t in enumerate(time_vector):
            M1_pos = P_M1(np.array([t]))[0]
            cloud_pos = P_c(t)
            is_shielded_flag[j] = isTargetFullyShielded(M1_pos, cloud_pos, target_points, R_c)
        
        shielded_indices = np.where(is_shielded_flag)[0]
        if len(shielded_indices) > 0:
            intervals[i, 0] = time_vector[shielded_indices[0]]
            intervals[i, 1] = time_vector[shielded_indices[-1]]
    
    total_union_time = calculate_union_of_intervals(intervals)
    
    if return_intervals:
        return total_union_time, intervals
    return total_union_time

# 保存结果函数
def save_results_to_excel(best_X, max_shielding_time, filename):
    v_F = best_X[0]
    theta_F_rad = best_X[1]
    t_launch_vec = [best_X[2], best_X[4], best_X[6]]
    t_fuse_vec = [best_X[3], best_X[5], best_X[7]]
    
    u_F1_h = np.array([np.cos(theta_F_rad), np.sin(theta_F_rad), 0])
    P_F1 = lambda t: P_F1_0 + v_F * t * u_F1_h
    V_F = v_F * u_F1_h
    
    P_launch_coords = np.zeros((3, 3))
    P_exp_coords = np.zeros((3, 3))
    
    for i in range(3):
        t_launch = t_launch_vec[i]
        P_launch_coords[i] = P_F1(t_launch)
        P_exp_coords[i] = P_launch_coords[i] + V_F * t_fuse_vec[i] + np.array([0, 0, -0.5 * g * t_fuse_vec[i]**2])
    
    theta_F_deg = np.degrees(theta_F_rad)
    
    # 创建数据
    data = {
        '无人机运动方向': [theta_F_deg, np.nan, np.nan],
        '无人机运动速度(m/s)': [v_F, np.nan, np.nan],
        '烟幕干扰弹编号': [1, 2, 3],
        '烟幕干扰弹投放点的x坐标(m)': P_launch_coords[:, 0],
        '烟幕干扰弹投放点的y坐标(m)': P_launch_coords[:, 1],
        '烟幕干扰弹投放点的z坐标(m)': P_launch_coords[:, 2],
        '烟幕干扰弹起爆点的x坐标(m)': P_exp_coords[:, 0],
        '烟幕干扰弹起爆点的y坐标(m)': P_exp_coords[:, 1],
        '烟幕干扰弹起爆点的z坐标(m)': P_exp_coords[:, 2],
        '有效干扰时长(s)': [max_shielding_time, np.nan, np.nan]
    }
    
    df = pd.DataFrame(data)
    df.to_excel(filename, index=False)
    print(f'结果已成功保存到 {filename}')

# 可视化函数
def visualize_optimal_strategy_q3(optimal_X, max_shielding_time):
    _, intervals = fitness_function_q3(optimal_X, return_intervals=True)
    
    print('\n=== 调试信息：各弹药遮蔽区间 ===')
    for i in range(3):
        if intervals[i, 1] > intervals[i, 0]:
            print(f'弹药{i+1}: [{intervals[i, 0]:.2f}, {intervals[i, 1]:.2f}] 秒 (时长: {intervals[i, 1]-intervals[i, 0]:.4f} s)')
        else:
            print(f'弹药{i+1}: 无有效遮蔽 (区间: [{intervals[i, 0]:.2f}, {intervals[i, 1]:.2f}])')
    print('================================\n')
    
    # 图1: 时间轴遮蔽效果图
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [(0.8, 0.2, 0.2), (0.2, 0.8, 0.2), (0.2, 0.2, 0.8)]
    
    for i in range(3):
        if intervals[i, 1] > intervals[i, 0]:
            t_start, t_end = intervals[i, 0], intervals[i, 1]
            duration = t_end - t_start
            ax.plot([t_start, t_end], [i+1, i+1], linewidth=25, color=colors[i], 
                   label=f'弹药{i+1} ({duration:.3f}s)')
            ax.text((t_start + t_end)/2, i+1 + 0.15, f'{duration:.3f}s', 
                   ha='center', fontsize=10, fontweight='bold')
        else:
            ax.plot([0, 0.1], [i+1, i+1], linewidth=5, color=(0.7, 0.7, 0.7), 
                   linestyle='--', label=f'弹药{i+1} (无效)')
            ax.text(0.05, i+1 + 0.15, '无效', ha='center', fontsize=9, color=(0.5, 0.5, 0.5))
    
    # 合并区间并绘制总效果
    merged_intervals = calculate_union_of_intervals_verbose(intervals)
    if len(merged_intervals) > 0:
        for interval in merged_intervals:
            t_start, t_end = interval
            ax.fill([t_start, t_end, t_end, t_start], [0.3, 0.3, 3.7, 3.7], 
                   color=(0.5, 0.8, 0.5), alpha=0.15, edgecolor=(0.3, 0.6, 0.3), 
                   linewidth=2, label='总有效遮蔽')
    
    ax.set_ylim(0.5, 3.8)
    min_time = min(0, np.min(intervals[:, 0]) - 1) if np.any(intervals[:, 0] > 0) else 0
    max_time = np.max(intervals[:, 1]) + 2 if np.any(intervals[:, 1] > 0) else 5
    ax.set_xlim(min_time, max_time)
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(['弹药 1', '弹药 2', '弹药 3'])
    ax.grid(True)
    ax.set_title(f'最优策略时间轴分析 (总遮蔽: {max_shielding_time:.4f}s)', fontsize=16)
    ax.set_xlabel('时间 (s)', fontsize=12)
    ax.set_ylabel('弹药编号', fontsize=12)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    plt.show()

# 模拟退火优化函数
def simulated_annealing_optimization(initial_X, lb, ub, fitness_fcn, max_iter):
    current_X = initial_X.copy()
    current_fitness = -fitness_fcn(current_X)
    
    best_X = current_X.copy()
    best_fitness = current_fitness
    
    initial_temp = 10.0
    final_temp = 0.01
    cooling_rate = (final_temp/initial_temp)**(1/(max_iter-1))
    temp = initial_temp
    
    step_sizes = [(ub - lb) * 0.1, (ub - lb) * 0.05, (ub - lb) * 0.02]
    
    print('模拟退火优化进度:')
    accept_count = 0
    improve_count = 0
    
    for iter in range(max_iter):
        if iter <= max_iter/3:
            step_size = step_sizes[0]
        elif iter <= 2*max_iter/3:
            step_size = step_sizes[1]
        else:
            step_size = step_sizes[2]
        
        candidate_X = current_X + (np.random.rand(len(current_X)) - 0.5) * step_size
        candidate_X = np.clip(candidate_X, lb, ub)
        
        # 特殊约束：确保投放时间递增且间隔>=1秒
        t_launches = [candidate_X[2], candidate_X[4], candidate_X[6]]
        t_launches_sorted = sorted(t_launches)
        
        if t_launches_sorted[1] < t_launches_sorted[0] + 1.0:
            t_launches_sorted[1] = t_launches_sorted[0] + 1.0
        if t_launches_sorted[2] < t_launches_sorted[1] + 1.0:
            t_launches_sorted[2] = t_launches_sorted[1] + 1.0
        
        candidate_X[2], candidate_X[4], candidate_X[6] = t_launches_sorted
        candidate_X = np.clip(candidate_X, lb, ub)
        
        candidate_fitness = -fitness_fcn(candidate_X)
        delta = candidate_fitness - current_fitness
        
        if delta > 0 or np.random.rand() < np.exp(delta / temp):
            current_X = candidate_X
            current_fitness = candidate_fitness
            accept_count += 1
            
            if candidate_fitness > best_fitness:
                best_X = candidate_X
                best_fitness = candidate_fitness
                improve_count += 1
                print(f'  第{iter}轮: 发现更好解 {best_fitness:.6f} s (温度: {temp:.4f})')
        
        temp *= cooling_rate
        
        if (iter + 1) % (max_iter // 10) == 0:
            progress = round(100 * (iter + 1) / max_iter)
            accept_rate = 100 * accept_count / (iter + 1)
            print(f'  进度: {progress}% | 最佳: {best_fitness:.6f} s | 当前: {current_fitness:.6f} s | 温度: {temp:.4f} | 接受率: {accept_rate:.1f}%')
    
    print(f'模拟退火完成: 共改进{improve_count}次, 总接受率{100*accept_count/max_iter:.1f}%')
    return best_X, best_fitness

# 主程序
def main():
    print('问题三：最优三弹投放策略求解程序已启动...\n')
    
    # 定义优化问题
    n_vars = 8
    
    # 智能约束：基于导弹M1的TTI计算合理的时间边界
    P_M1_0 = np.array([20000, 0, 2000])
    v_M1 = 300
    M1_TTI = np.linalg.norm(P_M1_0) / v_M1
    t_launch_ub = max(5.0, M1_TTI - 20.0)
    print(f'导弹M1预计到达时间: {M1_TTI:.2f}s, 投放时间上限设为: {t_launch_ub:.2f}s')
    
    # 下界和上界
    lb = [70, 0, 0.1, 0.1, 1.1, 0.1, 2.1, 0.1]
    ub = [140, 2*np.pi, t_launch_ub, 20.0, t_launch_ub+5, 20.0, t_launch_ub+8, 20.0]
    
    # 使用差分进化算法替代MATLAB的ga
    bounds = [(lb[i], ub[i]) for i in range(n_vars)]
    
    print('开始差分进化算法优化...')
    start_time = time.time()
    
    # 定义带约束的适应度函数
    def constrained_fitness(X):
        # 检查时间间隔约束
        if X[4] < X[2] + 1 or X[6] < X[4] + 1:
            return 1e10  # 返回一个很大的值作为惩罚
        return -fitness_function_q3(X)
    
    # 使用差分进化算法
    result = differential_evolution(
        constrained_fitness, 
        bounds, 
        strategy='best1bin',
        maxiter=150,
        popsize=800,
        tol=0.01,
        mutation=(0.5, 1),
        recombination=0.8,
        seed=42,
        disp=True,
        polish=False
    )
    
    best_X_de = result.x
    max_shielding_time_de = -result.fun
    de_time = time.time() - start_time
    
    print(f'\n==================== 差分进化算法阶段完成 ====================')
    print(f'优化时间: {de_time:.2f}秒')
    print(f'差分进化算法找到的解: {max_shielding_time_de:.4f} s')
    print('开始模拟退火进一步优化...')
    
    # 模拟退火进一步优化
    start_time = time.time()
    best_X_sa, max_shielding_time_sa = simulated_annealing_optimization(
        best_X_de, np.array(lb), np.array(ub), fitness_function_q3, 2000
    )
    sa_time = time.time() - start_time
    
    print(f'\n==================== 问题三 最终优化结果 ====================')
    print(f'差分进化结果: {max_shielding_time_de:.4f} s -> SA优化后: {max_shielding_time_sa:.4f} s (提升: {max_shielding_time_sa - max_shielding_time_de:.4f} s)')
    print(f'总优化时间: {de_time + sa_time:.2f}秒')
    print('最终最优策略如下:')
    print(f'  - 无人机飞行速度 (v_F):      {best_X_sa[0]:.2f} m/s')
    print(f'  - 无人机飞行方向 (theta_F):  {np.degrees(best_X_sa[1]):.2f} 度')
    print('-------------------- 弹药 1 --------------------')
    print(f'  - 投放时刻 (t_launch1):      {best_X_sa[2]:.2f} s')
    print(f'  - 引信时长 (t_fuse1):        {best_X_sa[3]:.2f} s')
    print('-------------------- 弹药 2 --------------------')
    print(f'  - 投放时刻 (t_launch2):      {best_X_sa[4]:.2f} s')
    print(f'  - 引信时长 (t_fuse2):        {best_X_sa[5]:.2f} s')
    print('-------------------- 弹药 3 --------------------')
    print(f'  - 投放时刻 (t_launch3):      {best_X_sa[6]:.2f} s')
    print(f'  - 引信时长 (t_fuse3):        {best_X_sa[7]:.2f} s')
    print('------------------------------------------------------------------')
    print(f'  在此最优策略下，可实现的最大有效遮蔽总时长为: {max_shielding_time_sa:.4f} s')
    print('==================================================================\n')
    
    # 保存结果到Excel文件
    save_results_to_excel(best_X_sa, max_shielding_time_sa, 'result1.xlsx')
    
    # 可视化最优结果
    print('正在为最优策略生成高质量可视化结果...')
    visualize_optimal_strategy_q3(best_X_sa, max_shielding_time_sa)
    
    print('程序运行完毕。')

if __name__ == '__main__':
    main()
