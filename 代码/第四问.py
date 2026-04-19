from itertools import combinations
import numpy as np
from scipy.optimize import minimize
import random

# --- 无人机与导弹配置 ---
# 定义5架无人机的初始位置
UAV_INITIAL_POSITIONS = {
    "UAV1": np.array([17800.0, 0.0, 1800.0]),
    "UAV2": np.array([12000.0, 1400.0, 1400.0]),
    "UAV3": np.array([6000.0, -3000.0, 700.0]),
    "UAV4": np.array([11000.0, 2000.0, 1800.0]),
    "UAV5": np.array([13000.0, -2000.0, 1300.0]),
}
UAV_IDS = list(UAV_INITIAL_POSITIONS.keys())

# 定义单个导弹的配置
MISSILE_CONFIGS = [
    {"id": "M1", "initial_pos": np.array([20000.0, 0.0, 2000.0]), "speed": 300.0},
]

# 无人机的基础覆盖能力矩阵（这里只用第一列数据）
COVERAGE_MATRIX_FULL = np.array([
    [4.6, 7.686, 1.246],
    [3.08, 3.9, 1.746],
    [2.92, 2.471, 3.1],
    [3.501, 1.212, 0.989],
    [1.877, 3.215, 0.36]
])
COVERAGE_MATRIX = COVERAGE_MATRIX_FULL[:, 0].reshape(-1, 1)

# 全局优化参数
MAX_DISTANCE = 10000.0  # 无人机最大移动距离
WOA_POPULATION_SIZE = 30  # 鲸鱼优化算法的种群大小
WOA_ITERATIONS = 100  # 鲸鱼优化算法的迭代次数
RANDOM_SEED = 42  # 随机种子，用于结果复现
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# --- 适应度函数 ---
def compute_coverage(uav_index, uav_position, missile_position):
    """根据距离和基础覆盖矩阵计算无人机的实际覆盖能力。"""
    distance = np.linalg.norm(uav_position - missile_position)
    # 覆盖能力随距离指数衰减
    return COVERAGE_MATRIX[uav_index, 0] * np.exp(-distance / 5000.0)

def fitness_function_with_individual(positions, selected_indices):
    """
    计算给定无人机位置的总覆盖能力及个体贡献。
    目标是最大化总覆盖能力，因此返回其负值。
    """
    total_coverage = 0.0
    individual_coverages = []
    for i, uav_idx in enumerate(selected_indices):
        uav_pos = positions[3*i:3*i+3]
        # 计算无人机与其初始位置的距离
        dist_from_initial = np.linalg.norm(uav_pos - UAV_INITIAL_POSITIONS[UAV_IDS[uav_idx]])
        # 限制无人机的最大移动距离
        scale = min(1.0, MAX_DISTANCE / (dist_from_initial + 1e-6))
        coverage = compute_coverage(uav_idx, uav_pos, MISSILE_CONFIGS[0]['initial_pos']) * scale
        individual_coverages.append(coverage)
        total_coverage += coverage
    return -total_coverage, individual_coverages

# --- 鲸鱼优化算法 (WOA) ---
def initialize_population(pop_size, num_uavs):
    """在指定范围内初始化种群位置。"""
    pos_min, pos_max = [], []
    for uav_idx in range(num_uavs):
        for dim in range(3):
            # 搜索范围以初始位置为中心，最大移动距离为半径
            pos_min.append(UAV_INITIAL_POSITIONS[UAV_IDS[uav_idx]][dim] - MAX_DISTANCE)
            pos_max.append(UAV_INITIAL_POSITIONS[UAV_IDS[uav_idx]][dim] + MAX_DISTANCE)
    population = np.random.uniform(low=pos_min, high=pos_max, size=(pop_size, 3*num_uavs))
    return population, np.array(pos_min), np.array(pos_max)

def woa_update_population(population, best_position, decay_factor, pos_min, pos_max):
    """使用鲸鱼优化算法更新种群位置。"""
    new_population = np.zeros_like(population)
    for i in range(population.shape[0]):
        r1, r2 = np.random.rand(), np.random.rand()
        A = 2 * decay_factor * r1 - decay_factor
        C = 2 * r2
        D = np.abs(C * best_position - population[i])
        new_population[i] = best_position - A * D
        # 确保新位置不超出界限
        new_population[i] = np.minimum(np.maximum(new_population[i], pos_min), pos_max)
    return new_population

# --- 主优化循环：枚举组合并进行优化 ---
total_uav_count = len(UAV_IDS)
uav_selection_count = 3  # 从5架中选择3架
optimization_results = []

print("开始枚举无人机组合并进行优化...")

# 遍历所有可能的无人机3机组合
for selected_indices in combinations(range(total_uav_count), uav_selection_count):
    selected_indices = list(selected_indices)
    
    # 1. 初始化种群
    population, pos_min, pos_max = initialize_population(WOA_POPULATION_SIZE, uav_selection_count)
    fitness_values = np.array([fitness_function_with_individual(p, selected_indices)[0] for p in population])
    best_idx = np.argmin(fitness_values)
    best_position = population[best_idx]
    best_fitness = fitness_values[best_idx]
    
    # 2. 执行鲸鱼优化算法 (WOA)
    for iter in range(WOA_ITERATIONS):
        # 衰减因子a从2线性衰减到0
        decay_factor = 2 * (1 - iter / WOA_ITERATIONS)
        population = woa_update_population(population, best_position, decay_factor, pos_min, pos_max)
        
        fitness_values = np.array([fitness_function_with_individual(p, selected_indices)[0] for p in population])
        min_idx = np.argmin(fitness_values)
        if fitness_values[min_idx] < best_fitness:
            best_fitness = fitness_values[min_idx]
            best_position = population[min_idx]
    
    # 3. 使用SLSQP进行局部精炼
    optimization_result = minimize(
        lambda x: fitness_function_with_individual(x, selected_indices)[0],
        best_position,
        method='SLSQP',
        bounds=np.stack((pos_min, pos_max), axis=1)
    )
    
    # 计算最终的覆盖能力
    total_coverage, individual_coverages = fitness_function_with_individual(optimization_result.x, selected_indices)
    
    result = {
        "uav_combination": [UAV_IDS[i] for i in selected_indices],
        "total_coverage": -total_coverage,
        "positions": [optimization_result.x[3*i:3*i+3] for i in range(uav_selection_count)],
        "individual_coverages": individual_coverages
    }
    optimization_results.append(result)
    print(f"完成组合 {result['uav_combination']} 的优化。")

# --- 输出优化结果 ---
print("\n" + "="*80)
print("              无人机小组最优配置与投放点分析")
print("="*80)

# 找到总覆盖能力最大的组合
best_overall_result = max(optimization_results, key=lambda x: x["total_coverage"])

# 打印最佳组合的详细信息
print("\n最佳无人机小组配置:")
print(f"  > 小组成员: {', '.join(best_overall_result['uav_combination'])}")
print(f"  > 总覆盖能力: {best_overall_result['total_coverage']:.4f}")

print("\n最佳小组各无人机投放点与贡献:")
for uav_id, pos, cov in zip(best_overall_result["uav_combination"], best_overall_result["positions"], best_overall_result["individual_coverages"]):
    print(f"  - {uav_id}:")
    print(f"    - 最终投放点: x={pos[0]:.1f}, y={pos[1]:.1f}, z={pos[2]:.1f}")
    print(f"    - 单机覆盖贡献: {cov:.4f}")

# 打印所有组合的简要对比
print("\n" + "-"*80)
print("所有无人机组合的总覆盖能力对比:")
print("-" * 80)
sorted_results = sorted(optimization_results, key=lambda x: x["total_coverage"], reverse=True)
for i, result in enumerate(sorted_results):
    rank_str = f"#{i+1}"
    combo_str = ", ".join(result['uav_combination'])
    coverage_str = f"{result['total_coverage']:.4f}"
    print(f"  {rank_str:<4} | 组合: {combo_str:<18} | 总覆盖能力: {coverage_str}")

print("="*80)
