# 该脚本的目标是解决一个多无人机、多导弹的协同防御优化问题。
# 通过两阶段混合优化方法（WOA + SLSQP），为无人机找到最佳投放点，以最大化对所有来袭导弹的联合覆盖能力。

import numpy as np
import os
from scipy.optimize import minimize
import random

# --- 全局常量和配置 ---

# 无人机、导弹和目标圆柱体的参数
CYLINDER_CENTER = np.array([0.0, 200.0, 0.0])  # 真目标圆柱体的底面圆心
CYLINDER_RADIUS = 7.0  # 圆柱体半径
CYLINDER_HEIGHT = 10.0  # 圆柱体高度

# 三枚来袭导弹的初始配置
MISSILE_CONFIGS = [
    {"id": "M1", "initial_pos": np.array([20000.0, 0.0, 2000.0]), "speed": 300.0},
    {"id": "M2", "initial_pos": np.array([19000.0, 600.0, 2100.0]), "speed": 300.0},
    {"id": "M3", "initial_pos": np.array([18000.0, -600.0, 1900.0]), "speed": 300.0},
]

# 五架无人机的初始位置
UAV_INITIAL_POSITIONS = {
    "UAV1": np.array([17800.0, 0.0, 1800.0]),
    "UAV2": np.array([12000.0, 1400.0, 1400.0]),
    "UAV3": np.array([6000.0, -3000.0, 700.0]),
    "UAV4": np.array([11000.0, 2000.0, 1800.0]),
    "UAV5": np.array([13000.0, -2000.0, 1300.0]),
}
UAV_IDS = list(UAV_INITIAL_POSITIONS.keys())

# 烟幕和无人机移动参数
SMOKE_RADIUS = 10.0  # 烟幕有效半径 (m)
MAX_COVERAGE_DURATION = 20.0  # 烟幕最大遮挡时长 (s)
MAX_UAV_DISTANCE = 10000.0  # 无人机最大可移动距离 (m)

# 鲸鱼优化算法（WOA）参数
WOA_POPULATION_SIZE = 60  # 种群大小
WOA_ITERATIONS = 400  # 迭代次数
RANDOM_SEED = 42  # 随机种子，确保结果可复现
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# 输出目录设置
OUTPUT_DIRECTORY = "uav_smoke_outputs_optimized"
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

# 无人机对不同导弹的基础覆盖能力矩阵
# 行代表无人机（UAV1-UAV5），列代表导弹（M1-M3）
COVERAGE_MATRIX = np.array([
    [4.6, 7.686, 1.246],
    [3.08, 3.9, 1.746],
    [2.92, 2.471, 3.1],
    [3.501, 1.212, 0.989],
    [1.877, 3.215, 0.36]
])

# --- 核心功能：适应度函数 ---

def compute_coverage(uav_index, missile_index, uav_position, missile_position):
    """
    计算单个无人机对单个导弹的覆盖贡献。
    覆盖能力随距离指数衰减。
    """
    distance = np.linalg.norm(uav_position - missile_position)
    return COVERAGE_MATRIX[uav_index, missile_index] * np.exp(-distance / 5000.0)

def fitness_function(positions):
    """
    计算所有无人机对所有导弹的总覆盖能力。
    这是优化的目标函数。
    positions 是一个包含所有无人机三维坐标的一维数组。
    """
    total_coverage = 0.0
    for uav_idx, uav_id in enumerate(UAV_IDS):
        # 从一维数组中提取当前无人机的三维坐标
        uav_pos = positions[3 * uav_idx:3 * uav_idx + 3]
        
        # 计算无人机与其初始位置的距离，用于惩罚超出移动范围的情况
        dist_from_initial = np.linalg.norm(uav_pos - UAV_INITIAL_POSITIONS[uav_id])
        
        # 限制无人机的最大移动距离，如果超出则按比例减少覆盖贡献
        scale = min(1.0, MAX_UAV_DISTANCE / (dist_from_initial + 1e-6))
        
        # 遍历所有导弹，计算并累加覆盖贡献
        for m_idx, missile in enumerate(MISSILE_CONFIGS):
            coverage = compute_coverage(uav_idx, m_idx, uav_pos, missile['initial_pos'])
            total_coverage += coverage * scale
            
    # 优化算法寻找最小值，因此返回总覆盖能力的负值
    return -total_coverage

# --- 优化器函数 ---

def initialize_woa_population(pop_size):
    """
    初始化鲸鱼优化算法（WOA）的种群。
    每个个体（鲸鱼）代表一组无人机投放点。
    """
    # 定义每个无人机在三维空间中的搜索范围
    pos_min = np.array([pos[dim] - MAX_UAV_DISTANCE for pos in UAV_INITIAL_POSITIONS.values() for dim in range(3)])
    pos_max = np.array([pos[dim] + MAX_UAV_DISTANCE for pos in UAV_INITIAL_POSITIONS.values() for dim in range(3)])
    
    # 在搜索范围内随机生成初始种群
    return np.random.uniform(low=pos_min, high=pos_max, size=(pop_size, len(pos_min)))

def woa_update_population(population, best_position, decay_factor):
    """
    根据鲸鱼优化算法的规则更新种群位置。
    模拟鲸鱼向最优解移动的行为。
    """
    new_population = np.zeros_like(population)
    for i in range(population.shape[0]):
        r1, r2 = np.random.rand(), np.random.rand()
        A = 2 * decay_factor * r1 - decay_factor
        C = 2 * r2
        D = np.abs(C * best_position - population[i])
        new_population[i] = best_position - A * D
        
        # 确保更新后的位置仍在合法范围内
        new_population[i] = np.minimum(np.maximum(new_population[i], POSITION_MIN), POSITION_MAX)
    return new_population

# --- 主程序入口 ---
if __name__ == "__main__":
    
    # 构建所有无人机位置的边界数组
    POSITION_MIN = np.array([UAV_INITIAL_POSITIONS[name][dim] - MAX_UAV_DISTANCE for name in UAV_IDS for dim in range(3)])
    POSITION_MAX = np.array([UAV_INITIAL_POSITIONS[name][dim] + MAX_UAV_DISTANCE for name in UAV_IDS for dim in range(3)])

    print("--- 多无人机协同防御优化开始 ---")
    
    # 1. 鲸鱼优化算法（WOA）全局搜索
    print("阶段一：使用鲸鱼优化算法进行全局搜索...")
    population = initialize_woa_population(WOA_POPULATION_SIZE)
    fitness_values = np.array([fitness_function(p) for p in population])
    best_index = np.argmin(fitness_values)
    best_position = population[best_index]
    best_fitness = fitness_values[best_index]

    for iteration in range(WOA_ITERATIONS):
        # 衰减因子a从2线性衰减到0，控制探索与开发平衡
        decay_factor = 2 * (1 - iteration / WOA_ITERATIONS)
        population = woa_update_population(population, best_position, decay_factor)
        fitness_values = np.array([fitness_function(p) for p in population])
        min_index = np.argmin(fitness_values)
        if fitness_values[min_index] < best_fitness:
            best_fitness = fitness_values[min_index]
            best_position = population[min_index]
        if (iteration + 1) % 50 == 0:
            print(f"  迭代 {iteration + 1}/{WOA_ITERATIONS}：当前最佳总覆盖能力: {-best_fitness:.4f}")

    print("全局搜索完成。")

    # 2. 序列二次规划（SLSQP）局部精炼
    print("\n阶段二：使用SLSQP进行局部精炼...")
    optimization_result = minimize(
        fitness_function,
        best_position,
        method='SLSQP',
        bounds=np.stack((POSITION_MIN, POSITION_MAX), axis=1)
    )
    best_final_position = optimization_result.x
    final_coverage = -fitness_function(best_final_position)
    
    print(f"优化终止信息: {optimization_result.message}")
    print("局部精炼完成。")

    # --- 结果展示 ---
    print("\n" + "=" * 60)
    print("           无人机最优投放点方案")
    print("=" * 60)
    
    print(f"  > 最终总覆盖能力: {final_coverage:.4f}")
    
    print("\n  > 各无人机最优投放点:")
    for uav_idx, uav_id in enumerate(UAV_IDS):
        x, y, z = best_final_position[3 * uav_idx:3 * uav_idx + 3]
        print(f"    - {uav_id}: (x={x:.1f}m, y={y:.1f}m, z={z:.1f}m)")
    
    print("\n" + "=" * 60)
    print("  该方案旨在通过优化无人机编队位置，最大化对来袭导弹的联合防御效果。")
    print("=" * 60)
