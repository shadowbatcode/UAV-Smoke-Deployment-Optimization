import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib as mpl
from scipy.spatial.distance import cdist

# --- 全局配置与参数 ---

# 绘图配置，支持中文显示和正确处理负号
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# 无人机和导弹的初始位置信息
MISSILE_POSITIONS = {
    "Missile1": np.array([20000.0, 0.0, 2000.0]),
    "Missile2": np.array([19000.0, 600.0, 2100.0]),
    "Missile3": np.array([18000.0, -600.0, 1900.0])
}

UAV_INITIAL_POSITIONS = {
    "UAV1": np.array([17800.0, 0.0, 1800.0]),
    "UAV2": np.array([12000.0, 1400.0, 1400.0]),
    "UAV3": np.array([6000.0, -3000.0, 700.0]),
    "UAV4": np.array([11000.0, 2000.0, 1800.0]),
    "UAV5": np.array([13000.0, -2000.0, 1300.0])
}

# 模型超参数
MAX_UAV_DISTANCE = 10000.0      # 无人机最大可移动距离
COST_WEIGHT = 0.2               # 部署成本权重
REDUNDANCY_WEIGHT = 0.5         # 冗余惩罚权重
DISTANCE_SCALE = 1000.0         # 距离缩放系数，用于计算冗余惩罚

# 无人机对导弹的基础覆盖能力矩阵
# 行: 无人机 (UAV1-5), 列: 导弹 (M1-3)
BASE_COVERAGE_MATRIX = np.array([
    [4.6, 7.686, 1.246],
    [3.08, 3.9, 1.746],
    [2.92, 2.471, 3.1],
    [3.501, 1.212, 0.989],
    [1.877, 3.215, 0.36]
])

UAV_IDS = list(UAV_INITIAL_POSITIONS.keys())
MISSILE_IDS = list(MISSILE_POSITIONS.keys())

# --- 数据计算逻辑 ---
def calculate_metrics():
    """计算部署成本、冗余惩罚和综合价值矩阵。"""
    # 计算部署成本（基于初始位置的距离）
    deployment_costs = np.zeros((len(UAV_IDS), len(MISSILE_IDS)))
    for i, uav_id in enumerate(UAV_IDS):
        for j, missile_id in enumerate(MISSILE_IDS):
            deployment_costs[i, j] = np.linalg.norm(UAV_INITIAL_POSITIONS[uav_id] - MISSILE_POSITIONS[missile_id])

    # 计算假设的最优投放点（用于评估冗余）
    optimal_target_points = {}
    for uav_id in UAV_IDS:
        optimal_target_points[uav_id] = {}
        for missile_id in MISSILE_IDS:
            optimal_target_points[uav_id][missile_id] = (UAV_INITIAL_POSITIONS[uav_id] + MISSILE_POSITIONS[missile_id]) / 2

    # 计算冗余惩罚
    redundancy_penalties = np.zeros((len(UAV_IDS), len(MISSILE_IDS)))
    for i, uav_id in enumerate(UAV_IDS):
        for j, missile_id in enumerate(MISSILE_IDS):
            current_pos = optimal_target_points[uav_id][missile_id]
            same_missile_dists = []
            for k, other_uav in enumerate(UAV_IDS):
                if k != i and BASE_COVERAGE_MATRIX[k, j] > 0:
                    other_pos = optimal_target_points[other_uav][missile_id]
                    dist = np.linalg.norm(current_pos - other_pos)
                    same_missile_dists.append(dist)
            if not same_missile_dists:
                redundancy_penalties[i, j] = 0
            else:
                avg_dist = np.mean(same_missile_dists)
                redundancy_penalties[i, j] = 1.0 / (1.0 + avg_dist / DISTANCE_SCALE)

    # 归一化所有矩阵
    normalized_coverage = normalize_matrix(BASE_COVERAGE_MATRIX)
    normalized_deployment_costs = normalize_matrix(deployment_costs)
    normalized_redundancy_penalties = normalize_matrix(redundancy_penalties)

    # 计算综合价值系数矩阵
    value_matrix = normalized_coverage - COST_WEIGHT * normalized_deployment_costs - REDUNDANCY_WEIGHT * normalized_redundancy_penalties

    return deployment_costs, redundancy_penalties, value_matrix

def normalize_matrix(matrix):
    """将矩阵数据归一化到 [0, 1] 范围。"""
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    if max_val - min_val < 1e-10:
        return np.zeros_like(matrix)
    return (matrix - min_val) / (max_val - min_val)

# --- 主程序入口 ---
if __name__ == "__main__":
    
    # 1. 计算所有任务指标矩阵
    deployment_costs, redundancy_penalties, value_matrix = calculate_metrics()

    # 2. 将数据转换为 DataFrame 以便可视化和输出
    value_df = pd.DataFrame(value_matrix, index=UAV_IDS, columns=MISSILE_IDS)

    # 3. 打印所有矩阵数据
    print("--- 任务指标矩阵 ---")
    print("\n基础遮蔽能力矩阵:")
    print(pd.DataFrame(BASE_COVERAGE_MATRIX, index=UAV_IDS, columns=MISSILE_IDS))
    print("\n部署成本矩阵 (距离):")
    print(pd.DataFrame(deployment_costs, index=UAV_IDS, columns=MISSILE_IDS))
    print("\n冗余惩罚矩阵:")
    print(pd.DataFrame(redundancy_penalties, index=UAV_IDS, columns=MISSILE_IDS))
    print("\n综合价值系数矩阵:")
    print(value_df)

    # 4. 可视化综合价值系数矩阵
    plt.figure(figsize=(14, 10))
    vmax = np.max(np.abs(value_matrix))
    vmin = -vmax

    # 创建热力图
    im = plt.imshow(value_matrix, cmap='RdYlGn', interpolation='bilinear', aspect='auto', vmin=vmin, vmax=vmax)

    # 为每个单元格添加数值标注
    for i in range(value_df.shape[0]):
        for j in range(value_df.shape[1]):
            value = value_df.iloc[i, j]
            text_color = 'black' if abs(value) < vmax * 0.6 else 'white'
            plt.text(j, i, f"{value:.2f}", ha='center', va='center', color=text_color, fontsize=10, fontweight='medium', alpha=0.9)

    # 设置图表标签和标题
    plt.xticks(ticks=np.arange(len(value_df.columns)), labels=value_df.columns, fontsize=12)
    plt.yticks(ticks=np.arange(len(value_df.index)), labels=value_df.index, fontsize=12)
    plt.title('协同多因子任务分配(CMFTA)价值系数矩阵', fontsize=16, pad=20, fontweight='bold')
    plt.xlabel('导弹目标', fontsize=14, labelpad=12)
    plt.ylabel('无人机平台', fontsize=14, labelpad=12)

    # 添加颜色条
    cbar = plt.colorbar(im, pad=0.05)
    cbar.set_label('综合价值系数', fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    # 添加网格线
    plt.grid(False)
    for i in range(len(value_df.index) + 1):
        plt.axhline(i - 0.5, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    for j in range(len(value_df.columns) + 1):
        plt.axvline(j - 0.5, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    # 调整布局和添加注释
    plt.subplots_adjust(bottom=0.15, right=0.85)
    plt.figtext(
        0.98, 0.02,
        f'注: 价值系数V_ij = B_ij - λ_cost·Cost_ij - λ_redund·Redund_ij\n'
        f'超参数: λ_cost={COST_WEIGHT}, λ_redund={REDUNDANCY_WEIGHT}\n'
        'B_ij: 遮蔽时长 | Cost_ij: 部署距离成本 | Redund_ij: 冗余惩罚',
        ha='right', va='bottom', fontsize=10, style='italic',
        bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 5}
    )

    plt.tight_layout(pad=2.0)
    plt.show()
