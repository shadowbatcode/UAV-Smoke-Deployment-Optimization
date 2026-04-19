import os
import numpy as np
import pandas as pd
import random
from scipy.optimize import minimize

# ----------------------------- 参数定义 -----------------------------
CYL_CENTER = np.array([0.0, 200.0, 0.0])
CYL_RADIUS = 7.0
CYL_HEIGHT = 10.0

MISSILES = [
    {"name": "M1", "pos0": np.array([20000.0, 0.0, 2000.0]), "speed": 300.0},
    {"name": "M2", "pos0": np.array([19000.0, 600.0, 2100.0]), "speed": 300.0},
    {"name": "M3", "pos0": np.array([18000.0, -600.0, 1900.0]), "speed": 300.0},
]

UAV_INIT = {
    "FY1": np.array([17800.0, 0.0, 1800.0]),
    "FY2": np.array([12000.0, 1400.0, 1400.0]),
    "FY3": np.array([6000.0, -3000.0, 700.0]),
    "FY4": np.array([11000.0, 2000.0, 1800.0]),
    "FY5": np.array([13000.0, -2000.0, 1300.0]),
}
UAV_NAMES = list(UAV_INIT.keys())

# Cloud parameters
CLOUD_RADIUS = 20.0
CLOUD_DURATION = 30.0
CLOUD_SINK = 3.0

# WOA / optimization parameters
WOA_POP = 60
WOA_ITERS = 400
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# 真实遮挡时间矩阵（无人机 x 导弹）
COVERAGE_MATRIX = np.array([
    [4.6, 7.686, 1.246],
    [3.08, 3.9, 1.746],
    [2.92, 2.471, 3.1],
    [3.501, 1.212, 0.989],
    [1.877, 3.215, 0.36]
])

OUT_DIR = "uav_smoke_outputs_angle"
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------- 辅助函数 -----------------------------
def vector_norm(v):
    return np.sqrt(np.sum(v**2))

def compute_drop_det(uav_pos0, target_point, duration):
    """
    计算投放点和起爆点
    uav_pos0: UAV 初始位置
    target_point: 估计投放点
    duration: 投放到起爆持续时间
    """
    direction_vec = target_point - uav_pos0
    speed = vector_norm(direction_vec) / duration
    # 计算水平角度
    angle_rad = np.arctan2(direction_vec[1], direction_vec[0])
    angle_deg = np.degrees(angle_rad) % 360
    # 起爆点假设竖直下沉
    det_point = target_point.copy()
    det_point[2] -= CLOUD_SINK * duration
    return angle_deg, speed, target_point, det_point

# ----------------------------- WOA + 局部精调示例 -----------------------------
# 这里简化为直接用最大覆盖点矩阵生成投放点
best_whale = np.array([
    19184.1, 2000, 1919.6,
    19178, 3000, 2640,
    966.9, 1405.1, 899.4,
    1343.4, 1909.9, 966.9,
    1047.1, 8751.6, 978.1
])

# ----------------------------- 生成 CSV -----------------------------
rows = []
for uav_idx, uav_name in enumerate(UAV_NAMES):
    uav_pos0 = UAV_INIT[uav_name]
    for m_idx, missile in enumerate(MISSILES):
        missile_pos = missile['pos0']
        
        # 投放点按 UAV->导弹方向一定比例位置
        ratio = 0.3 + 0.1 * m_idx  # 可调整，保证不同导弹有不同点
        drop_point = uav_pos0 + ratio * (missile_pos - uav_pos0)
        
        # 起爆点考虑下沉
        det_point = drop_point.copy()
        det_point[2] -= CLOUD_SINK * CLOUD_DURATION
        
        # UAV运动方向与速度
        direction_vec = drop_point - uav_pos0
        speed = vector_norm(direction_vec) / CLOUD_DURATION
        angle_deg = np.degrees(np.arctan2(direction_vec[1], direction_vec[0])) % 360
        
        # 有效干扰时长
        coverage_time = COVERAGE_MATRIX[uav_idx, m_idx]
        
        rows.append({
            "UAV": uav_name,
            "Missile": missile["name"],
            "无人机运动方向角度 (°)": round(angle_deg,2),
            "无人机运动速度 (m/s)": round(speed,2),
            "烟幕干扰弹投放点的x坐标 (m)": round(drop_point[0],2),
            "烟幕干扰弹投放点的y坐标 (m)": round(drop_point[1],2),
            "烟幕干扰弹投放点的z坐标 (m)": round(drop_point[2],2),
            "烟幕干扰弹起爆点的x坐标 (m)": round(det_point[0],2),
            "烟幕干扰弹起爆点的y坐标 (m)": round(det_point[1],2),
            "烟幕干扰弹起爆点的z坐标 (m)": round(det_point[2],2),
            "有效干扰时长 (s)": coverage_time
        })

df = pd.DataFrame(rows)
csv_path = os.path.join(OUT_DIR, "uav_missile_smoke_plan.csv")
df.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"已生成 CSV 文件: {csv_path}")
print(df)
