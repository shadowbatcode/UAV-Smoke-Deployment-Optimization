import os
import numpy as np
import pandas as pd

# ----------------------------- UAV 与导弹参数 -----------------------------
UAV_INIT = {
    "FY1": np.array([17800.0, 0.0, 1800.0]),
    "FY2": np.array([12000.0, 1400.0, 1400.0]),
    "FY3": np.array([6000.0, -3000.0, 700.0]),
    "FY4": np.array([11000.0, 2000.0, 1800.0]),
    "FY5": np.array([13000.0, -2000.0, 1300.0]),
}

MISSILES = [
    {"name": "M1"},
]

CLOUD_DURATION = 30.0
CLOUD_SINK = 3.0

# ----------------------------- 模拟优化结果 -----------------------------
# 组合数据: (组合列表, 总覆盖, 每架 UAV 投放点)
combo_results = [
    (['FY1', 'FY2', 'FY3'], 8.577894474383136, [(20000.0, -0.0, 2000.0), (20013.6, -3.0, 1951.7), (14651.3, 470.3, 5.4)]),
    (['FY1', 'FY2', 'FY4'], 7.1518459543835355, [(20000.0, 0.0, 2000.0), (19605.1, -138.3, 4110.4), (11191.1, -1979.4, -78.8)]),
    (['FY1', 'FY2', 'FY5'], 5.886732156734647, [(19997.1, 32.5, 2009.3), (19690.9, 2737.3, 8318.0), (14312.5, 1182.0, 830.1)]),
    (['FY1', 'FY3', 'FY4'], 7.4163734879103, [(20000.0, -0.0, 2000.0), (20036.1, -129.8, 1377.0), (14702.3, -335.0, 5097.5)]),
    (['FY1', 'FY3', 'FY5'], 5.860558677666613, [(19997.5, 184.6, 2009.6), (19236.7, 230.9, 50.0), (145.9, 3750.9, 72.6)]),
    (['FY1', 'FY4', 'FY5'], 7.92321152620949, [(20000.0, -0.0, 2000.0), (20008.0, -132.7, 2314.3), (4256.9, 5909.4, -174.9)]),
    (['FY2', 'FY3', 'FY4'], 4.532737243273563, [(19996.4, 325.2, 2048.9), (19422.2, 8.5, 509.0), (3188.7, 4149.8, 1.9)]),
    (['FY2', 'FY3', 'FY5'], 3.726099595923836, [(20012.5, 373.3, 1626.6), (18291.0, 3173.3, -8.9), (2738.3, -27.3, -0.2)]),
    (['FY2', 'FY4', 'FY5'], 6.640418496939123, [(19986.7, -19.1, 1673.6), (20034.7, 186.2, 2340.3), (14373.6, 3096.0, 2584.4)]),
    (['FY3', 'FY4', 'FY5'], 5.113450018347853, [(19595.2, 349.1, 1017.5), (19999.8, 135.9, 2032.8), (1812.6, -2908.0, 1865.4)])
]

# ----------------------------- 辅助函数 -----------------------------
def vector_norm(v):
    return np.sqrt(np.sum(v**2))

def compute_direction_speed(uav_pos0, drop_point):
    vec = np.array(drop_point) - np.array(uav_pos0)
    speed = vector_norm(vec) / CLOUD_DURATION
    angle_deg = np.degrees(np.arctan2(vec[1], vec[0])) % 360
    return round(angle_deg,2), round(speed,2)

# ----------------------------- 生成 CSV -----------------------------
rows = []
for combo, total_coverage, drop_points in combo_results:
    for uav_name, drop_point in zip(combo, drop_points):
        uav_pos0 = UAV_INIT[uav_name]
        angle_deg, speed = compute_direction_speed(uav_pos0, drop_point)
        det_point = (drop_point[0], drop_point[1], drop_point[2] - CLOUD_SINK * CLOUD_DURATION)
        
        rows.append({
            "组合": ','.join(combo),
            "总覆盖": total_coverage,
            "UAV": uav_name,
            "无人机运动方向角度 (°)": angle_deg,
            "无人机运动速度 (m/s)": speed,
            "投放点 x (m)": round(drop_point[0],2),
            "投放点 y (m)": round(drop_point[1],2),
            "投放点 z (m)": round(drop_point[2],2),
            "起爆点 x (m)": round(det_point[0],2),
            "起爆点 y (m)": round(det_point[1],2),
            "起爆点 z (m)": round(det_point[2],2),
        })

df = pd.DataFrame(rows)
OUT_DIR = "uav_smoke_outputs_angle"
os.makedirs(OUT_DIR, exist_ok=True)
csv_path = os.path.join(OUT_DIR, "uav_missile_smoke_combo.csv")
df.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"已生成 CSV 文件: {csv_path}")
print(df)
