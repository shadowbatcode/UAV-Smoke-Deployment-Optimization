import numpy as np
import random
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing
import pandas as pd
import matplotlib as mpl

# --- 全局配置 ---
# 设置绘图支持中文显示
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# 定义全局常量
GRAVITY_ACCEL = 9.81  # 重力加速度 (m/s²)
NUMERIC_THRESHOLD = 1e-15  # 数值运算的保护阈值
COARSE_TIME_STEP = 0.1  # 粗略计算时的时间步长
FINE_TIME_STEP = 0.005  # 关键时段的精细步长
MIN_DROP_INTERVAL = 1.0  # 投放烟幕弹的最小时间间隔
PARALLEL_JOBS = multiprocessing.cpu_count()  # 并行计算使用的CPU核心数

# 定义目标参数
FALSE_TARGET_POS = np.array([0.0, 0.0, 0.0])  # 假目标的坐标
TRUE_TARGET_SPEC = {
    "base_center": np.array([0.0, 200.0, 0.0]),  # 真目标的底面圆心
    "radius": 7.0,  # 真目标圆柱体的半径
    "height": 10.0  # 真目标圆柱体的高度
}

# 定义烟幕参数
SMOKE_CONFIG = {
    "effective_radius": 10.0,  # 烟幕的有效遮蔽半径 (m)
    "sink_velocity": 3.0,  # 烟幕下沉的速度 (m/s)
    "active_duration": 20.0  # 烟幕的有效遮蔽时间 (s)
}

# 定义无人机和导弹参数
UAV_INITIAL_POS = np.array([17800.0, 0.0, 1800.0])  # 无人机的初始位置
MISSILE_CONFIG = {
    "initial_pos": np.array([20000.0, 0.0, 2000.0]),  # 导弹的初始位置
    "flight_speed": 300.0  # 导弹的飞行速度 (m/s)
}

# 计算导弹的飞行方向和到达假目标的时间
MISSILE_DIRECTION = (FALSE_TARGET_POS - MISSILE_CONFIG["initial_pos"]) / np.linalg.norm(FALSE_TARGET_POS - MISSILE_CONFIG["initial_pos"])
MISSILE_ARRIVAL_TIME = np.linalg.norm(FALSE_TARGET_POS - MISSILE_CONFIG["initial_pos"]) / MISSILE_CONFIG["flight_speed"]

# --- 工具函数（未作改动） ---
def vector_magnitude(vector):
    """计算向量的模长。"""
    return np.sqrt(np.sum(vector**2))

def segment_sphere_intersection(start_point, end_point, sphere_center, sphere_radius):
    """判定高精度的线段与球体相交情况。"""
    segment_vec = end_point - start_point
    sphere_vec = sphere_center - start_point
    a = np.dot(segment_vec, segment_vec)
    if a < NUMERIC_THRESHOLD:
        dist = vector_magnitude(sphere_vec)
        return 1.0 if dist <= sphere_radius + NUMERIC_THRESHOLD else 0.0
    b = -2 * np.dot(segment_vec, sphere_vec)
    c = np.dot(sphere_vec, sphere_vec) - sphere_radius**2
    discriminant = b**2 - 4*a*c
    if discriminant < -NUMERIC_THRESHOLD:
        return 0.0
    if discriminant < 0:
        discriminant = 0.0
    sqrt_d = np.sqrt(discriminant)
    s1 = (-b - sqrt_d) / (2*a)
    s2 = (-b + sqrt_d) / (2*a)
    s_start = max(0.0, min(s1, s2))
    s_end = min(1.0, max(s1, s2))
    return max(0.0, s_end - s_start)

def is_fully_shielded_multi(missile_pos, smoke_centers, smoke_radius, target_samples):
    """判定真目标是否被多个烟幕完全遮蔽。"""
    for sample in target_samples:
        shielded = False
        for smoke_center in smoke_centers:
            if segment_sphere_intersection(missile_pos, sample, smoke_center, smoke_radius) >= NUMERIC_THRESHOLD:
                shielded = True
                break
        if not shielded:
            return False
    return True

def get_adaptive_time_steps(start_time, end_time, event_time=None):
    """生成自适应的时间步长序列。"""
    if event_time is None:
        return np.arange(start_time, end_time + COARSE_TIME_STEP, COARSE_TIME_STEP)
    fine_start = max(start_time, event_time - 1.0)
    fine_end = min(end_time, event_time + 1.0)
    times = []
    if start_time < fine_start:
        times.extend(np.arange(start_time, fine_start, COARSE_TIME_STEP))
    times.extend(np.arange(fine_start, fine_end + FINE_TIME_STEP, FINE_TIME_STEP))
    if fine_end < end_time:
        times.extend(np.arange(fine_end, end_time + COARSE_TIME_STEP, COARSE_TIME_STEP))
    return np.unique(times)

# --- 目标采样点生成（未作改动） ---
def generate_dense_samples(target_spec):
    """生成真目标的超高密度采样点。"""
    samples = []
    center, radius, height = target_spec["base_center"], target_spec["radius"], target_spec["height"]
    center_xy = center[:2]
    min_z, max_z = center[2], center[2] + height
    theta_dense = np.linspace(0, 2*np.pi, 120, endpoint=False)
    for z in [min_z, max_z]:
        for th in theta_dense:
            x = center_xy[0] + radius * np.cos(th)
            y = center_xy[1] + radius * np.sin(th)
            samples.append([x, y, z])
    heights_dense = np.linspace(min_z, max_z, 40, endpoint=True)
    for z in heights_dense:
        for th in theta_dense:
            x = center_xy[0] + radius * np.cos(th)
            y = center_xy[1] + radius * np.sin(th)
            samples.append([x, y, z])
    radii = np.linspace(0, radius, 10, endpoint=True)
    inner_heights = np.linspace(min_z, max_z, 30, endpoint=True)
    inner_thetas = np.linspace(0, 2*np.pi, 24, endpoint=False)
    for z in inner_heights:
        for rad in radii:
            for th in inner_thetas:
                x = center_xy[0] + rad * np.cos(th)
                y = center_xy[1] + rad * np.sin(th)
                samples.append([x, y, z])
    edge_radii = np.linspace(radius*0.95, radius*1.05, 5, endpoint=True)
    for z in np.linspace(min_z, max_z, 10):
        for rad in edge_radii:
            for th in np.linspace(0, 2*np.pi, 60, endpoint=False):
                x = center_xy[0] + rad * np.cos(th)
                y = center_xy[1] + rad * np.sin(th)
                samples.append([x, y, z])
    return np.unique(np.array(samples), axis=0)

# --- 适应度函数（未作改动） ---
def fitness_v_t_intervals(params, debug=False):
    """
    计算适应度，其值为距离加权和的负数。
    目的是寻找使烟幕起爆点与导弹轨迹距离最短的参数组合。
    """
    uav_speed, drop_delay_1, drop_delay_2, drop_delay_3 = params
    if not (70.0 <= uav_speed <= 140.0) or any(t < MIN_DROP_INTERVAL for t in (drop_delay_1, drop_delay_2, drop_delay_3)):
        return -1e6
    total_time = drop_delay_1 + drop_delay_2 + drop_delay_3
    total_time_p = total_time + uav_speed * drop_delay_2 / 298.5
    total_time_pp = total_time + uav_speed * (drop_delay_1 + drop_delay_2) / 298.5
    point_a = np.array([17800.0 + uav_speed * (drop_delay_1 + drop_delay_2), 0.0, 1800.0 - 3.0 * drop_delay_3])
    point_b = np.array([17800.0 + uav_speed * drop_delay_1, 0.0, 1800.0 - 3.0 * (drop_delay_3 + drop_delay_2 + uav_speed * drop_delay_2 / 298.5)])
    point_c = np.array([17800.0, 0.0, 1800.0 - 3.0 * total_time_pp])
    missile_pos_1 = np.array([20000.0 - 298.5 * total_time, 0.0, 2000.0 - 29.85 * total_time])
    missile_pos_2 = np.array([20000.0 - 298.5 * total_time_p, 0.0, 2000.0 - 29.85 * total_time_p])
    missile_pos_3 = np.array([20000.0 - 298.5 * total_time_pp, 0.0, 2000.0 - 29.85 * total_time_pp])
    dist_a = np.linalg.norm(point_a - missile_pos_1)
    dist_b = np.linalg.norm(point_b - missile_pos_2)
    dist_c = np.linalg.norm(point_c - missile_pos_3)
    weighted_dist_sum = 15*dist_a + 0.3*dist_b + 0.1*dist_c
    penalty = 0.0
    z_vals = [point_a[2], point_b[2], point_c[2], missile_pos_1[2], missile_pos_2[2], missile_pos_3[2]]
    if any(z < 0 for z in z_vals):
        penalty += 1e4 + 1e3 * sum(max(0.0, -z) for z in z_vals)
    if not np.isfinite(weighted_dist_sum):
        penalty += 1e6
    fitness = -weighted_dist_sum - penalty
    if debug:
        print(f"[调试] dist_a={dist_a:.2f}, dist_b={dist_b:.2f}, dist_c={dist_c:.2f}, "
              f"weighted_dist_sum={weighted_dist_sum:.2f}, penalty={penalty:.2f}, fitness={fitness:.2f}")
    return fitness

# --- 遮蔽时间计算函数（已更新） ---
def compute_shield_duration(params, target_samples):
    """
    计算各起爆点产生的烟幕的遮蔽时间以及总的有效遮蔽时长。
    该函数包含详细的调试输出。
    """
    uav_speed, drop_delay_1, drop_delay_2, drop_delay_3 = params
    total_time = drop_delay_1 + drop_delay_2 + drop_delay_3
    total_time_p = total_time + uav_speed * drop_delay_2 / 298.5
    total_time_pp = total_time + uav_speed * (drop_delay_1 + drop_delay_2) / 298.5

    if not (70.0 <= uav_speed <= 140.0) or any(t < 0 for t in [drop_delay_1, drop_delay_2, drop_delay_3]):
        print(f"[调试] 参数无效：uav_speed={uav_speed:.2f}, drop_delay_1={drop_delay_1:.2f}, drop_delay_2={drop_delay_2:.2f}, drop_delay_3={drop_delay_3:.2f}")
        return 0.0, [0.0, 0.0, 0.0], []

    # 计算烟幕起爆点和对应时刻的导弹位置
    explosion_points = [
        np.array([17800.0 + uav_speed * (drop_delay_1 + drop_delay_2), 0.0, 1800.0 - 3.0 * drop_delay_3]),
        np.array([17800.0 + uav_speed * drop_delay_1, 0.0, 1800.0 - 3.0 * (drop_delay_3 + drop_delay_2 + uav_speed * drop_delay_2 / 298.5)]),
        np.array([17800.0, 0.0, 1800.0 - 3.0 * total_time_pp])
    ]
    detonation_times = [drop_delay_1, drop_delay_1 + drop_delay_2, drop_delay_1 + drop_delay_2 + drop_delay_3]
    explosion_labels = ['A', 'B', 'C']

    # 调试：输出计算出的坐标
    print("[调试] 烟幕起爆点坐标：")
    for label, point in zip(explosion_labels, explosion_points):
        print(f"  {label}: {point.round(4)}")
    print(f"导弹位置:\n  M1: {np.array([20000.0 - 298.5 * total_time, 0.0, 2000.0 - 29.85 * total_time]).round(4)}\n  M2: {np.array([20000.0 - 298.5 * total_time_p, 0.0, 2000.0 - 29.85 * total_time_p]).round(4)}\n  M3: {np.array([20000.0 - 298.5 * total_time_pp, 0.0, 2000.0 - 29.85 * total_time_pp]).round(4)}")
    dist_a = np.linalg.norm(explosion_points[0] - np.array([20000.0 - 298.5 * total_time, 0.0, 2000.0 - 29.85 * total_time]))
    dist_b = np.linalg.norm(explosion_points[1] - np.array([20000.0 - 298.5 * total_time_p, 0.0, 2000.0 - 29.85 * total_time_p]))
    dist_c = np.linalg.norm(explosion_points[2] - np.array([20000.0 - 298.5 * total_time_pp, 0.0, 2000.0 - 29.85 * total_time_pp]))
    print(f"距离:\n  dist_a: {dist_a:.4f}\n  dist_b: {dist_b:.4f}\n  dist_c: {dist_c:.4f}")

    # 定义仿真时间范围
    start_time = min(detonation_times)
    smoke_end_times = [t_det + SMOKE_CONFIG["active_duration"] for t_det in detonation_times]
    end_time = min(max(smoke_end_times), MISSILE_ARRIVAL_TIME)
    if start_time >= end_time:
        print(f"[调试] 时间范围无效：start_time={start_time:.2f}, end_time={end_time:.2f}")
        return 0.0, [0.0, 0.0, 0.0], []

    # 计算导弹到达真目标投影点的时间
    missile_to_target = TRUE_TARGET_SPEC["base_center"] - MISSILE_CONFIG["initial_pos"]
    dist_proj = np.dot(missile_to_target, MISSILE_DIRECTION)
    event_time = dist_proj / MISSILE_CONFIG["flight_speed"]

    # 生成自适应时间步长序列
    time_steps = get_adaptive_time_steps(start_time, end_time, event_time)

    # 初始化遮蔽记录
    total_shield_duration = 0.0
    shield_durations = [0.0, 0.0, 0.0]
    shield_intervals = []
    was_shielded = False

    # 调试：输出时间范围
    print(f"[调试] 时间范围：start_time={start_time:.2f}, end_time={end_time:.2f}, event_time={event_time:.2f}")
    print("[调试] 开始时间步循环...")

    # 时间步循环
    prev_time = None
    for t in time_steps:
        if prev_time is not None:
            dt = t - prev_time
            missile_pos = MISSILE_CONFIG["initial_pos"] + MISSILE_CONFIG["flight_speed"] * t * MISSILE_DIRECTION

            # 确定当前时刻哪些烟幕是活跃的
            active_smoke_centers = []
            active_indices = []
            for i, t_det in enumerate(detonation_times):
                if t_det <= t < t_det + SMOKE_CONFIG["active_duration"]:
                    sink_time = t - t_det
                    smoke_z = explosion_points[i][2] - SMOKE_CONFIG["sink_velocity"] * sink_time
                    if smoke_z < 2.0:
                        continue
                    smoke_center = np.array([explosion_points[i][0], explosion_points[i][1], smoke_z])
                    active_smoke_centers.append(smoke_center)
                    active_indices.append(i)

            # 判定是否被整体遮蔽
            current_shielded = False
            if active_smoke_centers:
                current_shielded = is_fully_shielded_multi(missile_pos, active_smoke_centers,
                                                           SMOKE_CONFIG["effective_radius"], target_samples)

            # 记录总遮蔽时长
            if current_shielded:
                total_shield_duration += dt

            # 记录每个烟幕的遮蔽时长
            for i, t_det in enumerate(detonation_times):
                if t_det <= t < t_det + SMOKE_CONFIG["active_duration"]:
                    smoke_z = explosion_points[i][2] - SMOKE_CONFIG["sink_velocity"] * (t - t_det)
                    if smoke_z < 2.0:
                        continue
                    smoke_center = np.array([explosion_points[i][0], explosion_points[i][1], smoke_z])
                    if is_fully_shielded_multi(missile_pos, [smoke_center], SMOKE_CONFIG["effective_radius"], target_samples):
                        shield_durations[i] += dt

            # 记录遮蔽时间段
            if current_shielded and not was_shielded:
                shield_intervals.append({"start": t})
            elif not current_shielded and was_shielded:
                if shield_intervals:
                    shield_intervals[-1]["end"] = t - dt

            was_shielded = current_shielded
        prev_time = t

    # 处理最后一个未结束的遮蔽时间段
    if shield_intervals and "end" not in shield_intervals[-1]:
        shield_intervals[-1]["end"] = end_time

    return total_shield_duration, shield_durations, shield_intervals

# --- 粒子群优化器（未作改动） ---
class ParticleSwarmOptimizer:
    def __init__(self, objective_func, bounds, num_particles=50, max_iter=100,
                 c1=2.0, c2=2.0, w_start=0.9, w_end=0.4):
        """初始化粒子群优化器。"""
        self.objective_func = objective_func
        self.bounds = bounds
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.c1, self.c2 = c1, c2
        self.w_start, self.w_end = w_start, w_end
        self.dim = len(bounds)
        self.positions = np.array([
            [random.uniform(bounds[d][0], bounds[d][1]) for d in range(self.dim)]
            for _ in range(num_particles)
        ])
        self.velocities = np.random.uniform(-1, 1, (num_particles, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.array([float('-inf')] * num_particles)
        self.global_best_position = None
        self.global_best_score = float('-inf')
        self.history = []

    def optimize(self):
        """执行粒子群优化过程。"""
        for gen in range(self.max_iter):
            w = self.w_start - (self.w_start - self.w_end) * (gen / self.max_iter)
            scores = Parallel(n_jobs=PARALLEL_JOBS)(
                delayed(self.objective_func)(particle) for particle in self.positions
            )
            for i, score in enumerate(scores):
                if score > self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.positions[i]
                if score > self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.positions[i]
            for i in range(self.num_particles):
                r1, r2 = random.random(), random.random()
                cognitive = self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i])
                social = self.c2 * r2 * (self.global_best_position - self.positions[i])
                self.velocities[i] = w * self.velocities[i] + cognitive + social
                self.positions[i] += self.velocities[i]
                for d in range(self.dim):
                    if self.positions[i][d] < self.bounds[d][0]:
                        self.positions[i][d] = self.bounds[d][0]
                        self.velocities[i][d] *= -0.5
                    elif self.positions[i][d] > self.bounds[d][1]:
                        self.positions[i][d] = self.bounds[d][1]
                        self.velocities[i][d] *= -0.5
            self.history.append(self.global_best_score)
            if gen % 10 == 0:
                uav_speed, drop_delay_1, drop_delay_2, drop_delay_3 = self.global_best_position
                print(f"迭代 {gen}, 最优适应度 = {self.global_best_score:.6f}")
                print(f"参数: uav_speed={uav_speed:.2f}, drop_delay_1={drop_delay_1:.2f}, drop_delay_2={drop_delay_2:.2f}, drop_delay_3={drop_delay_3:.2f}")
                fitness_v_t_intervals(self.global_best_position, debug=True)
        return self.global_best_position, self.global_best_score, self.history

# --- 主程序（已更新） ---
if __name__ == "__main__":
    # 生成目标采样点
    print("生成目标采样点...")
    target_samples = generate_dense_samples(TRUE_TARGET_SPEC)
    print(f"采样点数量: {len(target_samples)}")

    # 设置优化参数的边界
    bounds = [
        (70.0, 140.0),  # 无人机速度
        (1.0, 80.0),    # 第一次投放延迟
        (1.0, 80.0),    # 第二次投放延迟
        (0, 80.0)       # 第三次投放延迟
    ]

    # 初始化并运行PSO
    pso = ParticleSwarmOptimizer(
        objective_func=fitness_v_t_intervals,
        bounds=bounds,
        num_particles=60,
        max_iter=150,
        c1=1.5, c2=1.5,
        w_start=0.9, w_end=0.4
    )
    best_params, best_fitness, history = pso.optimize()

    # 使用最优参数计算遮蔽时间
    opt_uav_speed, opt_drop_delay_1, opt_drop_delay_2, opt_drop_delay_3 = best_params
    min_weighted_dist_sum = -best_fitness if np.isfinite(best_fitness) else None
    total_shield_duration, shield_durations, shield_intervals = compute_shield_duration(best_params, target_samples)

    # --- 最终结果输出 ---
    print("\n" + "="*80)
    print("优化结果")
    print("="*80)

    # 显示最优参数和目标函数值
    print(f"目标函数值 (最小化加权距离和): {min_weighted_dist_sum:.6f}")
    print(f"找到的最优参数:")
    print(f"  无人机速度: {opt_uav_speed:.3f} 米/秒")
    print(f"  第一次投放延迟: {opt_drop_delay_1:.3f} 秒")
    print(f"  第二次投放延迟: {opt_drop_delay_2:.3f} 秒")
    print(f"  第三次投放延迟: {opt_drop_delay_3:.3f} 秒")

    # 显示遮蔽时长分析
    print("\n遮蔽时长分析:")
    print(f"总遮蔽时长: {total_shield_duration:.4f} 秒")
    print("各烟幕单点遮蔽时长:")
    for i, duration in enumerate(shield_durations):
        print(f"  起爆点 {['A', 'B', 'C'][i]}: {duration:.4f} 秒")
    
    # 显示遮蔽时间段
    print("\n遮蔽时间段详情:")
    if not shield_intervals:
        print("  未找到有效的遮蔽时间段。")
    else:
        for i, interval in enumerate(shield_intervals, 1):
            duration = interval["end"] - interval["start"]
            print(f"  第{i}段: {interval['start']:.4f}秒 ~ {interval['end']:.4f}秒 (时长: {duration:.4f}秒)")
    print("="*80)

    # 绘制收敛曲线
    plt.figure(figsize=(10, 6))
    plt.plot(history, label="最优适应度")
    plt.xlabel("迭代次数")
    plt.ylabel("适应度")
    plt.title("PSO 收敛曲线")
    plt.legend()
    plt.grid(True)
    plt.show()
