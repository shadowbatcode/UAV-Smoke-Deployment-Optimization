import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing
import time
import matplotlib as mpl

# 设置 matplotlib 的中文显示与美观样式
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# -------------------------- 1. 系统常量与参数定义 --------------------------
GRAVITY_ACCEL = 9.81
NUMERIC_THRESHOLD = 1e-15
COARSE_TIME_STEP = 0.1
FINE_TIME_STEP = 0.005
PARALLEL_JOBS = multiprocessing.cpu_count()

FALSE_TARGET_POS = np.array([0.0, 0.0, 0.0])
TRUE_TARGET_SPEC = {
    "base_center": np.array([0.0, 200.0, 0.0]),
    "radius": 7.0,
    "height": 10.0
}

UAV_START_POS = np.array([17800.0, 0.0, 1800.0])
SMOKE_SPEC = {
    "effective_radius": 10.0,
    "sink_velocity": 3.0,
    "active_duration": 20.0
}
MISSILE_CONFIG = {
    "start_pos": np.array([20000.0, 0.0, 2000.0]),
    "flight_speed": 300.0
}
MISSILE_DIRECTION = (FALSE_TARGET_POS - MISSILE_CONFIG["start_pos"]) / np.linalg.norm(FALSE_TARGET_POS - MISSILE_CONFIG["start_pos"])
MISSILE_ARRIVAL_TIME = np.linalg.norm(FALSE_TARGET_POS - MISSILE_CONFIG["start_pos"]) / MISSILE_CONFIG["flight_speed"]

# -------------------------- 2. 目标采样点生成 --------------------------
def generate_dense_samples(target_spec):
    """为目标生成密集采样点，用于模拟光线遮蔽。"""
    samples = []
    center = target_spec["base_center"]
    radius, height = target_spec["radius"], target_spec["height"]
    center_xy = center[:2]
    min_z, max_z = center[2], center[2] + height
    theta_dense = np.linspace(0, 2*np.pi, 120, endpoint=False)
    heights_dense = np.linspace(min_z, max_z, 40, endpoint=True)
    radii = np.linspace(0, radius, 10, endpoint=True)
    inner_heights = np.linspace(min_z, max_z, 30, endpoint=True)
    inner_thetas = np.linspace(0, 2*np.pi, 24, endpoint=False)
    edge_radii = np.linspace(radius*0.95, radius*1.05, 5, endpoint=True)

    for z in [min_z, max_z]:
        for th in theta_dense:
            x, y = center_xy[0] + radius * np.cos(th), center_xy[1] + radius * np.sin(th)
            samples.append([x, y, z])
    for z in heights_dense:
        for th in theta_dense:
            x, y = center_xy[0] + radius * np.cos(th), center_xy[1] + radius * np.sin(th)
            samples.append([x, y, z])
    for z in inner_heights:
        for rad in radii:
            for th in inner_thetas:
                x, y = center_xy[0] + rad * np.cos(th), center_xy[1] + rad * np.sin(th)
                samples.append([x, y, z])
    for z in np.linspace(min_z, max_z, 10):
        for rad in edge_radii:
            for th in np.linspace(0, 2*np.pi, 60, endpoint=False):
                x, y = center_xy[0] + rad * np.cos(th), center_xy[1] + rad * np.sin(th)
                samples.append([x, y, z])
    return np.unique(np.array(samples), axis=0)

# -------------------------- 3. 几何计算与判定函数 --------------------------
def vector_magnitude(vector):
    """计算向量的欧几里得范数（模长）。"""
    return np.sqrt(np.sum(vector**2))

def segment_sphere_intersection(start_point, end_point, sphere_center, sphere_radius):
    """判定线段与球体是否相交，并返回相交的线段长度比例。"""
    segment_vec = end_point - start_point
    sphere_vec = sphere_center - start_point
    a = np.dot(segment_vec, segment_vec)
    if a < NUMERIC_THRESHOLD:
        return 1.0 if vector_magnitude(sphere_vec) <= sphere_radius + NUMERIC_THRESHOLD else 0.0
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

def is_fully_shielded(missile_pos, smoke_center, smoke_radius, target_samples):
    """判定从导弹位置到真目标的所有采样点视线是否均被烟幕完全遮蔽。"""
    for sample in target_samples:
        if segment_sphere_intersection(missile_pos, sample, smoke_center, smoke_radius) < NUMERIC_THRESHOLD:
            return False
    return True

# -------------------------- 4. 自适应时间步长生成 --------------------------
def get_adaptive_time_steps(start_time, end_time, event_time=None):
    """生成一个自适应的时间序列，在关键事件前后使用更小的时间步长。"""
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

# -------------------------- 5. 适应度函数（目标函数） --------------------------
def fitness_function(params, target_samples):
    """评估无人机投放策略的有效性，适应度值为总有效遮蔽时长。"""
    direction_angle, uav_speed, drop_delay, det_delay = params
    uav_direction = np.array([np.cos(direction_angle), np.sin(direction_angle), 0.0])
    drop_position = UAV_START_POS + uav_speed * drop_delay * uav_direction
    det_xy = drop_position[:2] + uav_speed * det_delay * uav_direction[:2]
    det_z = drop_position[2] - 0.5 * GRAVITY_ACCEL * det_delay**2
    if det_z < 5.0:
        return 0.0 + np.random.uniform(-0.5, 0)
    det_position = np.array([det_xy[0], det_xy[1], det_z])
    detonation_time = drop_delay + det_delay
    missile_to_target = TRUE_TARGET_SPEC["base_center"] - MISSILE_CONFIG["start_pos"]
    dist_proj = np.dot(missile_to_target, MISSILE_DIRECTION)
    event_time = dist_proj / MISSILE_CONFIG["flight_speed"]
    smoke_end_time = detonation_time + SMOKE_SPEC["active_duration"]
    end_time = min(smoke_end_time, MISSILE_ARRIVAL_TIME)
    if detonation_time >= end_time:
        return 0.0 + np.random.uniform(-0.1, 0)
    time_steps = get_adaptive_time_steps(detonation_time, end_time, event_time)
    valid_duration = 0.0
    prev_time = None
    for t in time_steps:
        if prev_time is not None:
            dt_current = t - prev_time
            missile_pos = MISSILE_CONFIG["start_pos"] + MISSILE_CONFIG["flight_speed"] * t * MISSILE_DIRECTION
            sink_time = t - detonation_time
            smoke_z = det_position[2] - SMOKE_SPEC["sink_velocity"] * sink_time
            if smoke_z < 2.0:
                prev_time = t
                continue
            smoke_center = np.array([det_position[0], det_position[1], smoke_z])
            if is_fully_shielded(missile_pos, smoke_center, SMOKE_SPEC["effective_radius"], target_samples):
                valid_duration += dt_current
        prev_time = t
    boundary_bonus = 0.0
    if abs(uav_speed - 70) < 1 or abs(uav_speed - 140) < 1:
        boundary_bonus = 0.1
    if drop_delay < 1 or det_delay < 1:
        boundary_bonus += 0.1
    return valid_duration + boundary_bonus

# -------------------------- 6. 粒子群优化算法实现 --------------------------
class ParticleSwarmOptimizer:
    def __init__(self, objective_func, bounds, num_particles=30, max_iter=100, 
                 c1=2.0, c2=2.0, w_start=0.9, w_end=0.4):
        self.objective_func = objective_func
        self.bounds = bounds
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.c1 = c1
        self.c2 = c2
        self.w_start = w_start
        self.w_end = w_end
        self.dim = len(bounds)
        self.positions = np.zeros((num_particles, self.dim))
        self.velocities = np.zeros((num_particles, self.dim))
        self.pbest_positions = np.zeros((num_particles, self.dim))
        self.pbest_fitness = np.zeros(num_particles) - np.inf
        self.gbest_position = np.zeros(self.dim)
        self.gbest_fitness = -np.inf
        self.gbest_history = []
        self._initialize_particles()
    
    def _initialize_particles(self):
        for i in range(self.num_particles):
            for j in range(self.dim):
                self.positions[i, j] = np.random.uniform(self.bounds[j][0], self.bounds[j][1])
                vel_range = self.bounds[j][1] - self.bounds[j][0]
                self.velocities[i, j] = np.random.uniform(-0.1*vel_range, 0.1*vel_range)
            fitness = self.objective_func(self.positions[i])
            self.pbest_positions[i] = self.positions[i].copy()
            self.pbest_fitness[i] = fitness
            if fitness > self.gbest_fitness:
                self.gbest_fitness = fitness
                self.gbest_position = self.positions[i].copy()
    
    def _constrain_position(self, position, dim):
        min_val, max_val = self.bounds[dim]
        if position < min_val:
            return min_val + 0.01 * (np.random.random() - 0.5)
        elif position > max_val:
            return max_val + 0.01 * (np.random.random() - 0.5)
        return position
    
    def _constrain_velocity(self, velocity, dim):
        min_val, max_val = self.bounds[dim]
        vel_limit = 0.2 * (max_val - min_val)
        return np.clip(velocity, -vel_limit, vel_limit)
    
    def optimize(self):
        for iter in range(self.max_iter):
            mean_fitness = np.mean(self.pbest_fitness)
            w = self.w_start - (self.w_start - self.w_end) * (iter / self.max_iter) + (1 - self.w_end) * mean_fitness / np.max(self.pbest_fitness)
            fitness_values = Parallel(n_jobs=PARALLEL_JOBS)(
                delayed(self.objective_func)(self.positions[i])
                for i in range(self.num_particles)
            )
            for i in range(self.num_particles):
                fitness = fitness_values[i]
                if fitness > self.pbest_fitness[i]:
                    self.pbest_fitness[i] = fitness
                    self.pbest_positions[i] = self.positions[i].copy()
                if fitness > self.gbest_fitness:
                    self.gbest_fitness = fitness
                    self.gbest_position = self.positions[i].copy()
                r1 = np.random.random(self.dim)
                r2 = np.random.random(self.dim)
                cognitive_component = self.c1 * r1 * (self.pbest_positions[i] - self.positions[i])
                social_component = self.c2 * r2 * (self.gbest_position - self.positions[i])
                new_velocity = w * self.velocities[i] + cognitive_component + social_component
                for j in range(self.dim):
                    new_velocity[j] = self._constrain_velocity(new_velocity[j], j)
                self.velocities[i] = new_velocity
                new_position = self.positions[i] + new_velocity
                for j in range(self.dim):
                    new_position[j] = self._constrain_position(new_position[j], j)
                self.positions[i] = new_position
            self.gbest_history.append(self.gbest_fitness)
            if (iter + 1) % 10 == 0 or iter == 0:
                print(f"迭代 {iter+1}/{self.max_iter}, 当前最优适应度: {self.gbest_fitness:.6f}")
        return self.gbest_position, self.gbest_fitness, self.gbest_history

# -------------------------- 7. 主程序 --------------------------
if __name__ == "__main__":
    start_time = time.time()
    
    print("正在为真目标生成密集采样点...")
    target_samples = generate_dense_samples(TRUE_TARGET_SPEC)
    print(f"采样点总数: {len(target_samples)}")

    bounds = [
        (0.0, 2 * np.pi),
        (70.0, 140.0),
        (0.0, 80.0),
        (0.0, 25.0)
    ]

    def objective(params):
        return fitness_function(params, target_samples)

    print("\n开始执行粒子群优化算法...")
    pso = ParticleSwarmOptimizer(
        objective_func=objective,
        bounds=bounds,
        num_particles=50,
        max_iter=100,
        c1=1.5,
        c2=1.5,
        w_start=0.9,
        w_end=0.4
    )

    best_params, best_fitness, history = pso.optimize()

    opt_direction_angle, opt_uav_speed, opt_drop_delay, opt_det_delay = best_params
    uav_direction_opt = np.array([np.cos(opt_direction_angle), np.sin(opt_direction_angle), 0.0])
    opt_drop_position = UAV_START_POS + opt_uav_speed * opt_drop_delay * uav_direction_opt
    opt_det_xy = opt_drop_position[:2] + opt_uav_speed * opt_det_delay * uav_direction_opt[:2]
    opt_det_z = opt_drop_position[2] - 0.5 * GRAVITY_ACCEL * opt_det_delay**2
    opt_det_position = np.array([opt_det_xy[0], opt_det_xy[1], opt_det_z])
    opt_detonation_time = opt_drop_delay + opt_det_delay
    
    verify_fitness = fitness_function(best_params, target_samples)
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # --- 输出结果重新设计 ---
    
    print("\n" + "="*80)
    print("粒子群优化结果报告".center(80))
    print("="*80)
    print(f"\n[ 性能分析 ]")
    print(f"  - 总耗时: {elapsed_time:.2f} 秒")
    print(f"  - 最佳适应度 (有效遮蔽时长): {best_fitness:.4f} 秒")
    print(f"  - 验证阶段适应度: {verify_fitness:.4f} 秒")
    
    print(f"\n[ 最佳投放策略参数 ]")
    print(f"  - 无人机速度: {opt_uav_speed:.2f} m/s")
    print(f"  - 飞行方向角: {np.degrees(opt_direction_angle):.2f}° ({opt_direction_angle:.4f} rad)")
    print(f"  - 投放延迟时间: {opt_drop_delay:.2f} s")
    print(f"  - 起爆延迟时间: {opt_det_delay:.2f} s")

    print(f"\n[ 关键行动点坐标 ]")
    print(f"  - 投放点坐标: ({opt_drop_position[0]:.2f}, {opt_drop_position[1]:.2f}, {opt_drop_position[2]:.2f})")
    print(f"  - 起爆点坐标: ({opt_det_position[0]:.2f}, {opt_det_position[1]:.2f}, {opt_det_position[2]:.2f})")

    print(f"\n[ 烟幕生效时间 ]")
    print(f"  - 起爆时间: {opt_detonation_time:.2f} s")
    print(f"  - 结束时间: {opt_detonation_time + SMOKE_SPEC['active_duration']:.2f} s")
    print("="*80)
    
    # 绘制优化收敛曲线
    plt.figure(figsize=(10, 6))
    plt.plot(history)
    plt.title('粒子群优化收敛曲线')
    plt.xlabel('迭代次数')
    plt.ylabel('最优遮蔽时长 (s)')
    plt.grid(True)
    plt.savefig('pso_convergence.png')
    plt.show()
