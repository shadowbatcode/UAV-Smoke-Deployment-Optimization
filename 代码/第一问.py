import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl

# -------------------------- 1. 常量与初始参数定义 --------------------------
# 定义重力加速度和数值保护阈值
GRAVITY_ACCEL = 9.8  # 重力加速度 (m/s²)
NUMERIC_THRESHOLD = 1e-12  # 数值计算保护阈值

# 定义假目标（原点）
FALSE_TARGET_POS = np.array([0.0, 0.0, 0.0])  # 假目标（导弹/无人机指向目标）

# 定义真目标：圆柱体参数
TRUE_TARGET_SPEC = {
    "base_center": np.array([0.0, 200.0, 0.0]),  # 底面圆心
    "radius": 7.0,  # 圆柱半径
    "height": 10.0   # 圆柱高度
}

# 无人机参数
UAV_CONFIG = {
    "start_pos": np.array([17800.0, 0.0, 1800.0]),  # 初始位置
    "flight_speed": 120.0,  # 飞行速度(m/s)
    "drop_delay": 1.5,  # 受领任务到投放的时间(s)
    "detonation_delay": 3.6    # 投放至起爆的时间(s)
}

# 烟幕参数
SMOKE_SPEC = {
    "effective_radius": 10.0,  # 有效半径(m)
    "sink_velocity": 3.0,  # 起爆后下沉速度(m/s)
    "active_duration": 20.0  # 有效遮蔽时间(s)
}

# 导弹参数
MISSILE_CONFIG = {
    "start_pos": np.array([20000.0, 0.0, 2000.0]),  # 初始位置
    "flight_speed": 300.0  # 飞行速度(m/s)
}

TIME_STEP = 0.01  # 时间步长（高精度计算）

# -------------------------- 2. 核心位置计算函数 --------------------------
def calc_drop_position(uav_start_pos, uav_speed, drop_delay, false_target):
    """计算烟幕弹投放点（无人机等高度飞行，水平指向原点）"""
    # 水平方向向量（仅xy平面）
    uav_xy = uav_start_pos[:2]
    target_xy = false_target[:2]
    dist_xy = np.linalg.norm(target_xy - uav_xy)
    # 保护：避免除以零
    if dist_xy < NUMERIC_THRESHOLD:
        dir_vec_xy = np.array([0.0, 0.0])
    else:
        dir_vec_xy = (target_xy - uav_xy) / dist_xy
    
    # 投放点计算（z坐标不变）
    flight_dist = uav_speed * drop_delay
    drop_xy = uav_xy + dir_vec_xy * flight_dist
    drop_z = uav_start_pos[2]  # 等高度飞行
    
    return np.array([drop_xy[0], drop_xy[1], drop_z])

def calc_detonation_position(drop_pos, uav_speed, det_delay, gravity, false_target):
    """计算烟幕弹起爆点（投放后水平沿无人机方向，竖直自由落体）"""
    # 水平方向运动（继承无人机速度方向）
    drop_xy = drop_pos[:2]
    target_xy = false_target[:2]
    dist_xy = np.linalg.norm(target_xy - drop_xy)
    if dist_xy < NUMERIC_THRESHOLD:
        dir_vec_xy = np.array([0.0, 0.0])
    else:
        dir_vec_xy = (target_xy - drop_xy) / dist_xy
    
    horizontal_dist = uav_speed * det_delay
    det_xy = drop_xy + dir_vec_xy * horizontal_dist
    
    # 竖直方向自由落体
    drop_height = 0.5 * gravity * det_delay ** 2
    det_z = drop_pos[2] - drop_height
    
    return np.array([det_xy[0], det_xy[1], det_z])

# -------------------------- 3. 高密度真目标采样点生成 --------------------------
def generate_dense_samples(target_spec, num_circle=60, num_height=20):
    """
    生成超高密度采样点（覆盖目标所有表面和内部关键位置）
    - 每个圆周60个点（角度间隔6°）
    - 20个高度层（垂直间隔0.5m）
    - 增加内部网格点采样
    """
    samples = []
    center = target_spec["base_center"]
    radius = target_spec["radius"]
    height = target_spec["height"]
    center_xy = center[:2]
    min_z = center[2]
    max_z = center[2] + height
    
    # 1. 外表面采样
    # 1.1 底面圆周
    theta = np.linspace(0, 2*np.pi, num_circle, endpoint=False)
    for th in theta:
        x = center_xy[0] + radius * np.cos(th)
        y = center_xy[1] + radius * np.sin(th)
        samples.append([x, y, min_z])
    
    # 1.2 顶面圆周
    for th in theta:
        x = center_xy[0] + radius * np.cos(th)
        y = center_xy[1] + radius * np.sin(th)
        samples.append([x, y, max_z])
    
    # 1.3 侧面采样
    heights = np.linspace(min_z, max_z, num_height, endpoint=True)
    for z in heights:
        for th in theta:
            x = center_xy[0] + radius * np.cos(th)
            y = center_xy[1] + radius * np.sin(th)
            samples.append([x, y, z])
    
    # 2. 内部网格点采样
    radii = np.linspace(0, radius, 5, endpoint=True)
    inner_heights = np.linspace(min_z, max_z, 10, endpoint=True)
    inner_thetas = np.linspace(0, 2*np.pi, 12, endpoint=False)
    
    for z in inner_heights:
        for rad in radii:
            for th in inner_thetas:
                x = center_xy[0] + rad * np.cos(th)
                y = center_xy[1] + rad * np.sin(th)
                samples.append([x, y, z])
    
    # 3. 轴线关键点
    samples.extend([
        [center_xy[0], center_xy[1], min_z],
        [center_xy[0], center_xy[1], min_z + height/4],
        [center_xy[0], center_xy[1], min_z + height/2],
        [center_xy[0], center_xy[1], min_z + 3*height/4],
        [center_xy[0], center_xy[1], max_z]
    ])
    
    return np.unique(np.array(samples), axis=0)  # 去重

# -------------------------- 4. 高精度几何判定函数 --------------------------
def is_segment_intersect_sphere(start_point, end_point, sphere_center, sphere_radius):
    """高精度判定线段与球是否相交"""
    segment_vec = end_point - start_point
    sphere_vec = sphere_center - start_point
    
    a = np.dot(segment_vec, segment_vec)
    
    # 处理零长度线段
    if a < NUMERIC_THRESHOLD:
        return np.linalg.norm(sphere_vec) <= sphere_radius + NUMERIC_THRESHOLD
    
    b = -2 * np.dot(segment_vec, sphere_vec)
    c = np.dot(sphere_vec, sphere_vec) - sphere_radius ** 2
    
    discriminant = b ** 2 - 4 * a * c
    if discriminant < -NUMERIC_THRESHOLD:
        return False
    
    if discriminant < 0:
        discriminant = 0
    
    sqrt_d = np.sqrt(discriminant)
    s1 = (-b - sqrt_d) / (2 * a)
    s2 = (-b + sqrt_d) / (2 * a)
    
    return (s1 <= 1.0 + NUMERIC_THRESHOLD) and (s2 >= -NUMERIC_THRESHOLD)

def is_target_shielded(missile_pos, smoke_center, smoke_radius, target_samples):
    """判定真目标是否被完全遮蔽"""
    for sample in target_samples:
        if not is_segment_intersect_sphere(missile_pos, sample, smoke_center, smoke_radius):
            return False
    return True

# -------------------------- 5. 绘图函数 --------------------------
# 设置中文显示和美观样式
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体支持中文
mpl.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def validate_point(point, name):
    """验证点是否包含3个数字坐标"""
    if not isinstance(point, (list, tuple, np.ndarray)) or len(point) != 3:
        raise ValueError(f"{name} 必须是包含3个数字的序列")
    if not all(isinstance(x, (int, float)) for x in point):
        raise ValueError(f"{name} 必须包含数值")

def validate_params(uav_config, missile_config, target_spec):
    """验证输入参数"""
    if not isinstance(uav_config, dict) or not all(k in uav_config for k in ["start_pos", "flight_speed", "drop_delay", "detonation_delay"]):
        raise ValueError("uav_config 必须是包含 'start_pos', 'flight_speed', 'drop_delay', 'detonation_delay' 键的字典")
    if not isinstance(missile_config, dict) or "start_pos" not in missile_config:
        raise ValueError("missile_config 必须是包含 'start_pos' 键的字典")
    if not isinstance(target_spec, dict) or not all(k in target_spec for k in ["base_center", "radius", "height"]):
        raise ValueError("target_spec 必须是包含 'base_center', 'radius', 'height' 键的字典")
    
    validate_point(uav_config["start_pos"], "uav_config['start_pos']")
    validate_point(missile_config["start_pos"], "missile_config['start_pos']")
    validate_point(target_spec["base_center"], "target_spec['base_center']")

def compute_uav_trajectory(uav_config, drop_pos, dt=0.1):
    """计算无人机沿Z轴匀速直线运动轨迹"""
    validate_point(drop_pos, "drop_pos")
    init_pos = np.array(uav_config["start_pos"])
    speed = uav_config["flight_speed"]
    drop_delay = uav_config["drop_delay"]
    
    z_distance = drop_pos[2] - init_pos[2]
    time_to_drop = abs(z_distance / speed) if speed != 0 else drop_delay
    print(f"无人机轨迹计算：初始位置={init_pos}, 投放点={drop_pos}, Z距离={z_distance}, 时间={time_to_drop:.2f}s")
    
    t = np.arange(0, time_to_drop + dt, dt)
    trajectory = np.zeros((len(t), 3))
    trajectory[:, 0] = init_pos[0]
    trajectory[:, 1] = init_pos[1]
    trajectory[:, 2] = init_pos[2] + np.sign(z_distance) * speed * t
    
    print(f"无人机轨迹点数：{len(trajectory)}, 起点={trajectory[0]}, 终点={trajectory[-1]}")
    return trajectory

def compute_projectile_trajectory(drop_pos, det_pos, det_delay, dt=0.1):
    """计算投放点到引爆点的平抛运动轨迹"""
    validate_point(drop_pos, "drop_pos")
    validate_point(det_pos, "det_pos")
    drop_pos = np.array(drop_pos)
    det_pos = np.array(det_pos)
    
    t = np.arange(0, det_delay + dt, dt)
    v_xy = (det_pos[:2] - drop_pos[:2]) / det_delay if det_delay > 0 else np.zeros(2)
    gravity = 9.8
    z = drop_pos[2] - 0.5 * gravity * t**2
    trajectory = np.zeros((len(t), 3))
    trajectory[:, 0] = drop_pos[0] + v_xy[0] * t
    trajectory[:, 1] = drop_pos[1] + v_xy[1] * t
    trajectory[:, 2] = z
    return trajectory

def compute_smoke_trajectory(det_pos, smoke_center, dt=0.1):
    """计算引爆点到烟雾中心的直线运动轨迹"""
    if smoke_center is None:
        return None
    validate_point(det_pos, "det_pos")
    validate_point(smoke_center, "smoke_center")
    det_pos = np.array(det_pos)
    smoke_center = np.array(smoke_center)
    
    t = np.arange(0, 1.0 + dt, dt)
    trajectory = det_pos + (smoke_center - det_pos) * (t / 1.0)[:, np.newaxis]
    return trajectory

def plot_uav_path(ax, uav_config, drop_pos):
    """绘制无人机轨迹和方向箭头"""
    trajectory = compute_uav_trajectory(uav_config, drop_pos)
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
            'b-', label="无人机路径", linewidth=2)
    
    mid_idx = len(trajectory) // 2
    start = trajectory[mid_idx - 1]
    end = trajectory[mid_idx]
    ax.quiver(start[0], start[1], start[2], 
              end[0] - start[0], end[1] - start[1], end[2] - start[2],
              color='b', arrow_length_ratio=0.1)

def plot_points_and_trajectories(ax, drop_pos, det_pos, smoke_center, det_delay):
    """绘制关键点、平抛轨迹、烟雾轨迹和烟雾球体"""
    validate_point(drop_pos, "drop_pos")
    validate_point(det_pos, "det_pos")
    ax.scatter(*drop_pos, c='g', marker='o', s=100, label="投放点")
    ax.scatter(*det_pos, c='r', marker='^', s=100, label="引爆点")
    
    # 平抛轨迹
    projectile_trajectory = compute_projectile_trajectory(drop_pos, det_pos, det_delay)
    ax.plot(projectile_trajectory[:, 0], projectile_trajectory[:, 1], projectile_trajectory[:, 2],
            'b--', label="平抛轨迹", linewidth=1.5)
    
    # 烟雾轨迹和球体
    if smoke_center is not None:
        smoke_trajectory = compute_smoke_trajectory(det_pos, smoke_center)
        ax.plot(smoke_trajectory[:, 0], smoke_trajectory[:, 1], smoke_trajectory[:, 2],
                'b--', label="烟雾轨迹", linewidth=1.5)
        ax.scatter(*smoke_center, c='orange', marker='x', s=100, label="烟雾中心")
        
        # 绘制半径10m的半透明灰色球体
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        x = smoke_center[0] + 10 * np.outer(np.cos(u), np.sin(v))
        y = smoke_center[1] + 10 * np.outer(np.sin(u), np.sin(v))
        z = smoke_center[2] + 10 * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='gray', alpha=0.5, rstride=1, cstride=1)

def plot_missile_path(ax, missile_config):
    """绘制导弹轨迹"""
    ax.plot([missile_config["start_pos"][0], 0],
            [missile_config["start_pos"][1], 0],
            [missile_config["start_pos"][2], 0],
            'm--', label="导弹路径", linewidth=2)

def plot_target_cylinder(ax, target_spec):
    """绘制目标圆柱体"""
    theta = np.linspace(0, 2 * np.pi, 20)
    z = np.linspace(target_spec["base_center"][2], target_spec["base_center"][2] + target_spec["height"], 10)
    theta, z = np.meshgrid(theta, z)
    x = target_spec["base_center"][0] + target_spec["radius"] * np.cos(theta)
    y = target_spec["base_center"][1] + target_spec["radius"] * np.sin(theta)
    ax.plot_surface(x, y, z, alpha=0.3, color="cyan", rstride=1, cstride=1)

def set_axes_properties(ax, trajectory, drop_pos, det_pos, smoke_center):
    """设置坐标轴属性，聚焦于关键区域"""
    ax.set_xlabel("X (米)")
    ax.set_ylabel("Y (米)")
    ax.set_zlabel("Z (米)")
    ax.set_box_aspect([1, 1, 1])
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True)
    
    points = np.vstack([trajectory[-1:], [drop_pos], [det_pos]])
    if smoke_center is not None:
        points = np.vstack([points, [smoke_center]])
    
    margin = 50
    for i, (axis, label) in enumerate([(ax.set_xlim, "X"), (ax.set_ylim, "Y"), (ax.set_zlim, "Z")]):
        min_val, max_val = np.min(points[:, i]), np.max(points[:, i])
        if min_val == max_val:
            min_val -= 10
            max_val += 10
        axis(min_val - margin, max_val + margin)
    
    ax.view_init(elev=30, azim=45)

def set_equal_aspect_3d(ax):
    """让 3D 坐标轴等比例，避免球体显示成椭球"""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    max_range = max([x_range, y_range, z_range]) / 2.0

    mid_x = np.mean(x_limits)
    mid_y = np.mean(y_limits)
    mid_z = np.mean(z_limits)

    ax.set_xlim3d([mid_x - max_range, mid_x + max_range])
    ax.set_ylim3d([mid_y - max_range, mid_y + max_range])
    ax.set_zlim3d([mid_z - max_range, mid_z + max_range])

def plot_scene(uav_config, drop_pos, det_pos, missile_config, target_spec, smoke_center=None):
    """
    绘制3D场景，包括无人机沿Z轴运动轨迹、平抛轨迹、烟雾轨迹、导弹路径、目标圆柱体和烟雾球体
    """
    validate_params(uav_config, missile_config, target_spec)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    trajectory = compute_uav_trajectory(uav_config, drop_pos)
    
    plot_uav_path(ax, uav_config, drop_pos)
    plot_points_and_trajectories(ax, drop_pos, det_pos, smoke_center, uav_config["detonation_delay"])
    plot_missile_path(ax, missile_config)
    plot_target_cylinder(ax, target_spec)
    
    set_axes_properties(ax, trajectory, drop_pos, det_pos, smoke_center)
    set_equal_aspect_3d(ax)
    plt.tight_layout()
    plt.show()

# -------------------------- 6. 主计算流程 --------------------------
if __name__ == "__main__":
    # 步骤1：计算投放点和起爆点
    drop_position = calc_drop_position(
        uav_start_pos=UAV_CONFIG["start_pos"],
        uav_speed=UAV_CONFIG["flight_speed"],
        drop_delay=UAV_CONFIG["drop_delay"],
        false_target=FALSE_TARGET_POS
    )
    
    detonation_position = calc_detonation_position(
        drop_pos=drop_position,
        uav_speed=UAV_CONFIG["flight_speed"],
        det_delay=UAV_CONFIG["detonation_delay"],
        gravity=GRAVITY_ACCEL,
        false_target=FALSE_TARGET_POS
    )
    
    print("=== 基础位置信息 ===")
    print(f"无人机初始位置：{UAV_CONFIG['start_pos'].round(4)}")
    print(f"烟幕弹投放点：{drop_position.round(4)}")
    print(f"烟幕弹起爆点：{detonation_position.round(4)}")
    print(f"假目标位置：{FALSE_TARGET_POS}")

    # 步骤2：生成高密度真目标采样点
    target_samples = generate_dense_samples(TRUE_TARGET_SPEC)
    print(f"\n=== 采样点信息 ===")
    print(f"真目标采样点总数：{len(target_samples)}（含外表面和内部点）")

    # 步骤3：导弹飞行方向
    missile_vector = FALSE_TARGET_POS - MISSILE_CONFIG["start_pos"]
    missile_distance = np.linalg.norm(missile_vector)
    if missile_distance < NUMERIC_THRESHOLD:
        missile_direction = np.array([0.0, 0.0, 0.0])
    else:
        missile_direction = missile_vector / missile_distance
    print(f"\n=== 导弹信息 ===")
    print(f"导弹初始位置：{MISSILE_CONFIG['start_pos'].round(4)}")
    print(f"导弹飞行方向向量：{missile_direction.round(6)}")

    # 步骤4：时间范围定义
    detonation_time = UAV_CONFIG["drop_delay"] + UAV_CONFIG["detonation_delay"]
    time_start = detonation_time
    time_end = detonation_time + SMOKE_SPEC["active_duration"]
    time_steps = np.arange(time_start, time_end + TIME_STEP, TIME_STEP)
    print(f"\n=== 时间范围 ===")
    print(f"起爆时刻：{detonation_time:.2f}s")
    print(f"有效时间窗口：[{time_start:.2f}s, {time_end:.2f}s]，共{len(time_steps)}个时间步")

    # 步骤5：高精度迭代计算
    total_shield_time = 0.0
    shield_log = []
    was_shielded = False
    shield_intervals = []

    for t in time_steps:
        # 计算导弹位置
        missile_pos = MISSILE_CONFIG["start_pos"] + missile_direction * MISSILE_CONFIG["flight_speed"] * t

        # 计算烟幕位置
        sink_time = t - detonation_time
        smoke_center = np.array([
            detonation_position[0],
            detonation_position[1],
            detonation_position[2] - SMOKE_SPEC["sink_velocity"] * sink_time
        ])

        # 判定遮蔽状态
        is_shielded = is_target_shielded(missile_pos, smoke_center, SMOKE_SPEC["effective_radius"], target_samples)
        
        if is_shielded:
            total_shield_time += TIME_STEP
            shield_log.append({
                "t": round(t, 3),
                "missile_pos": missile_pos.round(4),
                "smoke_center": smoke_center.round(4)
            })
        
        if is_shielded and not was_shielded:
            shield_intervals.append({"start": t})
        elif not is_shielded and was_shielded:
            if shield_intervals:
                shield_intervals[-1]["end"] = t - TIME_STEP
        
        was_shielded = is_shielded

    if shield_intervals and "end" not in shield_intervals[-1]:
        shield_intervals[-1]["end"] = time_end

    # 步骤6：输出结果
    print("\n" + "="*80)
    print(f"【最终结果】真目标被有效遮蔽的总时长：{total_shield_time:.4f} 秒")
    print("="*80)

    print("\n=== 遮蔽时间段详情 ===")
    if not shield_intervals:
        print("无有效遮蔽时间段")
    else:
        for i, interval in enumerate(shield_intervals, 1):
            duration = interval["end"] - interval["start"]
            print(f"第{i}段：{interval['start']:.4f}s ~ {interval['end']:.4f}s，时长：{duration:.4f}s")

    print("\n=== 采样时刻状态示例 ===")
    if shield_log:
        print("前3个有效时刻：")
        for log in shield_log[:3]:
            print(f"t={log['t']}s | 导弹位置：{log['missile_pos']} | 烟幕中心：{log['smoke_center']}")
        
        print("\n最后3个有效时刻：")
        for log in shield_log[-3:]:
            print(f"t={log['t']}s | 导弹位置：{log['missile_pos']} | 烟幕中心：{log['smoke_center']}")
    else:
        print("无有效遮蔽时刻")

    plot_scene(UAV_CONFIG, drop_position, detonation_position, MISSILE_CONFIG, TRUE_TARGET_SPEC, smoke_center)
    print(UAV_CONFIG)
