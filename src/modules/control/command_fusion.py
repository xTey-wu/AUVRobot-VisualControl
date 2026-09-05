from typing import Tuple, Union

"""
简化的指令融合模块（Command Fusion）- v2.0

用途：
- 将各状态处理器返回的 y（深度轴）与 yaw（偏航轴）字段，选择对应的PID控制器进行融合
- 输出可直接下发的 0~255 电机值

支持的指令类型：
- 视觉控制：('vision', pixel_error) → 使用视觉PID
- 全局控制：'global_hold' → 使用全局PID  
- 数值：直接输出电机值
- 关闭：'no_yaw' → 输出中性值 128

简化设计：
- 移除复杂的控制模式切换机制
- 直接根据指令类型选择PID控制器
- 简化融合逻辑，提高可维护性
"""

# y 与 yaw 字段的联合类型：
# - 可以是数值（int）：直接视为 0~255 的电机值
# - 可以是字符串：用于表示特殊模式（'global_hold'、'no_yaw'）
# - 也可以是二元组 (标记, 值)：用于显式标记视觉控制（'vision', pixel_error）
YField = Union[int, str, Tuple[str, float]]
YawField = Union[int, str, Tuple[str, float]]


def clamp_u8(value: int) -> int:
    """将输入数值夹紧到 0~255 的无符号 8 位范围。确保任何来源的值都安全可下发。"""
    return int(max(0, min(255, int(value))))


def normalize_angle_error(target_angle: float, current_angle: float) -> float:
    """
    计算规范化的角度误差，处理0°/360°环绕问题。
    
    Args:
        target_angle: 目标角度 (0-360度)
        current_angle: 当前角度 (0-360度)
        
    Returns:
        规范化的角度误差 (-180到180度)，正值表示需要顺时针转动
    """
    error = target_angle - current_angle
    # 将误差规范化到 [-180°, 180°] 范围内，选择最短路径
    while error > 180:
        error -= 360
    while error < -180:
        error += 360
    return error


def fuse_depth_yaw(y: YField, yaw: YawField, auv, vision_yaw_pid, global_depth_pid, global_yaw_pid) -> Tuple[int, int, str]:
    """
    简化的深度与偏航融合函数 - v2.0
    
    参数：
    - y:   深度轴字段，可为：
           1) 'global_hold' → 使用全局PID基于传感器数据
           2) 数值 → 直接输出电机值
    - yaw: 偏航轴字段，可为：
           1) ('vision', pixel_error) → 使用视觉PID基于像素误差
           2) 'global_hold' → 使用全局PID基于传感器数据
           3) 'no_yaw' → 输出 128（关闭偏航）
           4) 数值 → 直接输出电机值
    
    返回：
    - (final_y, final_yaw, control_mode_str)
    """
    
    # --- 深度融合 ---
    final_y = 128
    depth_mode = "NEUTRAL"
    
    if y == 'global_hold':
        # 全局控深：使用当前阶段的目标深度
        target_depth = auv.get_current_target_depth()
        depth_error_m = auv.current_filtered_depth - target_depth
        depth_error_cm = depth_error_m * 100.0
        speed_y, _ = global_depth_pid.update(depth_error_cm)
        final_y = clamp_u8(speed_y)
        depth_mode = "GLOBAL"
        print(f"  [全局控深] 目标: {target_depth:.2f}m, 当前: {auv.current_filtered_depth:.2f}m, 误差: {depth_error_cm:.1f}cm")
        
    else:
        # 直接数值输出
        try:
            final_y = clamp_u8(int(y))
            depth_mode = "DIRECT"
        except (ValueError, TypeError):
            final_y = 128
            depth_mode = "ERROR"

    # --- 偏航融合 ---
    final_yaw = 128
    yaw_mode = "NEUTRAL"
    
    if isinstance(yaw, (tuple, list)) and len(yaw) == 2 and yaw[0] == 'vision':
        # 视觉控yaw：使用视觉PID处理像素误差
        pixel_error = yaw[1]
        speed_yaw, _ = vision_yaw_pid.update(pixel_error)
        final_yaw = clamp_u8(speed_yaw)
        yaw_mode = "VISION"
        print(f"  [视觉控yaw] 像素误差: {pixel_error:.1f}px, 输出: {final_yaw}")
        
    elif yaw == 'global_hold':
        # 全局控yaw：使用状态机的目标角度
        target_yaw = auv.target_yaw
        yaw_error = normalize_angle_error(target_yaw, auv.current_yaw)
        speed_yaw, _ = global_yaw_pid.update(yaw_error)
        final_yaw = clamp_u8(speed_yaw)
        yaw_mode = "GLOBAL"
        print(f"  [全局控yaw] 目标: {target_yaw:.1f}°, 当前: {auv.current_yaw:.1f}°, 误差: {yaw_error:.1f}°")
        
    elif yaw == 'no_yaw':
        # 关闭yaw
        final_yaw = 128
        yaw_mode = "OFF"
        
    else:
        # 直接数值输出
        try:
            final_yaw = clamp_u8(int(yaw))
            yaw_mode = "DIRECT"
        except (ValueError, TypeError):
            final_yaw = 128
            yaw_mode = "ERROR"
    
    # 生成控制模式字符串
    control_mode_str = f"D:{depth_mode}/Y:{yaw_mode}"

    return final_y, final_yaw, control_mode_str


def encode_command(fwd: int, x: int, y: int, yaw: int) -> bytes:
    """将 (fwd, x, y, yaw) 四通道打包为下发字节序列，带固定帧头/帧尾。"""
    return bytes([0xab, clamp_u8(fwd), clamp_u8(x), clamp_u8(y), clamp_u8(yaw), 0xcd])
