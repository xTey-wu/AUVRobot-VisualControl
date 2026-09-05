"""
简化的PID设置模块 - v2.0
移除复杂的预热和门控机制，只创建基本的PID控制器实例
"""
from typing import Dict
from src.modules.control.control import PIDController


def create_pids() -> Dict[str, PIDController]:
    """
    创建并返回系统使用的简化PID控制器实例
    
    控制器分类：
    - left_or_right_pid: 左右平移控制（基于像素误差）
    - vision_yaw_pid: 视觉偏航控制（基于像素误差）
    - global_depth_pid: 全局深度控制（基于传感器数据）
    - global_yaw_pid: 全局偏航控制（基于传感器数据）
    - forward_pid: 前进距离控制（基于面积误差）
    """
    # 视觉控制PID：基于像素误差，快速响应
    left_or_right_pid = PIDController(kp=0.5, ki=0, kd=0.17)
    vision_yaw_pid = PIDController(kp=0.028, ki=0, kd=0.015)
    forward_pid = PIDController(kp=0.0025, ki=0, kd=0.001)
    
    # 全局控制PID：基于传感器数据，精确保持
    global_depth_pid = PIDController(kp=30, ki=0.01, kd=2)  # cm误差参数
    global_yaw_pid = PIDController(kp=1.2, ki=0.02, kd=0.6)  # 角度误差参数
    
    print("[PID初始化] 简化PID控制器创建完成")
    print(f"  视觉PID: left_right(kp={left_or_right_pid.kp}), yaw(kp={vision_yaw_pid.kp})")
    print(f"  全局PID: depth(kp={global_depth_pid.kp}), yaw(kp={global_yaw_pid.kp})")
    print(f"  距离PID: forward(kp={forward_pid.kp})")
    
    return {
        'left_or_right_pid': left_or_right_pid,
        'vision_yaw_pid': vision_yaw_pid,
        'global_depth_pid': global_depth_pid,
        'global_yaw_pid': global_yaw_pid,
        'forward_pid': forward_pid,
    }
