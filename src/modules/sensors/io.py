from collections import deque
from typing import Optional, Tuple

"""
传感器读数处理模块：
- update_depth(auv, depth): 维护最近深度队列与滑动均值
- update_yaw(auv, yaw): 维护最近 yaw 队列，并在收集满 5 帧后设置 yaw_hold_ready 与 target_yaw
"""


def update_depth(auv, depth: Optional[float]) -> None:
    if depth is None:
        return
    # 维护最近 N 次深度读数
    if not hasattr(auv, 'depth_readings') or auv.depth_readings is None:
        auv.depth_readings = deque(maxlen=5)
    auv.depth_readings.append(depth)
    if len(auv.depth_readings) > 0:
        auv.current_filtered_depth = sum(auv.depth_readings) / len(auv.depth_readings)


def update_yaw(auv, yaw: Optional[float]) -> bool:
    """
    更新 yaw 读数并在首次收集满 5 帧时完成 yaw_hold 初始化。
    返回值：
      - True  表示本次调用刚完成了 5 帧初始化（只会触发一次）
      - False 表示未发生初始化事件
    """
    if yaw is None:
        return False
    # 保存当前 yaw
    auv.current_yaw = yaw

    # 维护最近 N 次 yaw 读数用于初始化保持目标
    if not hasattr(auv, 'yaw_readings') or auv.yaw_readings is None:
        auv.yaw_readings = deque(maxlen=5)
    auv.yaw_readings.append(yaw)

    # 启动阶段：采集前 5 帧，取均值作为 yaw_hold 目标，然后再启用 yaw 保持
    if not getattr(auv, 'yaw_hold_ready', False) and len(auv.yaw_readings) >= 5:
        avg_yaw = sum(auv.yaw_readings) / len(auv.yaw_readings)
        auv.target_yaw = avg_yaw
        auv.initial_yaw = avg_yaw  # 记录程序启动时的初始yaw
        auv.yaw_hold_ready = True
        return True
    return False
