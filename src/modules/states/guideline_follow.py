"""
GUIDELINE_FOLLOW状态模块 - 简化版v2.0
实现状态切换优先级 + 右转逻辑 + 简化循线控制
"""
from typing import Optional, Dict, Any, Tuple, Union
import time as _t
from src.modules.control.command_fusion import normalize_angle_error

# 指令类型定义
YField = Union[int, str, Tuple[str, float]]
YawField = Union[int, str, Tuple[str, float]]
Command = Tuple[int, int, YField, YawField]


def handle_state_transitions(auv, detections_forward, is_target_ball, State) -> bool:
    """
    处理最高优先级的状态切换逻辑
    返回True表示发生了状态切换，False表示继续当前状态
    """
    # 1. 未撞球状态：检测目标小球
    if auv.circle_kicked == 0:
        ball_found = any(is_target_ball(r) for r in detections_forward)
        if ball_found:
            auv.ball_detection_counter += 1
            print(f"    [状态切换] 检测到目标小球，计数器: {auv.ball_detection_counter}")
        else:
            auv.ball_detection_counter = 0
        
        if auv.ball_detection_counter >= auv.detection_confidence_threshold:
            print("    [状态切换] 连续检测到目标小球，切换到CIRCLE_KICK状态")
            auv.ball_detection_counter = 0
            auv._transition(State.CIRCLE_KICK)
            return True
    
    # 2. 已撞球状态：检测门框
    elif auv.circle_kicked == 1:
        doorframe_found = any(r.get('class_name') == 'doorframe' for r in detections_forward)
        if doorframe_found:
            auv.doorframe_detection_counter += 1
            print(f"    [状态切换] 检测到门框，计数器: {auv.doorframe_detection_counter}")
        else:
            auv.doorframe_detection_counter = 0
        
        if auv.doorframe_detection_counter >= auv.detection_confidence_threshold:
            print("    [状态切换] 连续检测到门框，切换到DOORFRAME_PASSTHROUGH状态")
            auv.doorframe_detection_counter = 0
            auv._transition(State.DOORFRAME_PASSTHROUGH)
            return True
    
    return False


def handle_from_circle_kick_logic(auv, global_yaw_pid) -> Optional[Command]:
    """
    处理从CIRCLE_KICK进入时的特殊逻辑：后退3秒 + 右转90度
    返回None表示特殊处理完成，返回Command表示仍在执行特殊处理
    """
    if not auv.from_circle_kick:
        return None
    
    # 初始化子状态（仅在第一次进入时）
    if auv.from_circle_kick_sub_state is None:
        auv.from_circle_kick_sub_state = "BACKWARD"
        auv.backward_start_time = None
        # 设置目标yaw为初始yaw（保持启动时的角度）
        auv.target_yaw = auv.initial_yaw
        print(f"    [特殊处理开始] 进入后退阶段，目标yaw: {auv.target_yaw:.1f}°（初始yaw）")
        # 重置全局yaw_pid
        global_yaw_pid.reset()
    
    # ========== 阶段1：后退3秒 ==========
    if auv.from_circle_kick_sub_state == "BACKWARD":
        # 记录后退开始时间
        if auv.backward_start_time is None:
            auv.backward_start_time = _t.time()
            print(f"    [后退阶段] 开始后退，持续 {auv.backward_duration}s")
        
        backward_elapsed = _t.time() - auv.backward_start_time
        
        if backward_elapsed < auv.backward_duration:
            # 后退中
            print(f"    [后退中] {backward_elapsed:.1f}s / {auv.backward_duration}s，保持初始yaw: {auv.target_yaw:.1f}°")
            # 后退指令：speed_forward < 128表示后退
            return (50, 128, "global_hold", "global_hold")
        else:
            # 后退完成，切换到右转阶段
            print("    [后退完成] 切换到右转阶段")
            # 计算右转目标yaw（右转是角度减）
            auv.target_yaw = (auv.current_yaw - 90) % 360
            print(f"    [右转阶段] 设置右转目标yaw: {auv.current_yaw:.1f}° - 90° = {auv.target_yaw:.1f}°")
            # 重置全局yaw_pid（清除后退阶段的积分误差）
            global_yaw_pid.reset()
            # 切换子状态
            auv.from_circle_kick_sub_state = "TURN_RIGHT"
            # 重置稳定计时器
            auv.yaw_stable_start_time = None
            # 继续执行右转逻辑
    
    # ========== 阶段2：右转90度 + 稳定判断 ==========
    if auv.from_circle_kick_sub_state == "TURN_RIGHT":
        # 检查是否已经完成右转（需要yaw误差稳定保持2秒）
        yaw_error = normalize_angle_error(auv.target_yaw, auv.current_yaw)
        print(f"    [右转逻辑] 目标yaw: {auv.target_yaw:.1f}°, 当前yaw: {auv.current_yaw:.1f}°, 误差: {yaw_error:.1f}°")
        
        # 右转稳定判断逻辑
        yaw_stable_threshold = 10  # 误差阈值
        yaw_stable_duration = 2.0   # 需要稳定的时间
        
        if abs(yaw_error) < yaw_stable_threshold:
            # 误差在阈值内，检查稳定时间
            if auv.yaw_stable_start_time is None:
                auv.yaw_stable_start_time = _t.time()
                print(f"    [右转稳定] 开始计时，需要稳定 {yaw_stable_duration}s")
            
            stable_elapsed = _t.time() - auv.yaw_stable_start_time
            if stable_elapsed >= yaw_stable_duration:
                # 右转稳定完成，清理并完成特殊处理
                print(f"    [右转完成] 稳定 {stable_elapsed:.1f}s，特殊处理完成，进入正常循线模式")
                # 清理所有特殊处理相关的变量
                auv.from_circle_kick = False
                auv.from_circle_kick_sub_state = None
                auv.backward_start_time = None
                auv.yaw_stable_start_time = None
                return None  # 进入正常循线逻辑
            else:
                print(f"    [右转稳定中] {stable_elapsed:.1f}s / {yaw_stable_duration}s")
                # 继续等待稳定，保持中性前进
                return (128, 128, "global_hold", "global_hold")
        else:
            # 误差超出阈值，重置稳定计时器
            auv.yaw_stable_start_time = None
            print("    [右转中] 继续执行右转...")
            return (128, 128, "global_hold", "global_hold")  # 保持中性前进 + 全局控yaw
    
    # 兜底返回（理论上不会执行到这里）
    return (128, 128, "global_hold", "global_hold")


def handle_lateral_search_logic(auv, detections_forward, img_forward) -> Optional[Command]:
    """
    处理过门后的左右移搜索逻辑
    返回None表示搜索完成，返回Command表示仍在执行搜索
    """
    if not hasattr(auv, 'lateral_search_active') or not auv.lateral_search_active:
        return None
    
    # 初始化开始时间
    if auv.lateral_search_start_time is None:
        auv.lateral_search_start_time = _t.time()
        direction_str = "左移" if auv.lateral_search_speed > 128 else "右移"
        print(f"    [左右移搜索] 开始{direction_str}搜索，持续{auv.lateral_search_duration}秒")
    
    elapsed_time = _t.time() - auv.lateral_search_start_time
    
    # 检测门框是否在视野中心±120px范围内
    doorframes = [r for r in detections_forward if r.get('class_name') == 'doorframe']
    if doorframes:
        # 找出最大门框
        target_doorframe = max(doorframes, key=lambda x: x.get('area', 0))
        center_x = target_doorframe['center'][0]
        img_center_x = img_forward.shape[1] / 2
        error_x = abs(img_center_x - center_x)
        
        # 检查门框是否在中心±120px范围内
        if error_x < 120:
            auv.lateral_search_doorframe_counter += 1
            print(f"    [左右移搜索] 门框在中心范围内（误差{error_x:.1f}px），计数器: {auv.lateral_search_doorframe_counter}/3")
            
            # 连续3帧检测到门框在中心，提前结束搜索
            if auv.lateral_search_doorframe_counter >= 3:
                print(f"    [左右移搜索] 连续3帧检测到门框对准，提前完成搜索（已用时{elapsed_time:.1f}s）")
                # 清理标志
                auv.lateral_search_active = False
                auv.lateral_search_start_time = None
                auv.lateral_search_doorframe_counter = 0
                return None  # 搜索完成
        else:
            # 门框不在范围内，重置计数器
            auv.lateral_search_doorframe_counter = 0
            print(f"    [左右移搜索] 门框偏离中心（误差{error_x:.1f}px > 120px），重置计数器")
    else:
        # 未检测到门框，重置计数器
        auv.lateral_search_doorframe_counter = 0
    
    # 检查是否超时
    if elapsed_time >= auv.lateral_search_duration:
        print(f"    [左右移搜索] 时间到（{elapsed_time:.1f}s），搜索完成")
        # 清理标志
        auv.lateral_search_active = False
        auv.lateral_search_start_time = None
        auv.lateral_search_doorframe_counter = 0
        return None  # 搜索完成
    
    # 继续左右移搜索
    print(f"    [左右移搜索] 搜索中... {elapsed_time:.1f}s / {auv.lateral_search_duration}s")
    # 返回控制指令：定速前进 + 左右移 + 全局控深 + 全局控yaw
    return (180, auv.lateral_search_speed, "global_hold", "global_hold")


def handle_front_cam_follow(auv, img_forward, detections_forward, left_or_right_pid, pid_viz_enabled) -> Optional[Command]:
    """
    处理前置摄像头循线逻辑
    """
    guidelines = [r for r in detections_forward if r.get('class_name') == 'guideline']
    guideline_f = max(guidelines, key=lambda x: x.get('area', 0)) if guidelines else None
    
    if guideline_f:
        # 发现引导线
        auv.front_cam_has_seen_guideline = True
        center_x, center_y = guideline_f['center']
        print(f"    [前置循线] 发现引导线，中心坐标: ({center_x}, {center_y})")
        
        # 计算x轴误差（线x - 视野中心x）
        img_center_x = img_forward.shape[1] / 2
        error_x = img_center_x - center_x # 引导线在左侧，error_x为正，此时pid输出为正，表示向左移动
        speed_x, pid_out_x = left_or_right_pid.update(error_x)
        
        # PID可视化
        if pid_viz_enabled and hasattr(auv, 'pid_visualizer'):
            auv.pid_visualizer.update_data('x', error_x, pid_out_x)
        
        print(f"    [前置循线] X误差: {error_x:.1f}px, PID输出: {speed_x}")
        
        # 固定前进速度
        speed_forward = 180
        
        # 保存有效指令
        command = (speed_forward, speed_x, "global_hold", "global_hold")
        auv.last_known_command = command
        return command
    
    else:
        # 引导线丢失
        if auv.front_cam_has_seen_guideline:
            print("    [前置循线] 丢失曾跟踪的引导线，准备切换到下置摄像头")
            return None  # 触发切换到下置摄像头逻辑
        else:
            print("    [前置循线] 搜索引导线中，定速前进")
            return (190, 128, "global_hold", "global_hold")


def handle_bottom_cam_align(auv, guideline_bottom_result, bottom_cam_available, State) -> Optional[Command]:
    """
    处理下置摄像头对准逻辑
    """
    if not bottom_cam_available:
        print("    [下置对准] 下置摄像头不可用，继续定速前进")
        return (180, 128, "global_hold", "global_hold")
    
    # 检查搜索超时
    if auv.bottom_cam_search_start_time is None:
        auv.bottom_cam_search_start_time = _t.time()
    
    search_elapsed = _t.time() - auv.bottom_cam_search_start_time
    if search_elapsed > 10.0:  # 10秒超时
        print("    [下置对准] 搜索超时，返回前置摄像头循线")
        auv.guideline_sub_state = "FRONT_CAM_FOLLOW"
        auv.bottom_cam_search_start_time = None
        return (160, 128, "global_hold", "global_hold")
    
    if guideline_bottom_result and guideline_bottom_result.get('detected'):
        # 找到引导线，执行对准
        x_error = guideline_bottom_result.get('x_error', 0)
        y_error = guideline_bottom_result.get('y_error', 0)
        yaw_error = guideline_bottom_result.get('yaw_error', 0)
        
        print(f"    [下置对准] 引导线误差 - X:{x_error:.1f}px, Y:{y_error:.1f}px, Yaw:{yaw_error:.1f}°")
        
        # 检查是否对准完成
        if abs(x_error) < 30 and abs(y_error) < 30 and abs(yaw_error) < 3:
            print("    [下置对准] 对准完成，切换到ACTION_FORWARD执行冲刺前进")
            # 切换到ACTION_FORWARD状态执行1.5秒冲刺
            auv._transition(State.ACTION_FORWARD)
            return None
        else:
            # 继续对准（这里简化处理，实际需要三轴PID）
            return (128, 128, "global_hold", "global_hold")  # 悬停对准
    else:
        print("    [下置对准] 搜索引导线中...")
        return (128, 128, "global_hold", "global_hold")  # 悬停搜索


def handle(auv, perception_bundle: Dict[str, Any], *, left_or_right_pid, vision_yaw_pid, global_yaw_pid, forward_pid, bottom_cam_available: bool, pid_viz_enabled: bool, is_target_ball, State) -> Optional[Command]:
    """
    GUIDELINE_FOLLOW状态处理器 - 简化版v2.0
    
    处理流程：
    1. 最高优先级：状态切换检测
    2. 过门后的左右移搜索（备用方案，可配置）
    3. 从CIRCLE_KICK进入的特殊处理（右转+右移）
    4. 正常循线逻辑（前置摄像头 + 下置摄像头对准）
    """
    print("  [GUIDELINE_FOLLOW] 循线状态...")
    
    # 获取感知数据
    img_forward = perception_bundle['frame_forward']
    img_bottom = perception_bundle['frame_bottom']
    detections_forward = perception_bundle['detections_forward']
    guideline_bottom_result = perception_bundle['guideline_bottom']
    
    # 1. 最高优先级：状态切换
    if handle_state_transitions(auv, detections_forward, is_target_ball, State):
        return None  # 发生状态切换，退出当前处理
    
    # 2. 过门后的左右移搜索（备用方案）
    lateral_search_command = handle_lateral_search_logic(auv, detections_forward, img_forward)
    if lateral_search_command is not None:
        return lateral_search_command
    
    # 3. 从CIRCLE_KICK进入的特殊处理
    special_command = handle_from_circle_kick_logic(auv, global_yaw_pid)
    if special_command is not None:
        return special_command
    
    # 4. 正常循线逻辑
    if auv.guideline_sub_state == "FRONT_CAM_FOLLOW":
        command = handle_front_cam_follow(auv, img_forward, detections_forward, left_or_right_pid, pid_viz_enabled)
        
        if command is None and auv.front_cam_has_seen_guideline:
            # 需要切换到下置摄像头对准
            print("    [状态切换] 切换到下置摄像头对准模式")
            auv.guideline_sub_state = "BOTTOM_CAM_ALIGN"
            auv.bottom_cam_search_start_time = None
            return handle_bottom_cam_align(auv, guideline_bottom_result, bottom_cam_available, State)
        else:
            return command
    
    elif auv.guideline_sub_state == "BOTTOM_CAM_ALIGN":
        return handle_bottom_cam_align(auv, guideline_bottom_result, bottom_cam_available, State)
    
    else:
        # 未知子状态，重置为前置摄像头循线
        print("    [错误] 未知子状态，重置为前置摄像头循线")
        auv.guideline_sub_state = "FRONT_CAM_FOLLOW"
        return (180, 128, "global_hold", "global_hold")