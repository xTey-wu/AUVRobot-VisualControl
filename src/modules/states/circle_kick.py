"""
CIRCLE_KICK状态模块 - 简化版v2.0
实现全局depth_pid + 视觉yaw_pid + 定速前进的撞球逻辑

支持两种小球识别方案：
方案1：YOLO模型识别（默认）
方案2：颜色阈值识别（备选）
"""
from typing import Optional, Dict, Any, Tuple, Union
import time as _t
import cv2
import numpy as np

# 指令类型定义
YField = Union[int, str, Tuple[str, float]]
YawField = Union[int, str, Tuple[str, float]]
Command = Tuple[int, int, YField, YawField]

# ==================== 配置参数 ====================
# 小球识别方案选择：1=YOLO模型识别，2=颜色阈值识别
BALL_DETECTION_METHOD = 1

# 颜色阈值配置（HSV色彩空间）
# 红色小球的HSV阈值范围（红色在HSV中分为两段）
RED_HSV_LOWER1 = np.array([0, 30, 30])      # 低色调红色下限
RED_HSV_UPPER1 = np.array([20, 255, 255])     # 低色调红色上限
RED_HSV_LOWER2 = np.array([120, 30, 30])    # 高色调红色下限
RED_HSV_UPPER2 = np.array([180, 255, 255])    # 高色调红色上限

# 蓝色小球的HSV阈值范围
BLUE_HSV_LOWER = np.array([100, 100, 100])    # 蓝色下限
BLUE_HSV_UPPER = np.array([130, 255, 255])    # 蓝色上限

# 最小轮廓面积阈值（过滤噪点）
MIN_CONTOUR_AREA = 500


def detect_ball_by_color(image: np.ndarray, target_color: str = 'red') -> Optional[Dict[str, Any]]:
    """
    使用颜色阈值检测小球
    
    参数:
        image: 输入图像（BGR格式）
        target_color: 目标颜色 ('red' 或 'blue')
    
    返回:
        包含小球信息的字典，如果未检测到则返回None
        {'center': [x, y], 'area': float, 'bbox': [x1, y1, x2, y2]}
    """
    # 转换到HSV色彩空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 根据目标颜色创建掩码
    if target_color == 'red':
        # 红色需要两个范围的掩码
        mask1 = cv2.inRange(hsv, RED_HSV_LOWER1, RED_HSV_UPPER1)
        mask2 = cv2.inRange(hsv, RED_HSV_LOWER2, RED_HSV_UPPER2)
        mask = cv2.bitwise_or(mask1, mask2)
    elif target_color == 'blue':
        mask = cv2.inRange(hsv, BLUE_HSV_LOWER, BLUE_HSV_UPPER)
    else:
        print(f"    [颜色识别] 未知的目标颜色: {target_color}")
        return None
    
    # 形态学操作去除噪点
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # 找出面积最大的轮廓
    max_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(max_contour)
    
    # 过滤太小的轮廓
    if area < MIN_CONTOUR_AREA:
        return None
    
    # 计算轮廓的边界框和中心
    x, y, w, h = cv2.boundingRect(max_contour)
    center_x = x + w // 2
    center_y = y + h // 2
    
    return {
        'center': [center_x, center_y],
        'area': area,
        'bbox': [x, y, x + w, y + h],
        'class_name': f'{target_color}_circle',
        'confidence': 1.0  # 颜色识别不提供置信度，设为1.0
    }


def handle(auv, perception_bundle: Dict[str, Any], *, vision_yaw_pid, forward_pid, is_target_ball, pid_viz_enabled: bool, State) -> Optional[Command]:
    """
    CIRCLE_KICK状态处理器 - 简化版v2.0
    
    控制策略：
    - 全局depth_pid锁定深度为0.4m
    - 有小球时：视觉yaw_pid控制（error = 小球x - 屏幕中心x）
    - 无小球时：全局yaw_pid锁定初始yaw角度
    - 定速前进
    - 完成条件：小球面积连续3次>阈值后等待一段时间
    
    支持两种识别方案：
    - 方案1：YOLO模型识别
    - 方案2：颜色阈值识别
    """
    print(f"  [CIRCLE_KICK] 执行撞球动作... (识别方案: {'YOLO模型' if BALL_DETECTION_METHOD == 1 else '颜色阈值'})")
    
    # 获取图像
    img = perception_bundle['frame_forward']
    
    # 根据选择的方案进行小球识别
    target_ball = None
    
    if BALL_DETECTION_METHOD == 1:
        # 方案1：使用YOLO模型识别
        detections = perception_bundle['detections_forward']
        
        # 找出视野中最靠下的目标小球
        lowest_y = 0
        for result in detections:
            if is_target_ball(result):
                ball_y = result.get('center', [0, 0])[1]
                if ball_y > lowest_y:
                    lowest_y = ball_y
                    target_ball = result
    
    elif BALL_DETECTION_METHOD == 1:
        # 方案2：使用颜色阈值识别
        # 从main.py的TARGET_BALL获取目标颜色（假设是'red_circle'或'blue_circle'）
        # 这里需要从auv对象获取目标颜色配置
        target_color = getattr(auv, 'target_ball_color', 'red')  # 默认红色
        if target_color.endswith('_circle'):
            target_color = target_color.replace('_circle', '')
        
        # 应用图像增强（提升水下暗环境的识别效果）
        # 注意：perception_bundle['frame_forward'] 已经在main.py中进行了图像增强
        # 这里直接使用增强后的图像即可
        target_ball = detect_ball_by_color(img, target_color)
        
        if target_ball:
            print(f"    [颜色识别] 检测到{target_color}色小球")
    
    else:
        print(f"    [错误] 未知的识别方案: {BALL_DETECTION_METHOD}，使用方案1")
        # 回退到方案1
        detections = perception_bundle['detections_forward']
        lowest_y = 0
        for result in detections:
            if is_target_ball(result):
                ball_y = result.get('center', [0, 0])[1]
                if ball_y > lowest_y:
                    lowest_y = ball_y
                    target_ball = result
    
    # 控制变量
    speed_forward = 200  # 定速前进
    speed_x = 128       # 左右平移中性值
    
    if target_ball:
        # 找到小球，执行视觉yaw控制
        center_x, center_y = target_ball['center']
        ball_area = target_ball.get('area', 0)
        
        print(f"    [目标锁定] 小球中心: ({center_x}, {center_y}), 面积: {ball_area:.0f}")
        
        # 计算yaw误差（小球x - 屏幕中心x）
        img_center_x = img.shape[1] / 2
        error_x = img_center_x - center_x # 小球在左侧，error_x为正，此时pid输出为正，表示向左摆头
        
        # 视觉yaw_pid控制
        speed_yaw_raw, pid_out_yaw = vision_yaw_pid.update(error_x)
        
        # PID可视化
        if pid_viz_enabled and hasattr(auv, 'pid_visualizer'):
            auv.pid_visualizer.update_data('yaw', error_x, pid_out_yaw)
        
        print(f"    [Yaw控制] 误差: {error_x:.1f}px, PID输出: {speed_yaw_raw}")
        
        # 检查撞球完成条件
        if ball_area > auv.ball_area_threshold:
            auv.ball_area_count += 1
            print(f"    [完成条件] 面积满足阈值，计数: {auv.ball_area_count}/{auv.ball_area_required_count}")
            
            if auv.ball_area_count >= auv.ball_area_required_count:
                # 连续满足条件，开始等待
                if auv.ball_area_start_time is None:
                    auv.ball_area_start_time = _t.time()
                    print(f"    [等待阶段] 开始等待 {auv.ball_area_wait_time}s...")
                
                # 检查等待时间是否足够
                elapsed = _t.time() - auv.ball_area_start_time
                if elapsed >= auv.ball_area_wait_time:
                    print("    [撞球完成] 切换到GUIDELINE_FOLLOW状态")
                    # 防止重复设置：只有在未撞球时才设置为1
                    if auv.circle_kicked == 0:
                        auv.circle_kicked = 1  # 标记撞球完成
                        print(f"    [标志位设置] circle_kicked: 0 -> 1")
                    else:
                        print(f"    [警告] circle_kicked已经是1，跳过重复设置（可能是重复进入CIRCLE_KICK状态）")
                    auv.ball_area_count = 0
                    auv.ball_area_start_time = None
                    auv._transition(State.GUIDELINE_FOLLOW)
                    return None
                else:
                    print(f"    [等待中] {elapsed:.1f}s / {auv.ball_area_wait_time}s，继续追踪小球")
        else:
            # 面积不满足，重置计数
            auv.ball_area_count = 0
            auv.ball_area_start_time = None
        
        # 无论是否在等待期间，都要继续追踪小球
        # 返回控制指令：定速前进 + 中性左右 + 全局控深 + 视觉控yaw
        return (speed_forward, speed_x, "global_hold", ("vision", error_x))
    
    else:
        # 没有找到小球
        # 重要：如果已经进入等待阶段，即使丢失目标也不重置计时器，继续等待
        if auv.ball_area_start_time is not None:
            elapsed = _t.time() - auv.ball_area_start_time
            if elapsed >= auv.ball_area_wait_time:
                print("    [撞球完成] 等待期间丢失目标，但时间已到，切换到GUIDELINE_FOLLOW状态")
                # 防止重复设置：只有在未撞球时才设置为1
                if auv.circle_kicked == 0:
                    auv.circle_kicked = 1  # 标记撞球完成
                    print(f"    [标志位设置] circle_kicked: 0 -> 1")
                else:
                    print(f"    [警告] circle_kicked已经是1，跳过重复设置")
                auv.ball_area_count = 0
                auv.ball_area_start_time = None
                auv._transition(State.GUIDELINE_FOLLOW)
                return None
            else:
                print(f"    [等待中-目标丢失] {elapsed:.1f}s / {auv.ball_area_wait_time}s，保持前进等待撞击")
                # 保持计时器，不重置
                return (speed_forward, speed_x, "global_hold", "global_hold")
        else:
            # 未进入等待阶段，正常搜索模式
            print(f"    [搜索模式] 未发现目标小球，锁定初始yaw: {auv.initial_yaw:.1f}°，保持定速定深前进")
            auv.ball_area_count = 0
            
            # 返回控制指令：定速前进 + 中性左右 + 全局控深 + 全局控yaw（锁定初始yaw）
            return (speed_forward, speed_x, "global_hold", "global_hold")