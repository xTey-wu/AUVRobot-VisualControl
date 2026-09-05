"""
DOORFRAME_PASSTHROUGH状态模块 - 简化版v2.0
实现全局depth/yaw + 视觉forward/left_right的过门逻辑
"""
from typing import Optional, Dict, Any, Tuple, Union
import time as _t
from src.modules.control.command_fusion import normalize_angle_error

# 指令类型定义
YField = Union[int, str, Tuple[str, float]]
YawField = Union[int, str, Tuple[str, float]]
Command = Tuple[int, int, YField, YawField]


def check_action_forward_conditions(auv, error_x: float, error_y: float, area: float, error_yaw: float) -> bool:
    """
    检查是否满足进入ACTION_FORWARD的条件
    必须同时满足：error_x<50px & error_y<50px & 25000<area<35000 & error_yaw<5° 保持1.5s
    """
    conditions_met = (
        abs(error_x) < 120 and
        abs(error_y) < 120 and
        75000 < area and
        abs(error_yaw) < 5.0
    )
    
    if conditions_met:
        if auv.door_align_start_time is None:
            auv.door_align_start_time = _t.time()
            print(f"    [对准条件] 开始满足，开始计时...")
        
        elapsed = _t.time() - auv.door_align_start_time
        print(f"    [对准条件] ✅ 已满足 {elapsed:.1f}s / {auv.door_align_duration}s")
        print(f"      X:{abs(error_x):.1f}<50px ✅ | Y:{abs(error_y):.1f}<50px ✅ | Area:{area:.0f}∈[25000,35000] ✅ | Yaw:{abs(error_yaw):.1f}<5° ✅")
        
        if elapsed >= auv.door_align_duration:
            return True
    else:
        # 条件不满足，重置计时器
        auv.door_align_start_time = None
        print(f"    [对准条件] ❌ 不满足")
        print(f"      X:{abs(error_x):.1f}<50px:{abs(error_x)<50} | Y:{abs(error_y):.1f}<50px:{abs(error_y)<50} | Area:{area:.0f}∈[25000,35000]:{25000<area<35000} | Yaw:{abs(error_yaw):.1f}<5°:{abs(error_yaw)<5}")
    
    return False


def handle(auv, perception_bundle: Dict[str, Any], *, left_or_right_pid, vision_yaw_pid, forward_pid, DOORFRAME_LOW_STANDARD_MODE_ENABLED, DOORFRAME_ALIGNMENT_MODE, pid_viz_enabled: bool, State) -> Optional[Command]:
    """
    DOORFRAME_PASSTHROUGH状态处理器 - 简化版v2.0 (双方案版本)
    
    控制策略：
    【方案1】DOORFRAME_ALIGNMENT_MODE=1 (主方案):
    - 全局yaw保持 + 视觉left_right_pid控制x轴对准（稳定可靠）
    - 使用forward_pid控制前进距离
    - 满足对准条件后切换到ACTION_FORWARD状态
    - 切换条件：error_x<120px & error_y<120px & area>75000 & error_yaw<5° 保持1.5s
    
    【方案3】DOORFRAME_ALIGNMENT_MODE=3:
    - 全局yaw保持 + 视觉left_right_pid控制x轴对准
    - 定速前进（不使用forward_pid）
    - 不使用action_forward阶段，直接切换到GUIDELINE_FOLLOW状态
    - 切换条件：与方案1相同（error_x<120px & error_y<120px & area>75000 & error_yaw<5° 保持1.5s）
    
    共同策略：
    - 全局depth_pid（根据过门数索引）
    - 统一使用check_action_forward_conditions函数进行对准判断
    """
    mode_name = {1: "方案1", 3: "方案3-定速穿门"}.get(DOORFRAME_ALIGNMENT_MODE, "未知方案")
    print(f"  [DOORFRAME_PASSTHROUGH] 执行过门动作... ({mode_name})")
    
    # 获取检测结果
    img = perception_bundle['frame_forward']
    detections = perception_bundle['detections_forward']
    
    # 找出面积最大的门框
    doorframes = [r for r in detections if r.get('class_name') == 'doorframe']
    target_doorframe = max(doorframes, key=lambda x: x.get('area', 0)) if doorframes else None
    
    if target_doorframe:
        # 找到门框，执行对准控制
        center_x, center_y = target_doorframe['center']
        area = target_doorframe.get('area', 0)
        
        # 视觉接管计数器更新
        auv.doorframe_vision_counter = getattr(auv, 'doorframe_vision_counter', 0) + 1
        vision_active = auv.doorframe_vision_counter >= 3  # 连续3帧才接管
        
        print(f"    [目标锁定] 门框中心: ({center_x}, {center_y}), 面积: {area:.0f} | 视觉计数: {auv.doorframe_vision_counter}/3")
        
        # 计算各轴误差
        img_center_x = img.shape[1] / 2
        img_center_y = img.shape[0] / 2
        error_x = img_center_x - center_x # 门框在左侧，error_x为正，此时pid输出为正，表示向左移动
        error_y = img_center_y - center_y # 门框在上方，error_y为正，此时pid输出为正，表示向上移动（由于目前使用定深，error_y仅做对准判据，实际不控制）
        error_area = 150000 - area # 门框面积小于30000，error_area为正，此时pid输出为正，表示前进

        # 前进控制（两个方案共用）
        speed_forward, pid_out_forward = forward_pid.update(error_area)
        
        # ==================== 方案选择 ====================
        if DOORFRAME_ALIGNMENT_MODE == 1:
            # 【方案1】全局yaw保持 + 左右平移对准
            print(f"    [方案1] 全局yaw保持 + 左右平移对准")
            
            # X轴对准控制（左右平移）
            speed_x, pid_out_x = left_or_right_pid.update(error_x)
            print(f"    [X轴对准] 误差: {error_x:.1f}px, PID输出: {speed_x}")
            
            # Yaw轴误差计算（使用全局yaw控制器的角度误差）
            error_yaw = normalize_angle_error(auv.target_yaw, auv.current_yaw)
            print(f"    [全局yaw误差] 目标: {auv.target_yaw:.1f}°, 当前: {auv.current_yaw:.1f}°, 误差: {error_yaw:.1f}°")
            
            # PID可视化
            if pid_viz_enabled and hasattr(auv, 'pid_visualizer'):
                auv.pid_visualizer.update_data('x', error_x, pid_out_x)
                auv.pid_visualizer.update_data('forward', error_area, pid_out_forward)
            
            # 检查ACTION_FORWARD条件（方案1标准：error_x<50px, error_yaw<5°）
            if check_action_forward_conditions(auv, error_x, error_y, area, error_yaw):
                print("    [切换条件满足] 切换到ACTION_FORWARD状态")
                auv.door_align_start_time = None
                auv.action_source_state = State.DOORFRAME_PASSTHROUGH
                auv.next_state_after_action = State.GUIDELINE_FOLLOW
                auv.action_start_time = _t.time()
                auv._transition(State.ACTION_FORWARD)
                return None
            
            # 返回控制指令：可变前进 + 视觉左右对准 + 全局控深 + 全局控yaw
            return (speed_forward, speed_x, "global_hold", "global_hold")
        
        elif DOORFRAME_ALIGNMENT_MODE == 3:
            # 【方案3】定速前进 + 左右对准（不使用action_forward）
            print(f"    [方案3] 定速前进 + 左右对准")
            
            # 定速前进（不使用forward_pid）
            speed_forward = 175  # 定速前进
            
            # X轴对准控制（左右平移）
            speed_x, pid_out_x = left_or_right_pid.update(error_x)
            print(f"    [X轴对准] 误差: {error_x:.1f}px, PID输出: {speed_x}")
            
            # Yaw轴误差计算（使用全局yaw控制器的角度误差）
            error_yaw = normalize_angle_error(auv.target_yaw, auv.current_yaw)
            print(f"    [全局yaw误差] 目标: {auv.target_yaw:.1f}°, 当前: {auv.current_yaw:.1f}°, 误差: {error_yaw:.1f}°")
            
            # PID可视化
            if pid_viz_enabled and hasattr(auv, 'pid_visualizer'):
                auv.pid_visualizer.update_data('x', error_x, pid_out_x)
            
            # 检查对准条件（使用统一的check_action_forward_conditions函数）
            if check_action_forward_conditions(auv, error_x, error_y, area, error_yaw):
                print("    [穿门完成] 直接切换到GUIDELINE_FOLLOW状态（跳过ACTION_FORWARD）")
                auv.door_align_start_time = None
                auv.task_finished += 1  # 增加完成的门框数
                auv._transition(State.GUIDELINE_FOLLOW)
                return None
            
            # 返回控制指令：定速前进 + 视觉左右对准 + 全局控深 + 全局控yaw
            return (speed_forward, speed_x, "global_hold", "global_hold")
        
        else:
            # 未知方案，默认使用方案1
            print(f"    [警告] 未知的DOORFRAME_ALIGNMENT_MODE: {DOORFRAME_ALIGNMENT_MODE}，使用方案1")
            speed_x, pid_out_x = left_or_right_pid.update(error_x)
            speed_forward, _ = forward_pid.update(error_area)
            return (speed_forward, speed_x, "global_hold", "global_hold")
    
    else:
        # 没有找到门框，保持低速定深前进
        print("    [搜索模式] 未发现门框，保持低速定深前进")
        auv.door_align_start_time = None
        auv.doorframe_vision_counter = 0  # 重置视觉接管计数器
        
        # 返回控制指令：低速前进 + 中性左右 + 全局控深 + 全局控yaw
        return (180, 128, "global_hold", "global_hold")


if __name__ == "__main__":
    """
    独立测试模式：仅测试视觉yaw对准门框功能
    - 视觉yaw轴：根据门框x偏差自动调整航向
    - 深度轴：使用全局控制器保持0.4m
    - 前进轴：固定128（中性，悬停）
    - 左右轴：固定128（中性，不平移）
    """
    import sys
    import os
    import cv2
    
    # 添加项目根目录到路径
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from src.modules.vision.object_recognize import ObjectRecognizer
    from src.modules.control.control import PIDController
    from src.modules.communicate.serial_communicate import (
        send_command, start_serial_reader, stop_serial_reader, 
        get_latest_serial_data, close_serial_sender
    )
    from src.modules.vision.image_enhancement import ImageEnhancer
    
    print("="*60)
    print("门框对准测试模式 - X轴3档定速 + Yaw角度锁定")
    print("="*60)
    print("功能说明：")
    print("  - X轴（左右平移）: 3档定速对准门框中心")
    print("  - Yaw轴: 全局PID锁定目标角度")
    print("  - 深度轴: 全局PID保持0.2m")
    print("  - 前进轴: 固定160（缓慢前进）")
    print("  - 图像增强: 启用（CLAHE，提升水下亮度）")
    print("\n按 'q' 或 ESC 退出")
    print("="*60 + "\n")
    
    # 1. 初始化摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[错误] 无法打开摄像头")
        sys.exit(1)
    print("[✓] 摄像头已初始化")
    
    # 2. 初始化目标识别模型
    model = ObjectRecognizer()
    print("[✓] 目标识别模型已加载")
    
    # 2.5. 初始化图像增强器（用于水下暗环境）
    image_enhancer = ImageEnhancer(
        enable_clahe=True,
        enable_gamma=False,
        enable_brightness=False,
        clahe_clip_limit=3.0
    )
    print("[✓] 图像增强器已初始化（CLAHE方法）")
    
    # 3. 初始化PID控制器和分档控制参数
    # 3档X轴（左右平移）速度控制配置（基于error_x分档定速）
    X_DEAD_ZONE = 15            # 死区：±15px内不平移
    X_SLOW_THRESHOLD = 80       # 慢速档位阈值：15-80px
    X_FAST_THRESHOLD = 80       # 快速档位阈值：>80px
    
    X_SPEED_FAST = 20           # 快速平移速度（相对128的偏移量）
    X_SPEED_SLOW = 10           # 慢速平移速度（相对128的偏移量）
    
    def calculate_x_speed_by_error(error_x):
        """
        根据x轴误差计算3档左右平移速度
        error_x > 0: 门框在左侧，需要左移（speed > 128）
        error_x < 0: 门框在右侧，需要右移（speed < 128）
        
        返回: (speed, 档位描述)
        """
        abs_error = abs(error_x)
        
        if abs_error <= X_DEAD_ZONE:
            # 死区：不平移
            return 128, "死区停止"
        elif abs_error <= X_SLOW_THRESHOLD:
            # 慢速档
            if error_x > 0:
                return 128 + X_SPEED_SLOW, "慢速左移"
            else:
                return 128 - X_SPEED_SLOW, "慢速右移"
        else:
            # 快速档
            if error_x > 0:
                return 128 + X_SPEED_FAST, "快速左移"
            else:
                return 128 - X_SPEED_FAST, "快速右移"
    
    # Global Yaw PID (用于锁定目标航向角度)
    global_yaw_pid = PIDController(
        kp=5.0,
        ki=0.01,
        kd=0,
        output_limits=(-127, 127)
    )
    
    # Global Depth PID (用于保持深度)
    global_depth_pid = PIDController(
        kp=30,
        ki=0.01,
        kd=0,
        output_limits=(-127, 127)
    )
    print("[✓] PID控制器已初始化")
    
    # 4. 启动串口通信
    serial_reader = start_serial_reader('/dev/ttyTHS1', 115200)
    print("[✓] 串口通信已启动")
    
    # 5. 测试状态变量
    target_depth = 0.35  # 目标深度0.2m
    current_depth = 0.0
    current_yaw = 0.0
    target_yaw = None   # 目标航向角度（首次获取yaw时设定）
    yaw_initialized = False  # yaw是否已初始化
    
    try:
        print("\n开始测试循环...\n")
        frame_count = 0
        
        while True:
            # 读取摄像头
            ret, frame = cap.read()
            if not ret:
                print("[错误] 读取摄像头失败")
                break
            
            # 翻转图像（如果摄像头装反）
            frame = cv2.flip(frame, -1)
            
            # 应用图像增强（提升水下暗环境的图像质量）
            frame = image_enhancer.enhance(frame)
            
            # 目标识别
            detections = model.recognize(frame)
            
            # 读取传感器数据
            yaw, depth = get_latest_serial_data(max_age_ms=100)
            if depth is not None:
                current_depth = depth
            if yaw is not None:
                current_yaw = yaw
                # 首次获取yaw时，设定为目标角度并锁定
                if not yaw_initialized:
                    target_yaw = current_yaw
                    yaw_initialized = True
                    print(f"[Yaw初始化] 锁定目标航向角度: {target_yaw:.1f}°")
            
            # 找出门框
            doorframes = [r for r in detections if r.get('class_name') == 'doorframe']
            target_doorframe = max(doorframes, key=lambda x: x.get('area', 0)) if doorframes else None
            
            # 默认指令：前进160（缓慢前进），左右128，yaw128
            speed_forward = 170
            speed_x = 128
            speed_yaw = 128
            
            # 计算深度控制
            depth_error_cm = (current_depth - target_depth) * 100.0
            speed_depth, pid_out_depth = global_depth_pid.update(depth_error_cm)
            
            # 计算Yaw控制（全局角度锁定）
            if yaw_initialized and target_yaw is not None:
                # 计算yaw误差（处理360度环绕）
                yaw_error = target_yaw - current_yaw
                if yaw_error > 180.0:
                    yaw_error -= 360.0
                elif yaw_error < -180.0:
                    yaw_error += 360.0
                speed_yaw, pid_out_yaw = global_yaw_pid.update(yaw_error)
            else:
                yaw_error = 0
                pid_out_yaw = 0
            
            if target_doorframe:
                # 检测到门框
                center_x, center_y = target_doorframe['center']
                area = target_doorframe.get('area', 0)
                
                # 计算x轴误差
                img_center_x = frame.shape[1] / 2
                error_x = img_center_x - center_x
                
                # 使用3档分速控制计算X轴（左右平移）速度
                speed_x, x_gear = calculate_x_speed_by_error(error_x)
                status = f"✓ {x_gear}"
                
                # 在图像上绘制门框
                bbox = target_doorframe['bbox']
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.circle(frame, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)
                cv2.line(frame, (int(img_center_x), 0), (int(img_center_x), frame.shape[0]), (255, 0, 0), 1)
                
                # 显示信息
                info_text = f"Status: {status} | Error_X: {error_x:.1f}px"
                cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                yaw_info = f"Yaw锁定:{target_yaw:.1f}° 误差:{yaw_error:.1f}°" if target_yaw is not None else "Yaw未初始化"
                print(f"[门框] 中心:({center_x:.0f},{center_y:.0f}) 面积:{area:.0f} | X误差:{error_x:.1f}px → X档位:{x_gear} X_Speed:{speed_x} | {yaw_info} Yaw_Speed:{speed_yaw}")
            else:
                # 未检测到门框，X轴停止平移，Yaw继续锁定角度
                speed_x = 128
                
                cv2.putText(frame, "No Doorframe Detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                yaw_info = f"Yaw锁定:{target_yaw:.1f}° 误差:{yaw_error:.1f}°" if target_yaw is not None else "Yaw未初始化"
                print(f"[搜索中] 未检测到门框 | X轴停止 | {yaw_info} | Depth_PID输出:{pid_out_depth:.2f}")
            
            # 显示深度信息
            depth_text = f"Depth: {current_depth:.2f}m / {target_depth:.2f}m | Yaw: {current_yaw:.1f}deg"
            cv2.putText(frame, depth_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # 显示控制输出
            control_text = f"Forward:{speed_forward} X:{speed_x} Depth:{int(speed_depth)} Yaw:{int(speed_yaw)}"
            cv2.putText(frame, control_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # 发送控制指令
            command_bytes = bytes([0xab, speed_forward, speed_x, int(speed_depth), int(speed_yaw), 0xcd])
            send_command(command_bytes, debug_print=False)
            
            # 打印串口发送数据和传感器数据
            print(f"[串口发送] 前进:{speed_forward} 左右:{speed_x} 深度:{int(speed_depth)} 航向:{int(speed_yaw)} | [传感器] 深度:{current_depth:.2f}m(目标:{target_depth:.2f}m) 航向:{current_yaw:.1f}°(目标:{target_yaw if target_yaw else 0:.1f}°) | Depth误差:{depth_error_cm:.1f}cm Depth_PID:{pid_out_depth:.2f} Yaw_PID:{pid_out_yaw:.2f}")
            
            # 显示图像
            cv2.imshow("Doorframe X-Axis Alignment Test (Yaw Locked)", frame)
            
            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("\n[退出] 用户手动退出")
                break
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"  已运行 {frame_count} 帧")
    
    except KeyboardInterrupt:
        print("\n[中断] Ctrl+C 检测到")
    
    finally:
        print("\n正在清理资源...")
        
        # 发送悬停指令
        hover_command = bytes([0xab, 128, 128, 128, 128, 0xcd])
        send_command(hover_command, debug_print=True)
        print("[✓] 已发送悬停指令")
        
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()
        stop_serial_reader()
        close_serial_sender()
        
        print("[✓] 资源已清理")
        print("="*60)
        print("测试结束")
        print("="*60)