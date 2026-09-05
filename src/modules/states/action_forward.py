from typing import Optional, Dict, Any, Tuple, Union
import time

YField = Union[int, str, Tuple[str, int]]
YawField = Union[int, str]
Command = Tuple[int, int, YField, YawField]

def handle(auv, perception_bundle: Dict[str, Any], *, State) -> Optional[Command]:
    """
    ACTION_FORWARD：执行冲刺动作，用于撞球、穿门以及经过引导线
    
    控制策略：
    - 前进：定速200
    - 左右：中性128
    - 深度：全局depth_pid锁定目标深度
    - Yaw：全局yaw_pid锁定进入时的稳定角度（auv.target_yaw已在进入时设置为current_yaw）
    
    注意：auv.target_yaw在进入ACTION_FORWARD时被锁定为进入时的current_yaw，
         确保整个冲刺期间保持稳定的航向，不受其他状态的视觉对准影响。
    """

    # 检查冲刺状态是否已经执行完毕
    if auv.action_start_time is None:
        print("  [调试] ACTION_FORWARD状态已完成，发送悬停指令")
        return (155, 128, "global_hold", "global_hold")

    # 使用单调时钟计算冲刺时长，避免NTP时间同步影响
    elapsed_time = time.monotonic() - auv.action_start_time
    # 根据来源状态决定冲刺时长：从 GUIDELINE_FOLLOW 进入则 1.5s，否则 6.5s
    sprint_duration = 1.5 if auv.action_source_state == State.GUIDELINE_FOLLOW else 6
    if elapsed_time < sprint_duration:
        print(f"  [状态] ACTION_FORWARD: 冲刺中... (剩余 {max(0.0, sprint_duration - elapsed_time):.1f} 秒)")
        print(f"  [控制] 定速冲刺 - Depth & Yaw 锁定稳定目标值 (Yaw={auv.action_forward_locked_yaw:.1f}°)")
        return (225, 128, "global_hold", "global_hold")
    else:
        print(f"  [状态] ACTION_FORWARD: {auv.action_source_state.name} 引发的{int(sprint_duration)}秒冲刺完成。")
        if auv.action_source_state == State.CIRCLE_KICK:
            auv.circle_kicked = 1
            print(f"  [计数] 撞球任务完成。 `circle_kicked` = {auv.circle_kicked}.")
        elif auv.action_source_state == State.DOORFRAME_PASSTHROUGH:
            auv.task_finished += 1
            print(f"  [计数] 穿门任务完成。累计完成任务: {auv.task_finished}")
            
            # 根据过门次数调整目标yaw角度
            # 可配置的角度变化列表：每次过门后的角度变化量（正数为逆时针，负数为顺时针）
            # 索引对应：[第1次过门, 第2次过门, 第3次过门, 第4次过门]
            yaw_adjustments = [0, 0, 0, 0]  # 预设为0，可根据实际情况调整
            
            old_yaw = auv.target_yaw
            if auv.task_finished <= len(yaw_adjustments):
                yaw_change = yaw_adjustments[auv.task_finished - 1]
                auv.target_yaw = (auv.target_yaw + yaw_change) % 360
                
                if yaw_change > 0:
                    direction = f"逆时针旋转{yaw_change}°"
                elif yaw_change < 0:
                    direction = f"顺时针旋转{abs(yaw_change)}°"
                else:
                    direction = "保持不变"
                
                print(f"  [Yaw调整] 第{auv.task_finished}次过门，{direction}: {old_yaw:.1f}° -> {auv.target_yaw:.1f}°")
            
            # 【备用方案】配置过门后的左右移搜索
            # 启用/禁用标志：0=禁用，1=启用
            LATERAL_SEARCH_ENABLED = 0
            
            if LATERAL_SEARCH_ENABLED:
                # 左右移配置列表：[方向(正数=左移，负数=右移，0=不移动), 持续时间(秒)]
                # 索引对应：[第1次过门, 第2次过门, 第3次过门, 第4次过门]
                lateral_search_configs = [
                    (60, 0.5),   # 第1次过门后：左移速度20，持续3秒
                    (60, 0.1),  # 第2次过门后：右移速度20，持续3秒
                    (60, 0.1),   # 第3次过门后：左移速度20，持续3秒
                    (1, 0.1),  # 第4次过门后：右移速度1，持续0.1秒
                ]
                
                if auv.task_finished <= len(lateral_search_configs):
                    speed_offset, duration = lateral_search_configs[auv.task_finished - 1]
                    if speed_offset != 0:
                        # 设置左右移搜索标志
                        auv.lateral_search_active = True
                        auv.lateral_search_speed = 128 + speed_offset  # 转换为绝对速度值
                        auv.lateral_search_duration = duration
                        auv.lateral_search_start_time = None  # 将在GUIDELINE_FOLLOW中初始化
                        auv.lateral_search_doorframe_counter = 0  # 门框对准计数器
                        
                        direction_str = "左移" if speed_offset > 0 else "右移"
                        print(f"  [左右移搜索] 已启用备用方案：{direction_str}，速度={auv.lateral_search_speed}，持续{duration}秒")
                    else:
                        print(f"  [左右移搜索] 第{auv.task_finished}次过门后不执行左右移")

        if auv.action_source_state == State.DOORFRAME_PASSTHROUGH:
            if auv.task_finished >= 4:
                print("  [流程] 所有门已通过，切换到返回状态。")
                auv.next_state_after_action = State.RETURN_TO_BEGINNING_AREA
            else:
                print(f"  [流程] 门{auv.task_finished}/4通过，继续寻找下一个目标。")

        next_state = auv.next_state_after_action
        auv.action_source_state = None
        auv.next_state_after_action = None
        auv.action_start_time = None
        auv._transition(next_state)

        # 切换后当帧返回低速前进，避免悬停停住
        return (155, 128, "global_hold", "global_hold")


