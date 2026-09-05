from typing import Optional, Dict, Any, Tuple, Union
import time

YField = Union[int, str, Tuple[str, int]]
YawField = Union[int, str]
Command = Tuple[int, int, YField, YawField]

def handle(auv, perception_bundle: Dict[str, Any], *, State) -> Optional[Command]:
    """
    RETURN_TO_BEGINNING_AREA state's logic delegated.
    
    执行逻辑：
    - 每隔10秒向电控连续发送 [BB, AA, CC] 10次
    - 每次发送间隔 0.02s（20ms）
    - 停留在此阶段，不再切换状态
    - 返回None，主循环会跳过常规控制指令发送（只发送BBAACC特殊指令）
    """
    # 初始化返回区域状态变量（只在第一次进入时执行）
    if not hasattr(auv, 'return_last_send_time'):
        auv.return_last_send_time = 0  # 上次发送完整序列的时间
        auv.return_send_interval = 10.0  # 每隔10秒发送一次
        auv.return_single_send_delay = 0.02  # 单次发送间隔20ms
        auv.return_send_count = 10  # 每次发送10次
        print("  [RETURN_TO_BEGINNING_AREA] 进入返回起始区域状态")
        print("  [配置] 每隔10秒向电控发送特殊指令序列 [BB, AA, CC] x 10次")
    
    # 使用单调时钟避免NTP时间同步影响
    current_time = time.monotonic()
    
    # 检查是否到达发送时间
    if current_time - auv.return_last_send_time >= auv.return_send_interval:
        print(f"\n  [特殊指令] 开始发送指令序列...")
        
        # 导入串口发送函数
        from src.modules.communicate.serial_communicate import send_command
        
        # 连续发送10次 [BB, AA, CC]
        special_command = bytes([0xBB, 0xAA, 0xCC])
        for i in range(auv.return_send_count):
            send_command(special_command, debug_print=True)
            print(f"    [发送 {i+1}/{auv.return_send_count}] 指令: BB AA CC")
            if i < auv.return_send_count - 1:  # 最后一次不需要等待
                time.sleep(auv.return_single_send_delay)
        
        auv.return_last_send_time = current_time
        print(f"  [特殊指令] 序列发送完成，下次发送时间: {auv.return_send_interval:.1f}秒后\n")
    
    # 返回None，主循环会跳过常规控制指令发送（只发送BBAACC特殊指令）
    return None


