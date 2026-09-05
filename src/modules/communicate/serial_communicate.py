import serial
import time
import threading
import select
import sys

def send_data(ser:serial.Serial, data:bytes, debug_print:bool=False):
    try:
        # 直接发送字节数据
        ser.write(data)
        if debug_print:
            print(f"发送数据 (hex): {data.hex()}")
        return True
        
    except serial.SerialException as e:
        print(f"串口通信错误: {e}")
        return False
    except Exception as e:
        print(f"发生未知错误: {e}")
        return False
    
def receive_data(ser:serial.Serial, read_timeout:float=0.02):
    """
    读取“最新”一帧：先清空输入缓冲，然后在短超时内阻塞读取一个完整包。

    通信协议:
    - 字节 1: 0xAA (起始标志)
    - 字节 2: Yaw 高八位
    - 字节 3: Yaw 低八位
    - 字节 4: 深度传感器原始值（厘米）
    - 字节 5: 0xBB (结束标志)

    :param ser: 已打开的 serial.Serial 对象。
    :param read_timeout: 本次“阻塞读取最新值”的最长时间（秒），默认 20ms。
    :return: (yaw, depth)。若在超时内未取得完整包，返回 (None, None)。
    """
    # 1) 清空输入缓冲：丢弃所有积压的旧数据（只关心最新值）
    try:
        ser.reset_input_buffer()
    except Exception:
        try:
            ser.flushInput()
        except Exception:
            pass

    # 2) 在短超时内读取一个完整包；期间临时修改 timeout 并在结束后恢复
    original_timeout = getattr(ser, "timeout", None)
    try:
        ser.timeout = read_timeout
        deadline = time.monotonic() + read_timeout

        while time.monotonic() < deadline:
            # 查找帧头 0xAA
            b = ser.read(1)
            if not b:
                # 本轮读不到数据，超时或暂时无数据
                break
            if b == b'\xAA':
                # 读取剩余 4 字节（尽量不超过总超时时间）
                remaining_time = max(0.0, deadline - time.monotonic())
                if remaining_time == 0.0:
                    break
                ser.timeout = remaining_time
                data_packet = ser.read(4)
                if len(data_packet) == 4 and data_packet[3] == 0xBB:
                    yaw_high = data_packet[0]
                    yaw_low = data_packet[1]
                    depth_raw = data_packet[2]

                    yaw = (yaw_high << 8) | yaw_low
                    depth = depth_raw / 100.0
                    return float(yaw), depth
                # 若不合法，继续在剩余时间内寻找下一帧头
                ser.timeout = max(0.0, deadline - time.monotonic())

        return None, None
    finally:
        if original_timeout is not None:
            ser.timeout = original_timeout


class SerialReader(threading.Thread):
    """
    独立线程持续读取串口数据，维护最新的陀螺仪和深度传感器样本。
    主循环可通过 get_latest() 获取最新值，避免阻塞等待。
    """
    
    def __init__(self, port:str='/dev/ttyTHS1', baudrate:int=115200, 
                 frame_head:int=0xAA, frame_tail:int=0xBB, packet_len:int=5,
                 timeout_ms:float=3.0, inter_byte_timeout_ms:float=2.0,
                 stale_threshold_ms:float=100.0):
        super().__init__(daemon=True)
        self.port = port
        self.baudrate = baudrate
        self.frame_head = frame_head
        self.frame_tail = frame_tail
        self.packet_len = packet_len
        self.timeout_s = timeout_ms / 1000.0
        self.inter_byte_timeout_s = inter_byte_timeout_ms / 1000.0
        self.stale_threshold_s = stale_threshold_ms / 1000.0
        
        # 线程控制
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        
        # 串口对象（线程内部创建）
        self._serial = None
        
        # 最新样本数据 (线程安全访问)
        self._latest_sample = None  # (yaw: float, depth: float, seq: int, timestamp: float)
        self._seq_counter = 0
        
        # 统计信息
        self._last_valid_time = 0.0
        self._packets_received = 0
        self._drops = 0
        self._parse_buffer = bytearray()
        
    def run(self):
        """线程主循环：持续读取和解析串口数据"""
        try:
            # 在线程内部打开串口
            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout_s,
                inter_byte_timeout=self.inter_byte_timeout_s
            )
            print(f"[SerialReader] 串口已打开: {self.port} @ {self.baudrate}")
            
            self._parse_buffer.clear()
            last_sync_attempt = 0.0
            
            while not self._stop_event.is_set():
                try:
                    # 使用 select 等待数据可读（非阻塞检查）
                    if sys.platform != 'win32':
                        ready, _, _ = select.select([self._serial], [], [], self.timeout_s)
                        if not ready:
                            time.sleep(0.001)  # 短暂睡眠避免忙等
                            continue
                    
                    # 批量读取当前可用数据
                    available = self._serial.in_waiting
                    if available > 0:
                        chunk = self._serial.read(min(available, 1024))  # 限制单次读取量
                        if chunk:
                            self._parse_buffer.extend(chunk)
                    else:
                        # 尝试读取1字节（带超时）
                        byte = self._serial.read(1)
                        if byte:
                            self._parse_buffer.extend(byte)
                        else:
                            time.sleep(0.001)
                            continue
                    
                    # 解析缓冲区中的所有完整帧，只保留最后一帧
                    self._parse_frames()
                    
                    # 如果缓冲区过大且长时间无有效帧，清空重新同步
                    now = time.monotonic()
                    if (len(self._parse_buffer) > 1000 and 
                        now - last_sync_attempt > 1.0 and 
                        now - self._last_valid_time > 1.0):
                        print(f"[SerialReader] 缓冲区过大且长时间无有效帧，执行重同步")
                        self._parse_buffer.clear()
                        self._serial.reset_input_buffer()
                        self._drops += 1
                        last_sync_attempt = now
                        
                except Exception as e:
                    print(f"[SerialReader] 读取异常: {e}")
                    time.sleep(0.01)
                    
        except Exception as e:
            print(f"[SerialReader] 串口初始化失败: {e}")
        finally:
            if self._serial and self._serial.is_open:
                self._serial.close()
                print(f"[SerialReader] 串口已关闭")
    
    def _parse_frames(self):
        """解析缓冲区中的所有完整帧，只保留最后一帧"""
        latest_frame = None
        frames_parsed = 0
        
        while len(self._parse_buffer) >= self.packet_len and frames_parsed < 64:  # 限制单次解析帧数
            # 寻找帧头
            head_idx = -1
            for i in range(len(self._parse_buffer) - self.packet_len + 1):
                if self._parse_buffer[i] == self.frame_head:
                    head_idx = i
                    break
            
            if head_idx == -1:
                # 没找到帧头，保留最后几个字节避免帧头被截断
                self._parse_buffer = self._parse_buffer[-self.packet_len+1:]
                break
            
            # 移除帧头之前的垃圾数据
            if head_idx > 0:
                self._parse_buffer = self._parse_buffer[head_idx:]
            
            # 检查是否有完整帧
            if len(self._parse_buffer) < self.packet_len:
                break
                
            # 验证帧尾
            if self._parse_buffer[self.packet_len - 1] == self.frame_tail:
                # 解析有效帧
                packet_data = self._parse_buffer[1:self.packet_len-1]  # 去掉帧头和帧尾
                
                if len(packet_data) == 3:  # yaw_high, yaw_low, depth_raw
                    yaw_high = packet_data[0]
                    yaw_low = packet_data[1]
                    depth_raw = packet_data[2]
                    
                    yaw = float((yaw_high << 8) | yaw_low)
                    depth = depth_raw / 100.0
                    
                    # 只保留最后一帧
                    latest_frame = (yaw, depth)
                    frames_parsed += 1
                
                # 移除已处理的帧
                self._parse_buffer = self._parse_buffer[self.packet_len:]
            else:
                # 帧尾不对，移除当前帧头，继续寻找
                self._parse_buffer = self._parse_buffer[1:]
        
        # 如果解析到了帧，更新最新样本
        if latest_frame:
            now = time.monotonic()
            with self._lock:
                self._seq_counter += 1
                self._latest_sample = (latest_frame[0], latest_frame[1], self._seq_counter, now)
                self._last_valid_time = now
                self._packets_received += frames_parsed
    
    def stop(self):
        """停止线程"""
        self._stop_event.set()
    
    def get_latest(self):
        """获取最新样本，返回 (yaw, depth, seq, timestamp) 或 None"""
        with self._lock:
            return self._latest_sample
    
    def get_latest_if_fresh(self, max_age_ms:float=100.0):
        """获取新鲜的最新样本，超过指定时间则返回 None"""
        sample = self.get_latest()
        if sample is None:
            return None
        
        _, _, _, timestamp = sample
        age_s = time.monotonic() - timestamp
        if age_s > (max_age_ms / 1000.0):
            return None
        return sample
    
    def get_stats(self):
        """获取统计信息"""
        with self._lock:
            now = time.monotonic()
            last_age_ms = (now - self._last_valid_time) * 1000.0 if self._last_valid_time > 0 else float('inf')
            # 简单的包速率估算（基于总接收包数和运行时间）
            runtime = now - (self._last_valid_time - 1.0) if self._last_valid_time > 1.0 else 1.0
            pps = self._packets_received / max(runtime, 0.1)
            
            return {
                'pps': pps,
                'last_age_ms': last_age_ms,
                'drops': self._drops,
                'packets_total': self._packets_received
            }

# 全局串口读取器实例（单例模式）
_global_serial_reader = None
# 全局串口发送器实例（单例模式）
_global_serial_sender = None

def get_serial_reader(port:str='/dev/ttyTHS1', baudrate:int=115200, **kwargs):
    """获取全局串口读取器实例（单例）"""
    global _global_serial_reader
    if _global_serial_reader is None:
        _global_serial_reader = SerialReader(port=port, baudrate=baudrate, **kwargs)
    return _global_serial_reader

def start_serial_reader(port:str='/dev/ttyTHS1', baudrate:int=115200, **kwargs):
    """启动全局串口读取器"""
    reader = get_serial_reader(port=port, baudrate=baudrate, **kwargs)
    if not reader.is_alive():
        reader.start()
    return reader

def stop_serial_reader():
    """停止全局串口读取器"""
    global _global_serial_reader
    if _global_serial_reader and _global_serial_reader.is_alive():
        _global_serial_reader.stop()
        _global_serial_reader.join(timeout=1.0)
        _global_serial_reader = None

def get_serial_sender(port:str='/dev/ttyTHS1', baudrate:int=115200):
    """获取全局串口发送器实例（单例）"""
    global _global_serial_sender
    if _global_serial_sender is None:
        _global_serial_sender = serial.Serial(port, baudrate, timeout=0.1)
    return _global_serial_sender

def close_serial_sender():
    """关闭全局串口发送器"""
    global _global_serial_sender
    if _global_serial_sender and _global_serial_sender.is_open:
        _global_serial_sender.close()
        _global_serial_sender = None

def send_command(data:bytes, debug_print:bool=False):
    """使用全局发送器发送命令"""
    sender = get_serial_sender()
    return send_data(sender, data, debug_print)

def get_latest_serial_data(max_age_ms:float=100.0):
    """便捷函数：获取最新的串口数据 (yaw, depth)"""
    if _global_serial_reader is None:
        return None, None
    
    sample = _global_serial_reader.get_latest_if_fresh(max_age_ms)
    if sample is None:
        return None, None
    
    yaw, depth, _, _ = sample
    return yaw, depth


if __name__ == "__main__":
    # 测试新的线程化串口读取
    print("启动串口读取线程测试...")
    reader = start_serial_reader()
    
    try:
        for i in range(100):
            time.sleep(0.1)
            yaw, depth = get_latest_serial_data()
            stats = reader.get_stats()
            print(f"Frame {i}: yaw={yaw}, depth={depth}, age={stats['last_age_ms']:.1f}ms, pps={stats['pps']:.1f}")
    finally:
        stop_serial_reader()
        print("测试完成")
