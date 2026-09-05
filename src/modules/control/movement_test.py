import sys
import os
import time
import serial

# 将项目根目录添加到Python路径中，以解决模块导入问题
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.modules.communicate.serial_communicate import send_data, receive_data


class MovementTester:
    """
    四方向运动测试器
    
    用于测试AUV的前后左右四个方向的单独运动功能。
    基于PRD中定义的通讯协议进行控制。
    
    通讯协议:
    Jetson Nano到STM32: [AB][前后速度][左右速度][上下速度][偏航速度][CD]
    STM32到Jetson Nano: [AA][偏航_H][偏航_L][深度_H][深度_L][BB]
    
    速度值范围: 0-255，其中128为中心零点
    - 前后: <128向后, >128向前
    - 左右: <128向左, >128向右
    - 上下: <128向下, >128向上
    - 偏航: <128逆时针, >128顺时针
    """
    
    def __init__(self, serial_port="/dev/ttyTHS1", baud_rate=115200):
        """
        初始化运动测试器
        
        Args:
            serial_port (str): 串口设备路径
            baud_rate (int): 波特率
        """
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.ser = None
        self.test_speed = 200  # 测试速度 (128为零点，所以180表示中等正向速度)
        self.zero_speed = 128  # 零点速度
        
    def connect(self):
        """
        连接串口
        
        Returns:
            bool: 连接成功返回True，失败返回False
        """
        try:
            self.ser = serial.Serial(self.serial_port, self.baud_rate, timeout=0.1)
            print(f"串口 {self.serial_port} 连接成功，波特率: {self.baud_rate}")
            return True
        except serial.SerialException as e:
            print(f"错误：无法打开串口 {self.serial_port}: {e}")
            return False
    
    def disconnect(self):
        """
        断开串口连接
        """
        if self.ser and self.ser.is_open:
            # 发送停止指令
            self.stop_all_movement()
            self.ser.close()
            print("串口已关闭")
    
    def stop_all_movement(self):
        """
        停止所有运动 - 发送全零指令
        """
        command = bytes([0xAB, self.zero_speed, self.zero_speed, self.zero_speed, self.zero_speed, 0xCD])
        send_data(self.ser, command)
        print("已发送停止指令")
    
    def move_forward(self, duration=2.0):
        """
        向前运动
        
        Args:
            duration (float): 运动持续时间（秒）
        """
        print(f"开始向前运动 {duration} 秒...")
        command = bytes([0xAB, self.test_speed, self.zero_speed, self.zero_speed, self.zero_speed, 0xCD])
        start_time = time.time()
        while time.time() - start_time < duration:
            send_data(self.ser, command)
            print(f"发送指令：{command}")
            # 尝试接收传感器数据并显示
            yaw, depth = receive_data(self.ser)
            if yaw is not None:
                print(f"  状态 - 偏航角: {yaw:.2f}°, 深度: {depth:.2f}")
            time.sleep(0.1)
        
        self.stop_all_movement()
        print("向前运动完成\n")
    
    def move_backward(self, duration=2.0):
        """
        向后运动
        
        Args:
            duration (float): 运动持续时间（秒）
        """
        print(f"开始向后运动 {duration} 秒...")
        backward_speed = self.zero_speed - (self.test_speed - self.zero_speed)  # 76 (128-52)
        command = bytes([0xAB, backward_speed, self.zero_speed, self.zero_speed, self.zero_speed, 0xCD])
        
        start_time = time.time()
        while time.time() - start_time < duration:
            send_data(self.ser, command)
            # 尝试接收传感器数据并显示
            yaw, depth = receive_data(self.ser)
            if yaw is not None:
                print(f"  状态 - 偏航角: {yaw:.2f}°, 深度: {depth:.2f}")
            time.sleep(0.1)
        
        self.stop_all_movement()
        print("向后运动完成\n")
    
    def move_left(self, duration=2.0):
        """
        向左运动
        
        Args:
            duration (float): 运动持续时间（秒）
        """
        print(f"开始向左运动 {duration} 秒...")
        left_speed = self.zero_speed - (self.test_speed - self.zero_speed)  # 76
        command = bytes([0xAB, self.zero_speed, left_speed, self.zero_speed, self.zero_speed, 0xCD])
        
        start_time = time.time()
        while time.time() - start_time < duration:
            send_data(self.ser, command)
            # 尝试接收传感器数据并显示
            yaw, depth = receive_data(self.ser)
            if yaw is not None:
                print(f"  状态 - 偏航角: {yaw:.2f}°, 深度: {depth:.2f}")
            time.sleep(0.1)
        
        self.stop_all_movement()
        print("向左运动完成\n")
    
    def move_right(self, duration=2.0):
        """
        向右运动
        
        Args:
            duration (float): 运动持续时间（秒）
        """
        print(f"开始向右运动 {duration} 秒...")
        command = bytes([0xAB, self.zero_speed, self.test_speed, self.zero_speed, self.zero_speed, 0xCD])
        
        start_time = time.time()
        while time.time() - start_time < duration:
            send_data(self.ser, command)
            # 尝试接收传感器数据并显示
            yaw, depth = receive_data(self.ser)
            if yaw is not None:
                print(f"  状态 - 偏航角: {yaw:.2f}°, 深度: {depth:.2f}")
            time.sleep(0.1)
        
        self.stop_all_movement()
        print("向右运动完成\n")
    
    def turn_left(self, duration=2.0):
        """
        左转（逆时针偏航）
        
        Args:
            duration (float): 转动持续时间（秒）
        """
        print(f"开始左转 {duration} 秒...")
        left_turn_speed = self.zero_speed - (self.test_speed - self.zero_speed)  # 逆时针转动
        command = bytes([0xAB, self.zero_speed, self.zero_speed, self.zero_speed, left_turn_speed, 0xCD])
        
        start_time = time.time()
        while time.time() - start_time < duration:
            send_data(self.ser, command)
            # 尝试接收传感器数据并显示
            yaw, depth = receive_data(self.ser)
            if yaw is not None:
                print(f"  状态 - 偏航角: {yaw:.2f}°, 深度: {depth:.2f}")
            time.sleep(0.1)
        
        self.stop_all_movement()
        print("左转完成\n")
    
    def turn_right(self, duration=2.0):
        """
        右转（顺时针偏航）
        
        Args:
            duration (float): 转动持续时间（秒）
        """
        print(f"开始右转 {duration} 秒...")
        command = bytes([0xAB, self.zero_speed, self.zero_speed, self.zero_speed, self.test_speed, 0xCD])
        
        start_time = time.time()
        while time.time() - start_time < duration:
            send_data(self.ser, command)
            # 尝试接收传感器数据并显示
            yaw, depth = receive_data(self.ser)
            if yaw is not None:
                print(f"  状态 - 偏航角: {yaw:.2f}°, 深度: {depth:.2f}")
            time.sleep(0.1)
        
        self.stop_all_movement()
        print("右转完成\n")
    
    def move_up(self, duration=2.0):
        """
        上升（向上运动）
        
        Args:
            duration (float): 上升持续时间（秒）
        """
        print(f"开始上升 {duration} 秒...")
        command = bytes([0xAB, self.zero_speed, self.zero_speed, self.test_speed, self.zero_speed, 0xCD])
        
        start_time = time.time()
        while time.time() - start_time < duration:
            send_data(self.ser, command)
            # 尝试接收传感器数据并显示
            yaw, depth = receive_data(self.ser)
            if yaw is not None:
                print(f"  状态 - 偏航角: {yaw:.2f}°, 深度: {depth:.2f}")
            time.sleep(0.1)
        
        self.stop_all_movement()
        print("上升完成\n")
    
    def move_down(self, duration=2.0):
        """
        下降（向下运动）
        
        Args:
            duration (float): 下降持续时间（秒）
        """
        print(f"开始下降 {duration} 秒...")
        down_speed = self.zero_speed - (self.test_speed - self.zero_speed)  # 向下运动
        command = bytes([0xAB, self.zero_speed, self.zero_speed, down_speed, self.zero_speed, 0xCD])
        
        start_time = time.time()
        while time.time() - start_time < duration:
            send_data(self.ser, command)
            # 尝试接收传感器数据并显示
            yaw, depth = receive_data(self.ser)
            if yaw is not None:
                print(f"  状态 - 偏航角: {yaw:.2f}°, 深度: {depth:.2f}")
            time.sleep(0.1)
        
        self.stop_all_movement()
        print("下降完成\n")
    
    def run_test_sequence(self):
        """
        运行完整的八方向测试序列（前后左右+上下+左转右转）
        """
        print("="*60)
        print("AUV 八方向运动测试开始")
        print("="*60)
        print(f"测试速度: {self.test_speed} (零点: {self.zero_speed})")
        print("每个方向测试 2 秒\n")
        
        try:
            # 测试前进
            self.move_forward(2.0)
            time.sleep(1)  # 暂停1秒
            
            # 测试后退
            self.move_backward(2.0)
            time.sleep(1)
            
            # 测试左移
            self.move_left(2.0)
            time.sleep(1)
            
            # 测试右移
            self.move_right(2.0)
            time.sleep(1)
            
            # 测试上升
            self.move_up(2.0)
            time.sleep(1)
            
            # 测试下降
            self.move_down(2.0)
            time.sleep(1)
            
            # 测试左转
            self.turn_left(2.0)
            time.sleep(1)
            
            # 测试右转
            self.turn_right(2.0)
            
            print("="*60)
            print("八方向运动测试完成!")
            print("="*60)
            
        except KeyboardInterrupt:
            print("\n测试被用户中断")
            self.stop_all_movement()
    
    def interactive_test(self):
        """
        交互式测试模式 - 用户可以选择测试方向
        """
        print("="*60)
        print("AUV 交互式运动测试")
        print("="*60)
        print("控制说明:")
        print("  w - 向前运动")
        print("  s - 向后运动")
        print("  a - 向左运动")
        print("  d - 向右运动")
        print("  r - 上升（向上运动）")
        print("  f - 下降（向下运动）")
        print("  j - 左转（逆时针偏航）")
        print("  l - 右转（顺时针偏航）")
        print("  x - 停止运动")
        print("  q - 退出测试")
        print("="*60)
        
        try:
            while True:
                choice = input("\n请选择运动方向 (w/s/a/d/r/f/j/l/x/q): ").lower().strip()
                
                if choice == 'w':
                    self.move_forward(200)
                elif choice == 's':
                    self.move_backward(200)
                elif choice == 'a':
                    self.move_left(200)
                elif choice == 'd':
                    self.move_right(200)
                elif choice == 'r':
                    self.move_up(200)
                elif choice == 'f':
                    self.move_down(200)
                elif choice == 'j':
                    self.turn_left(200)
                elif choice == 'l':
                    self.turn_right(200)
                elif choice == 'x':
                    self.stop_all_movement()
                    print("已停止所有运动")
                elif choice == 'q':
                    print("退出交互式测试")
                    break
                else:
                    print("无效输入，请重新选择")
                    
        except KeyboardInterrupt:
            print("\n交互式测试被用户中断")
            self.stop_all_movement()


def main():
    """
    主测试函数
    """
    # 创建运动测试器实例
    tester = MovementTester()
    
    # 连接串口
    if not tester.connect():
        return
    
    try:
        print("\n选择测试模式:")
        print("1 - 自动测试序列 (依次测试八个方向)")
        print("2 - 交互式测试 (手动选择方向)")
        
        mode = input("\n请选择模式 (1/2): ").strip()
        
        if mode == '1':
            tester.run_test_sequence()
        elif mode == '2':
            tester.interactive_test()
        else:
            print("无效选择，使用默认自动测试模式")
            tester.run_test_sequence()
            
    finally:
        # 确保断开连接
        tester.disconnect()


if __name__ == '__main__':
    main()
