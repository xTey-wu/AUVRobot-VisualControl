import sys
import os
import time
# 新增: 导入matplotlib和deque用于绘图
import matplotlib
# 在非GUI环境下，可能需要指定一个兼容的后端
try:
    matplotlib.use('TkAgg')
except ImportError:
    print("[警告] TkAgg后端不可用，将使用默认后端。在某些系统上图表可能无法显示。")
import matplotlib.pyplot as plt
from collections import deque

# 将项目根目录添加到Python路径中，以解决模块导入问题
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.modules.communicate.serial_communicate import send_data, receive_data
import serial

# def forward_or_backward(speed:int, ser:serial.Serial):
#     try:
#         if speed > 255 or speed < 0:
#             raise ValueError("速度范围错误")
#         send_data(ser, bytes([0xAB, speed, 0x00, 0x00, 0x00, 0xCD]))
#     except ValueError:
#         print("速度范围错误")
#     except Exception as e:
#         print(f"前进失败: {e}")

# def left_or_right(speed:int, ser:serial.Serial):
#     try:
#         if speed > 255 or speed < 0:
#             raise ValueError("速度范围错误")
#         send_data(ser, bytes([0xAB, 0x00, speed, 0x00, 0x00, 0xCD]))
#     except ValueError:
#         print("速度范围错误")
#     except Exception as e:
#         print(f"前进失败: {e}")

# def up_or_down(speed:int, ser:serial.Serial):
#     try:
#         if speed > 255 or speed < 0:
#             raise ValueError("速度范围错误")
#         send_data(ser, bytes([0xAB, 0x00, 0x00, speed, 0x00, 0xCD]))
#     except ValueError:
#         print("速度范围错误")
#     except Exception as e:
#         print(f"前进失败: {e}")

# def yaw_or_turn(speed:int, ser:serial.Serial):
#     try:
#         if speed > 255 or speed < 0:
#             raise ValueError("速度范围错误")
#         send_data(ser, bytes([0xAB, 0x00, 0x00, 0x00, speed, 0xCD]))
#     except ValueError:
#         print("速度范围错误")
#     except Exception as e:
#         print(f"前进失败: {e}")

class PIDController:
    """
    一个通用的PID控制器类。

    这个控制器被设计为可独立用于多个自由度。
    为每个需要PID控制的自由度（如偏航、俯仰、推力等）创建一个此类的实例。

    使用示例:
        # 1. 为偏航角创建一个PID控制器实例
        yaw_pid = PIDController(kp=0.5, ki=0.1, kd=0.05)

        # 2. 在主控制循环中
        while True:
            # error_yaw = 目标角度 - 当前角度
            error_yaw = calculate_yaw_error() 
            
            # 计算控制速度 (自动处理时间差 dt)
            yaw_speed = yaw_pid.update(error_yaw)  # 输出将在0-255之间

            # 将 yaw_speed 发送到电机控制器
            send_command(yaw_speed)
            
            time.sleep(0.01) # 循环等待
    """
    def __init__(self, kp: float, ki: float, kd: float, 
                 output_limits: tuple = (-127, 127), 
                 integral_limits: tuple = (-1000, 1000)):
        """
        初始化PID控制器

        Args:
            kp (float): 比例增益
            ki (float): 积分增益
            kd (float): 微分增益
            output_limits (tuple): 控制器原始输出的限制范围 (min, max)
            integral_limits (tuple): 积分项的抗饱和限制范围 (min, max)
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        self.output_min, self.output_max = output_limits
        self.integral_min, self.integral_max = integral_limits
        
        self._last_error = 0.0
        self._integral = 0.0
        self._last_time = None

        # --- 控制增强：积分抑制与抗风up（Anti-Windup）---
        # 说明：这些参数用于在“阶段切换/大误差/饱和”等情况下，抑制积分造成的偏置与过冲。
        # 1) 预热/禁积与Ki渐放
        self._warmup_frames_remaining = 0         # 预热禁积剩余帧数（>0时不进行积分）
        self._ki_ramp_frames_remaining = 0        # Ki从0到基准值的渐放帧数
        self._ki_scale = 1.0                      # 当前Ki缩放（0.0~1.0）

        # 2) 条件积分（小误差门控）
        self._small_error_threshold = None        # 只有当 |error| < 阈值 时才允许积分
        self._small_error_required_frames = 0     # 连续满足的帧数要求
        self._small_error_count = 0               # 连续小误差累计帧计数
        self._integral_decay_factor = 1.0         # 当不满足小误差时，积分按该因子衰减（<1.0 代表衰减）

        # 3) 抗饱和回卷（Back-Calculation）
        self._anti_windup_gain = 0.0              # >0 启用回卷：integral += gain * (u_sat - u_raw)

        # 4) 误差符号翻转时的积分释放
        self._decay_on_sign_flip = 0.0            # 0.0~1.0：当误差换向时，integral *= factor
        self._last_error_sign = 0                 # 上一帧误差符号（-1/0/+1）

    def update(self, error: float) -> tuple[int, float]:
        """
        根据新的误差值计算PID输出。

        Args:
            error (float): 当前的误差值 (目标值 - 当前值)

        Returns:
            tuple[int, float]: 包含两个值的元组:
                - int: 映射到0-255范围的速度控制值，128代表零点。
                - float: 钳位后的原始PID输出值 (在output_limits范围内)。
        """
        # 使用单调时钟避免NTP时间同步导致dt计算异常
        current_time = time.monotonic()
        
        # 计算时间差 (dt)
        if self._last_time is None:
            # 第一次调用时，dt为0，避免微分项过大
            dt = 0.0
        else:
            dt = current_time - self._last_time
        
        self._last_time = current_time

        # 如果时间间隔为0，不进行计算，直接返回零点速度
        if dt == 0:
            return 128, 0.0

        # --- 符号翻转：快速释放历史积分，避免反向过冲 ---
        current_sign = 0 if error == 0 else (1 if error > 0 else -1)
        if self._last_error_sign != 0 and current_sign != 0 and current_sign != self._last_error_sign:
            if 0.0 < self._decay_on_sign_flip < 1.0:
                self._integral *= self._decay_on_sign_flip
        self._last_error_sign = current_sign

        # --- 预热/禁积与Ki渐放 ---
        effective_ki = self.ki
        if self._warmup_frames_remaining > 0:
            # 预热阶段：禁用积分
            effective_ki = 0.0
            self._warmup_frames_remaining -= 1
        elif self._ki_ramp_frames_remaining > 0:
            # Ki渐放：从0逐步增加到基准Ki
            total = max(1, self._ki_ramp_frames_remaining)
            # 按剩余帧进行线性插值：先提升再递减可读性，我们采用累计比例来简化
            # 这里使用 scale 在 0.0~1.0 之间线性趋近
            progressed = 1.0 - (self._ki_ramp_frames_remaining / float(total))
            self._ki_scale = max(0.0, min(1.0, progressed))
            effective_ki = self.ki * self._ki_scale
            self._ki_ramp_frames_remaining -= 1
        else:
            self._ki_scale = 1.0

        # --- 条件积分：仅在小误差区持续M帧时才允许积分；否则对积分做衰减 ---
        allow_integrate = True
        if self._small_error_threshold is not None and self._small_error_required_frames > 0:
            if abs(error) < self._small_error_threshold:
                self._small_error_count += 1
            else:
                self._small_error_count = 0
                # 大误差区：对历史积分做衰减（可温和释放偏置）
                if 0.0 < self._integral_decay_factor < 1.0:
                    self._integral *= self._integral_decay_factor
            allow_integrate = self._small_error_count >= self._small_error_required_frames

        # 1. 积分项 (I)
        if allow_integrate and effective_ki > 0.0:
            self._integral += error * dt
            self._integral = max(min(self._integral, self.integral_max), self.integral_min)
        
        # 2. 微分项 (D)
        derivative = (error - self._last_error) / dt
        
        # 3. 计算PID原始输出
        output_raw = (self.kp * error) + (effective_ki * self._integral) + (self.kd * derivative)
        
        # 更新上一次误差，用于下次微分计算
        self._last_error = error
        
        # 限制原始输出范围
        clamped_output = max(min(output_raw, self.output_max), self.output_min)

        # --- 抗饱和回卷：当发生裁剪时，回卷多余至积分 ---
        if self._anti_windup_gain > 0.0 and clamped_output != output_raw:
            self._integral += self._anti_windup_gain * (clamped_output - output_raw)
            # 再次约束积分边界
            self._integral = max(min(self._integral, self.integral_max), self.integral_min)
        
        # 将输出映射到0-255的速度范围，128为中心零点
        speed = int(128 + clamped_output)
        
        # 最终确保值在0-255的字节范围内
        final_speed = max(0, min(255, speed))

        return final_speed, clamped_output

    def reset(self):
        """
        重置PID控制器的内部状态（积分项、历史误差）。
        当目标发生重大变化或需要重新开始控制时调用。
        """
        self._integral = 0.0
        self._last_error = 0.0
        self._last_time = None
        print("PID controller has been reset.")

    # --------------------------- 配置接口（供外部阶段管理使用） ---------------------------
    def begin_warmup(self, warmup_frames: int = 0, ki_ramp_frames: int = 0):
        """开始预热：warmup期间禁用积分；随后对Ki进行线性渐放。"""
        self._warmup_frames_remaining = max(0, int(warmup_frames))
        self._ki_ramp_frames_remaining = max(0, int(ki_ramp_frames))
        self._ki_scale = 0.0 if (warmup_frames > 0 or ki_ramp_frames > 0) else 1.0

    def configure_small_error_gate(self, threshold: float | None, required_frames: int = 0, integral_decay_factor: float = 1.0):
        """配置小误差门控：仅在误差足够小且连续满足时才允许积分；否则可按比例衰减积分。"""
        self._small_error_threshold = threshold
        self._small_error_required_frames = max(0, int(required_frames))
        self._integral_decay_factor = float(integral_decay_factor)
        self._small_error_count = 0

    def configure_anti_windup(self, backcalc_gain: float = 0.0):
        """配置抗饱和回卷增益，>0 启用回卷抑制积分风up。"""
        self._anti_windup_gain = max(0.0, float(backcalc_gain))

    def configure_sign_flip_decay(self, factor: float = 0.0):
        """配置误差符号翻转时对积分的衰减比例（0.0~1.0，越小释放越快）。"""
        self._decay_on_sign_flip = max(0.0, min(1.0, float(factor)))


# ==============================================================================
# 新增: PID实时数据可视化模块
# ==============================================================================
class PIDVisualizer:
    """
    一个用于实时绘制PID控制器数据的可视化工具。
    """
    def __init__(self, history_size=200):
        """
        初始化可视化工具。
        Args:
            history_size (int): 图表上显示的数据点数量。
        """
        self.history_size = history_size
        self._last_plot_time = 0.0

        # 为每个PID轴创建数据存储
        self.data = {
            'x': {'time': deque(maxlen=history_size), 'error': deque(maxlen=history_size), 'output': deque(maxlen=history_size)},
            'y': {'time': deque(maxlen=history_size), 'error': deque(maxlen=history_size), 'output': deque(maxlen=history_size)},
            'yaw': {'time': deque(maxlen=history_size), 'error': deque(maxlen=history_size), 'output': deque(maxlen=history_size)},
            'depth': {'time': deque(maxlen=history_size), 'error': deque(maxlen=history_size), 'output': deque(maxlen=history_size)}
        }
        
        # 打开交互模式
        plt.ion()
        self.fig, self.axs = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
        self.fig.suptitle('PID Controller Real-time Tuning', fontsize=16)

        self.plot_titles = {
            'x': 'X-Axis (Left/Right) PID',
            'y': 'Y-Axis (Up/Down) PID',
            'yaw': 'Yaw-Axis PID',
            'depth': 'Depth-Hold PID'
        }
        
        self.lines = {}
        for i, (key, title) in enumerate(self.plot_titles.items()):
            ax = self.axs[i]
            # 设置图表标题
            ax.set_title(title, fontsize=12)
            ax.grid(True)
            ax.axhline(0, color='r', linestyle='--', label='Target (Error=0)')
            
            # 绘制误差和输出线
            line_error, = ax.plot([], [], 'b-', label='Error')
            line_output, = ax.plot([], [], 'g-', label='Output')
            self.lines[key] = {'error': line_error, 'output': line_output}
            ax.legend(loc='upper right')

        self.axs[-1].set_xlabel('Time (s)')
        self.fig.tight_layout(rect=[0, 0, 1, 0.96])
        # 使用单调时钟避免NTP时间同步影响可视化时间轴
        self._start_time = time.monotonic()

    def update_data(self, pid_name, error, output):
        """
        向可视化工具添加新的数据点。
        Args:
            pid_name (str): PID轴的名称 ('x', 'y', 'yaw', 'depth')。
            error (float): 当前误差。
            output (float): 当前PID的原始输出 (通常在-127到127之间)。
        """
        if pid_name in self.data:
            # 使用单调时钟计算相对时间
            current_time = time.monotonic() - self._start_time
            self.data[pid_name]['time'].append(current_time)
            self.data[pid_name]['error'].append(error)
            self.data[pid_name]['output'].append(output)

    def plot(self):
        """
        更新图表。为避免性能问题，此函数会节流，每0.3秒才实际重绘。
        """
        # 使用单调时钟进行节流控制
        current_time = time.monotonic()
        if current_time - self._last_plot_time < 0.3:
            return

        self._last_plot_time = current_time

        for key, ax in zip(self.plot_titles.keys(), self.axs):
            if not self.data[key]['time']:
                continue

            # 更新数据
            self.lines[key]['error'].set_data(self.data[key]['time'], self.data[key]['error'])
            self.lines[key]['output'].set_data(self.data[key]['time'], self.data[key]['output'])
            
            # 自动调整坐标轴范围
            ax.relim()
            ax.autoscale_view()
        
        # 重绘画布
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        """
        关闭图表窗口。
        """
        plt.ioff()
        plt.close(self.fig)


if __name__ == '__main__':
    """
    偏航轴(Yaw)闭环PID控制硬件在环测试。

    本测试流程:
    1. 从STM32通过串口接收当前的偏航角度(yaw)和深度(depth)。
    2. 设置一个目标偏航角。
    3. 计算目标角度与当前角度的误差 (处理角度环绕问题)。
    4. 将误差输入PID控制器，计算出控制指令。
    5. 将控制指令通过串口发送给STM32，驱动电机进行角度校正。
    """
    try:
        # 1. 初始化串口
        # 根据 serial_communicate.py, 端口为 /dev/ttyTHS1, 波特率 115200
        ser = serial.Serial("/dev/ttyTHS1", 115200, timeout=0.1)
        print("串口 /dev/ttyTHS1 打开成功。")
    except serial.SerialException as e:
        print(f"错误：无法打开串口: {e}")
        sys.exit(1)

    # 2. 初始化Yaw轴的PID控制器
    # 这些增益值可以作为初始值，实际应用中需要调试以获得最佳性能
    yaw_pid = PIDController(kp=5, ki=0.05, kd=0.1, output_limits=(-100, 100))
    print(f"Yaw轴PID控制器初始化成功. Kp={yaw_pid.kp}, Ki={yaw_pid.ki}, Kd={yaw_pid.kd}")

    # 3. 设置目标角度
    target_yaw = 160  # 假设目标角度为180度
    # yaw_pid.reset() # 如果需要，可以重置PID状态

    print("="*60)
    print(f"开始Yaw轴闭环控制测试... 目标角度: {target_yaw}°")
    print("="*60)
    print(f"{'当前角度':<12} | {'误差':<12} | {'PID输出(0-255)':<18}")
    print("-"*50)

    # 初始化PID可视化器，仅用于yaw轴（通过开关控制显示）
    PID_VISUALIZE = 0  # 0 关闭，1 开启
    pid_visualizer = PIDVisualizer() if PID_VISUALIZE == 1 else None

    try:
        while True:
            # 4. 从串口接收数据
            current_yaw, current_depth = receive_data(ser)

            if current_yaw is not None:
                # 5. 计算角度误差 (处理360度环绕)
                error = target_yaw - current_yaw
                if error > 180.0:
                    error -= 360.0
                elif error < -180.0:
                    error += 360.0

                # 6. 更新PID控制器并获取控制速度
                yaw_speed, pid_output = yaw_pid.update(error)

                # 7. 构建并发送控制指令
                # 协议: [0xAB, 前后, 左右, 上下, 偏航, 0xCD]
                # 这里只控制偏航，其他轴设为不动 (128为零点)
                command = bytes([0xAB, 128, 128, 128, yaw_speed, 0xCD])
                send_data(ser, command)

                # 打印实时状态
                print(f"{current_yaw:<12.2f} | {error:<12.2f} | {yaw_speed:<18}")

                # --- 新增：更新PID可视化数据（可视化开关控制） ---
                if pid_visualizer is not None:
                    pid_visualizer.update_data('yaw', error, pid_output)
                    pid_visualizer.plot()

            # 稍微延时，避免CPU占用过高，并匹配传感器更新频率
            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\n测试被用户中断。")
    finally:
        # 8. 测试结束时，发送停止指令并关闭串口
        print("发送停止指令...")
        stop_command = bytes([0xAB, 128, 128, 128, 128, 0xCD])
        send_data(ser, stop_command)
        ser.close()
        print("串口已关闭。")
        # --- 新增：关闭PID可视化窗口（可视化开关控制） ---
        if pid_visualizer is not None:
            pid_visualizer.close()

