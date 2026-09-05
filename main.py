import time
import os
import sys
import math
import json
from enum import Enum, auto
from collections import deque # 导入deque用于实现滑动平均滤波器
import torch
import cv2
import serial
import numpy as np
import time

# 将项目根目录添加到Python路径中，以解决模块导入问题
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.modules.vision.object_recognize import ObjectRecognizer
from src.modules.control.control import PIDController
from src.modules.control.pid_setup import create_pids
# 移除复杂的稳定记录器导入
from src.modules.communicate.serial_communicate import send_command, start_serial_reader, stop_serial_reader, get_latest_serial_data, close_serial_sender
from src.modules.vision.guideline_detector import GuideLineDetector
from src.modules.vision.image_enhancement import ImageEnhancer
from src.modules.states import start as state_start
from src.modules.states import circle_kick as state_circle_kick
from src.modules.states import guideline_follow as state_guideline_follow
from src.modules.states import doorframe_passthrough as state_doorframe
from src.modules.states import action_forward as state_action_forward
from src.modules.states import return_to_beginning_area as state_return
from src.modules.states import end as state_end
from src.modules.visualization.osd import draw_forward_osd, draw_bottom_osd
from src.modules.videocapture.recorder import DualVideoRecorder
from src.modules.control.command_fusion import fuse_depth_yaw, encode_command, normalize_angle_error
from src.modules.sensors.io import update_depth, update_yaw
from src.modules.metrics.monitor import Metrics

# --- PID调参可视化开关 ---
# 设置为 1 开启实时PID图表, 设置为 0 则完全禁用
PID_TUNING_VISUALIZER_ENABLED = 0

# --- 性能模式开关 ---
# 设置为 1 启用低频打印以提升性能；设置为 0 恢复逐帧打印
PERFORMANCE_MODE_ENABLED = 0

# --- 调试模式开关 ---
# 设置为 1 开启调试模式, 设置为 0 则正常运行模式
DEBUG_MODE_ENABLED = 0

# --- 图像增强开关 ---
# 设置为 1 启用图像亮度增强（推荐用于水下暗环境）, 设置为 0 则禁用
IMAGE_ENHANCEMENT_ENABLED = 1
# 图像增强方法：'clahe'(推荐), 'gamma', 'brightness', 'auto'(组合方法)
IMAGE_ENHANCEMENT_METHOD = 'clahe'
# CLAHE参数：对比度限制值（1.0-5.0，值越大对比度越强）
IMAGE_ENHANCEMENT_CLAHE_CLIP = 3.0
# 伽马校正值（仅在method='gamma'或'auto'时有效，>1增亮）
IMAGE_ENHANCEMENT_GAMMA = 1.5
# 亮度增加值（仅在method='brightness'时有效，0-100）
IMAGE_ENHANCEMENT_BRIGHTNESS = 30

# --- 调试命令映射表 ---
DEBUG_COMMANDS = {
    ord('1'): 'START',
    ord('2'): 'CIRCLE_KICK',
    ord('3'): 'GUIDELINE_FOLLOW', 
    ord('4'): 'DOORFRAME_PASSTHROUGH',
    ord('5'): 'ACTION_FORWARD',
    ord('6'): 'RETURN_TO_BEGINNING_AREA',
    ord('0'): 'END',
    ord('r'): 'RESET_PID',
    ord('s'): 'EMERGENCY_STOP',
    27: 'EXIT'  # ESC键
}

if PID_TUNING_VISUALIZER_ENABLED:
    from src.modules.control.control import PIDVisualizer

# --- 穿门低标准模式开关 ---
# 设置为 1 启用：仅检查门框中心对齐与面积，不使用四角点与偏航判断
DOORFRAME_LOW_STANDARD_MODE_ENABLED = 1

# --- 穿门对准方案选择 ---
# 1 = 方案1：全局yaw保持 + 左右平移对准 + forward_pid（稳定但较慢）
# 2 = 方案2：视觉yaw对准 + forward_pid（快速但可能抖动，连续3帧后启用）
# 3 = 方案3：全局yaw保持 + 左右平移对准 + 定速前进（简单直接，不使用action_forward）
DOORFRAME_ALIGNMENT_MODE = 1

# --- 单轴PID开关（便于单独调试） ---
# True 启用该轴PID；False 关闭（输出固定为128中性）
PID_AXIS_ENABLE = {
    'x': True,        # 左右纠偏
    'depth': True,    # 竖直/深度轴
    'forward': True,  # 前后轴
    'yaw': True       # 偏航轴
}

# --- 任务目标配置 ---
# 目标小球类别：在 'red_circle' 与 'blue_circle' 中二选一，直接修改下行常量即可。
BALL_CLASSES = {"red_circle", "blue_circle"}
TARGET_BALL = 'red_circle'

def is_ball(result):
    return result.get('class_name') in BALL_CLASSES

def is_target_ball(result):
    return result.get('class_name') == TARGET_BALL

# ==============================================================================
# AUV控制逻辑 v2.0 - 简化重构版
# ==============================================================================
# 本脚本实现了一个基于简化状态机的自治水下航行器（AUV）主控程序，重构后大幅简化了控制逻辑，
# 移除了复杂的稳定记录器和控制模式切换机制，提高了可维护性和调试效率。
#
# 主要简化内容：
# 1. 状态机简化：
#    - 移除复杂的稳定记录器和控制模式切换
#    - 采用简单的列表索引深度管理系统
#    - 直接的状态切换逻辑，无复杂的过渡机制
#
# 2. 控制逻辑简化：
#    - CIRCLE_KICK: 全局depth_pid + 视觉yaw_pid + 定速前进
#    - GUIDELINE_FOLLOW: 状态切换优先级 + 右转逻辑 + 简化循线
#    - DOORFRAME_PASSTHROUGH: 全局depth/yaw + 视觉forward/left_right
#
# 3. PID系统简化：
#    - 移除复杂的预热和门控机制
#    - 直接创建和使用PID控制器
#    - 简化的指令融合逻辑
#
# 4. 架构优势：
#    - 降低复杂度：移除不必要的过渡状态和复杂判断
#    - 提高可维护性：每个阶段的控制逻辑清晰明确
#    - 增强可调试性：简化的逻辑便于问题定位
#    - 提升稳定性：减少状态切换的不确定性
# ==============================================================================

# --- 程序初始化和GPU预热 ---

# 摄像头和模型初始化
# 前置摄像头：强制要求，初始化失败则程序退出
cap_forward = cv2.VideoCapture(0) # 正向摄像头
if not cap_forward.isOpened():
    print("[错误] 前置摄像头初始化失败，程序无法运行！")
    sys.exit(1)

# 下置摄像头：可选，初始化失败则标记为不可用
cap_bottom = cv2.VideoCapture(2) # 底部摄像头
bottom_cam_available = False
if cap_bottom.isOpened():
    ret_test, _ = cap_bottom.read()
    if ret_test:
        bottom_cam_available = True
        print("[信息] 下置摄像头可用")
    else:
        cap_bottom.release()
        cap_bottom = None
        print("[警告] 下置摄像头无法读取帧，将禁用下置摄像头功能")
else:
    cap_bottom = None
    print("[警告] 下置摄像头未连接，将禁用下置摄像头功能")

model = ObjectRecognizer()
guideline_detector = GuideLineDetector()

# 图像增强器初始化
if IMAGE_ENHANCEMENT_ENABLED:
    if IMAGE_ENHANCEMENT_METHOD == 'clahe':
        image_enhancer = ImageEnhancer(
            enable_clahe=True,
            enable_gamma=False,
            enable_brightness=False,
            clahe_clip_limit=IMAGE_ENHANCEMENT_CLAHE_CLIP
        )
    elif IMAGE_ENHANCEMENT_METHOD == 'gamma':
        image_enhancer = ImageEnhancer(
            enable_clahe=False,
            enable_gamma=True,
            enable_brightness=False,
            gamma_value=IMAGE_ENHANCEMENT_GAMMA
        )
    elif IMAGE_ENHANCEMENT_METHOD == 'brightness':
        image_enhancer = ImageEnhancer(
            enable_clahe=False,
            enable_gamma=False,
            enable_brightness=True,
            brightness_value=IMAGE_ENHANCEMENT_BRIGHTNESS
        )
    elif IMAGE_ENHANCEMENT_METHOD == 'auto':
        image_enhancer = ImageEnhancer(
            enable_clahe=True,
            enable_gamma=True,
            enable_brightness=False,
            clahe_clip_limit=IMAGE_ENHANCEMENT_CLAHE_CLIP,
            gamma_value=IMAGE_ENHANCEMENT_GAMMA
        )
    else:
        print(f"[警告] 未知的图像增强方法: {IMAGE_ENHANCEMENT_METHOD}，将禁用图像增强")
        IMAGE_ENHANCEMENT_ENABLED = 0
        image_enhancer = None
    
    if IMAGE_ENHANCEMENT_ENABLED:
        print(f"[信息] 图像增强已启用，方法: {IMAGE_ENHANCEMENT_METHOD}")
else:
    image_enhancer = None
    print("[信息] 图像增强已禁用")

# PID控制器初始化（解耦到模块）
pids = create_pids()
left_or_right_pid = pids['left_or_right_pid']
vision_yaw_pid = pids['vision_yaw_pid']
global_depth_pid = pids['global_depth_pid']
global_yaw_pid = pids['global_yaw_pid']
forward_pid = pids['forward_pid']

# 启动串口读取线程（替代直接的串口操作）
serial_reader = start_serial_reader('/dev/ttyTHS1', 115200)

# GPU预热
torch.cuda.empty_cache()

# 状态枚举
class State(Enum):
    """定义AUV的可能状态。"""
    START = auto()
    CIRCLE_KICK = auto()
    GUIDELINE_FOLLOW = auto()
    DOORFRAME_PASSTHROUGH = auto()
    ACTION_FORWARD = auto() # <--- 新增的通用前进状态
    RETURN_TO_BEGINNING_AREA = auto()
    END = auto()


# 状态机类定义
class AUVStateMachine:
    """
    简化的AUV状态机，基于新的控制逻辑设计v2.0
    """
    def __init__(self):
        """初始化状态机和所有需要的变量。"""
        self.current_state = State.START
        self.circle_kicked = 0  # 0: 未撞, 1: 已撞
        self.task_finished = 0  # 通过的门框计数器
        self.last_known_command = None # 用于存储上一条有效指令
        
        # --- 程序运行时间控制 ---
        # 使用单调时钟避免NTP时间同步导致的时间跳变
        self.program_start_time = time.monotonic()  # 程序启动时间（单调时钟，不受系统时间调整影响）
        self.mission_timeout = 180.0           # 任务超时时间（3分0秒）

        # --- 用于非阻塞动作的变量 ---
        self.action_source_state = None      # 记录动作发起的状态
        self.next_state_after_action = None  # 记录动作结束后要转换的状态
        self.action_start_time = None        # 记录动作开始的时间
        self.action_forward_locked_yaw = 0.0 # ACTION_FORWARD期间锁定的yaw目标值
        
        # --- 传感器数据 ---
        self.depth_readings = deque(maxlen=1) # 存储最近1次的深度读数
        self.current_filtered_depth = 0.0      # 滤波后的当前深度
        self.current_yaw = 0.0                 # 当前从陀螺仪获取的yaw值
        self.initial_yaw = 0.0                 # 程序启动时的初始yaw值
        
        # # --- 简化的深度管理系统 ---
        # self.depth_circle_kick = 0.6  # CIRCLE_KICK阶段固定深度
        # self.depth_doorframe_list = [0.7, 0.6, 0.7, 0.6]  # 四个门的不同高度
        # self.depth_guideline_list = [0.6, 0.7, 0.6, 0.7]  # GUIDELINE_FOLLOW阶段深度
        # self.target_depth = self.depth_circle_kick       # 当前目标深度（供OSD显示使用）
        
        # --- 简化的深度管理系统 ---(测试深度10.14)
        self.depth_circle_kick = 0.30 # CIRCLE_KICK阶段固定深度
        self.depth_doorframe_list = [0.17, 0.45, 0.17, 0.45]  # 四个门的不同高度
        self.depth_guideline_list = [0.17, 0.45, 0.17, 0.45]  # GUIDELINE_FOLLOW阶段深度
        self.target_depth = self.depth_circle_kick       # 当前目标深度（供OSD显示使用）
        
        # --- 简化的yaw管理系统 ---
        self.target_yaw = 0.0                  # 目标航向角度
        self.yaw_readings = deque(maxlen=5)    # yaw读数缓冲区
        self.yaw_hold_ready = False            # yaw保持是否就绪
        
        # --- 目标检测计数器 ---
        self.ball_detection_counter = 0
        self.doorframe_detection_counter = 0
        self.detection_confidence_threshold = 3  # 连续检测到3帧才确认
        
        # --- CIRCLE_KICK阶段控制变量 ---
        self.ball_area_count = 0               # 小球面积大于阈值的连续计数
        self.ball_area_threshold = 50000       # 小球面积阈值
        self.ball_area_required_count = 3      # 需要连续满足的次数
        self.ball_area_wait_time = 3         # 满足条件后的等待时间
        self.ball_area_start_time = None       # 开始等待的时间
   
        # --- GUIDELINE_FOLLOW状态变量 ---
        self.guideline_sub_state = "FRONT_CAM_FOLLOW" # "FRONT_CAM_FOLLOW", "BOTTOM_CAM_ALIGN"
        self.front_cam_has_seen_guideline = False # 前置摄像头是否已看到过引导线
        self.from_circle_kick = False             # 是否从CIRCLE_KICK进入
        self.from_circle_kick_sub_state = None    # 从CIRCLE_KICK进入的子状态："BACKWARD", "TURN_RIGHT"
        self.backward_start_time = None           # 后退开始时间
        self.backward_duration = 3.5              # 后退持续时间（秒）
        self.turn_completed = False               # 右转是否完成（已废弃，保留兼容）
        self.yaw_stable_start_time = None         # 右转稳定计时器
        self.right_move_start_time = None         # 右移开始时间（已废弃，保留兼容）
        self.right_move_duration = 0              # 右移持续时间（已废弃，保留兼容）

        # --- DOORFRAME阶段控制变量 ---
        self.door_align_start_time = None         # 门框对准开始时间
        self.door_align_duration = 0.5            # 需要保持对准的时间
        
        # --- 计时器 ---
        self.bottom_cam_search_start_time = None

        print("AUV状态机已初始化 (简化版v2.0)。起始状态：START")
        print(f"程序启动时间: {time.strftime('%H:%M:%S', time.localtime())}")
        print(f"使用单调时钟计时，避免NTP时间同步影响")
        print(f"任务超时设置: {self.mission_timeout:.0f}秒 ({self.mission_timeout/60:.1f}分钟)")
        print("-" * 50)

    def _transition(self, new_state):
        """辅助函数，用于处理状态转换并打印变化。在调试模式下会拦截转换。"""
        if DEBUG_MODE_ENABLED:
            # 调试模式：拦截状态转换，只记录意图
            print(f"\n[调试-拦截] {self.current_state.name} 尝试切换到 {new_state.name}，但被调试模式拦截\n")
            return
        
        # 正常模式：执行实际的状态转换
        print(f"\n>>> 正在从 {self.current_state.name} 转换到 {new_state.name} <<<\n")
        
        previous_state = self.current_state

        # 状态切换时的特殊处理
        if new_state == State.GUIDELINE_FOLLOW:
            print("  [重置] 进入GUIDELINE_FOLLOW状态")
            self.front_cam_has_seen_guideline = False
            self.guideline_sub_state = "FRONT_CAM_FOLLOW"
            self.target_depth = self.depth_guideline_list[self.task_finished % len(self.depth_guideline_list)]
            
            # 检查是否从CIRCLE_KICK进入，需要特殊处理
            if previous_state == State.CIRCLE_KICK:
                self.from_circle_kick = True
                self.from_circle_kick_sub_state = None  # 初始化子状态，由guideline_follow.py设置
                self.backward_start_time = None
                self.yaw_stable_start_time = None
                print(f"  [从CIRCLE_KICK进入] 启用特殊处理：后退3秒 + 右转90度")
            else:
                self.from_circle_kick = False
                self.from_circle_kick_sub_state = None
                self.target_yaw = self.current_yaw
                print(f"  [普通进入] 保持当前yaw: {self.target_yaw:.1f}°")
        
        elif new_state == State.CIRCLE_KICK:
            print("  [重置] 进入CIRCLE_KICK状态")
            self.ball_area_count = 0
            self.ball_area_start_time = None
            print(f"  [深度设定] CIRCLE_KICK深度: {self.depth_circle_kick:.2f}m")
            self.target_depth = self.depth_circle_kick
            # 设置目标yaw为初始yaw（小球丢失时会锁定此角度）
            self.target_yaw = self.initial_yaw
            print(f"  [Yaw设定] 锁定初始yaw: {self.initial_yaw:.1f}°")
        
        elif new_state == State.DOORFRAME_PASSTHROUGH:
            print("  [重置] 进入DOORFRAME_PASSTHROUGH状态")
            self.door_align_start_time = None
            self.doorframe_vision_counter = 0  # 重置视觉接管计数器
            target_depth = self.depth_doorframe_list[self.task_finished % len(self.depth_doorframe_list)]
            print(f"  [深度设定] DOORFRAME深度: {target_depth:.2f}m (第{self.task_finished+1}个门)")
            self.target_depth = target_depth
        
        elif new_state == State.ACTION_FORWARD:
            print("  [进入] ACTION_FORWARD冲刺状态")
            # 设置冲刺状态变量
            self.action_source_state = previous_state
            self.next_state_after_action = State.GUIDELINE_FOLLOW  # 冲刺完成后返回GUIDELINE_FOLLOW
            self.action_start_time = time.monotonic()
            # 锁定当前yaw值作为ACTION_FORWARD期间的目标（保持进入时的稳定角度）
            self.action_forward_locked_yaw = self.current_yaw
            self.target_yaw = self.action_forward_locked_yaw  # 设置为全局目标，供global_hold使用
            print(f"  [冲刺设置] 来源状态: {previous_state.name}，完成后返回: {self.next_state_after_action.name}")
            print(f"  [Yaw锁定] 锁定当前yaw: {self.action_forward_locked_yaw:.1f}° 作为冲刺期间的稳定目标")

        self.current_state = new_state

    def _force_transition(self, new_state):
        """强制状态转换，绕过调试模式拦截机制，用于手动调试。"""
        print(f"\n[调试-手动] 强制切换到状态: {new_state.name}\n")
        
        # 重置相关的状态变量
        self._reset_state_variables_for_debug()
        
        # 为ACTION_FORWARD状态提供特殊处理
        if new_state == State.ACTION_FORWARD:
            self.action_start_time = time.monotonic()
            self.action_source_state = State.START  # 设置一个默认来源状态
            self.next_state_after_action = State.GUIDELINE_FOLLOW  # 设置默认的下一个状态
            # 锁定当前yaw值
            self.action_forward_locked_yaw = self.current_yaw
            self.target_yaw = self.action_forward_locked_yaw  # 设置为全局目标
            print("  [调试] ACTION_FORWARD状态已初始化，设置3秒冲刺计时器")
            print(f"  [调试-Yaw锁定] 锁定当前yaw: {self.action_forward_locked_yaw:.1f}°")
        
        # 执行相同的状态切换逻辑
        if new_state == State.GUIDELINE_FOLLOW:
            print("  [调试-重置] 进入GUIDELINE_FOLLOW状态")
            self.front_cam_has_seen_guideline = False
            self.guideline_sub_state = "FRONT_CAM_FOLLOW"
            self.from_circle_kick = False  # 调试模式下默认不从CIRCLE_KICK进入
            self.from_circle_kick_sub_state = None
            self.backward_start_time = None
            self.yaw_stable_start_time = None
            self.target_yaw = self.current_yaw
        
        elif new_state == State.CIRCLE_KICK:
            print("  [调试-重置] 进入CIRCLE_KICK状态")
            self.ball_area_count = 0
            self.ball_area_start_time = None
            # 设置目标yaw为初始yaw
            self.target_yaw = self.initial_yaw
        
        elif new_state == State.DOORFRAME_PASSTHROUGH:
            print("  [调试-重置] 进入DOORFRAME_PASSTHROUGH状态")
            self.door_align_start_time = None
            self.doorframe_vision_counter = 0  # 重置视觉接管计数器
        
        self.current_state = new_state

    def _reset_state_variables_for_debug(self):
        """重置调试时的状态相关变量。"""
        # 重置动作相关变量
        self.action_source_state = None
        self.next_state_after_action = None
        self.action_start_time = None
        
        # 重置计时器
        self.bottom_cam_search_start_time = None
        self.door_align_start_time = None
        self.ball_area_start_time = None
        self.yaw_stable_start_time = None
        self.right_move_start_time = None
        
        # 重置计数器
        self.ball_detection_counter = 0
        self.doorframe_detection_counter = 0
        self.ball_area_count = 0
        
        # 重置状态标志
        self.front_cam_has_seen_guideline = False
        self.from_circle_kick = False
        self.from_circle_kick_sub_state = None
        self.backward_start_time = None
        self.yaw_stable_start_time = None
        self.turn_completed = False
        
        # 清除历史指令
        self.last_known_command = None
        
        print("  [调试] 状态变量已重置")

    def update(self, perception_bundle):
        """根据传入的感知数据，更新状态机的状态，并返回控制指令。"""
        if self.current_state == State.END:
            self._handle_end()
            return None # 结束状态不返回指令

        # 根据性能模式切换状态日志频率
        if not hasattr(self, '_debug_frame_count'):
            self._debug_frame_count = 0
        self._debug_frame_count += 1
        if PERFORMANCE_MODE_ENABLED:
            if self._debug_frame_count % 5 == 0:
                print(f"--- 当前状态: {self.current_state.name} (已撞球: {self.circle_kicked}, 任务数: {self.task_finished}) ---")
        else:
            print(f"--- 当前状态: {self.current_state.name} (已撞球: {self.circle_kicked}, 任务数: {self.task_finished}) ---")
        
        # 调用当前状态的处理程序，并传入感知数据
        handler = getattr(self, f"_handle_{self.current_state.name.lower()}")
        # 让处理函数返回指令，而不是直接发送
        command = handler(perception_bundle)
        return command

    def _handle_start(self, perception_bundle):
        """START状态委托。"""
        return state_start.handle(self, perception_bundle, is_target_ball=is_target_ball, State=State)

    def _handle_circle_kick(self, perception_bundle):
        """CIRCLE_KICK状态委托。"""
        return state_circle_kick.handle(self, perception_bundle,
                                        vision_yaw_pid=vision_yaw_pid,
                                        forward_pid=forward_pid,
                                        is_target_ball=is_target_ball,
                                        pid_viz_enabled=PID_TUNING_VISUALIZER_ENABLED,
                                        State=State)

    def _handle_guideline_follow(self, perception_bundle):
        """GUIDELINE_FOLLOW状态委托。"""
        return state_guideline_follow.handle(self, perception_bundle,
                                             left_or_right_pid=left_or_right_pid,
                                             vision_yaw_pid=vision_yaw_pid,
                                             global_yaw_pid=global_yaw_pid,
                                             forward_pid=forward_pid,
                                             bottom_cam_available=bottom_cam_available,
                                             pid_viz_enabled=PID_TUNING_VISUALIZER_ENABLED,
                                             is_target_ball=is_target_ball,
                                             State=State)
    
    def _handle_action_forward(self, perception_bundle):
        """ACTION_FORWARD状态委托。"""
        # 记录进入前的来源状态
        previous_source = self.action_source_state
        
        result = state_action_forward.handle(self, perception_bundle, State=State)
        
        # 如果ACTION_FORWARD刚完成（action_source_state被清空）且来源是穿门，重置相关PID
        if (previous_source == State.DOORFRAME_PASSTHROUGH and 
            self.action_source_state is None):  # action_forward已清空source_state，表示完成
            left_or_right_pid.reset()
            forward_pid.reset()
            print("  [PID重置] 穿门完成，重置left_or_right_pid和forward_pid")
            
        return result


    def _handle_doorframe_passthrough(self, perception_bundle):
        """DOORFRAME_PASSTHROUGH状态委托。"""
        return state_doorframe.handle(self, perception_bundle,
                                      left_or_right_pid=left_or_right_pid,
                                      vision_yaw_pid=vision_yaw_pid,
                                      forward_pid=forward_pid,
                                      DOORFRAME_LOW_STANDARD_MODE_ENABLED=DOORFRAME_LOW_STANDARD_MODE_ENABLED,
                                      DOORFRAME_ALIGNMENT_MODE=DOORFRAME_ALIGNMENT_MODE,
                                      pid_viz_enabled=PID_TUNING_VISUALIZER_ENABLED,
                                      State=State)

    def _handle_return_to_beginning_area(self, perception_bundle):
        """RETURN_TO_BEGINNING_AREA状态委托。"""
        return state_return.handle(self, perception_bundle, State=State)

    def _handle_end(self):
        """END状态委托。"""
        return state_end.handle(self, {})

    def get_current_target_depth(self):
        """获取当前阶段的目标深度"""
        if self.current_state == State.CIRCLE_KICK:
            return self.depth_circle_kick
        elif self.current_state == State.DOORFRAME_PASSTHROUGH:
            return self.depth_doorframe_list[self.task_finished % len(self.depth_doorframe_list)]
        elif self.current_state == State.GUIDELINE_FOLLOW:
            return self.depth_guideline_list[self.task_finished % len(self.depth_guideline_list)]
        else:
            return 0.4  # 默认深度


    def _emergency_stop(self):
        """紧急停止，发送悬停指令。"""
        print("[调试] 紧急停止！发送悬停指令")
        emergency_command = bytes([0xab, 128, 128, 128, 128, 0xcd])
        send_command(emergency_command, debug_print=not PERFORMANCE_MODE_ENABLED)
        return emergency_command

def run_auv_mission():
    """
    初始化并运行AUV任务的主函数。
    它包含感知、决策和可视化循环。
    """
    # 系统状态报告
    print("=== AUV系统状态报告 v2.0 ===")
    print(f"前置摄像头: 可用 ✓")
    print(f"下置摄像头: {'可用 ✓' if bottom_cam_available else '不可用 ✗'}")
    print(f"控制逻辑: 简化版v2.0")
    print(f"PID调参可视化: {'启用' if PID_TUNING_VISUALIZER_ENABLED else '禁用'}")
    print(f"性能模式: {'启用' if PERFORMANCE_MODE_ENABLED else '禁用'}")
    print(f"图像增强: {'启用 (' + IMAGE_ENHANCEMENT_METHOD + ')' if IMAGE_ENHANCEMENT_ENABLED else '禁用'}")
    if not bottom_cam_available:
        print("注意: 下置摄像头不可用，将跳过精确对准功能")
    print("=============================")
    
    # 调试模式状态报告
    if DEBUG_MODE_ENABLED:
        print("\n=== 调试模式已启用 ===")
        print("状态转换将被拦截，仅记录转换意图")
        print("可用调试命令:")
        print("  1=START  2=CIRCLE_KICK  3=GUIDELINE_FOLLOW")
        print("  4=DOORFRAME_PASSTHROUGH  5=ACTION_FORWARD")
        print("  6=RETURN_TO_BEGINNING_AREA  0=END")
        print("  r=重置PID  s=紧急停止  ESC=退出")
        print("====================\n")
    else:
        print("\n=== 正常运行模式 v2.0 ===")
        print("简化状态机将自动执行任务")
        print("=========================\n")
    
    # 状态机实例化
    auv_controller = AUVStateMachine()

    # --- PID可视化工具初始化 (如果已启用) ---
    if PID_TUNING_VISUALIZER_ENABLED:
        pid_visualizer = PIDVisualizer()
        # 将实例附加到控制器以便在方法内部访问
        auv_controller.pid_visualizer = pid_visualizer
    # --- 结束PID可视化工具初始化 ---

    # --- 性能监控数据初始化 ---
    metrics = Metrics()
    # --- 结束性能监控数据初始化 ---

    # --- 视频录制设置 ---
    # 从摄像头获取帧的宽度和高度
    frame_width_forward = int(cap_forward.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height_forward = int(cap_forward.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 下置摄像头参数（仅在可用时获取）
    frame_width_bottom = 640  # 默认值
    frame_height_bottom = 480  # 默认值
    if bottom_cam_available and cap_bottom is not None:
        frame_width_bottom = int(cap_bottom.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height_bottom = int(cap_bottom.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fps = cap_forward.get(cv2.CAP_PROP_FPS)
    if fps == 0: # 如果获取不到FPS，设置一个默认值
        fps = 24

    # 创建一个唯一的文件名与录像器
    timestr = time.strftime("%Y%m%d-%H%M%S")
    recorder = DualVideoRecorder(
        timestr=timestr,
        fps=float(fps),
        frame_size_forward=(frame_width_forward, frame_height_forward),
        frame_size_bottom=(frame_width_bottom, frame_height_bottom) if bottom_cam_available and cap_bottom is not None else None,
    )
    fwd_path, bottom_path = recorder.paths()
    print(f"[INFO] 前置摄像头视频录制已启动，将保存至: {fwd_path}")
    if bottom_path is not None:
        print(f"[INFO] 下置摄像头视频录制已启动，将保存至: {bottom_path}")
    else:
        print(f"[INFO] 下置摄像头视频录制已跳过（摄像头不可用）")
    # --- 结束视频录制设置 ---
    
    try:
        while cap_forward.isOpened():
            # --- 性能监控：记录循环开始时间 ---
            metrics.loop_start()

            # 特殊处理：RETURN_TO_BEGINNING_AREA状态下，跳过图像识别，只发送返航指令
            if auv_controller.current_state == State.RETURN_TO_BEGINNING_AREA:
                # 只执行状态处理器（发送BBAACC指令），不执行任何图像处理和控制
                state_return.handle(auv_controller, None, State=State)
                
                # 检查是否需要结束程序
                if auv_controller.current_state == State.END:
                    print("状态机已到达END状态，程序结束。")
                    break
                
                # 短暂延时，避免CPU占用过高
                time.sleep(0.05)
                continue  # 跳过后续所有处理，直接进入下一次循环

            # 1. 统一的感知阶段
            # 前置摄像头：必须成功
            ret_f, frame_f = cap_forward.read()
            if not ret_f:
                print("[错误] 前置摄像头读取失败，程序退出")
                break
            
            # 上下左右翻转前置摄像头画面（因为摄像头装反，相当于旋转180度）
            frame_f = cv2.flip(frame_f, -1)
            
            # 应用图像增强（提升水下暗环境的图像质量）
            if IMAGE_ENHANCEMENT_ENABLED and image_enhancer is not None:
                frame_f = image_enhancer.enhance(frame_f)

            # 下置摄像头：可选
            frame_b = None
            if bottom_cam_available and cap_bottom is not None:
                ret_b, frame_b = cap_bottom.read()
                if not ret_b:
                    print("[警告] 下置摄像头读取失败，此帧跳过下置摄像头处理")
                    frame_b = None
                else:
                    # 上下左右翻转下置摄像头画面（因为摄像头装反，相当于旋转180度）
                    frame_b = cv2.flip(frame_b, -1)
                    
                    # 应用图像增强（提升水下暗环境的图像质量）
                    if IMAGE_ENHANCEMENT_ENABLED and image_enhancer is not None:
                        frame_b = image_enhancer.enhance(frame_b)

            # --- 从串口接收陀螺仪与深度传感器数据并滤波 ---
            yaw, depth = get_latest_serial_data(max_age_ms=100)

            if depth is not None:
                update_depth(auv_controller, depth)
                if PERFORMANCE_MODE_ENABLED:
                    if metrics.loop_durations and len(metrics.loop_durations) % 10 == 0:
                        print(f"  [串口接收] 深度: {depth:.2f}m, 滤波后: {auv_controller.current_filtered_depth:.2f}m")
                else:
                    print(f"  [串口接收] 深度: {depth:.2f}m, 滤波后: {auv_controller.current_filtered_depth:.2f}m")

            if yaw is not None:
                just_initialized = update_yaw(auv_controller, yaw)
                if just_initialized:
                    # 仅在刚达成 5 帧初始化的时刻打印一次提示
                    print(f"  [Yaw初始化] 收集5帧完成，锁定目标航向: {auv_controller.target_yaw:.1f}")
                if PERFORMANCE_MODE_ENABLED:
                    if metrics.loop_durations and len(metrics.loop_durations) % 10 == 0:
                        print(f"  [串口接收] Yaw: {yaw}")
                else:
                    print(f"  [串口接收] Yaw: {yaw}")
            
            # 每次循环实时打印 yaw 与深度信息（不受性能模式开关影响）
            print(f"  [传感器] 实时: Yaw={auv_controller.current_yaw:.1f}deg | Depth={auv_controller.current_filtered_depth:.2f}m")

            # 在一帧上运行所有需要的识别
            metrics.infer_start()
            detections_forward = model.recognize(frame_f) # 前置摄像头图像识别均由model.recognize()完成
            inference_duration = metrics.infer_end()
            
            # 仅在特定状态下处理下置摄像头图像
            guideline_bottom = None
            if (auv_controller.current_state == State.GUIDELINE_FOLLOW and 
                auv_controller.guideline_sub_state == "BOTTOM_CAM_ALIGN" and
                bottom_cam_available and frame_b is not None):
                guideline_bottom = guideline_detector.detect(frame_b)

            perception_bundle = {
                'frame_forward': frame_f,
                'frame_bottom': frame_b,
                'detections_forward': detections_forward,
                'guideline_bottom': guideline_bottom,
                'current_depth': auv_controller.current_filtered_depth
            }

            # 2. 状态机决策阶段
            
            # 检查任务完成条件或超时条件
            # 使用单调时钟计算真实运行时间，不受系统时间调整影响
            elapsed_time = time.monotonic() - auv_controller.program_start_time
            
            # 每次循环打印当前运行时间（调试信息）
            minutes = int(elapsed_time // 60)
            seconds = elapsed_time % 60
            print(f"  [运行计时] 当前运行时间: {minutes}:{seconds:05.2f} / 总超时时间: {auv_controller.mission_timeout:.0f}秒")
            
            if (auv_controller.current_state != State.RETURN_TO_BEGINNING_AREA and 
                auv_controller.current_state != State.END):
                if elapsed_time >= auv_controller.mission_timeout:
                    print(f"\n[任务超时] 程序运行时间: {elapsed_time:.1f}s >= {auv_controller.mission_timeout:.1f}s")
                    print("[自动切换] 切换到 RETURN_TO_BEGINNING_AREA 状态\n")
                    auv_controller._transition(State.RETURN_TO_BEGINNING_AREA)
                elif auv_controller.task_finished >= 4:
                    print(f"\n[任务完成] 已完成 {auv_controller.task_finished} 个门框")
                    print("[自动切换] 切换到 RETURN_TO_BEGINNING_AREA 状态\n")
                    auv_controller._transition(State.RETURN_TO_BEGINNING_AREA)
            
            # 状态机返回一个指令元组 (forward, x, y, yaw)
            main_command_tuple = auv_controller.update(perception_bundle)
            
            # 简化控制模式处理
            target_depth = auv_controller.get_current_target_depth()
            if PERFORMANCE_MODE_ENABLED:
                if len(metrics.loop_durations) % 10 == 0:  # 每10帧打印一次
                    print(f"  [目标深度] {target_depth:.2f}m | 当前深度: {auv_controller.current_filtered_depth:.2f}m")
                    print(f"  [目标角度] {auv_controller.target_yaw:.1f}° | 当前角度: {auv_controller.current_yaw:.1f}°")
            else:
                print(f"  [目标深度] {target_depth:.2f}m | 当前深度: {auv_controller.current_filtered_depth:.2f}m")
                print(f"  [目标角度] {auv_controller.target_yaw:.1f}° | 当前角度: {auv_controller.current_yaw:.1f}°")

            # 3. 指令融合与执行阶段
            # 特殊处理：RETURN_TO_BEGINNING_AREA状态下，只发送特殊指令，不发送常规控制指令
            if auv_controller.current_state == State.RETURN_TO_BEGINNING_AREA:
                # 跳过常规控制指令发送，由return_to_beginning_area.py自行发送BBAACC指令
                pass
            elif main_command_tuple:
                fwd, x, y, yaw = main_command_tuple
                # --- 单轴PID开关覆盖 ---
                try:
                    if not PID_AXIS_ENABLE.get('forward', True):
                        fwd = 128
                    if not PID_AXIS_ENABLE.get('x', True):
                        x = 128
                    if not PID_AXIS_ENABLE.get('depth', True):
                        # 以视觉直通的方式强制输出中性，绕过depth_pid
                        y = ('vision_depth', 128)
                    if not PID_AXIS_ENABLE.get('yaw', True):
                        # 强制关闭yaw输出
                        yaw = 'no_yaw'
                except Exception:
                    pass

                final_y, final_yaw, control_mode_str = fuse_depth_yaw(y, yaw, auv_controller, vision_yaw_pid, global_depth_pid, global_yaw_pid)
                final_command_bytes = encode_command(fwd, x, final_y, final_yaw)
                if PID_TUNING_VISUALIZER_ENABLED:
                    # 简化的PID可视化
                    if hasattr(auv_controller, 'pid_visualizer'):
                        # 只记录全局控深（因为不再使用视觉控深）
                        depth_error = auv_controller.current_filtered_depth - target_depth
                        depth_error_cm = depth_error * 100.0
                        _, pid_out_depth = global_depth_pid.update(depth_error_cm)
                        auv_controller.pid_visualizer.update_data('depth', depth_error_cm, pid_out_depth)
                send_command(final_command_bytes, debug_print=not PERFORMANCE_MODE_ENABLED)

            # 当状态为END时，跳出循环
            if auv_controller.current_state == State.END:
                print("状态机已到达END状态，程序结束。")
                break

            # 4. 可视化模块：调用已解耦的osd.py模块
            try:
                yaw_info_mode = control_mode_str
            except Exception:
                yaw_info_mode = "-"

            display_f = draw_forward_osd(
                frame_f,
                detections_forward,
                is_ball=is_ball,
                current_state_name=auv_controller.current_state.name,
                guideline_sub_state=getattr(auv_controller, 'guideline_sub_state', 'N/A'),
                circle_kicked=auv_controller.circle_kicked,
                task_finished=auv_controller.task_finished,
                current_depth=auv_controller.current_filtered_depth,
                target_depth=auv_controller.target_depth,
                current_yaw=auv_controller.current_yaw,
                yaw_mode_str=yaw_info_mode,
                target_yaw=auv_controller.target_yaw,
                debug_front_follow=getattr(auv_controller, 'debug_front_follow', None) if DEBUG_MODE_ENABLED else None,
            )

            display_b = draw_bottom_osd(
                frame_b,
                guideline_bottom,
                bottom_cam_available=bottom_cam_available,
                frame_width_bottom=frame_width_bottom,
                frame_height_bottom=frame_height_bottom,
            )

            cv2.imshow("AUV Real-time Vision", display_f)
            cv2.imshow("AUV Bottom View", display_b)

            # --- 新增: 更新PID图表 (如果已启用) ---
            if PID_TUNING_VISUALIZER_ENABLED:
                # 强制每帧刷新图表
                pid_visualizer._last_plot_time = 0
                pid_visualizer.plot()
            # --- 结束 ---

            # 写入录像
            recorder.write_forward(display_f)
            recorder.write_bottom(display_b)

            # --- 性能监控：计算并记录耗时 ---
            loop_duration = metrics.loop_end()
            # 根据性能模式切换性能监控打印频率
            if PERFORMANCE_MODE_ENABLED:
                if len(metrics.loop_durations) % 20 == 0:
                    avg_loop = sum(metrics.loop_durations[-20:]) / 20
                    avg_inference = sum(metrics.inference_durations[-20:]) / 20
                    stats = serial_reader.get_stats() if serial_reader else {'last_age_ms': float('inf'), 'pps': 0}
                    print(f"  [性能监控] 循环耗时: {avg_loop:.4f}s | 推理: {avg_inference:.4f}s | 串口延迟: {stats['last_age_ms']:.1f}ms | 串口PPS: {stats['pps']:.1f}")
            else:
                stats = serial_reader.get_stats() if serial_reader else {'last_age_ms': float('inf'), 'pps': 0}
                print(f"  [性能监控] 本次循环耗时: {loop_duration:.4f}s | 模型识别耗时: {inference_duration:.4f}s | 串口延迟: {stats['last_age_ms']:.1f}ms | 串口PPS: {stats['pps']:.1f}")
            
            # 处理按键输入
            key = cv2.waitKey(1) & 0xFF
            
            # 调试模式下的命令处理
            if DEBUG_MODE_ENABLED and key in DEBUG_COMMANDS:
                command = DEBUG_COMMANDS[key]
                
                if command == 'EXIT':
                    print("[调试] 用户请求退出。")
                    break
                elif command == 'RESET_PID':
                    # 针对性PID重置，根据当前状态决定
                    if auv_controller.current_state == State.GUIDELINE_FOLLOW:
                        global_yaw_pid.reset()
                        print("[调试] 重置global_yaw_pid")
                    elif auv_controller.current_state == State.DOORFRAME_PASSTHROUGH:
                        left_or_right_pid.reset()
                        forward_pid.reset()
                        print("[调试] 重置left_or_right_pid和forward_pid")
                    else:
                        print("[调试] 当前状态无需特定PID重置")
                elif command == 'EMERGENCY_STOP':
                    auv_controller._emergency_stop()
                elif hasattr(State, command):
                    # 状态切换命令
                    target_state = getattr(State, command)
                    auv_controller._force_transition(target_state)
                else:
                    print(f"[调试] 未知命令: {command}")
            
            # 常规退出键 (适用于所有模式)
            elif key == ord('q'):
                print("用户手动退出。")
                break
    finally:
        # 清理资源
        print("\n正在清理资源...")
        cap_forward.release()
        if cap_bottom is not None:
            cap_bottom.release()
        recorder.release()
        
        # --- 新增: 关闭PID图表 (如果已启用) ---
        if PID_TUNING_VISUALIZER_ENABLED:
            pid_visualizer.close()
        # --- 结束 ---

        cv2.destroyAllWindows()
        
        # 发送悬停指令，确保AUV停止（性能模式关闭时打印）
        send_command(bytes([0xab, 128, 128, 128, 128, 0xcd]), debug_print=not PERFORMANCE_MODE_ENABLED)
        
        # 关闭串口线程和发送器
        stop_serial_reader()
        close_serial_sender()

        # --- 保存性能监控数据到JSON文件 ---
        print("\n正在保存性能监控数据...")
        try:
            perf_output_filename = metrics.save_json(timestr)
            print(f"性能数据已成功保存至: {perf_output_filename}")
        except Exception as e:
            print(f"[错误] 保存性能数据时发生错误: {e}")
        # --- 结束数据保存 ---
        
        print("程序已安全关闭。")


if __name__ == "__main__":
    run_auv_mission()
