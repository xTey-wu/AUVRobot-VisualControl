#!/usr/bin/env python3
"""
HSV颜色阈值调试工具
用于实时调整HSV阈值，方便调试小球颜色识别参数
支持图像增强功能
"""
import cv2
import numpy as np
import sys
import os

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.modules.vision.image_enhancement import ImageEnhancer


class HSVThresholdTuner:
    """HSV颜色阈值调试器"""
    
    def __init__(self, camera_id=0, initial_preset='red'):
        """
        初始化调试器
        
        参数:
            camera_id: 摄像头ID
            initial_preset: 初始预设 ('red', 'blue', 'custom')
        """
        self.camera_id = camera_id
        self.cap = None
        
        # HSV阈值范围（默认值）
        self.h_min = 0
        self.h_max = 180
        self.s_min = 0
        self.s_max = 255
        self.v_min = 0
        self.v_max = 255
        
        # 预设阈值
        self.presets = {
            'red_low': {
                'h_min': 0, 'h_max': 10,
                's_min': 100, 's_max': 255,
                'v_min': 100, 'v_max': 255
            },
            'red_high': {
                'h_min': 160, 'h_max': 180,
                's_min': 100, 's_max': 255,
                'v_min': 100, 'v_max': 255
            },
            'blue': {
                'h_min': 100, 'h_max': 130,
                's_min': 100, 's_max': 255,
                'v_min': 100, 'v_max': 255
            },
            'green': {
                'h_min': 40, 'h_max': 80,
                's_min': 100, 's_max': 255,
                'v_min': 100, 'v_max': 255
            },
            'yellow': {
                'h_min': 20, 'h_max': 40,
                's_min': 100, 's_max': 255,
                'v_min': 100, 'v_max': 255
            }
        }
        
        # 应用初始预设
        if initial_preset in self.presets:
            self.apply_preset(initial_preset)
        
        # 形态学操作参数
        self.morph_kernel_size = 5
        self.enable_morph = True
        
        # 显示模式
        self.show_mode = 0  # 0=原图+掩码, 1=HSV图, 2=各通道分离
        
        # 图像增强
        self.enable_enhancement = True
        self.image_enhancer = ImageEnhancer(
            enable_clahe=True,
            enable_gamma=False,
            enable_brightness=False,
            clahe_clip_limit=3.0
        )
    
    def apply_preset(self, preset_name):
        """应用预设阈值"""
        if preset_name in self.presets:
            preset = self.presets[preset_name]
            self.h_min = preset['h_min']
            self.h_max = preset['h_max']
            self.s_min = preset['s_min']
            self.s_max = preset['s_max']
            self.v_min = preset['v_min']
            self.v_max = preset['v_max']
            print(f"[预设] 已应用 '{preset_name}' 预设")
    
    def nothing(self, x):
        """滑块回调函数（占位）"""
        pass
    
    def create_trackbars(self):
        """创建控制滑块窗口"""
        cv2.namedWindow('HSV Controls')
        
        # H通道滑块
        cv2.createTrackbar('H Min', 'HSV Controls', self.h_min, 180, self.nothing)
        cv2.createTrackbar('H Max', 'HSV Controls', self.h_max, 180, self.nothing)
        
        # S通道滑块
        cv2.createTrackbar('S Min', 'HSV Controls', self.s_min, 255, self.nothing)
        cv2.createTrackbar('S Max', 'HSV Controls', self.s_max, 255, self.nothing)
        
        # V通道滑块
        cv2.createTrackbar('V Min', 'HSV Controls', self.v_min, 255, self.nothing)
        cv2.createTrackbar('V Max', 'HSV Controls', self.v_max, 255, self.nothing)
        
        # 形态学操作滑块
        cv2.createTrackbar('Morph Kernel', 'HSV Controls', self.morph_kernel_size, 15, self.nothing)
        cv2.createTrackbar('Enable Morph', 'HSV Controls', 1 if self.enable_morph else 0, 1, self.nothing)
        
        # 图像增强滑块
        cv2.createTrackbar('Enable Enhancement', 'HSV Controls', 1 if self.enable_enhancement else 0, 1, self.nothing)
    
    def get_trackbar_values(self):
        """获取当前滑块值"""
        self.h_min = cv2.getTrackbarPos('H Min', 'HSV Controls')
        self.h_max = cv2.getTrackbarPos('H Max', 'HSV Controls')
        self.s_min = cv2.getTrackbarPos('S Min', 'HSV Controls')
        self.s_max = cv2.getTrackbarPos('S Max', 'HSV Controls')
        self.v_min = cv2.getTrackbarPos('V Min', 'HSV Controls')
        self.v_max = cv2.getTrackbarPos('V Max', 'HSV Controls')
        self.morph_kernel_size = max(1, cv2.getTrackbarPos('Morph Kernel', 'HSV Controls'))
        self.enable_morph = cv2.getTrackbarPos('Enable Morph', 'HSV Controls') == 1
        self.enable_enhancement = cv2.getTrackbarPos('Enable Enhancement', 'HSV Controls') == 1
    
    def create_mask(self, hsv_image):
        """根据当前阈值创建掩码"""
        lower = np.array([self.h_min, self.s_min, self.v_min])
        upper = np.array([self.h_max, self.s_max, self.v_max])
        
        mask = cv2.inRange(hsv_image, lower, upper)
        
        # 应用形态学操作
        if self.enable_morph and self.morph_kernel_size > 0:
            kernel = np.ones((self.morph_kernel_size, self.morph_kernel_size), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def print_current_values(self):
        """打印当前阈值参数"""
        print("\n" + "="*60)
        print("当前HSV阈值参数：")
        print("="*60)
        print(f"H通道: [{self.h_min:3d}, {self.h_max:3d}]")
        print(f"S通道: [{self.s_min:3d}, {self.s_max:3d}]")
        print(f"V通道: [{self.v_min:3d}, {self.v_max:3d}]")
        print(f"形态学核大小: {self.morph_kernel_size}")
        print(f"形态学操作: {'启用' if self.enable_morph else '禁用'}")
        print("="*60)
        print("代码格式：")
        print(f"LOWER = np.array([{self.h_min}, {self.s_min}, {self.v_min}])")
        print(f"UPPER = np.array([{self.h_max}, {self.s_max}, {self.v_max}])")
        print("="*60 + "\n")
    
    def run(self):
        """运行调试器主循环"""
        # 打开摄像头
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print(f"[错误] 无法打开摄像头 {self.camera_id}")
            return
        
        print("="*60)
        print("HSV颜色阈值调试工具（支持图像增强）")
        print("="*60)
        print("功能说明：")
        print("  - 拖动滑块调整HSV阈值")
        print("  - 实时查看掩码效果")
        print("  - 图像增强功能（CLAHE）")
        print("  - 快速找到最佳颜色识别参数")
        print("\n快捷键：")
        print("  p - 打印当前参数")
        print("  1 - 应用红色(低色调)预设")
        print("  2 - 应用红色(高色调)预设")
        print("  3 - 应用蓝色预设")
        print("  4 - 应用绿色预设")
        print("  5 - 应用黄色预设")
        print("  m - 切换显示模式")
        print("  f - 翻转图像")
        print("  s - 保存当前帧")
        print("  q/ESC - 退出")
        print("="*60 + "\n")
        
        # 创建控制滑块
        self.create_trackbars()
        
        frame_count = 0
        flip_image = False
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("[错误] 无法读取摄像头帧")
                    break
                
                # 可选翻转
                if flip_image:
                    frame = cv2.flip(frame, -1)
                
                # 获取当前滑块值
                self.get_trackbar_values()
                
                # 应用图像增强（如果启用）
                if self.enable_enhancement:
                    frame = self.image_enhancer.enhance(frame)
                
                # 转换到HSV色彩空间
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                
                # 创建掩码
                mask = self.create_mask(hsv)
                
                # 应用掩码到原图
                result = cv2.bitwise_and(frame, frame, mask=mask)
                
                # 根据显示模式显示不同内容
                if self.show_mode == 0:
                    # 模式0：原图 + 掩码 + 结果
                    mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                    top_row = np.hstack([frame, mask_3channel])
                    bottom_row = np.hstack([result, result])
                    display = np.vstack([top_row, bottom_row])
                    
                    # 添加文字说明
                    cv2.putText(display, "Original", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(display, "Mask", (frame.shape[1] + 10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(display, "Result", (10, frame.shape[0] + 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                elif self.show_mode == 1:
                    # 模式1：HSV各通道
                    h, s, v = cv2.split(hsv)
                    h_colored = cv2.applyColorMap(h, cv2.COLORMAP_HSV)
                    s_colored = cv2.applyColorMap(s, cv2.COLORMAP_BONE)
                    v_colored = cv2.applyColorMap(v, cv2.COLORMAP_BONE)
                    
                    top_row = np.hstack([frame, h_colored])
                    bottom_row = np.hstack([s_colored, v_colored])
                    display = np.vstack([top_row, bottom_row])
                    
                    cv2.putText(display, "Original", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(display, "H Channel", (frame.shape[1] + 10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(display, "S Channel", (10, frame.shape[0] + 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(display, "V Channel", (frame.shape[1] + 10, frame.shape[0] + 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                else:
                    # 模式2：大图对比
                    mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                    display = np.hstack([frame, mask_3channel, result])
                    
                    cv2.putText(display, "Original", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(display, "Mask", (frame.shape[1] + 10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(display, "Result", (frame.shape[1]*2 + 10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 显示当前参数
                param_text = f"HSV: [{self.h_min},{self.h_max}] [{self.s_min},{self.s_max}] [{self.v_min},{self.v_max}]"
                cv2.putText(display, param_text, (10, display.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # 显示画面
                cv2.imshow('HSV Threshold Tuner', display)
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # q或ESC
                    print("\n[退出] 用户退出")
                    break
                elif key == ord('p'):  # 打印参数
                    self.print_current_values()
                elif key == ord('1'):  # 红色低色调预设
                    self.apply_preset('red_low')
                    self.create_trackbars()  # 更新滑块
                elif key == ord('2'):  # 红色高色调预设
                    self.apply_preset('red_high')
                    self.create_trackbars()
                elif key == ord('3'):  # 蓝色预设
                    self.apply_preset('blue')
                    self.create_trackbars()
                elif key == ord('4'):  # 绿色预设
                    self.apply_preset('green')
                    self.create_trackbars()
                elif key == ord('5'):  # 黄色预设
                    self.apply_preset('yellow')
                    self.create_trackbars()
                elif key == ord('m'):  # 切换显示模式
                    self.show_mode = (self.show_mode + 1) % 3
                    print(f"[显示模式] 切换到模式 {self.show_mode}")
                elif key == ord('f'):  # 翻转图像
                    flip_image = not flip_image
                    print(f"[翻转] {'启用' if flip_image else '禁用'}")
                elif key == ord('s'):  # 保存当前帧
                    filename = f"hsv_tuner_frame_{frame_count}.jpg"
                    cv2.imwrite(filename, display)
                    print(f"[保存] 已保存到 {filename}")
                    frame_count += 1
        
        except KeyboardInterrupt:
            print("\n[中断] Ctrl+C 检测到")
        
        finally:
            # 打印最终参数
            print("\n" + "="*60)
            print("最终参数：")
            print("="*60)
            self.print_current_values()
            
            # 清理资源
            self.cap.release()
            cv2.destroyAllWindows()
            print("[✓] 资源已清理")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='HSV颜色阈值调试工具')
    parser.add_argument('--camera', type=int, default=0, help='摄像头ID (默认: 0)')
    parser.add_argument('--preset', type=str, default='red_low', 
                       choices=['red_low', 'red_high', 'blue', 'green', 'yellow', 'custom'],
                       help='初始预设 (默认: red_low)')
    
    args = parser.parse_args()
    
    # 创建并运行调试器
    tuner = HSVThresholdTuner(camera_id=args.camera, initial_preset=args.preset)
    tuner.run()


if __name__ == '__main__':
    main()

