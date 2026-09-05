#!/usr/bin/env python3
"""
图像增强调试工具
用于实时测试和调整图像增强参数，帮助找到最佳的水下图像增强设置
"""
import cv2
import sys
import os

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.modules.vision.image_enhancement import ImageEnhancer


# 全局变量存储参数
clahe_clip = 30  # 实际值 = clahe_clip / 10.0
gamma_value = 15  # 实际值 = gamma_value / 10.0
brightness_value = 30

# 增强模式
mode = 0  # 0=CLAHE, 1=Gamma, 2=Brightness, 3=Auto


def on_clahe_change(val):
    """CLAHE滑块回调"""
    global clahe_clip
    clahe_clip = val


def on_gamma_change(val):
    """Gamma滑块回调"""
    global gamma_value
    gamma_value = val


def on_brightness_change(val):
    """亮度滑块回调"""
    global brightness_value
    brightness_value = val


def on_mode_change(val):
    """模式滑块回调"""
    global mode
    mode = val


def print_help():
    """打印帮助信息"""
    print("\n" + "="*60)
    print("图像增强调试工具")
    print("="*60)
    print("使用方法:")
    print("  - 滑动条实时调整参数")
    print("  - 按 's' 保存当前增强后的图像")
    print("  - 按 'p' 打印当前参数")
    print("  - 按 'q' 或 ESC 退出")
    print("\n增强模式:")
    print("  0 = CLAHE (推荐用于水下)")
    print("  1 = Gamma (整体提亮)")
    print("  2 = Brightness (简单亮度增加)")
    print("  3 = Auto (CLAHE + Gamma组合)")
    print("\n参数说明:")
    print("  CLAHE Clip: 对比度限制 (1.0-5.0, 推荐2.5-3.5)")
    print("  Gamma: 伽马值 (0.5-3.0, >1增亮, <1变暗)")
    print("  Brightness: 亮度增加值 (0-100)")
    print("="*60 + "\n")


def print_current_params():
    """打印当前参数"""
    mode_names = ["CLAHE", "Gamma", "Brightness", "Auto"]
    print(f"\n当前参数:")
    print(f"  模式: {mode_names[mode]}")
    print(f"  CLAHE Clip: {clahe_clip/10.0:.1f}")
    print(f"  Gamma: {gamma_value/10.0:.1f}")
    print(f"  Brightness: {brightness_value}")
    print()


def test_camera():
    """使用摄像头实时测试图像增强"""
    print_help()
    
    # 尝试打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[错误] 无法打开摄像头")
        return
    
    print("[信息] 摄像头已打开")
    
    # 创建窗口和滑块
    cv2.namedWindow("Original")
    cv2.namedWindow("Enhanced")
    cv2.namedWindow("Controls")
    
    cv2.createTrackbar("Mode", "Controls", mode, 3, on_mode_change)
    cv2.createTrackbar("CLAHE Clip x10", "Controls", clahe_clip, 50, on_clahe_change)
    cv2.createTrackbar("Gamma x10", "Controls", gamma_value, 30, on_gamma_change)
    cv2.createTrackbar("Brightness", "Controls", brightness_value, 100, on_brightness_change)
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[错误] 无法读取摄像头帧")
                break
            
            # 翻转图像（如果需要）
            frame = cv2.flip(frame, -1)
            
            # 获取当前参数
            current_clahe = max(1.0, clahe_clip / 10.0)
            current_gamma = max(0.5, gamma_value / 10.0)
            current_brightness = brightness_value
            
            # 根据模式创建增强器
            if mode == 0:  # CLAHE
                enhancer = ImageEnhancer(
                    enable_clahe=True,
                    enable_gamma=False,
                    enable_brightness=False,
                    clahe_clip_limit=current_clahe
                )
            elif mode == 1:  # Gamma
                enhancer = ImageEnhancer(
                    enable_clahe=False,
                    enable_gamma=True,
                    enable_brightness=False,
                    gamma_value=current_gamma
                )
            elif mode == 2:  # Brightness
                enhancer = ImageEnhancer(
                    enable_clahe=False,
                    enable_gamma=False,
                    enable_brightness=True,
                    brightness_value=current_brightness
                )
            else:  # Auto
                enhancer = ImageEnhancer(
                    enable_clahe=True,
                    enable_gamma=True,
                    enable_brightness=False,
                    clahe_clip_limit=current_clahe,
                    gamma_value=current_gamma
                )
            
            # 应用增强
            enhanced = enhancer.enhance(frame)
            
            # 在图像上显示参数信息
            mode_names = ["CLAHE", "Gamma", "Brightness", "Auto"]
            info_text = f"Mode: {mode_names[mode]} | CLAHE: {current_clahe:.1f} | Gamma: {current_gamma:.1f} | Brightness: {current_brightness}"
            cv2.putText(enhanced, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 显示原始和增强后的图像
            cv2.imshow("Original", frame)
            cv2.imshow("Enhanced", enhanced)
            
            # 处理按键
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # q或ESC退出
                break
            elif key == ord('s'):  # 保存图像
                filename_orig = f"original_{frame_count}.jpg"
                filename_enhanced = f"enhanced_{frame_count}.jpg"
                cv2.imwrite(filename_orig, frame)
                cv2.imwrite(filename_enhanced, enhanced)
                print(f"[保存] 原始图像: {filename_orig}")
                print(f"[保存] 增强图像: {filename_enhanced}")
                frame_count += 1
            elif key == ord('p'):  # 打印参数
                print_current_params()
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n[信息] 程序已退出")


def test_image(image_path):
    """使用静态图像测试图像增强"""
    print_help()
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"[错误] 无法读取图像: {image_path}")
        return
    
    print(f"[信息] 已加载图像: {image_path}")
    print(f"[信息] 图像尺寸: {image.shape[1]}x{image.shape[0]}")
    
    # 创建窗口和滑块
    cv2.namedWindow("Original")
    cv2.namedWindow("Enhanced")
    cv2.namedWindow("Controls")
    
    cv2.createTrackbar("Mode", "Controls", mode, 3, on_mode_change)
    cv2.createTrackbar("CLAHE Clip x10", "Controls", clahe_clip, 50, on_clahe_change)
    cv2.createTrackbar("Gamma x10", "Controls", gamma_value, 30, on_gamma_change)
    cv2.createTrackbar("Brightness", "Controls", brightness_value, 100, on_brightness_change)
    
    frame_count = 0
    
    try:
        while True:
            # 获取当前参数
            current_clahe = max(1.0, clahe_clip / 10.0)
            current_gamma = max(0.5, gamma_value / 10.0)
            current_brightness = brightness_value
            
            # 根据模式创建增强器
            if mode == 0:  # CLAHE
                enhancer = ImageEnhancer(
                    enable_clahe=True,
                    enable_gamma=False,
                    enable_brightness=False,
                    clahe_clip_limit=current_clahe
                )
            elif mode == 1:  # Gamma
                enhancer = ImageEnhancer(
                    enable_clahe=False,
                    enable_gamma=True,
                    enable_brightness=False,
                    gamma_value=current_gamma
                )
            elif mode == 2:  # Brightness
                enhancer = ImageEnhancer(
                    enable_clahe=False,
                    enable_gamma=False,
                    enable_brightness=True,
                    brightness_value=current_brightness
                )
            else:  # Auto
                enhancer = ImageEnhancer(
                    enable_clahe=True,
                    enable_gamma=True,
                    enable_brightness=False,
                    clahe_clip_limit=current_clahe,
                    gamma_value=current_gamma
                )
            
            # 应用增强
            enhanced = enhancer.enhance(image)
            
            # 在图像上显示参数信息
            mode_names = ["CLAHE", "Gamma", "Brightness", "Auto"]
            info_text = f"Mode: {mode_names[mode]} | CLAHE: {current_clahe:.1f} | Gamma: {current_gamma:.1f} | Brightness: {current_brightness}"
            display_enhanced = enhanced.copy()
            cv2.putText(display_enhanced, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 显示原始和增强后的图像
            cv2.imshow("Original", image)
            cv2.imshow("Enhanced", display_enhanced)
            
            # 处理按键
            key = cv2.waitKey(10) & 0xFF
            
            if key == ord('q') or key == 27:  # q或ESC退出
                break
            elif key == ord('s'):  # 保存图像
                filename = f"enhanced_{os.path.basename(image_path).rsplit('.', 1)[0]}_{frame_count}.jpg"
                cv2.imwrite(filename, enhanced)
                print(f"[保存] {filename}")
                frame_count += 1
            elif key == ord('p'):  # 打印参数
                print_current_params()
    
    finally:
        cv2.destroyAllWindows()
        print("\n[信息] 程序已退出")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # 没有参数，使用摄像头
        print("[模式] 摄像头实时测试")
        test_camera()
    elif len(sys.argv) == 2:
        # 有参数，作为图像文件路径
        print("[模式] 静态图像测试")
        test_image(sys.argv[1])
    else:
        print("用法:")
        print("  实时摄像头测试: python test_image_enhancement.py")
        print("  静态图像测试:   python test_image_enhancement.py <图像路径>")
        sys.exit(1)

