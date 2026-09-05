import cv2
import numpy as np
import time

def test_cameras():
    """
    测试两个USB摄像头是否能正常打开和读取
    """
    print("开始测试USB摄像头...")
    
    # 尝试打开两个摄像头
    cameras = []
    for i in range(2):
        print(f"正在尝试打开摄像头 {i}...")
        try:
            cap = cv2.VideoCapture(i)
            
            if cap.isOpened():
                # 尝试读取一帧来确认摄像头真的可用
                ret, frame = cap.read()
                if ret:
                    # 获取摄像头信息
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    print(f"摄像头 {i} 打开成功!")
                    print(f"  分辨率: {width}x{height}")
                    print(f"  帧率: {fps}")
                    
                    cameras.append(cap)
                else:
                    print(f"摄像头 {i} 打开但无法读取帧!")
                    cap.release()
                    cameras.append(None)
            else:
                print(f"摄像头 {i} 打开失败!")
                cameras.append(None)
        except Exception as e:
            print(f"摄像头 {i} 打开时发生异常: {e}")
            cameras.append(None)
    
    if not any(cameras):
        print("没有找到可用的摄像头!")
        return
    
    print(f"\n成功打开 {sum(1 for cam in cameras if cam is not None)} 个摄像头")
    print("按 'q' 键退出测试")
    
    try:
        while True:
            frames = []
            
            # 读取所有摄像头的帧
            for i, cap in enumerate(cameras):
                if cap is not None:
                    ret, frame = cap.read()
                    if ret:
                        # 在图像上添加摄像头编号
                        cv2.putText(frame, f"Camera {i}", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        frames.append(frame)
                    else:
                        print(f"摄像头 {i} 读取失败")
                        frames.append(None)
                else:
                    frames.append(None)
            
            # 显示图像
            if len(frames) == 2 and all(frame is not None for frame in frames):
                # 水平拼接两个摄像头的图像
                combined_frame = np.hstack(frames)
                # cv2.imshow('USB Cameras Test', combined_frame)
                print("Two cameras are available")
            elif len(frames) == 1 and frames[0] is not None:
                # 只显示一个摄像头的图像
                # cv2.imshow('USB Cameras Test', frames[0])
                print("Only one camera is available")
            
            # 检查按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            time.sleep(0.01)  # 短暂延时
            
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    
    finally:
        # 释放资源
        for i, cap in enumerate(cameras):
            if cap is not None:
                cap.release()
                print(f"摄像头 {i} 已释放")
        
        cv2.destroyAllWindows()
        print("测试完成")

def list_camera_devices():
    """
    列出系统中可用的摄像头设备
    """
    print("检测系统中的摄像头设备...")
    
    # 检查前10个设备索引
    available_cameras = []
    for i in range(10):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # 尝试读取一帧来确认摄像头真的可用
                ret, frame = cap.read()
                if ret:
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    print(f"摄像头 {i}: {width}x{height}")
                    available_cameras.append(i)
                else:
                    print(f"摄像头 {i}: 打开但无法读取帧")
                cap.release()
            else:
                print(f"摄像头 {i}: 无法打开")
        except Exception as e:
            print(f"摄像头 {i}: 异常 - {e}")
    
    if available_cameras:
        print(f"找到 {len(available_cameras)} 个可用摄像头: {available_cameras}")
    else:
        print("未找到可用的摄像头设备")
    
    return available_cameras

if __name__ == "__main__":
    print("=" * 50)
    print("USB摄像头测试程序")
    print("=" * 50)
    
    # 首先列出可用的摄像头设备
    available_cameras = list_camera_devices()
    
    if available_cameras:
        print("\n开始摄像头测试...")
        test_cameras()
    else:
        print("没有找到可用的摄像头，请检查设备连接")
