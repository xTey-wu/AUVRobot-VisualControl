import torch
import cv2
import numpy as np
from ultralytics import YOLO
import os
from typing import List, Dict, Any


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "combine1014.pt")


class ObjectRecognizer:
    """
    目标识别类，使用YOLOv11模型进行目标检测
    """  
    
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        """
        初始化识别器
        
        Args:
            model_path: 模型文件路径
        """
        self.model_path = model_path
        self.model = None
        self.device = None
        self._load_model()
    
    def _load_model(self):
        """
        加载YOLOv11模型并配置GPU
        """
        try:
            # 检查GPU是否可用
            if torch.cuda.is_available():
                self.device = "cuda"
                print(f"使用GPU进行推理: {torch.cuda.get_device_name(0)}")
            else:
                self.device = "cpu"
                print("GPU不可用，使用CPU进行推理")

            # 加载YOLO .pt 模型
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
            
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            print(f"模型加载成功: {self.model_path}")

        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            raise
    
    def recognize(self, image: np.ndarray, confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        识别图片中的目标
        
        Args:
            image: 输入图片 (numpy数组格式)
            confidence_threshold: 置信度阈值
            
        Returns:
            List[Dict]: 检测结果列表，每个元素包含：
                - class_name: 物品类型名称
                - class_id: 类别ID
                - confidence: 置信度
                - bbox: 边界框坐标 [x1, y1, x2, y2]
                - center: 中心坐标 [x_center, y_center]
                - area: 识别框面积
        """
        if self.model is None:
            raise RuntimeError("模型未正确加载")
        
        if image is None or image.size == 0:
            raise ValueError("输入图片无效")
        
        try:
            # 进行推理
            results = self.model(image, device=self.device, verbose=False)
            detections = []
            
            for result in results:
                boxes = result.boxes
                
                if boxes is not None:
                    for box in boxes:
                        # 获取置信度
                        confidence = float(box.conf.item())
                        
                        # 过滤低置信度检测
                        if confidence < confidence_threshold:
                            continue
                        
                        # 获取边界框坐标
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        # 计算中心坐标
                        x_center = (x1 + x2) / 2
                        y_center = (y1 + y2) / 2
                        
                        # 计算面积
                        area = (x2 - x1) * (y2 - y1)
                        
                        # 获取类别ID和名称
                        class_id = int(box.cls.item())
                        class_name = self.model.names[class_id]
                        
                        detection = {
                            "class_name": class_name,
                            "class_id": class_id,
                            "confidence": confidence,
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "center": [int(x_center), int(y_center)],
                            "area": int(area)
                        }
                        
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            print(f"识别过程出错: {str(e)}")
            return []

# 创建全局识别器实例
_recognizer = None

def get_recognizer() -> ObjectRecognizer:
    """
    获取识别器实例（单例模式）
    """
    global _recognizer
    if _recognizer is None:
        _recognizer = ObjectRecognizer()
    return _recognizer

def recognize_objects(image: np.ndarray, confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
    """
    识别图片中的目标（便捷函数）
    
    Args:
        image: 输入图片 (numpy数组格式)
        confidence_threshold: 置信度阈值
        
    Returns:
        List[Dict]: 检测结果列表，每个元素包含：
            - class_name: 物品类型名称  
            - class_id: 类别ID
            - confidence: 置信度
            - bbox: 边界框坐标 [x1, y1, x2, y2]
            - center: 中心坐标 [x_center, y_center]
            - area: 识别框面积
    """
    recognizer = get_recognizer()
    return recognizer.recognize(image, confidence_threshold)


if __name__ == "__main__":
    # 创建识别器
    recognizer = ObjectRecognizer()

    # 输入/输出视频路径
    input_video = "451f648a9233209808e30cb92f5a75c7.mp4"
    base_name = os.path.splitext(os.path.basename(input_video))[0]
    output_video = os.path.join("testvideo", f"{base_name}_detected.avi")

    # 打开视频
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {input_video}")
        exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps == 0:
        fps = 24
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 创建VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(output_video, fourcc, float(fps), (width, height))
    if not writer.isOpened():
        print(f"❌ 无法创建输出视频: {output_video}")
        cap.release()
        exit(1)

    frame_idx = 0
    print(f"▶ 开始处理视频: {input_video} -> {output_video} @ {fps:.1f} FPS, 尺寸 {width}x{height}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 识别
            results = recognizer.recognize(frame)

            # 绘制检测结果
            for result in results:
                x1, y1, x2, y2 = result['bbox']
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{result['class_name']}: {result['confidence']:.2f}"
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            writer.write(frame)

            frame_idx += 1
            if frame_idx % 50 == 0:
                print(f"  - 已处理帧: {frame_idx}")
    except KeyboardInterrupt:
        print("⏹ 中断处理")
    finally:
        cap.release()
        writer.release()
        print(f"✅ 处理完成，总帧数: {frame_idx}，输出保存至: {output_video}")
