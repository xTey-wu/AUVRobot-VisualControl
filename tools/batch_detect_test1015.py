#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量视频目标检测脚本（YOLOv11 / Ultralytics）

功能：
- 加载指定YOLO权重（默认：models/combine1014.pt）
- 扫描输入目录下的所有视频文件，逐帧推理并可视化
- 将带框结果视频输出到指定目录（默认：outputs/detected_videos）

使用示例：
python tools/batch_detect_test1015.py \
  --model models/combine1014.pt \
  --input_dir examples/videos \
  --output_dir outputs/detected_videos \
  --imgsz 640 --conf 0.25 --device auto

说明：
- 该脚本依赖 ultralytics>=8，opencv-python。请确保 requirements 已安装。
- 为稳定输出，统一使用OpenCV进行视频写出，并从流式推理结果中取可视化帧。
"""

import argparse
import os
import sys
from typing import List

import cv2
import torch
from ultralytics import YOLO


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def is_video_file(filename: str) -> bool:
    """判断是否为常见视频文件后缀。"""
    lower = filename.lower()
    return lower.endswith((".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".m4v"))


def list_videos(input_dir: str) -> List[str]:
    """列出目录下的全部视频文件（不递归）。"""
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"输入目录不存在：{input_dir}")
    files = []
    for name in sorted(os.listdir(input_dir)):
        full_path = os.path.join(input_dir, name)
        if os.path.isfile(full_path) and is_video_file(name):
            files.append(full_path)
    return files


def ensure_dir(path: str) -> None:
    """确保输出目录存在。"""
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def select_device(arg: str) -> str:
    """根据参数和可用性选择推理设备。支持 auto/cpu/cuda。"""
    if arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if arg in ("cpu", "cuda"):
        if arg == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return arg
    # 兜底：未知输入按auto处理
    return "cuda" if torch.cuda.is_available() else "cpu"


def open_writer_like(src_video: str, output_path: str) -> cv2.VideoWriter:
    """按源视频FPS/尺寸打开一个VideoWriter，失败时采用合理默认值。"""
    cap = cv2.VideoCapture(src_video)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{src_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # 兼容某些容器缺失fps/尺寸元数据的情况
    if fps is None or fps <= 0:
        fps = 30.0
    if width <= 0 or height <= 0:
        width, height = 1280, 720

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"无法创建输出视频：{output_path}")
    return writer


def infer_single_video(model: YOLO, video_path: str, output_path: str, conf: float, imgsz: int, device: str) -> None:
    """对单个视频执行流式推理并写出可视化结果。"""
    writer = open_writer_like(video_path, output_path)
    try:
        # 使用Ultralytics的流式推理接口直接从源视频读取帧
        results_generator = model.predict(
            source=video_path,
            stream=True,
            conf=conf,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )
        for result in results_generator:
            # result.plot() 返回BGR格式的已绘制帧
            annotated_frame = result.plot()
            writer.write(annotated_frame)
    finally:
        writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="批量视频检测（YOLOv11/Ultralytics）")
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join(PROJECT_ROOT, "models", "combine1014.pt"),
        help="YOLO权重文件路径（.pt/.onnx/.engine等）",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "examples", "videos"),
        help="输入视频目录",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "outputs", "detected_videos"),
        help="输出结果视频目录",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="推理输入尺寸")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="推理设备选择",
    )

    args = parser.parse_args()
    device = select_device(args.device)

    if not os.path.isfile(args.model):
        print(f"权重文件不存在：{args.model}")
        sys.exit(1)

    ensure_dir(args.output_dir)

    # 加载YOLO模型（兼容YOLOv11权重）
    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"加载模型失败：{e}")
        sys.exit(1)

    videos = list_videos(args.input_dir)
    if not videos:
        print(f"未在目录中找到视频文件：{args.input_dir}")
        sys.exit(0)

    print(f"设备：{device} | 模型：{args.model}")
    print(f"输入：{args.input_dir} | 输出：{args.output_dir}")
    print(f"共 {len(videos)} 个视频待处理\n")

    # 逐个视频处理并写出到输出目录
    for idx, video_path in enumerate(videos, start=1):
        base = os.path.basename(video_path)
        name, _ = os.path.splitext(base)
        out_path = os.path.join(args.output_dir, f"{name}_detected.mp4")
        print(f"[{idx}/{len(videos)}] 处理：{base} -> {os.path.basename(out_path)}")
        try:
            infer_single_video(
                model=model,
                video_path=video_path,
                output_path=out_path,
                conf=args.conf,
                imgsz=args.imgsz,
                device=device,
            )
        except Exception as e:
            print(f"  处理失败：{e}")

    print("\n全部完成。")


if __name__ == "__main__":
    main()

