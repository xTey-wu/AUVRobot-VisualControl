import os
import cv2
import argparse
from typing import Optional


def extract_frames(
    input_path: str,
    output_dir: str,
    rate_fps: float = 1.0,
    start_time_s: float = 0.0,
    end_time_s: Optional[float] = None,
    max_frames: Optional[int] = None,
    image_ext: str = "jpg",
    jpeg_quality: int = 95,
) -> int:
    """
    按固定采样率从视频中导出帧（默认每秒一帧）。

    Args:
        input_path: 输入视频路径
        output_dir: 帧输出目录
        rate_fps: 导出帧率（帧/秒），默认 1.0
        start_time_s: 开始时间（秒），默认 0
        end_time_s: 结束时间（秒），默认到视频结束
        max_frames: 最多导出帧数，None 表示不限制
        image_ext: 导出图片格式，'jpg' 或 'png'
        jpeg_quality: JPEG 质量（1-100），仅在 jpg 时生效

    Returns:
        实际保存的帧数
    """
    if rate_fps <= 0:
        raise ValueError("rate_fps 必须大于 0")

    image_ext = image_ext.lower()
    if image_ext not in {"jpg", "jpeg", "png"}:
        raise ValueError("image_ext 仅支持 'jpg'/'jpeg'/'png'")

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {input_path}")

    # 定位到起始时间
    if start_time_s > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time_s * 1000.0)

    interval_ms = 1000.0 / rate_fps
    next_save_ms = start_time_s * 1000.0
    end_time_ms = end_time_s * 1000.0 if end_time_s is not None else None

    saved = 0
    frame_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        current_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

        # 结束时间判断
        if end_time_ms is not None and current_ms > end_time_ms:
            break

        if current_ms + 0.5 >= next_save_ms:
            # 命名与保存
            filename = f"frame_{saved:06d}.{image_ext if image_ext != 'jpeg' else 'jpg'}"
            save_path = os.path.join(output_dir, filename)

            if image_ext in {"jpg", "jpeg"}:
                cv2.imwrite(save_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
            else:
                cv2.imwrite(save_path, frame)

            saved += 1

            # 递增下一个保存时间，处理丢帧/跳帧的情况
            while next_save_ms <= current_ms + 0.5:
                next_save_ms += interval_ms

            # 最大帧数限制
            if max_frames is not None and saved >= max_frames:
                break

        frame_index += 1

    cap.release()
    return saved


def _default_output_dir_for(input_path: str) -> str:
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    default_dir = os.path.join("testvideo", f"frames_{base_name}")
    return default_dir


def main():
    parser = argparse.ArgumentParser(description="视频抽帧工具：默认每秒导出一帧")
    parser.add_argument(
        "-i",
        "--input",
        default=os.path.join("testvideo", "my_video-1.mp4"),
        help="输入视频路径，默认 testvideo/my_video-1.mp4",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="输出目录（默认 testvideo/frames_<视频名>）",
    )
    parser.add_argument(
        "-r",
        "--rate",
        type=float,
        default=1.0,
        help="导出帧率 (fps)，默认 1.0",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=0.0,
        help="开始时间（秒），默认 0",
    )
    parser.add_argument(
        "--end",
        type=float,
        default=None,
        help="结束时间（秒），默认到视频结束",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="最多导出帧数，默认不限制",
    )
    parser.add_argument(
        "--ext",
        choices=["jpg", "jpeg", "png"],
        default="jpg",
        help="导出图片格式，默认 jpg",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG 质量 (1-100)，仅对 jpg 有效，默认 95",
    )

    args = parser.parse_args()

    input_path = args.input
    output_dir = args.output or _default_output_dir_for(input_path)

    saved = extract_frames(
        input_path=input_path,
        output_dir=output_dir,
        rate_fps=args.rate,
        start_time_s=args.start,
        end_time_s=args.end,
        max_frames=args.max,
        image_ext=args.ext,
        jpeg_quality=args.quality,
    )

    print(f"✅ 完成：已保存 {saved} 帧 到 {output_dir}")


if __name__ == "__main__":
    main()


