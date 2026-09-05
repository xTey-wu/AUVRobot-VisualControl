#!/usr/bin/env python3
"""Run the packaged YOLO model on one image or an image directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = PROJECT_ROOT / "models" / "combine1014.pt"
DEFAULT_SOURCE = PROJECT_ROOT / "examples" / "images"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "detected_images"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def collect_images(source: Path) -> list[Path]:
    if source.is_file() and source.suffix.lower() in IMAGE_SUFFIXES:
        return [source]
    if source.is_dir():
        return sorted(
            path for path in source.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="AUV YOLO image detection demo")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    images = collect_images(args.source)
    if not images:
        raise SystemExit(f"No supported images found at: {args.source}")

    args.output.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model))
    results = model.predict(
        source=[str(path) for path in images],
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        verbose=False,
    )

    for source, result in zip(images, results):
        destination = args.output / source.name
        result.save(filename=str(destination))
        box_count = 0 if result.boxes is None else len(result.boxes)
        print(f"{source.name}: {box_count} detections -> {destination}")


if __name__ == "__main__":
    main()
