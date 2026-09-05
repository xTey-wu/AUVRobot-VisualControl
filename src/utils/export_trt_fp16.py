#!/usr/bin/env python3
"""
将已经训练好的 Ultralytics YOLO *.pt 模型导出为 TensorRT FP16 引擎文件。

支持两条导出路径：
1) Ultralytics 原生导出（Python API），直接生成 TensorRT 引擎并使用半精度（FP16）
2) 先导出 ONNX，再使用 trtexec（命令行）构建 FP16 引擎

使用示例：
  python src/utils/export_trt_fp16.py --weights models/ObjectRec.pt --imgsz 640 --device 0
  python src/utils/export_trt_fp16.py --weights models/ObjectRec.pt --imgsz 640 --use-trtexec --workspace 4096

说明：
- 需要安装 TensorRT（Python 包或提供 trtexec 的 SDK）
- RTX 4060 Laptop 支持 FP16 Tensor Core 加速
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


def log_info(message: str) -> None:
    print(f"[INFO] {message}")


def log_warn(message: str) -> None:
    print(f"[WARN] {message}")


def log_error(message: str) -> None:
    print(f"[ERROR] {message}")


def export_engine_with_ultralytics(weights: Path, imgsz: int, device: int, dynamic: bool) -> Path:
    """
    使用 Ultralytics 原生导出器导出 TensorRT 引擎（需要安装 Python 版 tensorrt）。
    返回生成的 *.engine 文件路径。
    """
    from ultralytics import YOLO  # 放在函数内部以便在 --use-trtexec 路径下不强依赖

    log_info("使用 Ultralytics 原生 TensorRT 导出（半精度 FP16）...")
    model = YOLO(str(weights))
    out = model.export(
        format="engine",
        half=True,  # 使用 FP16 半精度
        device=device,
        imgsz=imgsz,
        dynamic=dynamic,
    )
    # Ultralytics 返回输出文件路径字符串
    engine_path = Path(out) if isinstance(out, str) else Path(str(out))
    if not engine_path.exists():
        raise FileNotFoundError(f"导出后未找到引擎文件: {engine_path}")
    return engine_path


def export_onnx_with_ultralytics(weights: Path, imgsz: int, dynamic: bool) -> Path:
    from ultralytics import YOLO

    log_info("使用 Ultralytics 导出 ONNX...")
    model = YOLO(str(weights))
    out = model.export(
        format="onnx",
        imgsz=imgsz,
        dynamic=dynamic,
    )
    onnx_path = Path(out) if isinstance(out, str) else Path(str(out))
    if not onnx_path.exists():
        raise FileNotFoundError(f"导出后未找到 ONNX 文件: {onnx_path}")
    return onnx_path


def build_engine_with_trtexec(onnx_path: Path, engine_path: Path, imgsz: int, batch: int, workspace: int, dynamic: bool) -> Path:
    """
    通过 trtexec 构建 TensorRT 引擎。
    - imgsz：方形输入尺寸（如 640）
    - batch：用于形状配置的批大小
    - workspace：构建器可用内存（MB）
    - dynamic：若为 True，提供最小/最优/最大形状；否则使用固定 --shapes
    """
    trtexec = shutil.which("trtexec")
    if trtexec is None:
        raise RuntimeError("未在 PATH 中找到 trtexec。请安装 TensorRT 并确保 trtexec 可用。")

    input_name = "images"
    c = 3
    h = imgsz
    w = imgsz

    if dynamic:
        min_shapes = f"{input_name}:{max(1, batch//4) if batch > 1 else 1}x{c}x{h//2}x{w//2}"
        opt_shapes = f"{input_name}:{batch}x{c}x{h}x{w}"
        max_shapes = f"{input_name}:{max(batch*2, batch+1)}x{c}x{h*2}x{w*2}"
        shape_args = [
            f"--minShapes={min_shapes}",
            f"--optShapes={opt_shapes}",
            f"--maxShapes={max_shapes}",
        ]
    else:
        shapes = f"{input_name}:{batch}x{c}x{h}x{w}"
        shape_args = [f"--shapes={shapes}"]

    cmd = [
        trtexec,
        f"--onnx={str(onnx_path)}",
        f"--saveEngine={str(engine_path)}",
        "--fp16",
        f"--workspace={workspace}",
        "--noDataTransfers",  # 略微减少构建时的输出噪声，不影响最终结果
        *shape_args,
    ]

    log_info("使用 trtexec 构建 FP16 引擎...")
    log_info(" ".join(cmd))
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError("trtexec 构建引擎失败，详见上述日志。")
    if not engine_path.exists():
        raise FileNotFoundError(f"未生成引擎文件: {engine_path}")
    return engine_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 YOLO *.pt 导出为 TensorRT FP16 引擎")
    parser.add_argument("--weights", type=str, default="models/ObjectRec.pt", help=".pt 权重文件路径")
    parser.add_argument("--imgsz", type=int, default=640, help="推理尺寸（方形）")
    parser.add_argument("--device", type=int, default=0, help="CUDA 设备序号")
    parser.add_argument("--dynamic", action="store_true", help="启用动态形状导出")
    parser.add_argument("--batch", type=int, default=1, help="用于形状配置的批大小（trtexec）")
    parser.add_argument("--workspace", type=int, default=4096, help="构建器可用内存 MB（trtexec）")
    parser.add_argument("--use-trtexec", action="store_true", help="强制走 ONNX→trtexec 构建，跳过 Python TensorRT API")
    parser.add_argument("--onnx", type=str, default="", help="复用已有 ONNX 路径（跳过导出）")
    parser.add_argument("--force", action="store_true", help="若引擎已存在则覆盖")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    weights = Path(args.weights).resolve()
    if not weights.exists():
        log_error(f"未找到权重文件: {weights}")
        sys.exit(1)

    engine_path = weights.with_suffix(".engine")
    if engine_path.exists() and not args.force:
        log_warn(f"引擎已存在: {engine_path}。使用 --force 可覆盖。")
        print(str(engine_path))
        return

    try:
        if not args.use_trtexec:
            engine = export_engine_with_ultralytics(weights=weights, imgsz=args.imgsz, device=args.device, dynamic=args.dynamic)
            log_info(f"已导出 TensorRT FP16 引擎: {engine}")
            print(str(engine))
            return
        else:
            log_info("使用 ONNX → trtexec 构建路径...")
            onnx_path: Optional[Path] = Path(args.onnx).resolve() if args.onnx else None
            if onnx_path is None or not onnx_path.exists():
                onnx_path = export_onnx_with_ultralytics(weights=weights, imgsz=args.imgsz, dynamic=args.dynamic)
            engine = build_engine_with_trtexec(
                onnx_path=onnx_path,
                engine_path=engine_path,
                imgsz=args.imgsz,
                batch=args.batch,
                workspace=args.workspace,
                dynamic=args.dynamic,
            )
            log_info(f"已导出 TensorRT FP16 引擎: {engine}")
            print(str(engine))
            return
    except Exception as exc:
        if not args.use_trtexec:
            log_warn(f"原生导出失败: {exc}。回退到 ONNX → trtexec...")
            try:
                onnx_path = export_onnx_with_ultralytics(weights=weights, imgsz=args.imgsz, dynamic=args.dynamic)
                engine = build_engine_with_trtexec(
                    onnx_path=onnx_path,
                    engine_path=engine_path,
                    imgsz=args.imgsz,
                    batch=args.batch,
                    workspace=args.workspace,
                    dynamic=args.dynamic,
                )
                log_info(f"已导出 TensorRT FP16 引擎: {engine}")
                print(str(engine))
                return
            except Exception as exc2:
                log_error(f"回退的 ONNX → trtexec 也失败: {exc2}")
                sys.exit(2)
        else:
            log_error(f"ONNX → trtexec 路径失败: {exc}")
            sys.exit(2)


if __name__ == "__main__":
    main()
