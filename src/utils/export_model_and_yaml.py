import os
import argparse
import yaml
from pathlib import Path
from ultralytics import YOLO


def read_dataset_names(dataset_yaml_path: str) -> list:
    with open(dataset_yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    names = data.get('names')
    if isinstance(names, dict):
        # 如果names是{id: name}的dict，转换为list
        max_k = max(int(k) for k in names.keys())
        names_list = [None] * (max_k + 1)
        for k, v in names.items():
            names_list[int(k)] = v
        return names_list
    return list(names)


def write_deploy_yaml(
    out_yaml_path: str,
    *,
    yaml_type: str,
    name: str | None,
    display_name: str | None,
    model_path: str,
    names: list,
    classes_format: str = 'dict',
    iou_threshold: float | None = None,
    conf_threshold: float | None = None,
) -> None:
    from datetime import datetime
    doc = {
        'type': yaml_type,
        'name': name if name else f'yolo11s-{datetime.now().strftime("r%Y%m%d")}',
        'provider': 'Ultralytics',
        'display_name': display_name if display_name else 'YOLO11s',
        'model_path': model_path,
    }
    if classes_format == 'list':
        doc['classes'] = names
    else:
        doc['classes'] = {i: n for i, n in enumerate(names)}
    if iou_threshold is not None:
        doc['iou_threshold'] = float(iou_threshold)
    if conf_threshold is not None:
        doc['conf_threshold'] = float(conf_threshold)

    with open(out_yaml_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True)


def read_classes_file(path: str) -> list:
    names: list[str] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            name = line.strip()
            if name:
                names.append(name)
    return names


COCO80: list[str] = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light',
    'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow',
    'elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
    'skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle',
    'wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange',
    'broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed',
    'dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven',
    'toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
]


def main():
    parser = argparse.ArgumentParser(description='导出YOLO模型并生成部署YAML')
    parser.add_argument('--run_dir', type=str, default='runs/train/yolov11s_new', help='训练运行目录')
    parser.add_argument('--weights', type=str, default='weights/best.pt', help='权重相对路径（相对于run_dir）')
    parser.add_argument('--dataset_yaml', type=str, default='dataset_merged/dataset.yaml', help='数据集配置，读取类别名')
    parser.add_argument('--imgsz', type=int, default=640, help='导出尺寸')
    parser.add_argument('--opset', type=int, default=12, help='ONNX opset')
    parser.add_argument('--simplify', action='store_true', help='简化ONNX')
    parser.add_argument('--dynamic', action='store_true', help='动态尺寸ONNX')
    parser.add_argument('--export_engine', action='store_true', help='同时导出TensorRT engine')
    parser.add_argument('--half', action='store_true', help='engine使用FP16')
    parser.add_argument('--out_yaml_name', type=str, default='deploy_det.yaml', help='输出部署YAML文件名')
    parser.add_argument('--skip_export', action='store_true', help='仅生成YAML，不导出模型')
    parser.add_argument('--yaml_type', type=str, default='yolo11_det', help='YAML类型，如 yolo11 / yolo11_det / yolo11_cls')
    parser.add_argument('--name', type=str, default=None, help='YAML中的 name 字段')
    parser.add_argument('--display_name', type=str, default=None, help='YAML中的 display_name 字段')
    parser.add_argument('--model_path_override', type=str, default=None, help='覆盖YAML中的 model_path（URL或绝对路径）')
    parser.add_argument('--iou_threshold', type=float, default=None, help='检测阈值（可选）')
    parser.add_argument('--conf_threshold', type=float, default=None, help='置信度阈值（可选）')
    parser.add_argument('--classes_file', type=str, default=None, help='从文件读取类别（每行一个）')
    parser.add_argument('--classes_format', type=str, default='dict', choices=['dict','list'], help='YAML中 classes 写法')
    parser.add_argument('--use_coco80', action='store_true', help='使用COCO80类别并按列表格式输出')
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    weights_path = run_dir / args.weights
    if not weights_path.exists():
        raise FileNotFoundError(f'未找到权重: {weights_path}')

    model = None
    onnx_path = run_dir / 'weights' / 'best.onnx'
    if not args.skip_export:
        print(f'加载模型: {weights_path}')
        model = YOLO(str(weights_path))
        # 导出ONNX
        print('开始导出ONNX...')
        onnx_results = model.export(
            format='onnx',
            imgsz=args.imgsz,
            opset=args.opset,
            simplify=args.simplify,
            dynamic=args.dynamic,
        )
        if not onnx_path.exists() and isinstance(onnx_results, (list, tuple)):
            # 兼容不同返回
            for p in onnx_results:
                if str(p).endswith('.onnx'):
                    onnx_path = Path(p)
                    break
        print(f'ONNX 导出完成: {onnx_path}')

        # 可选导出TensorRT
        if args.export_engine:
            print('开始导出 TensorRT engine...')
            engine_results = model.export(
                format='engine',
                half=args.half,
                imgsz=args.imgsz,
                dynamic=args.dynamic,
            )
            engine_path = run_dir / 'weights' / 'best.engine'
            if not engine_path.exists() and isinstance(engine_results, (list, tuple)):
                for p in engine_results:
                    if str(p).endswith('.engine'):
                        engine_path = Path(p)
                        break
            print(f'Engine 导出完成: {engine_path}')

    # 读取类别名
    if args.use_coco80:
        names = COCO80
        args.classes_format = 'list'
        if args.yaml_type == 'yolo11_det':
            args.yaml_type = 'yolo11'
    elif args.classes_file:
        names = read_classes_file(args.classes_file)
    else:
        names = read_dataset_names(args.dataset_yaml)

    # 写部署YAML（指向ONNX；如需要engine可改路径）
    model_out_path = args.model_path_override if args.model_path_override else str(onnx_path)
    out_yaml_path = run_dir / args.out_yaml_name
    write_deploy_yaml(
        str(out_yaml_path),
        yaml_type=args.yaml_type,
        name=args.name,
        display_name=args.display_name,
        model_path=model_out_path,
        names=names,
        classes_format=args.classes_format,
        iou_threshold=args.iou_threshold,
        conf_threshold=args.conf_threshold,
    )
    print(f'部署YAML已生成: {out_yaml_path}')


if __name__ == '__main__':
    main()
