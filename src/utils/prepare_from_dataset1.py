import os
import glob
import argparse
import random
import shutil
import yaml
from typing import List, Tuple


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_names_from_dataset_yaml(yaml_path: str) -> List[str]:
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    names = data.get('names', [])
    if isinstance(names, dict):
        # convert dict {id:name} to list
        max_k = max(int(k) for k in names.keys())
        arr = [None] * (max_k + 1)
        for k, v in names.items():
            arr[int(k)] = v
        return arr
    return list(names)


def list_images(folder: str) -> List[str]:
    exts = ['*.jpg', '*.jpeg', '*.png']
    res: List[str] = []
    for e in exts:
        res.extend(glob.glob(os.path.join(folder, e)))
    return sorted(res)


def unique_image_label_paths(dst_images: str, dst_labels: str, base: str, img_ext: str) -> Tuple[str, str]:
    candidate = base
    idx = 1
    while True:
        img_candidate = os.path.join(dst_images, candidate + img_ext)
        label_candidate = os.path.join(dst_labels, candidate + '.txt')
        if not os.path.exists(img_candidate) and not os.path.exists(label_candidate):
            return img_candidate, label_candidate
        candidate = f"{base}_{idx}"
        idx += 1


def write_dataset_yaml(out_yaml: str, root_abs: str, names: List[str]) -> None:
    with open(out_yaml, 'w', encoding='utf-8') as f:
        f.write('names:\n')
        for n in names:
            f.write(f'- {n}\n')
        f.write(f'nc: {len(names)}\n')
        f.write(f'path: {root_abs}\n')
        f.write('train: train/images\n')
        f.write('val: val/images\n')


def main():
    parser = argparse.ArgumentParser(description='从 dataset1 生成标准 YOLO 数据集结构')
    parser.add_argument('--source', default='dataset1', help='源目录，包含 images/ 与 labels/')
    parser.add_argument('--out_root', default='dataset1_prepared', help='输出根目录')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='验证集比例')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()

    random.seed(args.seed)

    src = args.source
    out_root = args.out_root

    src_images = os.path.join(src, 'images')
    src_labels = os.path.join(src, 'labels')
    src_yaml = os.path.join(src, 'dataset.yaml')

    if not os.path.isdir(src_images) or not os.path.isdir(src_labels):
        raise FileNotFoundError('源目录缺少 images/ 或 labels/ 子目录')

    # 读取类别
    names = read_names_from_dataset_yaml(src_yaml) if os.path.exists(src_yaml) else []

    # 目标目录
    train_images = os.path.join(out_root, 'train', 'images')
    train_labels = os.path.join(out_root, 'train', 'labels')
    val_images = os.path.join(out_root, 'val', 'images')
    val_labels = os.path.join(out_root, 'val', 'labels')
    for d in [train_images, train_labels, val_images, val_labels]:
        ensure_dir(d)

    # 列出图片并划分
    imgs = list_images(src_images)
    random.shuffle(imgs)
    n_total = len(imgs)
    n_val = max(1, int(n_total * args.val_ratio)) if n_total > 0 else 0
    val_set = set(imgs[:n_val])

    c_train = 0
    c_val = 0

    for img_path in imgs:
        base = os.path.splitext(os.path.basename(img_path))[0]
        img_ext = os.path.splitext(img_path)[1]
        label_src = os.path.join(src_labels, base + '.txt')

        # 读取标签，若无则创建空
        label_lines: List[str] = []
        if os.path.exists(label_src):
            with open(label_src, 'r', encoding='utf-8') as f:
                label_lines = [ln.rstrip('\n') for ln in f]

        # 目标集合 & 唯一命名
        if img_path in val_set:
            dst_img, dst_label = unique_image_label_paths(val_images, val_labels, base, img_ext)
            c_val += 1
        else:
            dst_img, dst_label = unique_image_label_paths(train_images, train_labels, base, img_ext)
            c_train += 1

        shutil.copy2(img_path, dst_img)
        with open(dst_label, 'w', encoding='utf-8') as f:
            if label_lines:
                f.write('\n'.join(label_lines).strip() + '\n')

    # 写 dataset.yaml
    root_abs = os.path.abspath(out_root)
    if not names:
        # 若源未提供，则从 labels 中最大id+1 推断为空占位名
        # 这里简单保留空列表，用户可后续覆盖
        pass
    write_dataset_yaml(os.path.join(out_root, 'dataset.yaml'), root_abs, names)

    print('✅ 生成完成:')
    print('  - total_images:', n_total)
    print('  - train_images:', c_train)
    print('  - val_images  :', c_val)
    print('  - out_root    :', root_abs)


if __name__ == '__main__':
    main()


