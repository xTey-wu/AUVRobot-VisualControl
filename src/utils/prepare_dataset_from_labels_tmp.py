import os
import argparse
import glob
import random
import shutil
from typing import List, Tuple


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_classes(source_dir: str) -> List[str]:
    classes_file = os.path.join(source_dir, "classes.txt")
    names: List[str] = []
    if os.path.exists(classes_file):
        with open(classes_file, "r", encoding="utf-8") as f:
            for line in f:
                name = line.strip()
                if name:
                    names.append(name)
    else:
        raise FileNotFoundError(f"未找到 classes.txt: {classes_file}")
    return names


def list_images(source_dir: str) -> List[str]:
    exts = ["*.jpg", "*.jpeg", "*.png"]
    files: List[str] = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(source_dir, ext)))
    return sorted(files)


def unique_image_label_paths(dst_images: str, dst_labels: str, base: str, img_ext: str) -> Tuple[str, str]:
    candidate_base = base
    idx = 1
    while True:
        img_candidate = os.path.join(dst_images, candidate_base + img_ext)
        label_candidate = os.path.join(dst_labels, candidate_base + ".txt")
        if not os.path.exists(img_candidate) and not os.path.exists(label_candidate):
            return img_candidate, label_candidate
        candidate_base = f"{base}_{idx}"
        idx += 1


def write_yaml(dst_yaml: str, root_abs: str, names: List[str]) -> None:
    with open(dst_yaml, "w", encoding="utf-8") as f:
        f.write("names:\n")
        for name in names:
            f.write(f"- {name}\n")
        f.write(f"nc: {len(names)}\n")
        f.write(f"path: {root_abs}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n")


def main():
    parser = argparse.ArgumentParser(description="将 labels_tmp 目录整理为 YOLO 数据集结构")
    parser.add_argument("--source", default="labels_tmp", help="源目录，含图片/同名txt与classes.txt")
    parser.add_argument("--out_root", default="dataset_from_labels_tmp", help="输出数据集根目录")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="验证集比例，默认0.2")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    random.seed(args.seed)

    source = args.source
    out_root = args.out_root

    # 读类别
    names = read_classes(source)

    # 目标目录结构
    train_images = os.path.join(out_root, "train", "images")
    train_labels = os.path.join(out_root, "train", "labels")
    val_images = os.path.join(out_root, "val", "images")
    val_labels = os.path.join(out_root, "val", "labels")
    for d in [train_images, train_labels, val_images, val_labels]:
        ensure_dir(d)

    # 收集图片
    imgs = list_images(source)
    random.shuffle(imgs)
    n_total = len(imgs)
    n_val = max(1, int(n_total * args.val_ratio)) if n_total > 0 else 0
    val_set = set(imgs[:n_val])

    c_train = 0
    c_val = 0

    for img_path in imgs:
        base = os.path.splitext(os.path.basename(img_path))[0]
        img_ext = os.path.splitext(img_path)[1]
        label_src = os.path.join(source, f"{base}.txt")

        # 读取标签（若不存在则视为空标签）
        label_lines: List[str] = []
        if os.path.exists(label_src):
            with open(label_src, "r", encoding="utf-8") as f:
                label_lines = [ln.rstrip("\n") for ln in f]

        # 决定目标集合并生成唯一文件名
        if img_path in val_set:
            dst_img, dst_label = unique_image_label_paths(val_images, val_labels, base, img_ext)
            c_val += 1
        else:
            dst_img, dst_label = unique_image_label_paths(train_images, train_labels, base, img_ext)
            c_train += 1

        # 复制图片
        shutil.copy2(img_path, dst_img)
        # 写入标签（即便空也创建，以保证一一对应）
        with open(dst_label, "w", encoding="utf-8") as f:
            if label_lines:
                f.write("\n".join(label_lines).strip() + "\n")

    # 写 dataset.yaml
    root_abs = os.path.abspath(out_root)
    write_yaml(os.path.join(out_root, "dataset.yaml"), root_abs, names)

    print("✅ 整理完成：")
    print(f"  - total_images: {n_total}")
    print(f"  - train_images: {c_train}")
    print(f"  - val_images: {c_val}")
    print(f"  - out_root: {root_abs}")


if __name__ == "__main__":
    main()


