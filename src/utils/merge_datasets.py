import os
import shutil
import argparse
import glob
import random
from typing import Dict, List, Tuple


"""
将 dataset1/ 与 dataset_new (1)/ 合并为统一的 YOLO 检测数据集，支持：
1) 动态合并类别（以 dataset1/dataset.yaml 的 names 为基准，追加新类别）
2) 对 dataset_new (1) 的标签进行基于类别名的重映射
3) 统一随机划分 train/val（默认 8:2）
4) 保证重名文件不会覆盖，图片与标签严格一一对应
"""


ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _list_image_files(folder: str) -> List[str]:
    paths = []
    for ext in ALLOWED_IMAGE_EXTS:
        paths.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))
    # 去重并排序
    return sorted(list(set(paths)))


def _parse_names_from_dataset_yaml(yaml_path: str) -> List[str]:
    """简易解析 dataset.yaml 中的 names 列表（避免额外依赖 pyyaml）。"""
    if not os.path.isfile(yaml_path):
        return []
    names: List[str] = []
    with open(yaml_path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line == "names:" or line.startswith("names:"):
            i += 1
            while i < len(lines):
                l = lines[i]
                s = l.strip()
                if s.startswith("-"):
                    # 形如 "- class_name"
                    part = s[1:].strip()
                    if part:
                        names.append(part)
                    i += 1
                    continue
                break
            break
        i += 1
    return names


def _parse_names_from_classes_txt(txt_path: str) -> List[str]:
    if not os.path.isfile(txt_path):
        return []
    names: List[str] = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            n = line.strip()
            if n:
                names.append(n)
    return names


def remap_circle_label_line(line: str, mapping: Dict[int, int]) -> str:
    parts = line.strip().split()
    if not parts:
        return ""
    try:
        cid = int(parts[0])
    except ValueError:
        return ""
    if cid not in mapping:
        return ""
    parts[0] = str(mapping[cid])
    return " ".join(parts)


def write_yaml(dst_yaml: str, merged_root_abs: str, names: List[str]) -> None:
    with open(dst_yaml, "w", encoding="utf-8") as f:
        f.write("names:\n")
        for name in names:
            f.write(f"- {name}\n")
        f.write(f"nc: {len(names)}\n")
        f.write(f"path: {merged_root_abs}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n")


def unique_image_label_paths(dst_images: str, dst_labels: str, base: str, img_ext: str) -> Tuple[str, str]:
    """为图像和标签生成唯一且配套的目标路径，避免与现有文件同名冲突。
    返回 (dst_img_path, dst_label_path)。
    """
    candidate_base = base
    idx = 1
    while True:
        img_candidate = os.path.join(dst_images, candidate_base + img_ext)
        label_candidate = os.path.join(dst_labels, candidate_base + ".txt")
        if not os.path.exists(img_candidate) and not os.path.exists(label_candidate):
            return img_candidate, label_candidate
        candidate_base = f"{base}_{idx}"
        idx += 1


def _copy_with_optional_remap(img_paths: List[str], src_labels_dir: str, dst_images: str, dst_labels: str, label_id_mapping: Dict[int, int] | None) -> Tuple[int, int]:
    ensure_dir(dst_images)
    ensure_dir(dst_labels)
    copied_imgs = 0
    copied_labels = 0
    for img_path in img_paths:
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_src = os.path.join(src_labels_dir, f"{base}.txt")
        ext = os.path.splitext(img_path)[1]
        dst_img, dst_label = unique_image_label_paths(dst_images, dst_labels, base, ext)
        # 复制图片
        shutil.copy2(img_path, dst_img)
        copied_imgs += 1
        # 写入/复制标签
        if os.path.exists(label_src):
            if label_id_mapping is None:
                # 直接复制
                shutil.copy2(label_src, dst_label)
            else:
                # 读取并重映射
                lines_out: List[str] = []
                with open(label_src, "r", encoding="utf-8") as f:
                    for line in f:
                        mapped = remap_circle_label_line(line, label_id_mapping)
                        if mapped:
                            lines_out.append(mapped)
                with open(dst_label, "w", encoding="utf-8") as f:
                    if lines_out:
                        f.write("\n".join(lines_out).strip() + "\n")
                    else:
                        # 保留空文件以保证一一对应
                        pass
        else:
            with open(dst_label, "w", encoding="utf-8") as f:
                pass
        copied_labels += 1
    return copied_imgs, copied_labels


def _copy_to_combined_with_suffix(img_paths: List[str], src_labels_dir: str, combined_images: str, combined_labels: str, fixed_suffix: str, label_id_mapping: Dict[int, int] | None) -> Tuple[int, int]:
    """将一组图片复制到 combined/ 并统一在基名后追加固定后缀，然后写入/重映射标签。
    例如 base.jpg -> base_1.jpg 或 base_2.jpg
    """
    ensure_dir(combined_images)
    ensure_dir(combined_labels)
    copied_imgs = 0
    copied_labels = 0
    for img_path in img_paths:
        base = os.path.splitext(os.path.basename(img_path))[0]
        base_with_suffix = f"{base}{fixed_suffix}"
        label_src = os.path.join(src_labels_dir, f"{base}.txt")
        ext = os.path.splitext(img_path)[1]
        dst_img = os.path.join(combined_images, base_with_suffix + ext)
        dst_label = os.path.join(combined_labels, base_with_suffix + ".txt")
        # 若目标已存在，尽量再附加计数避免覆盖
        if os.path.exists(dst_img) or os.path.exists(dst_label):
            idx = 1
            while True:
                candidate_base = f"{base_with_suffix}_{idx}"
                dst_img = os.path.join(combined_images, candidate_base + ext)
                dst_label = os.path.join(combined_labels, candidate_base + ".txt")
                if not os.path.exists(dst_img) and not os.path.exists(dst_label):
                    break
                idx += 1
        shutil.copy2(img_path, dst_img)
        copied_imgs += 1
        if os.path.exists(label_src):
            if label_id_mapping is None:
                shutil.copy2(label_src, dst_label)
            else:
                lines_out: List[str] = []
                with open(label_src, "r", encoding="utf-8") as f:
                    for line in f:
                        mapped = remap_circle_label_line(line, label_id_mapping)
                        if mapped:
                            lines_out.append(mapped)
                with open(dst_label, "w", encoding="utf-8") as f:
                    if lines_out:
                        f.write("\n".join(lines_out).strip() + "\n")
                    else:
                        pass
        else:
            with open(dst_label, "w", encoding="utf-8") as f:
                pass
        copied_labels += 1
    return copied_imgs, copied_labels


def main():
    parser = argparse.ArgumentParser(description="合并 dataset1/ 与 dataset_new (1)/ 为统一 YOLO 数据集")
    parser.add_argument("--ds1_root", default="dataset1", help="数据集1根目录，需包含 images/ 与 labels/ 以及 dataset.yaml")
    parser.add_argument("--ds2_root", default="dataset_new (1)", help="数据集2根目录，需包含 images/ 与 labels/ 以及 classes.txt 或 dataset.yaml")
    parser.add_argument("--out_root", default="dataset_merged", help="合并后输出目录")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="验证集比例，默认0.2")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    random.seed(args.seed)

    ds1_root = args.ds1_root
    ds2_root = args.ds2_root
    out_root = args.out_root

    # 目标目录结构
    train_images = os.path.join(out_root, "train", "images")
    train_labels = os.path.join(out_root, "train", "labels")
    val_images = os.path.join(out_root, "val", "images")
    val_labels = os.path.join(out_root, "val", "labels")
    for d in [train_images, train_labels, val_images, val_labels]:
        ensure_dir(d)

    # 1) 读取类别并建立合并后的 names
    ds1_yaml = os.path.join(ds1_root, "dataset.yaml")
    ds1_names = _parse_names_from_dataset_yaml(ds1_yaml)
    ds2_classes_txt = os.path.join(ds2_root, "classes.txt")
    ds2_yaml = os.path.join(ds2_root, "dataset.yaml")
    ds2_names = _parse_names_from_classes_txt(ds2_classes_txt)
    if not ds2_names:
        ds2_names = _parse_names_from_dataset_yaml(ds2_yaml)

    merged_names: List[str] = list(ds1_names)
    for n in ds2_names:
        if n not in merged_names:
            merged_names.append(n)

    # 构建 ds2 的类别重映射：local id -> merged id
    ds2_id_to_merged: Dict[int, int] = {}
    for local_id, name in enumerate(ds2_names):
        if name in merged_names:
            ds2_id_to_merged[local_id] = merged_names.index(name)

    # 2) 处理 ds1 平铺数据集 -> 复制到 combined 并统一追加后缀 _1
    ds1_images_dir = os.path.join(ds1_root, "images")
    ds1_labels_dir = os.path.join(ds1_root, "labels")
    ds1_imgs = _list_image_files(ds1_images_dir)
    combined_images = os.path.join(out_root, "combined", "images")
    combined_labels = os.path.join(out_root, "combined", "labels")
    ds1_copied = _copy_to_combined_with_suffix(ds1_imgs, ds1_labels_dir, combined_images, combined_labels, "_1", None)

    # 3) 处理 ds2 平铺数据集 -> 重映射后复制到 combined 并统一追加后缀 _2
    ds2_images_dir = os.path.join(ds2_root, "images")
    ds2_labels_dir = os.path.join(ds2_root, "labels")
    ds2_imgs = _list_image_files(ds2_images_dir)
    ds2_copied = _copy_to_combined_with_suffix(ds2_imgs, ds2_labels_dir, combined_images, combined_labels, "_2", ds2_id_to_merged)

    # 4) 在 combined 基础上统一划分 train/val，并移动文件
    combined_img_paths = _list_image_files(combined_images)
    random.shuffle(combined_img_paths)
    n_total = len(combined_img_paths)
    n_val = max(1, int(n_total * args.val_ratio)) if n_total > 0 else 0
    val_set = set(combined_img_paths[:n_val])

    moved_train = 0
    moved_val = 0
    for img_path in combined_img_paths:
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(combined_labels, base + ".txt")
        ext = os.path.splitext(img_path)[1]
        if img_path in val_set:
            dst_img = os.path.join(val_images, base + ext)
            dst_label = os.path.join(val_labels, base + ".txt")
            moved_val += 1
        else:
            dst_img = os.path.join(train_images, base + ext)
            dst_label = os.path.join(train_labels, base + ".txt")
            moved_train += 1
        ensure_dir(os.path.dirname(dst_img))
        ensure_dir(os.path.dirname(dst_label))
        shutil.move(img_path, dst_img)
        if os.path.exists(label_path):
            shutil.move(label_path, dst_label)
        else:
            with open(dst_label, "w", encoding="utf-8") as f:
                pass

    # 5) 写 dataset.yaml
    merged_root_abs = os.path.abspath(out_root)
    write_yaml(os.path.join(out_root, "dataset.yaml"), merged_root_abs, merged_names)

    # 统计
    def count_files(folder: str, pattern: str) -> int:
        return len(glob.glob(os.path.join(folder, pattern)))

    stats = {
        "train_images": count_files(train_images, "*.*"),
        "train_labels": count_files(train_labels, "*.txt"),
        "val_images": count_files(val_images, "*.*"),
        "val_labels": count_files(val_labels, "*.txt"),
        "combined_total": n_total,
        "moved_train": moved_train,
        "moved_val": moved_val,
        "ds1_copied": ds1_copied,
        "ds2_copied": ds2_copied,
        "merged_root": merged_root_abs,
    }

    print("✅ 合并完成：")
    for k, v in stats.items():
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()


