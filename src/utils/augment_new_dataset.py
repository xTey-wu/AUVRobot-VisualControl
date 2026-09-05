import os
import glob
import random
import argparse
from typing import List, Tuple

import cv2
import numpy as np


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_yolo_labels(label_path: str) -> List[Tuple[int, float, float, float, float]]:
    labels = []
    if not os.path.exists(label_path):
        return labels
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                cls_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])
                labels.append((cls_id, cx, cy, w, h))
            except Exception:
                continue
    return labels


def write_yolo_labels(label_path: str, labels: List[Tuple[int, float, float, float, float]]) -> None:
    with open(label_path, "w", encoding="utf-8") as f:
        for cls_id, cx, cy, w, h in labels:
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def hflip_image_and_labels(img: np.ndarray, labels: List[Tuple[int, float, float, float, float]]) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]:
    flipped = cv2.flip(img, 1)
    flipped_labels = []
    for cls_id, cx, cy, w, h in labels:
        flipped_cx = 1.0 - cx
        flipped_labels.append((cls_id, flipped_cx, cy, w, h))
    return flipped, flipped_labels


def adjust_brightness_contrast(img: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    # new = alpha*img + beta
    out = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return out


def adjust_hsv(img: np.ndarray, dh: int, ds: int, dv: int) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    h = (h.astype(np.int32) + dh) % 180
    s = np.clip(s.astype(np.int32) + ds, 0, 255)
    v = np.clip(v.astype(np.int32) + dv, 0, 255)
    hsv_aug = cv2.merge((h.astype(np.uint8), s.astype(np.uint8), v.astype(np.uint8)))
    bgr = cv2.cvtColor(hsv_aug, cv2.COLOR_HSV2BGR)
    return bgr


def gaussian_blur(img: np.ndarray, ksize: int) -> np.ndarray:
    ksize = max(1, ksize)
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def add_gaussian_noise(img: np.ndarray, sigma: float) -> np.ndarray:
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def jpeg_compress(img: np.ndarray, quality: int) -> np.ndarray:
    quality = int(np.clip(quality, 10, 100))
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', img, encode_param)
    if not result:
        return img
    dec = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
    return dec if dec is not None else img


def apply_augmentations(
    img: np.ndarray,
    labels: List[Tuple[int, float, float, float, float]],
    img_size: Tuple[int, int],
    cfg,
) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float]], str]:
    h, w = img_size
    aug_img = img.copy()
    aug_labels = list(labels)
    aug_suffix = []

    # 1) 随机水平翻转（需要同步bbox）
    if random.random() < cfg.flip_p:
        aug_img, aug_labels = hflip_image_and_labels(aug_img, aug_labels)
        aug_suffix.append("hf")

    # 2) 亮度/对比度
    if random.random() < cfg.brightness_contrast_p:
        alpha = random.uniform(0.8, 1.3)  # 对比度
        beta = random.uniform(-25, 25)    # 亮度
        aug_img = adjust_brightness_contrast(aug_img, alpha, beta)
        aug_suffix.append("bc")

    # 3) HSV 扰动（色偏）
    if random.random() < cfg.hsv_p:
        dh = random.randint(-5, 5)
        ds = random.randint(-30, 30)
        dv = random.randint(-30, 30)
        aug_img = adjust_hsv(aug_img, dh, ds, dv)
        aug_suffix.append("hsv")

    # 4) 模糊
    if random.random() < cfg.blur_p:
        k = random.choice([3, 5, 7])
        aug_img = gaussian_blur(aug_img, k)
        aug_suffix.append("bl")

    # 5) 高斯噪声
    if random.random() < cfg.noise_p:
        sigma = random.uniform(3, 8)
        aug_img = add_gaussian_noise(aug_img, sigma)
        aug_suffix.append("gn")

    # 6) JPEG 压缩失真
    if random.random() < cfg.jpeg_p:
        q = random.randint(40, 90)
        aug_img = jpeg_compress(aug_img, q)
        aug_suffix.append(f"jq{q}")

    suffix = "_" + "_".join(aug_suffix) if aug_suffix else ""
    return aug_img, aug_labels, suffix


def process_dataset(images_dir: str, labels_dir: str, out_images: str, out_labels: str, copies: int, seed: int) -> dict:
    random.seed(seed)
    ensure_dir(out_images)
    ensure_dir(out_labels)

    img_paths = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
    if not img_paths:
        img_paths = sorted(glob.glob(os.path.join(images_dir, "*.png")))

    class Cfg:
        flip_p = 0.5
        brightness_contrast_p = 0.7
        hsv_p = 0.5
        blur_p = 0.3
        noise_p = 0.2
        jpeg_p = 0.5

    total_imgs = 0
    total_labels = 0
    total_aug = 0

    for img_path in img_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(labels_dir, f"{base}.txt")
        labels = read_yolo_labels(label_path)

        total_imgs += 1
        total_labels += 1  # 我们总会输出对应标签（可为空）

        for i in range(copies):
            aug_img, aug_labels, suffix = apply_augmentations(img, labels, (h, w), Cfg)
            out_name = f"{base}_aug{i}{suffix}"
            out_img_path = os.path.join(out_images, out_name + ".jpg")
            out_label_path = os.path.join(out_labels, out_name + ".txt")

            cv2.imwrite(out_img_path, aug_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            write_yolo_labels(out_label_path, aug_labels)
            total_aug += 1

    return {
        "src_images": images_dir,
        "src_labels": labels_dir,
        "out_images": out_images,
        "out_labels": out_labels,
        "num_source_images": total_imgs,
        "num_augmented_pairs": total_aug,
        "copies_per_image": copies,
    }


def main():
    parser = argparse.ArgumentParser(description="对新数据集(图片+YOLO标签)进行离线增强，仅做不会改变bbox的变换+水平翻转")
    parser.add_argument("--images_dir", default=os.path.join("testvideo", "frames_my_video-1"), help="原始图片目录")
    parser.add_argument("--labels_dir", default=os.path.join("testvideo", "circle"), help="原始标签目录")
    parser.add_argument("--out_images", default=os.path.join("testvideo", "frames_my_video-1_aug"), help="增强图片输出目录")
    parser.add_argument("--out_labels", default=os.path.join("testvideo", "circle_aug"), help="增强标签输出目录")
    parser.add_argument("--copies", type=int, default=2, help="每张图片生成的增强副本数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    stats = process_dataset(args.images_dir, args.labels_dir, args.out_images, args.out_labels, args.copies, args.seed)
    print("✅ 增强完成：")
    for k, v in stats.items():
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()


