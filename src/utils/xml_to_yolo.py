#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XML标签转YOLO格式脚本
将XML格式的标注文件转换为YOLO v5可用的格式
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path

def parse_xml_to_yolo(xml_file, img_width, img_height, class_mapping=None):
    """
    解析XML文件并转换为YOLO格式
    
    Args:
        xml_file: XML文件路径
        img_width: 图片宽度
        img_height: 图片高度
    
    Returns:
        List of YOLO format strings
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    yolo_lines = []
    
    for obj in root.findall('object'):
        try:
            # 获取类别ID，尝试多种可能的标签名
            class_element = obj.find('n')
            if class_element is None:
                class_element = obj.find('name')
            if class_element is None:
                class_element = obj.find('class')
            
            if class_element is None or class_element.text is None:
                print(f"警告: 对象缺少类别信息，跳过该对象")
                continue
            
            raw_label = class_element.text.strip()
            if raw_label == "":
                print(f"警告: 类别名为空，跳过该对象")
                continue
            
            # 支持两种情况：
            # 1) XML里类别是数字（1-based），转为0-based
            # 2) XML里类别是字符串：
            #    - 若提供了映射则使用
            #    - 若未提供映射或映射为空，则在解析阶段动态新增映射
            if raw_label.isdigit():
                class_id = int(raw_label) - 1
            else:
                if class_mapping is None:
                    class_mapping = {}
                if raw_label not in class_mapping:
                    class_mapping[raw_label] = len(class_mapping)
                class_id = class_mapping[raw_label]
            
            # 获取边界框坐标
            bbox = obj.find('bndbox')
            if bbox is None:
                print(f"警告: 对象缺少边界框信息，跳过该对象")
                continue
            
            xmin_elem = bbox.find('xmin')
            ymin_elem = bbox.find('ymin')
            xmax_elem = bbox.find('xmax')
            ymax_elem = bbox.find('ymax')
            
            if any(elem is None or elem.text is None for elem in [xmin_elem, ymin_elem, xmax_elem, ymax_elem]):
                print(f"警告: 边界框坐标不完整，跳过该对象")
                continue
            
            xmin = float(xmin_elem.text)
            ymin = float(ymin_elem.text)
            box_width = float(xmax_elem.text)  # 这里看起来是宽度
            box_height = float(ymax_elem.text)  # 这里看起来是高度
            
            # 验证坐标合理性
            if box_width <= 0 or box_height <= 0:
                print(f"警告: 无效的边界框尺寸 (width={box_width}, height={box_height})，跳过该对象")
                continue
            
            # 计算中心点坐标
            center_x = xmin + box_width / 2
            center_y = ymin + box_height / 2
            
            # 归一化坐标（YOLO要求0-1之间）
            center_x_norm = center_x / img_width
            center_y_norm = center_y / img_height
            width_norm = box_width / img_width
            height_norm = box_height / img_height
            
            # 确保坐标在有效范围内
            center_x_norm = max(0, min(1, center_x_norm))
            center_y_norm = max(0, min(1, center_y_norm))
            width_norm = max(0, min(1, width_norm))
            height_norm = max(0, min(1, height_norm))
            
            # YOLO格式: class_id center_x center_y width height
            yolo_line = f"{class_id} {center_x_norm:.6f} {center_y_norm:.6f} {width_norm:.6f} {height_norm:.6f}"
            yolo_lines.append(yolo_line)
            
        except ValueError as e:
            print(f"警告: 数值转换错误: {e}，跳过该对象")
            continue
        except Exception as e:
            print(f"警告: 解析对象时出错: {e}，跳过该对象")
            continue
    
    return yolo_lines

def get_image_size_from_xml(xml_file):
    """
    从XML文件中获取图片尺寸
    
    Args:
        xml_file: XML文件路径
    
    Returns:
        Tuple of (width, height)
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    return width, height

def build_class_mapping(labels_dir):
    """
    扫描XML标签，构建 字符串类别名 -> 连续ID(0-based) 的映射
    仅对字符串类别生效；若XML中类别为数字，则仍按(值-1)处理，不参与该映射。
    """
    labels_path = Path(labels_dir)
    mapping = {}
    next_id = 0
    for xml_file in labels_path.glob("*.xml"):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for obj in root.findall('object'):
                class_element = obj.find('n') or obj.find('name') or obj.find('class')
                if class_element is None or class_element.text is None:
                    continue
                label = class_element.text.strip()
                if not label or label.isdigit():
                    # 数字类别不进入字符串映射，由解析阶段按(值-1)处理
                    continue
                if label not in mapping:
                    mapping[label] = next_id
                    next_id += 1
        except Exception as e:
            print(f"警告: 扫描 {xml_file.name} 构建类别映射时出错: {e}")
    return mapping

def convert_xml_to_yolo(images_dir, labels_dir, output_dir):
    """
    批量转换XML标签到YOLO格式
    
    Args:
        images_dir: 图片文件夹路径
        labels_dir: XML标签文件夹路径
        output_dir: YOLO格式标签输出文件夹路径
    """
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    output_path = Path(output_dir)
    
    # 创建输出目录
    output_path.mkdir(exist_ok=True)

    # 预扫描标签，生成字符串类别映射，并保存到classes.txt
    class_mapping = build_class_mapping(labels_dir)
    if class_mapping:
        classes_file = output_path / "classes.txt"
        with open(classes_file, 'w', encoding='utf-8') as f:
            # 按ID排序写入，以便行号=ID
            for name, idx in sorted(class_mapping.items(), key=lambda x: x[1]):
                f.write(name + "\n")
        print(f"已生成类别映射文件: {classes_file}")
    
    # 统计信息
    total_images = 0
    processed_images = 0
    
    # 遍历所有图片文件
    for img_file in images_path.glob("*.jpg"):
        total_images += 1
        
        # 构建对应的XML文件路径
        xml_file = labels_path / (img_file.stem + ".xml")
        
        # 检查是否存在对应的XML标签文件
        if not xml_file.exists():
            print(f"跳过图片 {img_file.name}：没有对应的标签文件")
            continue
        
        try:
            # 从XML文件获取图片尺寸
            img_width, img_height = get_image_size_from_xml(xml_file)
            
            # 转换XML到YOLO格式（传入类别映射以支持字符串类别）
            yolo_lines = parse_xml_to_yolo(xml_file, img_width, img_height, class_mapping)
            
            # 写入YOLO格式文件
            output_file = output_path / (img_file.stem + ".txt")
            with open(output_file, 'w', encoding='utf-8') as f:
                for line in yolo_lines:
                    f.write(line + '\n')
            
            processed_images += 1
            print(f"已处理: {img_file.name} -> {output_file.name} ({len(yolo_lines)} 个对象)")
            
        except Exception as e:
            print(f"处理文件 {img_file.name} 时出错: {str(e)}")
    
    print(f"\n转换完成！")
    print(f"总图片数: {total_images}")
    print(f"成功处理: {processed_images}")
    print(f"跳过图片: {total_images - processed_images}")

def main():
    """主函数"""
    # 设置路径
    images_dir = os.path.join("dataset_new", "images")
    labels_dir = os.path.join("dataset_new", "labels")
    output_dir = os.path.join("dataset_new", "yolo_labels")
    
    print("开始XML到YOLO格式转换...")
    print(f"图片目录: {images_dir}")
    print(f"标签目录: {labels_dir}")
    print(f"输出目录: {output_dir}")
    print("-" * 50)
    
    # 执行转换
    convert_xml_to_yolo(images_dir, labels_dir, output_dir)

if __name__ == "__main__":
    main() 
