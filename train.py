#!/usr/bin/env python3
"""
YOLOv11模型训练启动脚本
使用方法: python train.py
"""

import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == "__main__":
    # 导入训练模块
    from utils.train_models import YOLOTrainer
    
    print("🚀 YOLOv11模型训练")
    print("=" * 50)
    
    # 创建训练器
    trainer = YOLOTrainer(
        dataset_path="configs/dataset.yaml",
        model_size="s"  # 可选: n, s, m, l, x
    )
    
    # 开始训练
    try:
        model_path = trainer.train(
            epochs=100,        # 训练轮数
            batch_size=8,     # 批次大小 (根据GPU显存调整)
            imgsz=640,         # 输入图片尺寸
            patience=50,       # 早停耐心值
            save_period=10,    # 每10轮保存一次
            project="runs/train",
            name="yolov11s_custom_augfirst"  # 实验名称
        )
        
        print(f"\n🎉 训练成功完成!")
        print(f"📁 模型保存在: {model_path}")
        
        # 验证模型
        print("\n🔍 开始验证模型...")
        trainer.validate_model(model_path)
        
    except Exception as e:
        print(f"❌ 训练失败: {str(e)}")
        exit(1)
    
    exit(0) 
