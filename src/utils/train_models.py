import os
import yaml
import torch
from ultralytics import YOLO, settings
from pathlib import Path
import argparse
from datetime import datetime

class YOLOTrainer:
    """
    YOLOv11模型训练器
    """
    
    def __init__(self, dataset_path: str = "dataset/dataset.yaml", model_size: str = "s"):
        """
        初始化训练器
        
        Args:
            dataset_path: 数据集配置文件路径
            model_size: 模型大小 (n, s, m, l, x)
        """
        self.dataset_path = dataset_path
        self.model_size = model_size
        self.model_name = f"yolo11s.pt"
        
        # 检查GPU可用性
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    def validate_dataset(self) -> bool:
        """
        验证数据集配置
        
        Returns:
            bool: 数据集是否有效
        """
        try:
            # 检查数据集配置文件
            if not os.path.exists(self.dataset_path):
                print(f"❌ 数据集配置文件不存在: {self.dataset_path}")
                return False
            
            # 读取数据集配置
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                dataset_config = yaml.safe_load(f)
            
            print("📊 数据集配置:")
            print(f"  类别数量: {dataset_config.get('nc', 'N/A')}")
            print(f"  类别名称: {dataset_config.get('names', 'N/A')}")
            print(f"  训练集路径: {dataset_config.get('train', 'N/A')}")
            print(f"  验证集路径: {dataset_config.get('val', 'N/A')}")
            
            # 检查训练集和验证集
            train_path = os.path.join(os.path.dirname(self.dataset_path), dataset_config.get('train', ''))
            val_path = os.path.join(os.path.dirname(self.dataset_path), dataset_config.get('val', ''))
            
            if not os.path.exists(train_path):
                print(f"❌ 训练集路径不存在: {train_path}")
                return False
            
            if not os.path.exists(val_path):
                print(f"❌ 验证集路径不存在: {val_path}")
                return False
            
            # 统计图片数量
            train_images = len([f for f in os.listdir(train_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            val_images = len([f for f in os.listdir(val_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            
            print(f"✅ 训练集图片数量: {train_images}")
            print(f"✅ 验证集图片数量: {val_images}")
            
            return True
            
        except Exception as e:
            print(f"❌ 数据集验证失败: {str(e)}")
            return False
    
    def train(self, 
              epochs: int = 100,
              batch_size: int = 16,
              imgsz: int = 640,
              patience: int = 50,
              save_period: int = 10,
              project: str = "runs/train",
              name: str = None,
              augment_params: dict | None = None) -> str:
        """
        开始训练模型
        
        Args:
            epochs: 训练轮数
            batch_size: 批次大小
            imgsz: 输入图片尺寸
            patience: 早停耐心值
            save_period: 保存周期
            project: 项目保存路径
            name: 实验名称
            
        Returns:
            str: 训练完成后的模型路径
        """
        try:
            # 验证数据集
            if not self.validate_dataset():
                raise ValueError("数据集验证失败")
            
            # 显式启用 TensorBoard 日志（需要已安装 tensorboard 包）
            try:
                settings.update({"tensorboard": True})
                print("📝 TensorBoard 日志: 已启用")
            except Exception as _:
                print("[警告] 无法更新 Ultralytics 设置以启用 TensorBoard")

            # 生成实验名称
            if name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                name = f"yolov11{self.model_size}_{timestamp}"
            
            print(f"\n🚀 开始训练 YOLOv11{self.model_size}")
            print(f"📁 实验名称: {name}")
            print(f"📊 训练参数:")
            print(f"  轮数: {epochs}")
            print(f"  批次大小: {batch_size}")
            print(f"  图片尺寸: {imgsz}x{imgsz}")
            print(f"  早停耐心: {patience}")
            
            # 加载预训练模型
            print(f"\n📥 加载预训练模型: {self.model_name}")
            model = YOLO(self.model_name)
            
            # 禁用所有数据增强
            recommended_aug = dict(
                degrees=0,
                translate=0.0,
                scale=0.0,
                shear=0.0,
                perspective=0.0,
                fliplr=0.0,
                flipud=0.0,
                hsv_h=0.0,
                hsv_s=0.0,
                hsv_v=0.0,
                mosaic=0.0,
                mixup=0.0,
                copy_paste=0.0,
            )

            # 开始训练
            results = model.train(
                data=self.dataset_path,
                epochs=epochs,
                batch=batch_size,
                imgsz=imgsz,
                patience=patience,
                save_period=save_period,
                project=project,
                name=name,
                device=self.device,
                verbose=True,
                plots=True,  # 生成训练图表
                save=True,   # 保存模型
                exist_ok=True,  # 覆盖已存在的实验
                **recommended_aug
            )
            
            # 获取最佳模型路径
            best_model_path = results.save_dir / "weights" / "best.pt"
            
            if os.path.exists(best_model_path):
                print(f"\n✅ 训练完成!")
                print(f"📁 最佳模型保存路径: {best_model_path}")
                
                # 复制到models目录
                import shutil
                models_dir = Path("models")
                models_dir.mkdir(exist_ok=True)
                final_model_path = models_dir / "best.pt"
                shutil.copy2(best_model_path, final_model_path)
                print(f"📁 模型已复制到: {final_model_path}")
                
                return str(final_model_path)
            else:
                raise FileNotFoundError("训练完成但未找到最佳模型文件")
                
        except Exception as e:
            print(f"❌ 训练失败: {str(e)}")
            raise
    
    def validate_model(self, model_path: str) -> bool:
        """
        验证训练好的模型
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            bool: 模型是否有效
        """
        try:
            print(f"\n🔍 验证模型: {model_path}")
            
            if not os.path.exists(model_path):
                print(f"❌ 模型文件不存在: {model_path}")
                return False
            
            # 加载模型
            model = YOLO(model_path)
            
            # 在验证集上测试
            results = model.val(data=self.dataset_path, verbose=True)
            
            print(f"✅ 模型验证完成")
            print(f"📊 mAP50: {results.box.map50:.4f}")
            print(f"📊 mAP50-95: {results.box.map:.4f}")
            
            return True
            
        except Exception as e:
            print(f"❌ 模型验证失败: {str(e)}")
            return False

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description="YOLOv11模型训练")
    parser.add_argument("--dataset", type=str, default="dataset_merged/dataset.yaml", help="数据集配置文件路径")
    parser.add_argument("--model-size", type=str, default="s", choices=["n", "s", "m", "l", "x"], help="模型大小")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=16, help="批次大小")
    parser.add_argument("--img-size", type=int, default=640, help="输入图片尺寸")
    parser.add_argument("--patience", type=int, default=50, help="早停耐心值")
    parser.add_argument("--project", type=str, default="runs/train", help="项目保存路径")
    parser.add_argument("--name", type=str, default=None, help="实验名称")
    parser.add_argument("--validate-only", action="store_true", help="仅验证数据集")
    parser.add_argument("--simple", action="store_true", help="使用简单训练模式")
    
    args = parser.parse_args()
    
    # 创建训练器
    trainer = YOLOTrainer(args.dataset, args.model_size)
    
    if args.validate_only:
        # 仅验证数据集
        trainer.validate_dataset()
    elif args.simple:
        # 简单训练模式
        print("🚀 YOLOv11模型训练 (简单模式)")
        print("=" * 50)
        
        try:
            model_path = trainer.train(
                epochs=args.epochs,
                batch_size=args.batch_size,
                imgsz=args.img_size,
                patience=args.patience,
                save_period=10,
                project=args.project,
                name="yolov11s_new" if args.name is None else args.name
            )
            
            print(f"\n🎉 训练成功完成!")
            print(f"📁 模型保存在: {model_path}")
            
            # 验证模型
            print("\n🔍 开始验证模型...")
            trainer.validate_model(model_path)
            
        except Exception as e:
            print(f"❌ 训练失败: {str(e)}")
            return 1
    else:
        # 命令行参数训练模式
        try:
            model_path = trainer.train(
                epochs=args.epochs,
                batch_size=args.batch_size,
                imgsz=args.img_size,
                patience=args.patience,
                project=args.project,
                name=args.name
            )
            
            # 验证训练好的模型
            trainer.validate_model(model_path)
            
        except Exception as e:
            print(f"❌ 训练过程出错: {str(e)}")
            return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

