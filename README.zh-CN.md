[English](README.md) | **简体中文**

# AUV Vision & Control

面向水下机器人竞赛的视觉感知与自主控制系统。项目将 YOLO 目标检测、引导线识别、PID 控制、串口通信和任务状态机组合在同一套 Python 程序中，用于完成循线、撞球、穿门和返航流程。

![AUV 状态机](docs/flowcharts/AUV_StateMachine.png)

## 项目亮点

- 8 类水下目标检测：门框、四个角点、红球、蓝球和引导线。
- 分层状态机：循线、撞球、穿门、动作执行、返航与结束状态相互解耦。
- 多轴 PID：覆盖平移、深度和偏航控制，并包含抗积分饱和等稳定性处理。
- 水下图像增强：支持 CLAHE、Gamma、亮度补偿及组合模式。
- Jetson 部署接口：摄像头采集、串口收发、性能监控和视频记录均已模块化。
- 可复核模型资产：仓库包含最终 PyTorch 权重、示例素材和评测明细。

## 系统结构

```text
.
├── main.py                         # 机器人主循环与任务状态机
├── src/modules/
│   ├── states/                     # 循线、撞球、穿门、返航等状态
│   ├── vision/                     # YOLO、引导线检测和图像增强
│   ├── control/                    # PID 与控制指令融合
│   ├── communicate/                # STM32 串口通信
│   ├── sensors/                    # 深度与偏航数据处理
│   ├── visualization/              # OSD 调试界面
│   ├── videocapture/               # 双摄像头录像
│   └── metrics/                    # 运行性能记录
├── models/combine1014.pt           # 最终 8 类检测模型
├── configs/dataset.yaml            # YOLO 数据集结构模板
├── examples/                       # 两张图片与一段演示视频
├── tools/                          # 图片/视频推理及调试工具
├── evaluation/                     # 唯一一组评测结果与证据
└── docs/                           # 控制设计、流程图和调参说明
```

## 快速体验

建议使用 Python 3.10 或 3.11。先创建独立环境并安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

对仓库内两张示例图片运行检测：

```bash
python tools/detect_images.py
```

结果默认写入 `outputs/detected_images/`。也可以指定自己的图片、模型或设备：

```bash
python tools/detect_images.py \
  --source path/to/images \
  --model models/combine1014.pt \
  --device cpu
```

处理演示视频：

```bash
python tools/batch_detect_test1015.py
```

## 检测示例

以下结果由仓库内最终权重以 `imgsz=640`、`conf=0.25` 生成：

![门框与引导线检测示例 1](examples/results/239.jpg)

![门框与引导线检测示例 2](examples/results/443.jpg)

## 机器人运行

`main.py` 会直接访问摄像头和串口，适合在完成硬件接线与参数检查后于 Jetson 设备运行：

```bash
python main.py
```

默认接口包括前置摄像头、可选下置摄像头及 `/dev/ttyTHS1` 串口。首次上机前应检查摄像头编号、目标球颜色、PID 参数、面积阈值、串口协议和紧急停止逻辑。不要在未架空推进器或没有现场保护的情况下直接启动主程序。

## 训练数据

仓库没有包含完整训练数据集。若需复现训练，请按以下结构将数据放入 `dataset/`：

```text
dataset/
├── train/
│   ├── images/
│   └── labels/
└── val/
    ├── images/
    └── labels/
```

随后运行：

```bash
python train.py
```

类别定义见 `configs/dataset.yaml`。

## 现有评测结果

| 指标 | 结果 | 口径 |
| --- | ---: | --- |
| Precision | 0.99098 | 训练期验证集 |
| Recall | 0.99343 | 训练期验证集 |
| mAP@50 | 0.99497 | 训练期验证集 |
| mAP@50:95 | 0.88885 | 训练期验证集 |
| CPU 推理平均耗时 | 2488.820 ms/图 | 2 张无标注图片，各计时 10 次 |

这些准确率来自训练流程中的验证记录，并非独立测试集结果。当前材料没有独立、带标注且可证明未参与训练或调参的测试集，因此不报告独立测试 mAP，也不以示例图片的预测框数代替真实目标数。完整条件和证据见 [`evaluation/README.md`](evaluation/README.md)。

## 文档导航

- [`docs/PROJECT_REQUIREMENTS.md`](docs/PROJECT_REQUIREMENTS.md)：任务目标、状态流程和接口需求。
- [`docs/new_control_logic_design.md`](docs/new_control_logic_design.md)：控制逻辑设计。
- [`docs/pid_paras.md`](docs/pid_paras.md)：PID 参数说明。
- [`docs/IMAGE_ENHANCEMENT_GUIDE.md`](docs/IMAGE_ENHANCEMENT_GUIDE.md)：水下图像增强指南。
- [`evaluation/README.md`](evaluation/README.md)：结果口径、限制和复核信息。
