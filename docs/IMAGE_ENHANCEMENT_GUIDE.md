# 水下图像增强功能使用指南

## 概述

本系统已集成图像增强功能，专门用于改善水下暗环境下的图像质量。通过多种图像处理算法，可以显著提升摄像头采集图像的亮度和对比度，从而提高目标检测的准确性。

## 功能特性

### 支持的增强方法

1. **CLAHE (对比度限制自适应直方图均衡化)** ⭐推荐
   - 最适合水下环境
   - 能够增强局部对比度
   - 不会过度放大噪声
   - 处理效果自然

2. **Gamma校正**
   - 整体提升图像亮度
   - 适合均匀光照不足的场景
   - 可能会放大噪声

3. **简单亮度增强**
   - 直接增加像素值
   - 最快的处理速度
   - 容易导致过曝

4. **Auto模式 (组合方法)**
   - 结合CLAHE和Gamma校正
   - 综合增强效果
   - 适合复杂光照环境

## 配置参数

在 `main.py` 文件中的配置开关（第51-61行）：

```python
# --- 图像增强开关 ---
# 设置为 1 启用图像亮度增强（推荐用于水下暗环境）, 设置为 0 则禁用
IMAGE_ENHANCEMENT_ENABLED = 1

# 图像增强方法：'clahe'(推荐), 'gamma', 'brightness', 'auto'(组合方法)
IMAGE_ENHANCEMENT_METHOD = 'clahe'

# CLAHE参数：对比度限制值（1.0-5.0，值越大对比度越强）
IMAGE_ENHANCEMENT_CLAHE_CLIP = 3.0

# 伽马校正值（仅在method='gamma'或'auto'时有效，>1增亮）
IMAGE_ENHANCEMENT_GAMMA = 1.5

# 亮度增加值（仅在method='brightness'时有效，0-100）
IMAGE_ENHANCEMENT_BRIGHTNESS = 30
```

## 推荐配置

### 方案1：CLAHE (默认，推荐) ⭐
适用于大部分水下场景，效果最自然。

```python
IMAGE_ENHANCEMENT_ENABLED = 1
IMAGE_ENHANCEMENT_METHOD = 'clahe'
IMAGE_ENHANCEMENT_CLAHE_CLIP = 3.0  # 推荐范围: 2.5-4.0
```

**调整建议：**
- 如果图像还是太暗，增加 `CLAHE_CLIP` 到 3.5 或 4.0
- 如果图像噪点明显，减小 `CLAHE_CLIP` 到 2.5 或 2.0
- 不建议超过 5.0，会过度增强噪声

### 方案2：Gamma校正
适用于光线均匀但整体偏暗的场景。

```python
IMAGE_ENHANCEMENT_ENABLED = 1
IMAGE_ENHANCEMENT_METHOD = 'gamma'
IMAGE_ENHANCEMENT_GAMMA = 1.5  # 推荐范围: 1.2-2.0
```

**调整建议：**
- 值越大，图像越亮
- 推荐从 1.5 开始尝试
- 超过 2.0 可能导致严重的噪声问题

### 方案3：组合模式 (效果最强)
适用于极暗环境或复杂光照条件。

```python
IMAGE_ENHANCEMENT_ENABLED = 1
IMAGE_ENHANCEMENT_METHOD = 'auto'
IMAGE_ENHANCEMENT_CLAHE_CLIP = 2.5
IMAGE_ENHANCEMENT_GAMMA = 1.3
```

**调整建议：**
- 组合方法效果最强，但也最容易产生噪声
- 建议两个参数都不要设置太高
- 先调整 CLAHE，再调整 Gamma

## 调试工具

系统提供了一个实时调试工具，帮助您找到最佳参数：

### 使用摄像头实时测试

```bash
cd AUVRobot-VisualControl
python tools/test_image_enhancement.py
```

### 使用静态图像测试

```bash
python tools/test_image_enhancement.py path/to/your/image.jpg
```

### 工具功能

- **实时预览**：左侧显示原始图像，右侧显示增强后的图像
- **参数调整**：通过滑块实时调整参数，立即看到效果
- **保存图像**：按 `s` 键保存当前增强后的图像
- **打印参数**：按 `p` 键打印当前参数到控制台
- **退出**：按 `q` 或 `ESC` 键退出

### 调试流程建议

1. 先拍摄几张典型的水下暗环境图像
2. 使用调试工具加载图像
3. 尝试不同的模式和参数
4. 记录效果最好的参数组合
5. 更新 `main.py` 中的配置
6. 运行主程序验证效果

## 性能影响

不同增强方法的性能开销（在Jetson平台上测试）：

| 方法 | 处理耗时 (640x480) | 性能影响 |
|------|-------------------|----------|
| 无增强 | 0 ms | - |
| Brightness | ~0.5 ms | 极小 ✓ |
| Gamma | ~1 ms | 小 ✓ |
| CLAHE | ~5-8 ms | 中等 ⚠️ |
| Auto | ~6-9 ms | 中等 ⚠️ |

**注意：**
- CLAHE 处理时间较长，但效果最好
- 如果帧率下降明显，可以考虑：
  1. 降低摄像头分辨率
  2. 使用 Gamma 方法代替 CLAHE
  3. 仅对前置摄像头启用增强

## 常见问题

### Q1: 增强后图像噪点很多怎么办？
A: 降低 `CLAHE_CLIP` 参数，或者切换到 Gamma 方法。

### Q2: 图像还是太暗？
A: 
- 如果使用CLAHE，尝试增加 `CLAHE_CLIP` 到 4.0-5.0
- 或切换到 `auto` 模式
- 检查摄像头硬件设置（见下文）

### Q3: 图像过曝（太白）？
A: 
- 降低相关参数值
- CLAHE: 减小 `CLAHE_CLIP`
- Gamma: 减小 `GAMMA` 值

### Q4: 处理速度太慢？
A:
- 切换到 `gamma` 或 `brightness` 方法
- 或完全禁用增强：`IMAGE_ENHANCEMENT_ENABLED = 0`

## 硬件级优化建议

除了软件增强，还可以尝试调整摄像头硬件参数：

### 在 main.py 中添加摄像头参数设置

在摄像头初始化后添加以下代码（第129行附近）：

```python
cap_forward = cv2.VideoCapture(0)

# 尝试调整摄像头参数
cap_forward.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 手动曝光模式
cap_forward.set(cv2.CAP_PROP_EXPOSURE, -3)      # 曝光值（负数=更长曝光=更亮）
cap_forward.set(cv2.CAP_PROP_GAIN, 10)          # 增益值（0-100）
cap_forward.set(cv2.CAP_PROP_BRIGHTNESS, 150)   # 亮度（0-255）
```

**注意：** 不是所有摄像头都支持这些参数，需要实际测试。

## 最佳实践

1. **先硬后软**：优先尝试硬件参数调整，再使用软件增强
2. **参数保守**：从较小的增强值开始，逐步增加
3. **实际测试**：在真实水下环境中测试，不要只看陆地效果
4. **平衡取舍**：在图像质量和处理速度之间找到平衡点
5. **定期调整**：不同水域、不同深度可能需要不同的参数

## 更新日志

- 2024-10-12: 初始版本，添加CLAHE、Gamma、Brightness三种增强方法
- 支持前置和下置摄像头独立增强
- 提供实时调试工具

## 技术支持

如有问题，请检查：
1. 是否正确设置了 `IMAGE_ENHANCEMENT_ENABLED = 1`
2. 增强方法名称是否正确（小写）
3. 参数值是否在合理范围内
4. 查看控制台输出的系统状态报告

---

祝您的AUV项目顺利！🚀
