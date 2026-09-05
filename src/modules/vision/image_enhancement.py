"""
图像增强模块 - 用于水下图像的亮度和对比度增强
适用于光线不足的水下环境
"""
import cv2
import numpy as np
from typing import Optional


class ImageEnhancer:
    """
    图像增强器，提供多种增强方法以改善水下图像质量
    """
    
    def __init__(self, 
                 enable_clahe: bool = True,
                 enable_gamma: bool = False,
                 enable_brightness: bool = False,
                 clahe_clip_limit: float = 3.0,
                 clahe_tile_size: int = 8,
                 gamma_value: float = 1.5,
                 brightness_value: int = 30):
        """
        初始化图像增强器
        
        参数:
            enable_clahe: 是否启用CLAHE（对比度限制自适应直方图均衡化）
            enable_gamma: 是否启用伽马校正
            enable_brightness: 是否启用简单亮度增强
            clahe_clip_limit: CLAHE对比度限制值（1.0-5.0，默认3.0）
            clahe_tile_size: CLAHE分块大小（默认8x8）
            gamma_value: 伽马值（>1增亮，<1变暗，默认1.5）
            brightness_value: 亮度增加值（0-100，默认30）
        """
        self.enable_clahe = enable_clahe
        self.enable_gamma = enable_gamma
        self.enable_brightness = enable_brightness
        
        self.gamma_value = gamma_value
        self.brightness_value = brightness_value
        
        # 创建CLAHE对象（仅在启用时）
        if self.enable_clahe:
            self.clahe = cv2.createCLAHE(
                clipLimit=clahe_clip_limit, 
                tileGridSize=(clahe_tile_size, clahe_tile_size)
            )
        else:
            self.clahe = None
        
        # 预计算伽马查找表（加速处理）
        if self.enable_gamma:
            self.gamma_table = self._build_gamma_table(gamma_value)
        else:
            self.gamma_table = None
    
    def _build_gamma_table(self, gamma: float) -> np.ndarray:
        """
        构建伽马校正查找表
        
        参数:
            gamma: 伽马值
        
        返回:
            查找表数组
        """
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in range(256)]).astype("uint8")
        return table
    
    def enhance_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        使用CLAHE增强图像
        适合水下场景，能够增强局部对比度而不过度增强噪声
        
        参数:
            image: 输入图像（BGR格式）
        
        返回:
            增强后的图像
        """
        # 转换到LAB色彩空间
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 只对L通道应用CLAHE
        l_enhanced = self.clahe.apply(l)
        
        # 合并通道
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        
        # 转换回BGR
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def enhance_gamma(self, image: np.ndarray) -> np.ndarray:
        """
        使用伽马校正增强图像亮度
        
        参数:
            image: 输入图像
        
        返回:
            增强后的图像
        """
        return cv2.LUT(image, self.gamma_table)
    
    def enhance_brightness(self, image: np.ndarray) -> np.ndarray:
        """
        简单的亮度增强
        
        参数:
            image: 输入图像
        
        返回:
            增强后的图像
        """
        # 使用可饱和加法，避免溢出
        enhanced = cv2.add(image, np.array([self.brightness_value]))
        return enhanced
    
    def enhance(self, image: np.ndarray) -> np.ndarray:
        """
        应用所有启用的增强方法
        
        参数:
            image: 输入图像（BGR格式）
        
        返回:
            增强后的图像
        """
        if image is None or image.size == 0:
            return image
        
        enhanced = image.copy()
        
        # 按顺序应用增强
        if self.enable_brightness:
            enhanced = self.enhance_brightness(enhanced)
        
        if self.enable_gamma:
            enhanced = self.enhance_gamma(enhanced)
        
        if self.enable_clahe:
            enhanced = self.enhance_clahe(enhanced)
        
        return enhanced
    
    def update_params(self, 
                     clahe_clip_limit: Optional[float] = None,
                     gamma_value: Optional[float] = None,
                     brightness_value: Optional[int] = None):
        """
        动态更新增强参数
        
        参数:
            clahe_clip_limit: 新的CLAHE对比度限制值
            gamma_value: 新的伽马值
            brightness_value: 新的亮度值
        """
        if clahe_clip_limit is not None and self.enable_clahe:
            self.clahe = cv2.createCLAHE(
                clipLimit=clahe_clip_limit, 
                tileGridSize=(self.clahe.getTilesGridSize())
            )
        
        if gamma_value is not None and self.enable_gamma:
            self.gamma_value = gamma_value
            self.gamma_table = self._build_gamma_table(gamma_value)
        
        if brightness_value is not None and self.enable_brightness:
            self.brightness_value = brightness_value


def enhance_underwater_image(image: np.ndarray, 
                            method: str = 'clahe',
                            **kwargs) -> np.ndarray:
    """
    便捷函数：增强水下图像
    
    参数:
        image: 输入图像
        method: 增强方法 ('clahe', 'gamma', 'brightness', 'auto')
        **kwargs: 额外的参数
    
    返回:
        增强后的图像
    """
    if method == 'clahe':
        enhancer = ImageEnhancer(
            enable_clahe=True,
            enable_gamma=False,
            enable_brightness=False,
            **kwargs
        )
    elif method == 'gamma':
        enhancer = ImageEnhancer(
            enable_clahe=False,
            enable_gamma=True,
            enable_brightness=False,
            **kwargs
        )
    elif method == 'brightness':
        enhancer = ImageEnhancer(
            enable_clahe=False,
            enable_gamma=False,
            enable_brightness=True,
            **kwargs
        )
    elif method == 'auto':
        # 自动模式：组合多种方法
        enhancer = ImageEnhancer(
            enable_clahe=True,
            enable_gamma=True,
            enable_brightness=False,
            **kwargs
        )
    else:
        raise ValueError(f"未知的增强方法: {method}")
    
    return enhancer.enhance(image)


if __name__ == "__main__":
    """测试代码"""
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python image_enhancement.py <图像路径>")
        sys.exit(1)
    
    # 读取测试图像
    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"无法读取图像: {image_path}")
        sys.exit(1)
    
    # 创建增强器
    enhancer = ImageEnhancer(
        enable_clahe=True,
        enable_gamma=False,
        enable_brightness=False,
        clahe_clip_limit=3.0
    )
    
    # 增强图像
    enhanced = enhancer.enhance(image)
    
    # 显示对比
    comparison = np.hstack([image, enhanced])
    cv2.imshow("Original vs Enhanced", comparison)
    
    print("按任意键退出...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 保存结果
    output_path = image_path.rsplit('.', 1)[0] + '_enhanced.jpg'
    cv2.imwrite(output_path, enhanced)
    print(f"增强后的图像已保存至: {output_path}")

