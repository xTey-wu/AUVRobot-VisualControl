import cv2
import numpy as np
from typing import Tuple, Optional, List


class GuideLineDetector:
    """
    简化的引导线检测器
    只输出夹角和相对于图片中心的坐标
    """
    
    def __init__(self, 
                 lower_orange_red: Tuple[int, int, int] = (96, 0, 98),
                 upper_orange_red: Tuple[int, int, int] = (179, 142, 191),
                 min_area: int = 10,
                 max_area: int = 5000000):
        """
        初始化引导线检测器
        
        Args:
            lower_orange_red: 橙红色HSV下界 (H, S, V)
            upper_orange_red: 橙红色HSV上界 (H, S, V)
            min_area: 最小轮廓面积
            max_area: 最大轮廓面积
        """
        self.lower_orange_red = np.array(lower_orange_red)
        self.upper_orange_red = np.array(upper_orange_red)
        self.min_area = min_area
        self.max_area = max_area
        
        # 形态学操作核
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    
    def _color_detection(self, frame: np.ndarray) -> np.ndarray:
        """
        颜色识别 - 提取橙红色区域
        
        Args:
            frame: 输入图像 (BGR格式)
            
        Returns:
            二值化掩码
        """
        # 转换到HSV色彩空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 提取橙红色区域
        mask = cv2.inRange(hsv, self.lower_orange_red, self.upper_orange_red)
        
        return mask
    
    def _morphological_processing(self, mask: np.ndarray) -> np.ndarray:
        """
        形态学处理 - 去除噪声，填补空洞
        
        Args:
            mask: 二值化掩码
            
        Returns:
            处理后的掩码
        """
        # 开运算 - 去除小噪声
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        
        # 闭运算 - 填补空洞
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        
        return mask
    
    def _contour_detection(self, mask: np.ndarray) -> List[np.ndarray]:
        """
        轮廓检测
        
        Args:
            mask: 二值化掩码
            
        Returns:
            检测到的轮廓列表（只保留最大的）
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 筛选面积合适的轮廓
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_area <= area <= self.max_area:
                valid_contours.append(contour)
        
        # 只保留面积最大的轮廓
        if valid_contours:
            largest_contour = max(valid_contours, key=cv2.contourArea)
            return [largest_contour]
        else:
            return []
    
    def _fit_rectangle(self, contours: List[np.ndarray]) -> Optional[Tuple]:
        """
        矩形拟合与方向提取 - 使用四边形逼近
        
        Args:
            contours: 轮廓列表
            
        Returns:
            (center, angle_1) 或 None
            center: 引导线中心坐标
            angle_1: 与y轴的夹角（度，范围0-90°，带正负号）
        """
        if not contours:
            return None
        
        # 选择面积最大的轮廓
        target_contour = max(contours, key=cv2.contourArea)
        
        # 1. 轮廓逼近为多边形
        epsilon = 0.02 * cv2.arcLength(target_contour, True)
        approx_poly = cv2.approxPolyDP(target_contour, epsilon, True)
        
        # 2. 如果逼近后的多边形点数不是4，尝试不同的epsilon值
        if len(approx_poly) != 4:
            for epsilon_factor in [0.01, 0.03, 0.05, 0.1]:
                epsilon = epsilon_factor * cv2.arcLength(target_contour, True)
                approx_poly = cv2.approxPolyDP(target_contour, epsilon, True)
                if len(approx_poly) == 4:
                    break
        
        # 3. 如果仍然不是4个点，使用最小外接矩形
        if len(approx_poly) != 4:
            rect = cv2.minAreaRect(target_contour)
            center, (width, height), original_angle = rect
            
            # 计算与y轴的夹角angle_1（0-90度范围）
            angle_1 = abs(90 - original_angle)
            if angle_1 > 90:
                angle_1 = 180 - angle_1
            angle_1 = abs(angle_1)
            
            # 根据original_angle范围确定angle_1的正负号
            if (original_angle >= -90 and original_angle < 0) or (original_angle > 90 and original_angle <= 180):
                angle_1 = -angle_1
                
            return (center, angle_1)
        
        # 4. 获取四边形的四个角点
        quad_points = approx_poly.reshape(4, 2).astype(np.float32)
        
        # 5. 计算四边形的中心
        center = np.mean(quad_points, axis=0)
        
        # 6. 计算四边形的边长
        edges = []
        for i in range(4):
            next_i = (i + 1) % 4
            edge_length = np.linalg.norm(quad_points[next_i] - quad_points[i])
            edges.append(edge_length)
        
        # 7. 计算主方向（长边的方向）
        # 找到最长的边
        longest_edge_idx = np.argmax(edges)
        next_idx = (longest_edge_idx + 1) % 4
        
        # 计算长边的方向向量
        edge_vector = quad_points[next_idx] - quad_points[longest_edge_idx]
        edge_vector = edge_vector / np.linalg.norm(edge_vector)
        
        # 计算角度（相对于水平方向）
        angle_rad = np.arctan2(edge_vector[1], edge_vector[0])
        angle_deg = np.degrees(angle_rad)
        
        # 计算与y轴的夹角angle_1（0-90度范围）
        angle_1 = abs(90 - angle_deg)
        if angle_1 > 90:
            angle_1 = 180 - angle_1
        angle_1 = abs(angle_1)

        # 根据angle_deg范围确定angle_1的正负号
        if (angle_deg >= -90 and angle_deg < 0) or (angle_deg > 90 and angle_deg <= 180):
            angle_1 = -angle_1
        
        return (center, angle_1)
    
    def detect(self, frame: np.ndarray) -> Optional[Tuple[Tuple[float, float], float]]:
        """
        检测引导线，返回以左上角为原点的坐标和夹角
        
        Args:
            frame: 输入图像
            
        Returns:
            ((x, y), angle) 或 None
            x, y: 以左上角为原点的坐标
            angle: 与y轴的夹角（度，范围0-90°，带正负号）
        """
        # 1. 颜色识别
        mask = self._color_detection(frame)
        
        # 2. 形态学处理
        mask = self._morphological_processing(mask)
        
        # 3. 轮廓检测
        contours = self._contour_detection(mask)
        
        # 4. 矩形拟合
        result = self._fit_rectangle(contours)
        
        if result is None:
            return None
        
        # 解包结果
        center, angle_1 = result
        
        # 直接返回以左上角为原点的坐标
        return ((center[0], center[1]), angle_1)
    
    def detect_with_debug(self, frame: np.ndarray) -> Tuple[Optional[Tuple[Tuple[float, float], float]], np.ndarray]:
        """
        检测引导线并返回调试图像
        
        Args:
            frame: 输入图像
            
        Returns:
            (result, debug_image)
            result: ((x, y), angle) 或 None
            debug_image: 调试图像
        """
        # 1. 颜色识别
        mask = self._color_detection(frame)
        
        # 2. 形态学处理
        mask = self._morphological_processing(mask)
        
        # 3. 轮廓检测
        contours = self._contour_detection(mask)
        
        # 4. 矩形拟合
        result = self._fit_rectangle(contours)
        
        # 生成调试图像
        debug_image = self._create_debug_image(frame, result, contours)
        
        if result is None:
            return None, debug_image
        
        # 解包结果
        center, angle_1 = result
        
        # 直接返回以左上角为原点的坐标
        return ((center[0], center[1]), angle_1), debug_image
    
    def _create_debug_image(self, frame: np.ndarray, result: Optional[Tuple], contours: List[np.ndarray]) -> np.ndarray:
        """
        创建调试图像
        
        Args:
            frame: 原始图像
            result: 检测结果
            contours: 轮廓列表
            
        Returns:
            调试图像
        """
        result_img = frame.copy()
        
        if result is not None:
            center, angle_1 = result
            
            # 绘制中心点
            center_int = (int(center[0]), int(center[1]))
            cv2.circle(result_img, center_int, 5, (255, 0, 0), -1)
            
            # 显示角度信息
            text_angle = f"Angle: {angle_1:.1f}°"
            cv2.putText(result_img, text_angle, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 显示坐标信息
            text_coord = f"Coord: ({center[0]:.1f}, {center[1]:.1f})"
            cv2.putText(result_img, text_coord, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # 绘制拟合的四边形
            if contours:
                target_contour = max(contours, key=cv2.contourArea)
                
                # 轮廓逼近为多边形
                epsilon = 0.02 * cv2.arcLength(target_contour, True)
                approx_poly = cv2.approxPolyDP(target_contour, epsilon, True)
                
                # 如果逼近后的多边形点数不是4，尝试不同的epsilon值
                if len(approx_poly) != 4:
                    for epsilon_factor in [0.01, 0.03, 0.05, 0.1]:
                        epsilon = epsilon_factor * cv2.arcLength(target_contour, True)
                        approx_poly = cv2.approxPolyDP(target_contour, epsilon, True)
                        if len(approx_poly) == 4:
                            break
                
                # 绘制逼近的四边形
                if len(approx_poly) == 4:
                    cv2.drawContours(result_img, [approx_poly], 0, (0, 255, 0), 3)
                    
                    # 绘制四边形的角点
                    for i, point in enumerate(approx_poly):
                        cv2.circle(result_img, tuple(point[0]), 5, (0, 255, 255), -1)
                        cv2.putText(result_img, str(i), tuple(point[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                else:
                    # 如果无法逼近为四边形，绘制最小外接矩形
                    rect_cv = cv2.minAreaRect(target_contour)
                    box = cv2.boxPoints(rect_cv)
                    box = np.int0(box)
                    cv2.drawContours(result_img, [box], 0, (0, 255, 0), 3)
        
        return result_img 