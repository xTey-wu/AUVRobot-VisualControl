import math
import numpy as np
import cv2
from typing import Any, Dict, Optional, Tuple, Callable

# 用户偏好：OSD 文本采用英文，以避免显示问题
# 参考: 用户项目偏好说明

def draw_forward_osd(
    frame_forward: np.ndarray,
    detections_forward: list,
    *,
    is_ball: Callable[[Dict[str, Any]], bool],
    current_state_name: str,
    guideline_sub_state: str,
    circle_kicked: int,
    task_finished: int,
    current_depth: float,
    target_depth: float,
    current_yaw: float,
    yaw_mode_str: str,
    target_yaw: float,
    debug_front_follow: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Draw OSD overlays for the forward camera view and return a display frame."""
    display_f = frame_forward.copy()

    # Draw detections
    for r in detections_forward:
        class_name = r.get('class_name')
        if class_name == 'guideline':
            center = r.get('center')
            if center:
                radius = int((r['bbox'][2] - r['bbox'][0]) / 2)
                cv2.circle(display_f, (int(center[0]), int(center[1])), radius, (255, 255, 0), 2)
                cv2.circle(display_f, (int(center[0]), int(center[1])), 5, (255, 255, 0), -1)
        else:
            box = r.get('bbox')
            label = f"{class_name} ({r.get('confidence', 0):.2f})"
            if is_ball(r) or class_name == 'doorframe':
                area = r.get('area', 0)
                label += f" Area: {area:.0f}"
            if class_name == 'red_circle':
                color = (0, 0, 255)
            elif class_name == 'blue_circle':
                color = (255, 0, 0)
            elif class_name == 'doorframe':
                color = (255, 0, 255)
            else:
                color = (0, 255, 0)
            cv2.rectangle(display_f, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            cv2.putText(display_f, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # State texts (English OSD preferred)
    state_info_line1 = f"State: {current_state_name} | Sub: {guideline_sub_state}"
    state_info_line2 = f"Kicked: {circle_kicked} | Tasks: {task_finished}"
    depth_info = f"Depth: {current_depth:.2f}m (Target: {target_depth:.2f}m)"
    yaw_info = f"Yaw: {current_yaw:.1f}deg | Mode: {yaw_mode_str} | Target: {target_yaw:.1f}deg"

    cv2.putText(display_f, state_info_line1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(display_f, state_info_line2, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(display_f, depth_info, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(display_f, yaw_info, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    # Debug overlay (optional)
    if debug_front_follow is not None:
        debug_info = [
            f"DEBUG MODE - State Locked: {current_state_name}",
            f"Commands: 1=START 2=KICK 3=FOLLOW 4=DOOR 5=ACTION 6=RETURN 0=END",
            f"Helpers: r=Reset_PID s=Emergency_Stop ESC=Exit q=Quit",
            f"FrontFollow: fwd={int(debug_front_follow['fwd'])} band={debug_front_follow['band']} y_ratio={debug_front_follow['y_ratio']:.2f} x_err={debug_front_follow['x_err']:.1f} x_out={debug_front_follow['x_out']:.1f}",
        ]
        for i, info in enumerate(debug_info):
            cv2.putText(display_f, info, (10, 110 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)

    return display_f


def draw_bottom_osd(
    frame_bottom: Optional[np.ndarray],
    guideline_bottom: Optional[Tuple[Tuple[float, float], float]],
    *,
    bottom_cam_available: bool,
    frame_width_bottom: int,
    frame_height_bottom: int,
) -> np.ndarray:
    """Draw OSD overlays for the bottom camera view and return a display frame."""
    if bottom_cam_available and frame_bottom is not None:
        display_b = frame_bottom.copy()
        if guideline_bottom:
            (center_x, center_y), angle = guideline_bottom
            cv2.circle(display_b, (int(center_x), int(center_y)), 10, (255, 0, 0), -1)
            line_length = 80
            end_x = int(center_x + line_length * math.sin(math.radians(-angle)))
            end_y = int(center_y - line_length * math.cos(math.radians(-angle)))
            cv2.line(display_b, (int(center_x), int(center_y)), (end_x, end_y), (0, 0, 255), 3)
            cv2.putText(display_b, "Guideline Detected", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(display_b, f"Angle: {angle:.1f}deg", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        else:
            cv2.putText(display_b, "No Guideline Detected", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    else:
        display_b = np.zeros((frame_height_bottom, frame_width_bottom, 3), dtype=np.uint8)
        cv2.putText(display_b, "Bottom Camera Not Available", (50, int(frame_height_bottom/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return display_b
