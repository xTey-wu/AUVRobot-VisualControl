from typing import Optional, Dict, Any

def handle(ctx, perception_bundle: Dict[str, Any], *, is_target_ball, State) -> Optional[tuple]:
    """START: 决定程序开始执行时进入哪个状态"""
    results = perception_bundle['detections_forward']
    # 如果识别到目标小球，则进入CIRCLE_KICK状态
    for result in results:
        if is_target_ball(result):
            ctx._transition(State.CIRCLE_KICK)
            return None
    # 如果未识别到目标小球，则进入GUIDELINE_FOLLOW状态，通过引导线找球
    ctx._transition(State.GUIDELINE_FOLLOW)
    return None


