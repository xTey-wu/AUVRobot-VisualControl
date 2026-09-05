from typing import Optional, Dict, Any, Tuple, Union

YField = Union[int, str, Tuple[str, int]]
YawField = Union[int, str]
Command = Tuple[int, int, YField, YawField]

def handle(auv, perception_bundle: Dict[str, Any]) -> Optional[Command]:
    print("\n--- 任务完成。AUV正在关闭。 ---")
    return (128, 128, 200, "global_hold")


