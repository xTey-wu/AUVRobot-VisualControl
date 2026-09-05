import os
import json
import time
from typing import List, Dict, Any, Optional

class Metrics:
    def __init__(self) -> None:
        self.loop_durations: List[float] = []
        self.inference_durations: List[float] = []
        self._last_loop_start: Optional[float] = None
        self._last_infer_start: Optional[float] = None

    def loop_start(self) -> None:
        # 使用单调时钟避免NTP时间同步导致的性能统计异常
        self._last_loop_start = time.monotonic()

    def loop_end(self) -> float:
        if self._last_loop_start is None:
            return 0.0
        dur = time.monotonic() - self._last_loop_start
        self.loop_durations.append(dur)
        self._last_loop_start = None
        return dur

    def infer_start(self) -> None:
        # 使用单调时钟避免NTP时间同步导致的性能统计异常
        self._last_infer_start = time.monotonic()

    def infer_end(self) -> float:
        if self._last_infer_start is None:
            return 0.0
        dur = time.monotonic() - self._last_infer_start
        self.inference_durations.append(dur)
        self._last_infer_start = None
        return dur

    def tick(self, loop_duration: float, inference_duration: float) -> None:
        # 兼容旧接口：直接传入本次的loop/infer耗时
        self.loop_durations.append(loop_duration)
        self.inference_durations.append(inference_duration)

    def save_json(self, timestr: str, output_dir: str = 'test_performance_record') -> str:
        os.makedirs(output_dir, exist_ok=True)
        data: Dict[str, Any] = {
            'loop_durations_seconds': self.loop_durations,
            'inference_durations_seconds': self.inference_durations,
        }
        path = os.path.join(output_dir, f'performance_{timestr}.json')
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
        return path


