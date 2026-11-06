# -*- coding: utf-8 -*-
"""传感器仿真与读数质量门槛。

关键设计：**读数先过质量门槛，再过告警规则**。质量不合格（遮挡 / 超时 / 置信度低）
的读数返回 valid=False 并落到 UNKNOWN，绝不拿一个可疑数字去参与告警判断 ——
这是巡检场景里「漏报可以补测、误报会让人失去信任」的直接体现。
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass

from .model import Reading

# 各传感器的基线、噪声、单位、物理上限
SENSOR_SPEC = {
    # 温度基线刻意压在 41.9（阈值 45）：单次采样会因为噪声偶尔越界，
    # 这就是「持续时间过滤」要挡掉的那类噪声，也是误报的主要来源
    "temp":     {"base": 41.9, "noise": 2.2, "unit": "°C",    "max": 120.0},
    "humidity": {"base": 48.0, "noise": 2.0, "unit": "%RH",   "max": 100.0},
    "gas":      {"base": 18.0, "noise": 6.0, "unit": "ppm",   "max": 999.0},
    "meter":    {"base": 220.0,"noise": 1.5, "unit": "V",     "max": 400.0},
    "indicator":{"base": 1.0,  "noise": 0.0, "unit": "state", "max": 1.0},   # 1=正常 0=异常
}


@dataclass
class FaultInjection:
    """故障注入：bench 用它制造「真实异常」，用来评测告警的召回与误报。"""

    waypoint_id: str
    sensor: str
    delta: float          # 相对基线的偏移，gas 用 ppm 绝对值偏移
    start_ts: float
    end_ts: float


class SensorSuite:
    def __init__(self, seed: int = 7, quality_gate: float = 0.06):
        self.rng = random.Random(seed)
        self.quality_gate = quality_gate
        self.faults: list[FaultInjection] = []
        self.readings: list[Reading] = []

    def inject(self, fault: FaultInjection) -> None:
        self.faults.append(fault)

    def clear_faults(self) -> None:
        self.faults.clear()

    def _fault_delta(self, wp: str, sensor: str, now: float) -> float:
        d = 0.0
        for f in self.faults:
            if f.waypoint_id == wp and f.sensor == sensor and f.start_ts <= now <= f.end_ts:
                d += f.delta
        return d

    def read(self, waypoint_id: str, sensor: str, now: float | None = None) -> Reading:
        now = time.time() if now is None else now
        spec = SENSOR_SPEC[sensor]
        quality = self.rng.random()
        # 低质量读数的典型成因：遮挡、反光、超时。这里按概率模拟
        if quality < self.quality_gate:
            r = Reading(waypoint_id=waypoint_id, sensor=sensor, value=None,
                        unit=spec["unit"], ts=now, valid=False, quality=quality,
                        note="读数质量不足，返回 UNKNOWN")
            self.readings.append(r)
            return r

        value = spec["base"] + self.rng.gauss(0, spec["noise"]) \
            + self._fault_delta(waypoint_id, sensor, now)
        if sensor == "indicator":
            value = 0.0 if value < 0.5 else 1.0
        value = max(0.0, min(spec["max"], value))
        r = Reading(waypoint_id=waypoint_id, sensor=sensor, value=round(value, 2),
                    unit=spec["unit"], ts=now, valid=True, quality=quality)
        self.readings.append(r)
        return r

    def read_all(self, waypoint_id: str, sensors: list[str], now: float | None = None) -> list[Reading]:
        return [self.read(waypoint_id, s, now) for s in sensors]

    def unknown_ratio(self) -> float:
        if not self.readings:
            return 0.0
        return sum(1 for r in self.readings if not r.valid) / len(self.readings)
