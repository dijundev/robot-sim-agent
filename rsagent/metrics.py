# -*- coding: utf-8 -*-
"""指标聚合：把一堆执行结果收敛成可汇报的数字。

口径先写死在这里，避免「同一个指标两个说法」：
  - 任务成功率 = 全部步骤 SUCCEEDED 的任务数 / 总任务数
  - 导航成功率 = SUCCEEDED 的 navigate 步 / 全部 navigate 步
  - 告警召回   = 被触发的注入故障数 / 注入故障总数
  - 告警误报   = 无注入时触发的事件数
  - 恢复成功率 = 低电量中断后成功续跑完的任务数 / 发生低电量中断的任务数
"""
from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass, field

from .action import ExecutionResult


@dataclass
class Metrics:
    tasks: int = 0
    task_success: int = 0
    nav_steps: int = 0
    nav_success: int = 0
    step_fail_reasons: Counter = field(default_factory=Counter)
    alarms_total: int = 0
    alarms_by_level: Counter = field(default_factory=Counter)
    injected_faults: int = 0
    injected_hit: int = 0
    false_alarm_events: int = 0
    low_battery_interrupts: int = 0
    low_battery_recovered: int = 0
    unknown_readings: int = 0
    total_readings: int = 0
    wall_s: list[float] = field(default_factory=list)
    odometer_m: float = 0.0
    overruns: int = 0

    def summary(self) -> dict:
        return {
            "tasks": self.tasks,
            "task_success_rate": _rate(self.task_success, self.tasks),
            "nav_steps": self.nav_steps,
            "nav_success_rate": _rate(self.nav_success, self.nav_steps),
            "alarms_total": self.alarms_total,
            "alarms_by_level": dict(self.alarms_by_level),
            "alarm_recall": _rate(self.injected_hit, self.injected_faults),
            "alarm_false_positive_events": self.false_alarm_events,
            "injected_faults": self.injected_faults,
            "low_battery_interrupts": self.low_battery_interrupts,
            "low_battery_recovery_rate": _rate(self.low_battery_recovered,
                                               self.low_battery_interrupts),
            "unknown_reading_ratio": _rate(self.unknown_readings, self.total_readings),
            "step_fail_reasons": dict(self.step_fail_reasons),
            "wall_s_p50": round(statistics.median(self.wall_s), 3) if self.wall_s else 0.0,
            "wall_s_p95": round(_p95(self.wall_s), 3) if self.wall_s else 0.0,
            "odometer_m": round(self.odometer_m, 2),
            "feedback_overruns": self.overruns,
        }


def _rate(a: int, b: int) -> float:
    return round(a / b, 4) if b else 0.0


def _p95(vals: list[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    i = min(len(s) - 1, int(round(0.95 * (len(s) - 1))))
    return s[i]


def aggregate(results: list[ExecutionResult], injected: int,
              faults_hit: set[str], false_alarm_events: int,
              readings_total: int, readings_unknown: int) -> Metrics:
    m = Metrics()
    m.tasks = len(results)
    m.injected_faults = injected
    m.injected_hit = len(faults_hit)
    m.false_alarm_events = false_alarm_events
    m.total_readings = readings_total
    m.unknown_readings = readings_unknown

    for r in results:
        if r.success:
            m.task_success += 1
        if r.recovered_from_low_battery:
            m.low_battery_interrupts += 1
            if r.success:
                m.low_battery_recovered += 1
        m.wall_s.append(r.wall_s)
        m.odometer_m += r.odometer_m
        m.overruns += r.overruns
        for s in r.steps:
            if s.step_type == "navigate":
                m.nav_steps += 1
                if s.status.value == "succeeded":
                    m.nav_success += 1
                elif s.note:
                    m.step_fail_reasons[s.note] += 1
            if s.step_type == "inspect" and s.note == "readings_unknown":
                pass
        m.alarms_total += len(r.alarms)
        for a in r.alarms:
            m.alarms_by_level[a["level"]] += 1
    return m
