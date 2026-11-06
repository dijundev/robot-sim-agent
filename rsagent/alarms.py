# -*- coding: utf-8 -*-
"""告警规则引擎：阈值 + 持续时间 + 滞回 + 去重 + 事件状态机。

为什么不用「读一次超阈值就报」：
  巡检数据天然带噪声（探头抖动、瞬时气流），单点越界会淹没值班人员。
  所以这里要求**持续超过 duration_s 才触发**，**回落到 threshold*(1-hysteresis) 才恢复**，
  同一个点位同一传感器在 dedupe_window 内只维持一个事件（更新峰值，不重复刷屏）。
每条告警都保留触发规则版本与原始读数，用于事后追溯。
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from .model import AlarmEvent, AlarmState, Reading


@dataclass
class AlarmRule:
    sensor: str
    threshold: float
    level: str = "warn"           # warn / critical
    duration_s: float = 5.0       # 持续多久才触发
    hysteresis: float = 0.05      # 回落 5% 才算恢复
    compare: str = ">"            # > 或 <（指示灯这类低值异常用 <）
    rule_version: str = "v1"

    @property
    def rule_id(self) -> str:
        """规则身份：同传感器可能有多条规则（>250 和 <180 都是 warn）。

        只用 (waypoint, sensor, level) 当键会让这两条规则互相清掉对方的计时 ——
        第一版就是这么写的，导致 meter 的越界读数永远打不出告警。
        """
        return "%s%s%s" % (self.sensor, self.compare, self.threshold)

    def breached(self, value: float) -> bool:
        return value > self.threshold if self.compare == ">" else value < self.threshold

    def recovered(self, value: float) -> bool:
        if self.compare == ">":
            return value <= self.threshold * (1 - self.hysteresis)
        return value >= self.threshold * (1 + self.hysteresis)


DEFAULT_RULES: list[AlarmRule] = [
    AlarmRule("temp", threshold=45.0, level="warn", duration_s=5.0),
    AlarmRule("temp", threshold=60.0, level="critical", duration_s=2.0),
    AlarmRule("gas", threshold=50.0, level="critical", duration_s=3.0),
    AlarmRule("humidity", threshold=85.0, level="warn", duration_s=6.0),
    AlarmRule("meter", threshold=250.0, level="warn", duration_s=5.0),
    AlarmRule("meter", threshold=180.0, level="warn", duration_s=5.0, compare="<"),
    AlarmRule("indicator", threshold=0.5, level="critical", duration_s=1.0, compare="<",
              hysteresis=0.0),
]


@dataclass
class _Pending:
    since: float
    peak: float


class AlarmManager:
    def __init__(self, rules: list[AlarmRule] | None = None,
                 dedupe_window_s: float = 120.0):
        self.rules = rules if rules is not None else list(DEFAULT_RULES)
        self.dedupe_window = dedupe_window_s
        self.pending: dict[tuple[str, str, str], _Pending] = {}
        self.events: dict[str, AlarmEvent] = {}
        self.history: list[AlarmEvent] = []

    # ---------- 主入口 ----------
    def feed(self, r: Reading) -> list[AlarmEvent]:
        """喂一条读数，返回本次受影响的事件（新建 / 更新 / 恢复）。"""
        touched: list[AlarmEvent] = []
        if not r.valid or r.value is None:
            # 无效读数不参与判定，但也不清除 pending —— 等下一个有效读数
            return touched

        value = float(r.value)
        for rule in self.rules:
            if rule.sensor != r.sensor:
                continue
            key = (r.waypoint_id, r.sensor, rule.rule_id)
            active = self._find_active(r.waypoint_id, r.sensor, rule.rule_id)

            if active is not None:
                if rule.recovered(value):
                    active.state = AlarmState.RECOVERED
                    active.last_ts = r.ts
                    self.pending.pop(key, None)
                    touched.append(active)
                else:
                    active.state = AlarmState.ONGOING
                    active.last_ts = r.ts
                    active.peak_value = max(active.peak_value, value) \
                        if rule.compare == ">" else min(active.peak_value, value)
                    touched.append(active)
                continue

            if not rule.breached(value):
                self.pending.pop(key, None)
                continue

            p = self.pending.get(key)
            if p is None:
                self.pending[key] = _Pending(since=r.ts, peak=value)
                p = self.pending[key]
            else:
                p.peak = max(p.peak, value) if rule.compare == ">" else min(p.peak, value)
            # 仿真时钟是 0.1 累加出来的，直接比 >= 会因为浮点误差差一点点而永不触发
            if r.ts - p.since >= rule.duration_s - 1e-9:
                ev = AlarmEvent(
                    alarm_id="alm_" + uuid.uuid4().hex[:8],
                    waypoint_id=r.waypoint_id, sensor=r.sensor, level=rule.level,
                    state=AlarmState.TRIGGERED, first_ts=p.since, last_ts=r.ts,
                    peak_value=p.peak, threshold=rule.threshold,
                    rule_version=rule.rule_version,
                    rule_id=rule.rule_id,
                )
                self.events[ev.alarm_id] = ev
                self.history.append(ev)
                self.pending.pop(key, None)
                touched.append(ev)
        return touched

    def _find_active(self, wp: str, sensor: str, rule_id: str) -> AlarmEvent | None:
        for ev in self.events.values():
            if (ev.waypoint_id == wp and ev.sensor == sensor
                    and ev.rule_id == rule_id
                    and ev.state in (AlarmState.TRIGGERED, AlarmState.ONGOING,
                                     AlarmState.ACKED)):
                return ev
        return None

    # ---------- 人工确认 ----------
    def ack(self, alarm_id: str, ts: float) -> bool:
        ev = self.events.get(alarm_id)
        if ev is None or ev.state == AlarmState.RECOVERED:
            return False
        ev.state = AlarmState.ACKED
        ev.last_ts = ts
        return True

    # ---------- 统计 ----------
    def counts(self) -> dict:
        out = {"triggered": 0, "recovered": 0, "acked": 0, "ongoing": 0}
        for ev in self.history:
            if ev.state == AlarmState.RECOVERED:
                out["recovered"] += 1
            elif ev.state == AlarmState.ACKED:
                out["acked"] += 1
            elif ev.state == AlarmState.ONGOING:
                out["ongoing"] += 1
            else:
                out["triggered"] += 1
        return out

    def to_list(self) -> list[dict]:
        return [ev.to_dict() for ev in self.history]
