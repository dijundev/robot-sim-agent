# -*- coding: utf-8 -*-
"""领域模型：位姿、动作状态、巡检计划、事件。全部不可变 dataclass，便于回放与断言。"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class GoalStatus(str, Enum):
    """对齐 ROS 2 action_msgs/GoalStatus 的语义子集。"""

    PENDING = "pending"
    ACCEPTED = "accepted"
    EXECUTING = "executing"
    SUCCEEDED = "succeeded"
    ABORTED = "aborted"
    CANCELED = "canceled"
    REJECTED = "rejected"


TERMINAL = {GoalStatus.SUCCEEDED, GoalStatus.ABORTED, GoalStatus.CANCELED, GoalStatus.REJECTED}


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    yaw: float = 0.0

    def distance_to(self, other: "Pose") -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    def to_dict(self) -> dict:
        return {"x": round(self.x, 3), "y": round(self.y, 3), "yaw": round(self.yaw, 3)}


@dataclass
class Step:
    """巡检计划中的一步。step_type 只允许声明过的取值 —— 白名单在执行前再校验一次。"""

    index: int
    step_type: str          # navigate / inspect / dock / wait
    waypoint_id: str | None = None
    sensor: str | None = None
    timeout_s: float = 60.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Plan:
    """模型产出物的落地形态：结构化、可校验、可回放。"""

    plan_id: str
    task_text: str
    map_name: str
    map_digest: str
    steps: list[Step] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    source: str = "rule"     # rule / llm
    reject_reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "task_text": self.task_text,
            "map_name": self.map_name,
            "map_digest": self.map_digest,
            "steps": [s.to_dict() for s in self.steps],
            "created_at": self.created_at,
            "source": self.source,
            "reject_reason": self.reject_reason,
        }


@dataclass
class Feedback:
    """与 ROS 2 action feedback 对齐：执行过程中的高频状态回报。"""

    plan_id: str
    step_index: int
    status: GoalStatus
    pose: Pose
    progress: float          # 0~1，本步完成度
    battery: float
    elapsed_s: float

    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "step_index": self.step_index,
            "status": self.status.value,
            "pose": self.pose.to_dict(),
            "progress": round(self.progress, 3),
            "battery": round(self.battery, 2),
            "elapsed_s": round(self.elapsed_s, 2),
        }


@dataclass
class Reading:
    """一次传感器读数。valid=False 表示被质量门槛拦下（例如遮挡、超时、置信度低）。"""

    waypoint_id: str
    sensor: str
    value: Any
    unit: str
    ts: float
    valid: bool = True
    quality: float = 1.0
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "waypoint_id": self.waypoint_id,
            "sensor": self.sensor,
            "value": self.value,
            "unit": self.unit,
            "ts": round(self.ts, 3),
            "valid": self.valid,
            "quality": round(self.quality, 3),
            "note": self.note,
        }


class AlarmState(str, Enum):
    TRIGGERED = "triggered"
    ONGOING = "ongoing"
    RECOVERED = "recovered"
    ACKED = "acked"


@dataclass
class AlarmEvent:
    alarm_id: str
    waypoint_id: str
    sensor: str
    level: str               # warn / critical
    state: AlarmState
    first_ts: float
    last_ts: float
    peak_value: float
    threshold: float
    rule_version: str = "v1"
    rule_id: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["state"] = self.state.value
        d["first_ts"] = round(self.first_ts, 3)
        d["last_ts"] = round(self.last_ts, 3)
        d["peak_value"] = round(self.peak_value, 3)
        return d
