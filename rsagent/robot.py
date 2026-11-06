# -*- coding: utf-8 -*-
"""机器人本体仿真：差速运动学积分 + 电量模型 + 到点判定。

刻意只保留两层：底盘（速度、角速度、电量）与定位（位姿）。巡检逻辑一律放在上层，
这样换真机时只需替换本文件，上层状态机不动 —— 和真机联调时的分层一致。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .mapgrid import GridMap
from .model import Pose


@dataclass
class RobotConfig:
    max_linear: float = 1.2        # m/s
    max_angular: float = 1.6       # rad/s
    accel: float = 1.0             # m/s^2
    arrive_tolerance: float = 0.15 # m
    battery_full: float = 100.0
    drain_per_m: float = 0.42      # 每米耗电百分比
    drain_per_s_idle: float = 0.004
    charge_per_s: float = 2.2      # 回充速率
    low_battery: float = 25.0      # 触发回充
    resume_battery: float = 80.0   # 充到多少继续执行


@dataclass
class Robot:
    pose: Pose
    battery: float = 100.0
    cfg: RobotConfig = field(default_factory=RobotConfig)
    linear: float = 0.0
    odometer_m: float = 0.0
    stalls: int = 0

    def __post_init__(self) -> None:
        # 允许用 (x, y, yaw) 简写构造，避免每处都 import Pose
        if not isinstance(self.pose, Pose):
            x, y, *rest = tuple(self.pose)  # type: ignore[arg-type]
            self.pose = Pose(float(x), float(y), float(rest[0]) if rest else 0.0)

    # ---------- 运动 ----------
    def step_towards(self, target: Pose, dt: float, blocked_ahead: bool = False) -> bool:
        """朝目标走一步。返回是否已到达容差内。

        blocked_ahead 由上层根据局部避障给出：前方不可走时降速并计入 stall，
        连续 stall 由上层判定为导航失败（对应真机的「卡死超时」）。
        """
        dx, dy = target.x - self.pose.x, target.y - self.pose.y
        dist = math.hypot(dx, dy)
        if dist <= self.cfg.arrive_tolerance:
            self.linear = 0.0
            self.battery = max(0.0, self.battery - self.cfg.drain_per_s_idle * dt)
            return True

        desired_yaw = math.atan2(dy, dx)
        dyaw = _norm_angle(desired_yaw - self.pose.yaw)

        # 先转向再前进：模拟差速底盘在拐角处的减速行为
        if abs(dyaw) > 0.6:
            self.linear = max(0.0, self.linear - self.cfg.accel * dt)
            w = min(self.cfg.max_angular, abs(dyaw) * 2.0) * (1 if dyaw > 0 else -1)
            self.pose = Pose(self.pose.x, self.pose.y, _norm_angle(self.pose.yaw + w * dt))
            self.battery = max(0.0, self.battery - self.cfg.drain_per_s_idle * dt)
            return False

        vmax = self.cfg.max_linear * (0.45 if abs(dyaw) > 0.25 else 1.0)
        if blocked_ahead:
            vmax *= 0.35
        vmax = min(vmax, max(0.15, dist * 2.0))     # 临近目标平滑减速
        self.linear = min(vmax, self.linear + self.cfg.accel * dt)
        travelled = self.linear * dt
        self.pose = Pose(self.pose.x + math.cos(desired_yaw) * travelled,
                         self.pose.y + math.sin(desired_yaw) * travelled,
                         _norm_angle(self.pose.yaw + dyaw * 0.5))
        self.odometer_m += travelled
        self.battery = max(0.0, self.battery - travelled * self.cfg.drain_per_m
                           - self.cfg.drain_per_s_idle * dt)
        return False

    def charge(self, dt: float) -> bool:
        """在充电桩上充电。返回是否已到恢复阈值。"""
        self.linear = 0.0
        self.battery = min(self.cfg.battery_full, self.battery + self.cfg.charge_per_s * dt)
        return self.battery >= self.cfg.resume_battery

    def need_charge(self) -> bool:
        return self.battery < self.cfg.low_battery


def _norm_angle(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def cell_of(pose: Pose) -> tuple[int, int]:
    return (int(round(pose.x)), int(round(pose.y)))


def local_blocked(gmap: GridMap, pose: Pose, target: Pose) -> bool:
    """简易局部避障：检查前进方向 1 格内是否撞障碍。真机上是代价地图，这里用栅格近似。"""
    dx, dy = target.x - pose.x, target.y - pose.y
    norm = math.hypot(dx, dy) or 1.0
    ahead = (pose.x + dx / norm * 1.0, pose.y + dy / norm * 1.0)
    return not gmap.is_free((int(round(ahead[0])), int(round(ahead[1]))))
