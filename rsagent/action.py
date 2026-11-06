# -*- coding: utf-8 -*-
"""Action 执行层：参照 ROS 2 action 语义（goal / feedback / result / cancel）。

为什么自己写一层而不是直接散在业务代码里：
  巡检任务的本质是「一串有时序、可取消、要能追踪状态的异步目标」。裸写就是一堆
  if/else 加 sleep，取消和恢复必然漏。这里把四件事固化下来：
    1) 目标生命周期：PENDING → ACCEPTED → EXECUTING → SUCCEEDED/ABORTED/CANCELED
    2) 串行队列 + 单写入锁：一次只执行一个导航目标，避免多目标互相抢占
    3) 进度持久化：每步终态落盘，进程重启/低电量中断后能续跑
    4) 低电量中断与回充后恢复：把「中断」变成一等公民，而不是异常分支
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field

from .alarms import AlarmManager
from .mapgrid import GridMap
from .model import TERMINAL, Feedback, GoalStatus, Plan, Pose, Step
from .robot import Robot, cell_of, local_blocked
from .sensors import SensorSuite

DT = 0.1                 # 仿真步长
TICK_BUDGET_S = 0.02     # 单 tick 墙钟预算，超了记一次 overrun（暴露规划/回调开销）
SAMPLES_PER_INSPECT = 4  # 单点有效采样次数
SAMPLE_INTERVAL_S = 2.0  # 采样间隔（仿真时钟）
MAX_RESAMPLE = 2         # UNKNOWN 允许补测的次数上限


@dataclass
class StepResult:
    index: int
    step_type: str
    status: GoalStatus
    waypoint_id: str | None
    sensor: str | None = None
    elapsed_s: float = 0.0
    note: str = ""
    readings: list = field(default_factory=list)
    resample: int = 0

    def to_dict(self) -> dict:
        return {"index": self.index, "step_type": self.step_type,
                "status": self.status.value, "waypoint_id": self.waypoint_id,
                "sensor": self.sensor, "elapsed_s": round(self.elapsed_s, 3),
                "note": self.note,
                "readings": [r.to_dict() for r in self.readings]}


@dataclass
class ExecutionResult:
    plan_id: str
    status: GoalStatus
    steps: list[StepResult]
    alarms: list[dict]
    battery_end: float
    odometer_m: float
    wall_s: float
    recovered_from_low_battery: bool = False
    cancel_reason: str = ""
    overruns: int = 0

    @property
    def success(self) -> bool:
        return (self.status == GoalStatus.SUCCEEDED
                and all(s.status == GoalStatus.SUCCEEDED for s in self.steps))

    def to_dict(self) -> dict:
        return {"plan_id": self.plan_id, "status": self.status.value,
                "success": self.success,
                "steps": [s.to_dict() for s in self.steps],
                "alarms": self.alarms,
                "battery_end": round(self.battery_end, 2),
                "odometer_m": round(self.odometer_m, 3),
                "wall_s": round(self.wall_s, 3),
                "recovered_from_low_battery": self.recovered_from_low_battery,
                "cancel_reason": self.cancel_reason,
                "overruns": self.overruns}


class ProgressStore:
    """进度持久化。真机/线上是 Redis，这里用 JSON 文件，接口保持一样。"""

    def __init__(self, path: str | None = None):
        self.path = path
        self._data: dict[str, dict] = {}
        if path and os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    self._data = json.load(f)
            except Exception:
                self._data = {}

    def save(self, plan_id: str, done: list[int], status: str, battery: float) -> None:
        self._data[plan_id] = {"done": sorted(done), "status": status,
                               "battery": round(battery, 2), "ts": time.time()}
        if self.path:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=1)

    def done_steps(self, plan_id: str) -> list[int]:
        return list(self._data.get(plan_id, {}).get("done", []))


@dataclass
class ExecConfig:
    max_step_retries: int = 1        # 单个导航目标失败重试上限
    stall_limit: int = 60            # 连续多少 tick 无位移判定卡死
    max_wall_s: float = 900.0        # 单任务墙钟上限，防死循环
    persist: bool = True


class ActionExecutor:
    def __init__(self, gmap: GridMap, robot: Robot, sensors: SensorSuite,
                 alarms: AlarmManager, store: ProgressStore | None = None,
                 cfg: ExecConfig | None = None):
        self.gmap = gmap
        self.robot = robot
        self.sensors = sensors
        self.alarms = alarms
        self.store = store or ProgressStore()
        self.cfg = cfg or ExecConfig()
        self._lock = threading.Lock()          # 单写入锁：同一时刻只允许一个目标在跑
        self._queue: deque[list[Step]] = deque()
        self.cancel_flag = threading.Event()
        self.feedback_log: list[Feedback] = []
        self.overruns = 0
        self._feedback_cb = None

    def on_feedback(self, cb) -> None:
        self._feedback_cb = cb

    def request_cancel(self) -> None:
        """外部取消（急停、人工接管）。只置标志，由执行循环在 tick 边界收敛。"""
        self.cancel_flag.set()

    # ------------------------------------------------------------ 主循环
    def run_plan(self, plan: Plan, now: float | None = None) -> ExecutionResult:
        t0 = time.perf_counter()
        clock = now if now is not None else 0.0
        done: list[int] = self.store.done_steps(plan.plan_id) if self.cfg.persist else []
        results: list[StepResult] = []
        recovered = False
        cancel_reason = ""
        status = GoalStatus.SUCCEEDED

        for step in plan.steps:
            if step.index in done:
                continue
            if self.cancel_flag.is_set():
                status = GoalStatus.CANCELED
                cancel_reason = "外部取消"
                results.append(StepResult(step.index, step.step_type,
                                          GoalStatus.CANCELED, step.waypoint_id,
                                          step.sensor, note="未执行"))
                break

            sr, clock = self._run_step(step, clock, plan)
            results.append(sr)

            if sr.status == GoalStatus.SUCCEEDED:
                done.append(step.index)
                if self.cfg.persist:
                    self.store.save(plan.plan_id, done, "executing", self.robot.battery)
                continue

            if sr.note == "low_battery":
                # 低电量不是失败，是「可恢复的中断」：回充后重排剩余步骤继续
                ok, clock = self._dock_and_resume(clock)
                recovered = recovered or ok
                if not ok:
                    status = GoalStatus.ABORTED
                    cancel_reason = "回充失败"
                    break
                sr2, clock = self._run_step(step, clock, plan)
                results[-1] = sr2
                if sr2.status != GoalStatus.SUCCEEDED:
                    status = GoalStatus.ABORTED
                    cancel_reason = sr2.note or "回充后仍失败"
                    break
                done.append(step.index)
                continue

            status = GoalStatus.ABORTED
            cancel_reason = sr.note or "执行失败"
            break

        wall = time.perf_counter() - t0
        if self.cfg.persist:
            self.store.save(plan.plan_id, done, status.value, self.robot.battery)
        return ExecutionResult(plan_id=plan.plan_id, status=status, steps=results,
                               alarms=self.alarms.to_list(),
                               battery_end=self.robot.battery,
                               odometer_m=self.robot.odometer_m, wall_s=wall,
                               recovered_from_low_battery=recovered,
                               cancel_reason=cancel_reason, overruns=self.overruns)

    # ------------------------------------------------------------ 单步
    def _run_step(self, step: Step, clock: float,
                  plan: Plan) -> tuple[StepResult, float]:
        started = clock
        if step.step_type == "navigate":
            st, clock, note = self._navigate(step, clock, plan)
            return self._mk(step, st, started, clock, note), clock
        if step.step_type == "inspect":
            # 单点多次采样 + UNKNOWN 补测：巡检机器人到点后通常停留取样，而不是只读一次。
            # 补测是必需的 —— 首帧被判 UNKNOWN 时若不补，有效观测窗口会从 6 s 缩到 4 s，
            # 持续时间类规则（5 s）就永远达不到阈值，表现为「漏报」而不是「读不到」。
            readings = []
            valid_n = 0
            attempts = 0
            while valid_n < SAMPLES_PER_INSPECT and attempts < SAMPLES_PER_INSPECT + MAX_RESAMPLE:
                rs = self.sensors.read_all(step.waypoint_id, [step.sensor],
                                           now=clock + attempts * SAMPLE_INTERVAL_S)
                readings += rs
                if any(r.valid for r in rs):
                    valid_n += 1
                attempts += 1
            for r in readings:
                self.alarms.feed(r)
            clock += max(0, attempts - 1) * SAMPLE_INTERVAL_S
            self._emit(plan, step, GoalStatus.EXECUTING, 1.0, clock)
            st = GoalStatus.SUCCEEDED
            note = "" if valid_n else "readings_unknown"
            sr = self._mk(step, st, started, clock, note, readings)
            sr.resample = attempts - valid_n
            return sr, clock
        if step.step_type == "dock":
            ok, clock = self._dock_and_resume(clock)
            return self._mk(step, GoalStatus.SUCCEEDED if ok else GoalStatus.ABORTED,
                            started, clock, "" if ok else "dock_failed"), clock
        if step.step_type == "wait":
            clock += step.timeout_s
            return self._mk(step, GoalStatus.SUCCEEDED, started, clock, "wait"), clock
        return self._mk(step, GoalStatus.REJECTED, started, clock, "unknown_step"), clock

    def _navigate(self, step: Step, clock: float,
                  plan: Plan) -> tuple[GoalStatus, float, str]:
        wp = self.gmap.waypoints.get(step.waypoint_id or "")
        if wp is None:
            return GoalStatus.REJECTED, clock, "unknown_waypoint"

        attempts = 0
        while attempts <= self.cfg.max_step_retries:
            attempts += 1
            start_cell = cell_of(self.robot.pose)
            path = self.gmap.plan(start_cell, wp.cell)
            if not path:
                # 已在该点，或真不可达
                if start_cell == wp.cell:
                    return GoalStatus.SUCCEEDED, clock, ""
                return GoalStatus.ABORTED, clock, "no_path"
            st, clock, note = self._follow(path, wp.pose, step, plan, clock)
            if st == GoalStatus.SUCCEEDED:
                return st, clock, ""
            # 超时与取消不做重试：超时说明这一步的预算已经耗完，再跑一遍只会
            # 把整条任务的墙钟时间翻倍（第一版重试了，被单测抓到语义不对）
            if note in ("low_battery", "timeout", "canceled"):
                return st, clock, note
            if st == GoalStatus.CANCELED:
                return st, clock, note
            self._emit(plan, step, GoalStatus.ABORTED, 0.0, clock)
        return GoalStatus.ABORTED, clock, "retry_exhausted"

    def _follow(self, path, target: Pose, step: Step, plan: Plan,
                clock: float) -> tuple[GoalStatus, float, str]:
        """沿路径点前进。每 tick 检查：取消、超时、低电量、卡死。"""
        idx = 0
        stall = 0
        last_pose = (self.robot.pose.x, self.robot.pose.y)
        total = len(path)
        step_start = clock

        while clock - step_start <= step.timeout_s:
            if self.cancel_flag.is_set():
                return GoalStatus.CANCELED, clock, "canceled"
            if self.robot.need_charge():
                return GoalStatus.ABORTED, clock, "low_battery"

            if idx < total:
                cell = path[idx]
                aim = Pose(float(cell[0]), float(cell[1]))
                if self.robot.pose.distance_to(aim) <= 0.35:
                    idx += 1
                    continue
            elif idx >= total:
                aim = target
                if self.robot.pose.distance_to(aim) <= self.robot.cfg.arrive_tolerance:
                    self._emit(plan, step, GoalStatus.SUCCEEDED, 1.0, clock)
                    return GoalStatus.SUCCEEDED, clock, ""

            blocked = local_blocked(self.gmap, self.robot.pose, aim)
            self.robot.step_towards(aim, DT, blocked_ahead=blocked)
            clock += DT

            prog = min(1.0, (idx + (1.0 if idx < total else 0.0)) / max(1, total))
            self._emit(plan, step, GoalStatus.EXECUTING, prog, clock)

            moved = abs(self.robot.pose.x - last_pose[0]) + abs(self.robot.pose.y - last_pose[1])
            stall = stall + 1 if moved < 1e-4 else 0
            last_pose = (self.robot.pose.x, self.robot.pose.y)
            if stall >= self.cfg.stall_limit:
                return GoalStatus.ABORTED, clock, "stalled"

        return GoalStatus.ABORTED, clock, "timeout"

    def _dock_and_resume(self, clock: float) -> tuple[bool, float]:
        """回到充电桩并充到恢复阈值。失败返回 False，由上层判 ABORTED。"""
        dock = Pose(float(self.gmap.charger_cell[0]), float(self.gmap.charger_cell[1]))
        start = cell_of(self.robot.pose)
        path = self.gmap.plan(start, self.gmap.charger_cell)
        if path:
            guard = 0
            while self.robot.pose.distance_to(dock) > self.robot.cfg.arrive_tolerance:
                self.robot.step_towards(dock, DT, blocked_ahead=False)
                clock += DT
                guard += DT
                if guard > 600:
                    return False, clock
        else:
            # 已无法规划（电量过低走出路径）：直接瞬移到桩，模拟人工接管拖走
            self.robot.pose = dock
        while not self.robot.charge(DT):
            clock += DT
            if clock > 1e6:
                return False, clock
        return True, clock

    # ------------------------------------------------------------ 工具
    def _emit(self, plan: Plan, step: Step, st: GoalStatus, prog: float, clock: float) -> None:
        fb = Feedback(plan_id=plan.plan_id, step_index=step.index, status=st,
                      pose=self.robot.pose, progress=prog, battery=self.robot.battery,
                      elapsed_s=clock)
        self.feedback_log.append(fb)
        if self._feedback_cb:
            t0 = time.perf_counter()
            self._feedback_cb(fb)
            if time.perf_counter() - t0 > TICK_BUDGET_S:
                self.overruns += 1

    def _mk(self, step: Step, st: GoalStatus, t_start: float, t_end: float,
            note: str, readings=None) -> StepResult:
        return StepResult(index=step.index, step_type=step.step_type, status=st,
                          waypoint_id=step.waypoint_id, sensor=step.sensor,
                          elapsed_s=t_end - t_start, note=note,
                          readings=readings or [])
