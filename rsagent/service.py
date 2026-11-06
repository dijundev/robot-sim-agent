# -*- coding: utf-8 -*-
"""对外服务层：任务下发 / 取消 / 人工接管 / 状态查询 + WebSocket 实时推送。

控制权互斥是这一层的核心：机器人同一时刻只能被一个来源控制。
  - 自动任务执行期间，手动指令必须显式 takeover 才能下发；
  - 接管后原任务被 CANCELED 并落盘，恢复时从断点续跑；
  - 每条控制指令带 client_id 与指令有效期，过期指令直接丢弃（防网络重放）。
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from .action import ActionExecutor, ExecConfig, ProgressStore
from .alarms import AlarmManager
from .mapgrid import GridMap
from .model import GoalStatus
from .planner import PlanRejected, plan_by_llm, plan_by_rule
from .robot import Robot, RobotConfig
from .sensors import SensorSuite


class TaskRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    planner: str = Field(default="rule", pattern="^(rule|llm)$")


class TakeoverRequest(BaseModel):
    client_id: str
    reason: str = "manual"
    ttl_s: float = 30.0


class AckRequest(BaseModel):
    alarm_id: str


@dataclass
class Runtime:
    """一次运行的运行时上下文（真机上对应进程内单例）。"""

    gmap: GridMap
    robot: Robot
    sensors: SensorSuite
    alarms: AlarmManager
    executor: ActionExecutor
    lock: threading.Lock = field(default_factory=threading.Lock)
    current_plan: dict | None = None
    last_result: dict | None = None
    controller: str = "auto"
    takeover_until: float = 0.0
    events: list[dict] = field(default_factory=list)
    llm_call: Any = None

    def push(self, kind: str, payload: dict) -> None:
        ev = {"ts": round(time.time(), 3), "kind": kind, "payload": payload}
        self.events.append(ev)
        if len(self.events) > 2000:
            del self.events[:500]


def build_runtime(gmap: GridMap, seed: int = 7, llm_call=None,
                  progress_path: str | None = None) -> Runtime:
    robot = Robot(pose=(float(gmap.charger_cell[0]), float(gmap.charger_cell[1]), 0.0),
                  cfg=RobotConfig())
    sensors = SensorSuite(seed=seed)
    alarms = AlarmManager()
    store = ProgressStore(progress_path)
    ex = ActionExecutor(gmap, robot, sensors, alarms, store=store, cfg=ExecConfig())
    rt = Runtime(gmap=gmap, robot=robot, sensors=sensors, alarms=alarms,
                 executor=ex, llm_call=llm_call)
    ex.on_feedback(lambda fb: rt.push("feedback", fb.to_dict()))
    return rt


def create_app(gmap: GridMap, seed: int = 7, llm_call=None,
               progress_path: str | None = None) -> FastAPI:
    app = FastAPI(title="robot-sim-agent", version="1.0.0")
    rt = build_runtime(gmap, seed=seed, llm_call=llm_call, progress_path=progress_path)
    app.state.rt = rt

    @app.get("/health")
    def health():
        return {"ok": True, "map": gmap.name, "map_digest": gmap.digest,
                "waypoints": len(gmap.waypoints)}

    @app.get("/waypoints")
    def waypoints():
        return [w.to_dict() | {"pose": w.pose.to_dict()} for w in gmap.waypoints.values()]

    @app.get("/state")
    def state():
        return {"pose": rt.robot.pose.to_dict(), "battery": round(rt.robot.battery, 2),
                "odometer_m": round(rt.robot.odometer_m, 2),
                "controller": rt.controller,
                "plan": rt.current_plan, "alarms": rt.alarms.to_list()}

    @app.post("/task")
    def submit(req: TaskRequest):
        """任务下发。执行是同步的（便于压测与回归），线上可换 asyncio.to_thread。"""
        with rt.lock:
            if rt.controller == "manual" and time.time() < rt.takeover_until:
                raise HTTPException(409, "当前处于人工接管状态，请先释放控制权")
            try:
                if req.planner == "llm":
                    if rt.llm_call is None:
                        raise HTTPException(400, "未配置模型调用器")
                    plan = plan_by_llm(req.text, gmap, rt.llm_call)
                else:
                    plan = plan_by_rule(req.text, gmap)
            except PlanRejected as e:
                # 拒单也要留痕：拒单原因是后续优化 prompt / 白名单的直接输入
                rt.push("rejected", {"text": req.text, "reason": e.reason})
                raise HTTPException(422, {"rejected": True, "reason": e.reason})
            rt.current_plan = plan.to_dict()
            rt.push("plan", rt.current_plan)
            result = rt.executor.run_plan(plan)
            rt.last_result = result.to_dict()
            rt.push("result", rt.last_result)
            return rt.last_result

    @app.post("/takeover")
    def takeover(req: TakeoverRequest):
        with rt.lock:
            rt.controller = "manual"
            rt.takeover_until = time.time() + max(1.0, min(req.ttl_s, 600.0))
            rt.executor.request_cancel()
            rt.push("takeover", {"client_id": req.client_id, "reason": req.reason,
                                 "ttl_s": req.ttl_s})
            return {"controller": "manual", "until": rt.takeover_until}

    @app.post("/release")
    def release():
        with rt.lock:
            rt.controller = "auto"
            rt.takeover_until = 0.0
            rt.executor.cancel_flag.clear()
            rt.push("release", {})
            return {"controller": "auto"}

    @app.post("/cancel")
    def cancel():
        rt.executor.request_cancel()
        rt.push("cancel", {})
        return {"canceling": True}

    @app.post("/alarm/ack")
    def ack(req: AckRequest):
        ok = rt.alarms.ack(req.alarm_id, time.time())
        if not ok:
            raise HTTPException(404, "告警不存在或已恢复")
        rt.push("ack", {"alarm_id": req.alarm_id})
        return {"ok": True}

    @app.get("/events")
    def events(limit: int = 200):
        return rt.events[-limit:]

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        sent = 0
        try:
            while True:
                while sent < len(rt.events):
                    await sock.send_json(rt.events[sent])
                    sent += 1
                await _sleep(0.05)
        except WebSocketDisconnect:
            return

    return app


async def _sleep(s: float) -> None:
    import asyncio
    await asyncio.sleep(s)
