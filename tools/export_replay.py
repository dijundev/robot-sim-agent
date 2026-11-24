# -*- coding: utf-8 -*-
"""导出一段巡检执行的完整轨迹，供 web/replay.html 离线回放。

用法：
    python tools/export_replay.py                      # 默认剧本
    python tools/export_replay.py --battery 26 --seed 7
    python tools/export_replay.py --dry                # 只打印计划，不跑

产出：web/replay_data.json（位姿/电量逐 tick、告警、读数、步骤终态）。
剧本刻意制造三件事，回放页面才能看出「执行层到底做了什么」：
    1) 两个点位注入故障 → 告警按「持续时间 + 滞回」触发/恢复
    2) 起始电量压低 → 导航中途触发低电量中断 → 回桩充电 → 回到原步骤续跑
    3) 采样中混入质量不合格的读数 → UNKNOWN 补测
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from rsagent.action import ActionExecutor, ExecConfig, ProgressStore   # noqa: E402
from rsagent.alarms import AlarmManager                                 # noqa: E402
from rsagent.mapgrid import load_map                                    # noqa: E402
from rsagent.planner import plan_by_rule                                # noqa: E402
from rsagent.robot import Robot, RobotConfig                            # noqa: E402
from rsagent.sensors import FaultInjection, SensorSuite                 # noqa: E402

DEFAULT_MAP = os.path.join(ROOT, "config", "map_office.json")
DEFAULT_TASK = "先去配电柜A看温度，再去机柜B看温度和指示灯，然后去水泵房看湿度和气体，最后绕末端管廊看温度，回充"


def build_faults() -> list[FaultInjection]:
    """注入三类真实异常。窗口给足整个仿真时段，点位/传感器各自独立生效。"""
    far = (0.0, 9_999.0)
    return [
        # 机柜B 温度 +20 → 基线 41.9 越过 45（warn）和 60（critical）两条规则
        FaultInjection("wp_b2", "temp", 20.0, *far),
        # 机柜B 指示灯熄灭 → indicator < 0.5 的 critical 规则
        FaultInjection("wp_b2", "indicator", -1.0, *far),
        # 水泵房 气体 +45 ppm → 越过 50 ppm 的 critical 规则
        FaultInjection("wp_c3", "gas", 45.0, *far),
    ]


def derive_events(trace: list[dict], steps: list[dict], alarms: list[dict],
                  names: dict[str, str]) -> list[dict]:
    """把逐 tick 轨迹压成人类能读的事件流（页面右侧日志用）。"""
    ev: list[dict] = []
    seen_step: set[int] = set()
    for p in trace:
        si = p["s"]
        if si >= 0 and si not in seen_step:
            seen_step.add(si)
            ev.append({"t": round(p["t"], 2), "kind": "step", "step": si,
                       "text": "第 %d 步开始" % (si + 1)})

    # 回充段：s=-1 的连续区间。后面还有 s>=0 的＝低电量中断；最后一段＝收尾回充
    runs: list[dict] = []
    for i, p in enumerate(trace):
        if p["s"] == -1:
            if i == 0 or trace[i - 1]["s"] != -1:
                prev_s = trace[i - 1]["s"] if i else -1
                runs.append({"t0": p["t"], "t1": p["t"], "step": prev_s, "b0": p["b"]})
            runs[-1]["t1"] = p["t"]
            runs[-1]["btop"] = p["b"]
    for k, r in enumerate(runs):
        last = k == len(runs) - 1
        if not last:
            ev.append({"t": round(r["t0"], 2), "kind": "abort", "step": r["step"],
                       "text": "第 %d 步中断：电量 %.1f%% 触及低电量阈值（25%%），转回充"
                               % (r["step"] + 1, r["b0"])})
            ev.append({"t": round(r["t1"], 2), "kind": "charge",
                       "text": "回充至 %.0f%%，回到第 %d 步续跑（中断 %.1fs）"
                               % (r["btop"], r["step"] + 1, r["t1"] - r["t0"])})
        else:
            ev.append({"t": round(r["t0"], 2), "kind": "dock",
                       "text": "任务收尾：回桩充电（%.1fs）" % (r["t1"] - r["t0"])})

    for a in alarms:
        ev.append({"t": round(a["first_ts"], 2), "kind": "alarm",
                   "text": "%s 触发：%s %s（峰值 %.1f，阈值 %.0f）"
                           % (names.get(a["waypoint_id"], a["waypoint_id"]), a["sensor"],
                              a["level"], a["peak_value"], a["threshold"])})
    for s in steps:
        if s["status"] != "succeeded":
            ev.append({"t": 0.0, "kind": "fail",
                       "text": "第 %d 步 %s：%s" % (s["index"] + 1, s["status"], s["note"])})
    ev.sort(key=lambda e: e["t"])
    return ev


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="导出巡检回放数据")
    ap.add_argument("--map", default=DEFAULT_MAP)
    ap.add_argument("--task", default=DEFAULT_TASK)
    ap.add_argument("--battery", type=float, default=26.0, help="起始电量 %%")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(ROOT, "web", "replay_data.json"))
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args(argv)

    gmap = load_map(args.map)
    robot = Robot(pose=(float(gmap.charger_cell[0]), float(gmap.charger_cell[1]), 0.0),
                  cfg=RobotConfig(), battery=args.battery)
    sensors = SensorSuite(seed=args.seed)
    for f in build_faults():
        sensors.inject(f)
    alarms = AlarmManager()
    ex = ActionExecutor(gmap, robot, sensors, alarms,
                        store=ProgressStore(None),
                        cfg=ExecConfig(persist=False))

    try:
        plan = plan_by_rule(args.task, gmap)
    except Exception as e:                     # PlanRejected
        print("[拒单] %s" % e)
        return 2

    print("地图 %s（%dx%d，指纹 %s）" % (gmap.name, gmap.rows, gmap.cols, gmap.digest))
    print("任务 %s" % args.task)
    print("计划 %d 步：" % len(plan.steps))
    for s in plan.steps:
        print("  %2d %-9s %-8s %s" % (s.index, s.step_type, s.waypoint_id or "-", s.sensor or ""))
    if args.dry:
        return 0

    trace: list[dict] = []

    def collect(fb) -> None:
        trace.append({"t": round(fb.elapsed_s, 2),
                      "x": round(fb.pose.x, 2), "y": round(fb.pose.y, 2),
                      "yaw": round(fb.pose.yaw, 3),
                      "b": round(fb.battery, 1),
                      "s": fb.step_index,
                      "st": fb.status.value,
                      "p": round(fb.progress, 2)})

    ex.on_trace(collect)
    t0 = time.perf_counter()
    res = ex.run_plan(plan)
    wall = time.perf_counter() - t0

    steps = [s.to_dict() for s in res.steps]
    alarm_list = [a for a in res.alarms] if isinstance(res.alarms, list) else []
    payload = {
        "meta": {
            "map": gmap.name, "map_digest": gmap.digest,
            "task": args.task, "plan_id": plan.plan_id, "plan_source": plan.source,
            "result": res.status.value, "success": res.success,
            "sim_s": round(max([p["t"] for p in trace] + [0.0]), 2),
            "odometer_m": round(res.odometer_m, 2),
            "battery_start": args.battery,
            "battery_end": round(res.battery_end, 1),
            "recovered_from_low_battery": res.recovered_from_low_battery,
            "alarm_count": len(alarm_list),
            "step_count": len(plan.steps),
            "tick_count": len(trace),
            "wall_ms": round(wall * 1000, 1),
            "seed": args.seed,
            "generated": time.strftime("%Y-%m-%d %H:%M"),
        },
        "map": {
            "rows": gmap.rows, "cols": gmap.cols, "resolution": gmap.resolution,
            "grid": gmap.grid_rows,
            "charger": {"id": "dock_01", "name": gmap.charger_name,
                        "cell": list(gmap.charger_cell)},
            "waypoints": [w.to_dict() for w in gmap.waypoints.values()],
        },
        "plan": {"steps": [s.to_dict() for s in plan.steps]},
        "trace": trace,
        "steps": steps,
        "alarms": alarm_list,
        "events": derive_events(trace, steps, alarm_list,
                                 {w.id: w.name for w in gmap.waypoints.values()}),
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(raw)

    # 把数据内联进页面：单文件即可打开，不依赖 fetch / 本地服务器
    tpl = os.path.join(ROOT, "web", "replay_template.html")
    page = os.path.join(ROOT, "web", "replay.html")
    with open(tpl, encoding="utf-8") as f:
        html = f.read()
    if "__DATA__" not in html:
        print("[警告] 模板里没有 __DATA__ 占位符，页面未重建")
    else:
        with open(page, "w", encoding="utf-8") as f:
            f.write(html.replace("__DATA__", raw))

    print("-" * 56)
    print("结果 %s / 里程 %.2f m / 电量 %.1f%% → %.1f%% / 告警 %d"
          % (res.status.value, res.odometer_m, args.battery, res.battery_end, len(alarm_list)))
    print("仿真 %.1fs，%d 个 tick，墙钟 %.0f ms；低电量恢复：%s"
          % (payload["meta"]["sim_s"], len(trace), wall * 1000,
             res.recovered_from_low_battery))
    print("写出 %s（%.1f KB）" % (args.out, os.path.getsize(args.out) / 1024))
    if os.path.exists(page):
        print("写出 %s（%.1f KB）" % (page, os.path.getsize(page) / 1024))
    return 0 if res.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
