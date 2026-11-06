# -*- coding: utf-8 -*-
"""命令行：一条命令跑一轮完整任务，打印人类可读摘要。

用法：
    python -m rsagent.cli --task "去配电柜A和机柜B看温度，最后回充"
    python -m rsagent.cli --task "全点位巡检温度" --planner llm
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time

from .action import ActionExecutor, ExecConfig, ProgressStore
from .alarms import AlarmManager
from .mapgrid import load_map
from .planner import PlanRejected, plan_by_llm, plan_by_rule
from .robot import Robot, RobotConfig

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MAP = os.path.join(HERE, "config", "map_office.json")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="巡检机器人任务执行仿真")
    ap.add_argument("--task", default="去配电柜A和机柜B看温度，最后回充")
    ap.add_argument("--map", default=DEFAULT_MAP)
    ap.add_argument("--planner", choices=["rule", "llm"], default="rule")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--json", action="store_true", help="输出完整 JSON")
    args = ap.parse_args(argv)

    gmap = load_map(args.map)
    robot = Robot(pose=(float(gmap.charger_cell[0]), float(gmap.charger_cell[1]), 0.0),
                  cfg=RobotConfig())
    sensors = SensorSuiteFactory(args.seed)
    alarms = AlarmManager()
    store = ProgressStore(os.path.join(HERE, "runs", "progress.json"))
    ex = ActionExecutor(gmap, robot, sensors, alarms, store=store, cfg=ExecConfig())

    t0 = time.perf_counter()
    try:
        if args.planner == "llm":
            from .planner import fake_llm
            plan = plan_by_llm(args.task, gmap, fake_llm())
        else:
            plan = plan_by_rule(args.task, gmap)
    except PlanRejected as e:
        print("[拒单] %s" % e.reason)
        return 2
    plan_ms = (time.perf_counter() - t0) * 1000

    print("地图      : %s（%dx%d，指纹 %s）" % (gmap.name, gmap.rows, gmap.cols, gmap.digest))
    print("任务      : %s" % args.task)
    print("计划      : %d 步，来源 %s，规划耗时 %.2f ms" % (len(plan.steps), plan.source, plan_ms))
    for s in plan.steps:
        print("  %2d %-9s %-8s %s" % (s.index, s.step_type, s.waypoint_id or "-",
                                       s.sensor or ""))

    res = ex.run_plan(plan)
    print("-" * 56)
    print("结果      : %s（%s）" % (res.status.value, "成功" if res.success else "失败"))
    print("已完成步数: %d / %d" % (len(res.steps), len(plan.steps)))
    print("行驶里程  : %.2f m   末态电量 %.1f%%" % (res.odometer_m, res.battery_end))
    print("仿真耗时  : %.2f s（墙钟 %.3f s）" % (max(s.elapsed_s for s in res.steps) if res.steps else 0,
                                           res.wall_s))
    print("低电量恢复: %s" % ("是" if res.recovered_from_low_battery else "否"))
    print("告警      : %d 条 %s" % (len(res.alarms),
                                 json.dumps(alarms.counts(), ensure_ascii=False)))
    if res.cancel_reason:
        print("中断原因  : %s" % res.cancel_reason)

    if args.json:
        print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
    return 0 if res.success else 1


def SensorSuiteFactory(seed: int):
    from .sensors import SensorSuite
    return SensorSuite(seed=seed)


if __name__ == "__main__":
    raise SystemExit(main())
