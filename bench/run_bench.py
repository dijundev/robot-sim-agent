# -*- coding: utf-8 -*-
"""批量评测：跑 N 轮任务 + 拒单用例，产出可汇报的指标。

用法：
    python bench/run_bench.py            # 跑默认规模
    python bench/run_bench.py --rounds 60

输出：
    bench/out/metrics.json   全量指标
    bench/out/summary.md     人读摘要（简历/汇报用）

口径说明（和 rsagent/metrics.py 保持一致，不要另立一套）：
  - 告警召回：注入的故障里，至少有 1 条告警落在同一点位同一传感器上的比例
  - 误报：没有注入故障的轮次里，仍然触发的事件数
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from rsagent.action import ActionExecutor, ExecConfig, ProgressStore  # noqa: E402
from rsagent.alarms import AlarmManager  # noqa: E402
from rsagent.mapgrid import load_map  # noqa: E402
from rsagent.metrics import aggregate, _p95  # noqa: E402
from rsagent.planner import (PlanRejected, plan_by_llm, plan_by_rule,  # noqa: E402
                             validate_plan)
from rsagent.robot import Robot, RobotConfig  # noqa: E402
from rsagent.sensors import FaultInjection, SensorSuite  # noqa: E402

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

MAP = os.path.join(ROOT, "config", "map_office.json")
OUT = os.path.join(HERE, "out")

TASKS = [
    "巡检配电柜A的温度，最后回充",
    "去配电柜A和机柜B看温度",
    "巡检水泵房的气体浓度和湿度",
    "巡检气体阀组的气体，再去仪表盘看电压",
    "全点位巡检温度",
    "巡检末端管廊的温度、湿度和气体，最后回充",
]

# 拒单用例：应当 100% 被拦住。两类来源——
#   (a) 自然语言层：用户说了地图里没有的地方；
#   (b) 计划层：模型产出的 JSON 里有越权动作 / 未知点位 / 点位与检测错配 / 地图过期
REJECT_TEXT_CASES = [
    "去天台看看设备",
    "巡检不存在的机柜",
    "去充电桩旁边的配电柜Z看温度",
]


def bad_payloads(gmap):
    good_nav = {"step_type": "navigate", "waypoint_id": "wp_a1"}
    return [
        ("未声明动作类型", {"task_text": "t", "map_name": gmap.name, "steps":
                          [{"step_type": "open_door", "waypoint_id": "wp_a1"}]}),
        ("未知点位", {"task_text": "t", "map_name": gmap.name, "steps":
                     [{"step_type": "navigate", "waypoint_id": "wp_ghost"}]}),
        ("点位与检测错配", {"task_text": "t", "map_name": gmap.name, "steps":
                          [good_nav, {"step_type": "inspect", "waypoint_id": "wp_b2",
                                      "sensor": "temp"}]}),
        ("该点位没有该传感器", {"task_text": "t", "map_name": gmap.name, "steps":
                             [good_nav, {"step_type": "inspect", "waypoint_id": "wp_a1",
                                         "sensor": "gas"}]}),
        ("地图名不一致", {"task_text": "t", "map_name": "another_floor", "steps": [good_nav]}),
        ("步数超限", {"task_text": "t", "map_name": gmap.name, "steps":
                    [good_nav, {"step_type": "inspect", "waypoint_id": "wp_a1",
                                "sensor": "temp"}] * 20}),
        ("超时参数越界", {"task_text": "t", "map_name": gmap.name, "steps":
                        [{"step_type": "navigate", "waypoint_id": "wp_a1",
                          "timeout_s": 100000}]}),
    ]


def make_exec(gmap, battery: float, seed: int, store_path: str):
    robot = Robot(pose=(float(gmap.charger_cell[0]), float(gmap.charger_cell[1]), 0.0),
                  battery=battery, cfg=RobotConfig())
    sensors = SensorSuite(seed=seed)
    alarms = AlarmManager()
    ex = ActionExecutor(gmap, robot, sensors, alarms,
                        store=ProgressStore(store_path), cfg=ExecConfig())
    return ex, sensors, alarms


def pick_faults(gmap, plan, rng: random.Random, n: int) -> list[FaultInjection]:
    """只在「本轮真的会采样到」的 (点位, 传感器) 上注入故障。

    否则召回率会被拉低到无意义 —— 注入了压根没有采到的组合，不算漏报。
    """
    targets = [(s.waypoint_id, s.sensor) for s in plan.steps if s.step_type == "inspect"]
    if not targets:
        return []
    faults = []
    for _ in range(n):
        wid, sensor = rng.choice(targets)
        wp = gmap.waypoints[wid]
        # 幅度要能穿透阈值：温度至少 +6（45 阈值），气体至少 +45（50 阈值）
        if sensor == "temp":
            delta = rng.uniform(6.5, 14.0)
        elif sensor == "gas":
            delta = rng.uniform(45.0, 90.0)
        elif sensor == "humidity":
            delta = rng.uniform(40.0, 48.0)
        elif sensor == "meter":
            delta = rng.choice([rng.uniform(35.0, 55.0), -rng.uniform(45.0, 60.0)])
        else:
            delta = -1.0
        faults.append(FaultInjection(wp.id, sensor, delta, 0.0, 1e9))
    return faults


CORRIDOR_CELLS = [(i, 2) for i in range(4, 13)] + [(i, 8) for i in range(4, 13)] \
    + [(i, 14) for i in range(4, 13)] + [(5, j) for j in range(4, 19)] + [(11, j) for j in range(4, 19)]


def run_round(gmap, task: str, seed: int, with_fault: bool) -> tuple:
    rng = random.Random(seed)
    # 1/4 的轮次制造临时占道（人 / 推车），看规划层能不能绕开
    gmap.clear_blocks()
    if seed % 4 == 0:
        taken = {(w.cell) for w in gmap.waypoints.values()} | {gmap.charger_cell}
        cand = [c for c in CORRIDOR_CELLS if c not in taken and gmap.is_free(c)]
        if cand:
            gmap.block_cell(rng.choice(cand))
    battery = 27.0 if seed % 5 == 0 else 100.0
    store_path = os.path.join(OUT, "progress_%d.json" % seed)
    ex, sensors, alarms = make_exec(gmap, battery, seed, store_path)
    try:
        plan = plan_by_rule(task, gmap)
    except PlanRejected as e:
        return None, [], 0, 0, sensors, alarms

    faults = pick_faults(gmap, plan, rng, rng.randint(1, 2)) if with_fault else []
    for f in faults:
        sensors.inject(f)

    res = ex.run_plan(plan)

    hit = set()
    for f in faults:
        for a in res.alarms:
            if a["waypoint_id"] == f.waypoint_id and a["sensor"] == f.sensor:
                hit.add("%s|%s" % (f.waypoint_id, f.sensor))
    injected_keys = {"%s|%s" % (f.waypoint_id, f.sensor) for f in faults}
    return res, faults, len(injected_keys), len(hit & injected_keys), sensors, alarms


def bench(rounds: int) -> dict:
    gmap = load_map(MAP)
    os.makedirs(OUT, exist_ok=True)
    results, injected_total, hit_total = [], 0, 0
    readings_total = readings_unknown = 0
    false_alarm_events = 0
    false_rounds = 0

    t0 = time.perf_counter()
    for i in range(rounds):
        task = TASKS[i % len(TASKS)]
        with_fault = (i % 3 != 2)      # 约 1/3 的轮次不注入，用来测误报
        out = run_round(gmap, task, seed=i + 1, with_fault=with_fault)
        res, faults, inj, hit, sensors, alarms = out
        if res is None:
            continue
        results.append(res)
        injected_total += inj
        hit_total += hit
        readings_total += len(sensors.readings)
        readings_unknown += sum(1 for r in sensors.readings if not r.valid)
        if not with_fault:
            false_rounds += 1
            false_alarm_events += len(res.alarms)

    elapsed = time.perf_counter() - t0
    m = aggregate(results, injected_total, set(), 0, readings_total, readings_unknown)
    m.injected_hit = hit_total

    # 拒单评测：自然语言层
    rejects = []
    for text in REJECT_TEXT_CASES:
        try:
            plan_by_rule(text, gmap)
            rejects.append({"case": text, "layer": "text", "rejected": False, "reason": ""})
        except PlanRejected as e:
            rejects.append({"case": text, "layer": "text", "rejected": True, "reason": e.reason})
    # 拒单评测：计划层（模拟模型产出坏 JSON）
    for name, payload in bad_payloads(gmap):
        try:
            validate_plan(payload, gmap, expected_digest=gmap.digest)
            rejects.append({"case": name, "layer": "plan", "rejected": False, "reason": ""})
        except PlanRejected as e:
            rejects.append({"case": name, "layer": "plan", "rejected": True, "reason": e.reason})

    # 规划耗时 P95
    plan_ms = []
    for i in range(50):
        t = time.perf_counter()
        try:
            plan_by_rule(TASKS[i % len(TASKS)], gmap)
        except PlanRejected:
            pass
        plan_ms.append((time.perf_counter() - t) * 1000)

    summary = m.summary()
    # 用真实误报口径覆盖 aggregate 的占位值
    summary["alarm_false_positive_events"] = false_alarm_events
    summary["false_positive_rounds"] = false_rounds
    summary["plan_ms_p95"] = round(_p95(plan_ms), 3)
    summary["plan_ms_median"] = round(statistics.median(plan_ms), 3)
    summary["reject_cases"] = rejects
    summary["reject_block_rate"] = round(
        sum(1 for r in rejects if r["rejected"]) / len(rejects), 4)
    summary["resample_total"] = sum(getattr(x, "resample", 0)
                                    for r in results for x in r.steps)
    summary["wall_clock_s"] = round(elapsed, 2)
    summary["simulated_seconds_per_round"] = round(
        sum(max((s.elapsed_s for s in r.steps), default=0) for r in results) / max(1, len(results)), 1)

    with open(os.path.join(OUT, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    md = render_md(summary, rounds)
    with open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    return summary


def render_md(s: dict, rounds: int) -> str:
    lines = [
        "# 巡检机器人任务执行仿真 · 评测摘要",
        "",
        "> 自动生成，勿手改。数据来源：`python bench/run_bench.py`。",
        "",
        "## 任务执行",
        "",
        "| 指标 | 数值 |",
        "| --- | --- |",
        "| 轮次 | %d |" % s["tasks"],
        "| 任务成功率 | %.1f%% |" % (s["task_success_rate"] * 100),
        "| 导航成功率 | %.1f%%（%d 个导航目标）|" % (s["nav_success_rate"] * 100, s["nav_steps"]),
        "| 低电量恢复成功率 | %.1f%%（%d 次中断）|"
        % (s["low_battery_recovery_rate"] * 100, s["low_battery_interrupts"]),
        "| 单轮仿真时长中位 | %.0f s（压缩后墙钟 P95 %.2f s）|"
        % (s["simulated_seconds_per_round"], s["wall_s_p95"]),
        "| 累计行驶 | %.1f m |" % s["odometer_m"],
        "",
        "## 感知与告警",
        "",
        "| 指标 | 数值 |",
        "| --- | --- |",
        "| 读数 UNKNOWN 比例 | %.1f%% |" % (s["unknown_reading_ratio"] * 100),
        "| 注入故障数 | %d |" % s["injected_faults"],
        "| 告警召回 | %.1f%% |" % (s["alarm_recall"] * 100),
        "| 误报事件（无注入轮次 %d 轮）| %d |" % (s["false_positive_rounds"],
                                                s["alarm_false_positive_events"]),
        "| 告警总数 | %d（critical %d / warn %d）|"
        % (s["alarms_total"], s["alarms_by_level"].get("critical", 0),
           s["alarms_by_level"].get("warn", 0)),
        "",
        "## 规划与拒单",
        "",
        "| 指标 | 数值 |",
        "| --- | --- |",
        "| 规划耗时 中位 / P95 | %.3f ms / %.3f ms |" % (s["plan_ms_median"], s["plan_ms_p95"]),
        "| 拒单用例拦截率 | %.0f%%（%d 条：文本层 %d + 计划层 %d）|"
        % (s["reject_block_rate"] * 100, len(s["reject_cases"]),
           sum(1 for r in s["reject_cases"] if r["layer"] == "text"),
           sum(1 for r in s["reject_cases"] if r["layer"] == "plan")),
        "| UNKNOWN 补测次数 | %d |" % s["resample_total"],
        "| 反馈回调超预算次数 | %d |" % s["feedback_overruns"],
        "",
        "## 失败原因分布",
        "",
    ]
    reasons = s.get("step_fail_reasons") or {}
    if reasons:
        lines += ["| 原因 | 次数 |", "| --- | --- |"]
        for k, v in sorted(reasons.items(), key=lambda x: -x[1]):
            lines.append("| %s | %d |" % (k, v))
    else:
        lines.append("无（本轮全部导航目标达成）")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=30)
    a = ap.parse_args()
    bench(a.rounds)
