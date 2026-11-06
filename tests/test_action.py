# -*- coding: utf-8 -*-
"""执行层单测：任务成功、取消、超时、低电量中断与恢复、进度持久化。"""
import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rsagent.action import ActionExecutor, ExecConfig, ProgressStore  # noqa: E402
from rsagent.alarms import AlarmManager  # noqa: E402
from rsagent.mapgrid import load_map  # noqa: E402
from rsagent.model import GoalStatus, Plan, Pose, Step  # noqa: E402
from rsagent.planner import plan_by_rule  # noqa: E402
from rsagent.robot import Robot, RobotConfig  # noqa: E402
from rsagent.sensors import SensorSuite  # noqa: E402

MAP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "config", "map_office.json")


@pytest.fixture
def gmap():
    return load_map(MAP)


def make_exec(gmap, battery=100.0, store=None, cfg=None):
    robot = Robot(pose=(float(gmap.charger_cell[0]), float(gmap.charger_cell[1]), 0.0),
                  battery=battery, cfg=RobotConfig())
    sensors = SensorSuite(seed=11)
    alarms = AlarmManager()
    return ActionExecutor(gmap, robot, sensors, alarms,
                          store=store or ProgressStore(), cfg=cfg or ExecConfig())


def test_simple_task_succeeds(gmap):
    ex = make_exec(gmap)
    plan = plan_by_rule("巡检配电柜A温度", gmap)
    res = ex.run_plan(plan)
    assert res.success
    assert res.status == GoalStatus.SUCCEEDED
    assert res.odometer_m > 0
    assert len(res.steps) == len(plan.steps)


def test_all_points_task_succeeds(gmap):
    ex = make_exec(gmap)
    plan = plan_by_rule("全点位巡检温度", gmap)
    res = ex.run_plan(plan)
    assert res.success
    navs = [s for s in res.steps if s.step_type == "navigate"]
    assert len(navs) == 6
    assert all(s.status == GoalStatus.SUCCEEDED for s in navs)


def test_low_battery_triggers_dock_and_resume(gmap):
    ex = make_exec(gmap, battery=28.0)
    plan = plan_by_rule("巡检末端管廊温度", gmap)
    res = ex.run_plan(plan)
    assert res.recovered_from_low_battery is True
    assert res.success, "回充后应续跑完剩余步骤：%s" % res.cancel_reason
    assert res.battery_end > 50


def test_cancel_stops_execution(gmap):
    ex = make_exec(gmap)
    plan = plan_by_rule("全点位巡检温度", gmap)
    ex.request_cancel()
    res = ex.run_plan(plan)
    assert res.status == GoalStatus.CANCELED
    assert res.success is False
    assert res.cancel_reason


def test_timeout_aborts_step(gmap):
    ex = make_exec(gmap)
    # 人为把超时压到 0.5s，必然超时
    plan = Plan(plan_id="plan_timeout", task_text="t", map_name=gmap.name,
                map_digest=gmap.digest,
                steps=[Step(index=0, step_type="navigate", waypoint_id="wp_f6",
                            timeout_s=0.5)])
    res = ex.run_plan(plan)
    assert res.status == GoalStatus.ABORTED
    assert res.steps[0].note in ("timeout", "stalled")


def test_unreachable_waypoint_aborts(gmap):
    ex = make_exec(gmap)
    plan = Plan(plan_id="plan_unreach", task_text="t", map_name=gmap.name,
                map_digest=gmap.digest,
                steps=[Step(index=0, step_type="navigate", waypoint_id="wp_ghost")])
    res = ex.run_plan(plan)
    assert res.status == GoalStatus.ABORTED
    assert res.steps[0].note == "unknown_waypoint"


def test_progress_persisted_and_skipped_on_rerun(gmap):
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "progress.json")
        store = ProgressStore(p)
        ex = make_exec(gmap, store=store)
        plan = plan_by_rule("巡检配电柜A温度", gmap)
        res1 = ex.run_plan(plan)
        assert res1.success
        assert os.path.exists(p)
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        assert data[plan.plan_id]["status"] == "succeeded"
        assert len(data[plan.plan_id]["done"]) == len(plan.steps)

        # 同一 plan 再跑一次：全部步骤已完成 → 不重复执行
        ex2 = make_exec(gmap, store=ProgressStore(p))
        res2 = ex2.run_plan(plan)
        assert res2.steps == []
        assert res2.odometer_m == 0


def test_feedback_stream_is_monotonic_in_time(gmap):
    ex = make_exec(gmap)
    plan = plan_by_rule("巡检机柜B温度", gmap)
    ex.run_plan(plan)
    assert len(ex.feedback_log) > 10
    times = [fb.elapsed_s for fb in ex.feedback_log]
    assert times == sorted(times)


def test_inspect_collects_multiple_samples(gmap):
    ex = make_exec(gmap)
    plan = plan_by_rule("巡检配电柜A温度", gmap)
    res = ex.run_plan(plan)
    inspect = [s for s in res.steps if s.step_type == "inspect"][0]
    assert len(inspect.readings) >= 3
