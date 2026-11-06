# -*- coding: utf-8 -*-
"""规划层单测：重点不是「能生成计划」，而是「坏计划一定进不来」。"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rsagent.mapgrid import load_map  # noqa: E402
from rsagent.planner import (PlanRejected, fake_llm, plan_by_llm, plan_by_rule,  # noqa: E402
                             plan_digest, validate_plan)

MAP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "config", "map_office.json")


@pytest.fixture(scope="module")
def gmap():
    return load_map(MAP)


# ---------------------------------------------------------------- 规则路径
def test_rule_plan_keeps_text_order(gmap):
    plan = plan_by_rule("先去水泵房，再去配电柜A", gmap)
    navs = [s.waypoint_id for s in plan.steps if s.step_type == "navigate"]
    assert navs == ["wp_c3", "wp_a1"]


def test_rule_plan_filters_sensor_by_waypoint_capability(gmap):
    # 配电柜A 没有 gas 传感器，不该生成 gas 检测步
    plan = plan_by_rule("巡检配电柜A的气体浓度", gmap)
    sensors = [s.sensor for s in plan.steps if s.step_type == "inspect"]
    assert "gas" not in sensors


def test_rule_plan_empty_text_rejected(gmap):
    with pytest.raises(PlanRejected, match="为空"):
        plan_by_rule("   ", gmap)


def test_rule_plan_unknown_place_rejected(gmap):
    with pytest.raises(PlanRejected, match="白名单"):
        plan_by_rule("去天台看看", gmap)


def test_rule_plan_all_points(gmap):
    plan = plan_by_rule("全点位巡检温度", gmap)
    navs = [s.waypoint_id for s in plan.steps if s.step_type == "navigate"]
    assert set(navs) == set(gmap.waypoints)


def test_rule_plan_appends_dock_when_asked(gmap):
    plan = plan_by_rule("巡检机柜B温度，最后回充", gmap)
    assert plan.steps[-1].step_type == "dock"


# ---------------------------------------------------------------- 校验层
def _base_payload(gmap, steps):
    return {"task_text": "t", "map_name": gmap.name, "map_digest": gmap.digest,
            "steps": steps}


def test_validate_accepts_good_plan(gmap):
    p = validate_plan(_base_payload(gmap, [
        {"step_type": "navigate", "waypoint_id": "wp_a1"},
        {"step_type": "inspect", "waypoint_id": "wp_a1", "sensor": "temp"},
    ]), gmap)
    assert len(p.steps) == 2


@pytest.mark.parametrize("steps,match", [
    ([{"step_type": "fly", "waypoint_id": "wp_a1"}], "不在白名单"),
    ([{"step_type": "navigate", "waypoint_id": "wp_zz"}], "不在白名单"),
    ([{"step_type": "navigate"}], "缺少 waypoint_id"),
    ([{"step_type": "navigate", "waypoint_id": "wp_a1"},
      {"step_type": "inspect", "waypoint_id": "wp_a1", "sensor": "laser"}], "传感器"),
    ([{"step_type": "navigate", "waypoint_id": "wp_a1"},
      {"step_type": "inspect", "waypoint_id": "wp_a1", "sensor": "gas"}], "没有 gas"),
    ([{"step_type": "navigate", "waypoint_id": "wp_a1"},
      {"step_type": "inspect", "waypoint_id": "wp_b2", "sensor": "temp"}], "不一致"),
    ([{"step_type": "navigate", "waypoint_id": "wp_a1", "timeout_s": 9999}], "越界"),
])
def test_validate_rejects(gmap, steps, match):
    with pytest.raises(PlanRejected, match=match):
        validate_plan(_base_payload(gmap, steps), gmap)


def test_validate_rejects_map_mismatch(gmap):
    p = _base_payload(gmap, [{"step_type": "navigate", "waypoint_id": "wp_a1"}])
    p["map_name"] = "another_floor"
    with pytest.raises(PlanRejected, match="不一致"):
        validate_plan(p, gmap)


def test_validate_rejects_stale_map_digest(gmap):
    p = _base_payload(gmap, [{"step_type": "navigate", "waypoint_id": "wp_a1"}])
    with pytest.raises(PlanRejected, match="指纹"):
        validate_plan(p, gmap, expected_digest="deadbeefdeadbeef")


def test_validate_rejects_too_many_steps(gmap):
    steps = []
    for _ in range(20):
        steps += [{"step_type": "navigate", "waypoint_id": "wp_a1"},
                  {"step_type": "inspect", "waypoint_id": "wp_a1", "sensor": "temp"}]
    with pytest.raises(PlanRejected, match="上限"):
        validate_plan(_base_payload(gmap, steps), gmap)


def test_validate_rejects_missing_field(gmap):
    with pytest.raises(PlanRejected, match="缺少必填"):
        validate_plan({"map_name": gmap.name, "steps": []}, gmap)


# ---------------------------------------------------------------- 模型路径
def test_llm_path_success(gmap):
    plan = plan_by_llm("去配电柜A和机柜B看温度", gmap, fake_llm())
    assert plan.source == "llm"
    assert [s.waypoint_id for s in plan.steps if s.step_type == "navigate"] == ["wp_a1", "wp_b2"]


@pytest.mark.parametrize("mode", ["bad_json", "unknown_waypoint", "bad_action", "mismatch"])
def test_llm_path_rejects_after_retries(gmap, mode):
    calls = {"n": 0}

    def counting(messages):
        calls["n"] += 1
        return fake_llm(mode=mode)(messages)

    with pytest.raises(PlanRejected, match="重试"):
        plan_by_llm("巡检配电柜A温度", gmap, counting, retries=2)
    assert calls["n"] == 3, "重试上限 2 次，总调用应为 3"


def test_llm_retry_then_fix(gmap):
    """第一次给坏动作，第二次给正确计划 —— 回灌重生成链路必须真的生效。"""
    seq = [fake_llm(mode="bad_action"), fake_llm()]

    def flaky(messages):
        return seq.pop(0)(messages)

    plan = plan_by_llm("巡检配电柜A温度", gmap, flaky, retries=2)
    assert plan.steps[0].waypoint_id == "wp_a1"


def test_plan_digest_is_stable_and_distinct(gmap):
    a = plan_by_rule("巡检配电柜A温度", gmap)
    b = plan_by_rule("巡检配电柜A温度", gmap)
    c = plan_by_rule("巡检机柜B温度", gmap)
    assert plan_digest(a) == plan_digest(b)
    assert plan_digest(a) != plan_digest(c)
