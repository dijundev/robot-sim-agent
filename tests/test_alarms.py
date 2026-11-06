# -*- coding: utf-8 -*-
"""告警层单测：持续时间、滞回、去重、无效读数。"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rsagent.alarms import AlarmManager, AlarmRule  # noqa: E402
from rsagent.model import AlarmState, Reading  # noqa: E402


def reading(value, ts, sensor="temp", wp="wp_a1", valid=True):
    return Reading(waypoint_id=wp, sensor=sensor, value=value, unit="°C",
                   ts=ts, valid=valid)


def test_single_spike_does_not_trigger():
    am = AlarmManager([AlarmRule("temp", threshold=45, duration_s=5)])
    am.feed(reading(50, ts=0))
    assert am.to_list() == []


def test_sustained_breach_triggers():
    am = AlarmManager([AlarmRule("temp", threshold=45, duration_s=5)])
    am.feed(reading(50, ts=0))
    am.feed(reading(51, ts=2))
    am.feed(reading(52, ts=4))
    evs = am.feed(reading(53, ts=6))
    assert len(evs) == 1
    assert evs[0].state == AlarmState.TRIGGERED
    assert evs[0].peak_value == 53


def test_gap_resets_pending():
    am = AlarmManager([AlarmRule("temp", threshold=45, duration_s=5)])
    am.feed(reading(50, ts=0))
    am.feed(reading(30, ts=2))     # 回落到阈值下 → 计时清零
    am.feed(reading(50, ts=4))
    assert am.to_list() == []


def test_hysteresis_requires_falling_below_band():
    am = AlarmManager([AlarmRule("temp", threshold=45, duration_s=1, hysteresis=0.1)])
    am.feed(reading(50, ts=0))
    am.feed(reading(50, ts=1))     # 触发
    am.feed(reading(43, ts=2))     # 43 > 45*0.9=40.5 → 未恢复，仍进行中
    assert am.history[0].state == AlarmState.ONGOING
    am.feed(reading(40, ts=3))     # 40 <= 40.5 → 恢复
    assert am.history[0].state == AlarmState.RECOVERED


def test_no_duplicate_events_for_same_alarm():
    am = AlarmManager([AlarmRule("temp", threshold=45, duration_s=1)])
    for ts in range(0, 20):
        am.feed(reading(60, ts=ts))
    assert len(am.to_list()) == 1


def test_invalid_reading_never_triggers():
    am = AlarmManager([AlarmRule("temp", threshold=45, duration_s=1)])
    for ts in range(0, 10):
        am.feed(reading(None, ts=ts, valid=False))
    assert am.to_list() == []


def test_low_value_rule_for_indicator():
    am = AlarmManager([AlarmRule("indicator", threshold=0.5, duration_s=1, compare="<",
                                 hysteresis=0.0)])
    am.feed(reading(0.0, ts=0, sensor="indicator"))
    evs = am.feed(reading(0.0, ts=1, sensor="indicator"))
    assert evs and evs[0].level == "warn"


def test_ack_changes_state():
    am = AlarmManager([AlarmRule("gas", threshold=50, duration_s=1)])
    am.feed(reading(80, ts=0, sensor="gas"))
    am.feed(reading(80, ts=1, sensor="gas"))
    aid = am.history[0].alarm_id
    assert am.ack(aid, ts=2) is True
    assert am.events[aid].state == AlarmState.ACKED


def test_ack_missing_returns_false():
    am = AlarmManager()
    assert am.ack("nope", ts=0) is False


def test_levels_are_independent():
    """同传感器不同等级（warn 45 / critical 60）应各自计时、各自触发。"""
    am = AlarmManager([
        AlarmRule("temp", threshold=45, duration_s=2, level="warn"),
        AlarmRule("temp", threshold=60, duration_s=2, level="critical"),
    ])
    am.feed(reading(70, ts=0))
    am.feed(reading(70, ts=2))
    levels = {e.level for e in am.history}
    assert levels == {"warn", "critical"}
