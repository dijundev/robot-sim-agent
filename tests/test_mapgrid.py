# -*- coding: utf-8 -*-
"""地图与规划层单测。"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rsagent.mapgrid import GridMap, MapError, load_map  # noqa: E402

MAP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "config", "map_office.json")


@pytest.fixture(scope="module")
def gmap():
    return load_map(MAP)


def test_map_loads_and_has_six_waypoints(gmap):
    assert gmap.rows == 14 and gmap.cols == 20
    assert len(gmap.waypoints) == 6
    assert len(gmap.digest) == 16


def test_every_waypoint_reachable_from_charger(gmap):
    for wp in gmap.waypoints.values():
        path = gmap.plan(gmap.charger_cell, wp.cell)
        assert path, "%s 不可达" % wp.name


def test_astar_path_is_contiguous(gmap):
    path = gmap.plan(gmap.charger_cell, gmap.waypoints["wp_f6"].cell)
    pts = [gmap.charger_cell] + path
    for a, b in zip(pts, pts[1:]):
        assert abs(a[0] - b[0]) <= 1 and abs(a[1] - b[1]) <= 1
    assert path[-1] == gmap.waypoints["wp_f6"].cell


def test_astar_returns_empty_on_obstacle(gmap):
    assert gmap.plan(gmap.charger_cell, (0, 0)) == []


def test_path_length_positive(gmap):
    path = gmap.plan(gmap.charger_cell, gmap.waypoints["wp_c3"].cell)
    assert gmap.path_length_m(path, gmap.charger_cell) > 3.0


def test_waypoint_on_obstacle_rejected():
    raw = {
        "map_name": "bad", "resolution": 0.5,
        "grid": ["111", "101", "111"],
        "charger": {"cell": [1, 1], "name": "桩"},
        "waypoints": [{"id": "w", "name": "坏点", "cell": [0, 0], "sensors": ["temp"]}],
    }
    with pytest.raises(MapError, match="落在障碍上"):
        GridMap(raw)


def test_duplicate_waypoint_name_rejected():
    raw = {
        "map_name": "dup", "resolution": 0.5,
        "grid": ["11111111111",
                 "10000000001",
                 "10000000001",
                 "10000000001",
                 "11111111111"],
        "charger": {"cell": [1, 1], "name": "桩"},
        "waypoints": [
            {"id": "a", "name": "同名", "cell": [1, 5], "sensors": ["temp"]},
            {"id": "b", "name": "同名", "cell": [3, 5], "sensors": ["temp"]},
        ],
    }
    with pytest.raises(MapError, match="名称重复"):
        GridMap(raw)


def test_resolve_by_name_and_id(gmap):
    assert gmap.resolve_cell("配电柜A").id == "wp_a1"
    assert gmap.resolve_cell("wp_c3").name == "水泵房"
    assert gmap.resolve_cell("不存在") is None
