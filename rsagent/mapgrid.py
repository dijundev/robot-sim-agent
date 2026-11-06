# -*- coding: utf-8 -*-
"""栅格地图与全局规划：加载 → 障碍膨胀 → A* 求最短路 → 里程坐标换算。

只做「确定性、可单测」的部分：不碰网络、不碰模型。
"""
from __future__ import annotations

import hashlib
import heapq
import json
import math
from dataclasses import dataclass

from .model import Pose

Cell = tuple[int, int]

# 8 邻接：先走直边再走斜边，保证同代价路径的展开顺序稳定（评测可复现）
NEIGHBORS = [
    (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
    (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)),
    (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2)),
]


class MapError(Exception):
    pass


@dataclass
class Waypoint:
    id: str
    name: str
    cell: Cell
    sensors: list[str]

    @property
    def pose(self) -> Pose:
        return Pose(float(self.cell[0]), float(self.cell[1]))

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "cell": list(self.cell),
                "sensors": list(self.sensors)}


class GridMap:
    def __init__(self, raw: dict):
        self.name: str = raw["map_name"]
        self.resolution: float = float(raw.get("resolution", 0.5))
        self.grid_rows: list[str] = raw["grid"]
        self.rows = len(self.grid_rows)
        self.cols = len(self.grid_rows[0])
        for r, line in enumerate(self.grid_rows):
            if len(line) != self.cols:
                raise MapError("第 %d 行宽度不一致" % r)
        self.charger_cell: Cell = tuple(raw["charger"]["cell"])
        self.charger_name: str = raw["charger"]["name"]
        self.waypoints: dict[str, Waypoint] = {
            w["id"]: Waypoint(w["id"], w["name"], tuple(w["cell"]), list(w["sensors"]))
            for w in raw["waypoints"]
        }
        # 运行时临时占道（人、推车、临时堆放）：只影响规划与通行判断，不改静态地图
        self.runtime_blocks: set[Cell] = set()
        self._check_geometry()
        # 地图指纹：执行前用它和计划里记录的指纹比对，防止「按旧地图派单」
        self.digest = hashlib.sha256(
            ("|".join(self.grid_rows) + "|" + str(self.cols)).encode("utf-8")
        ).hexdigest()[:16]

    # ---------- 静态校验 ----------
    def _check_geometry(self) -> None:
        for wp in self.waypoints.values():
            if not self.is_free(wp.cell):
                raise MapError("点位 %s 落在障碍上 %s" % (wp.name, wp.cell))
        if not self.is_free(self.charger_cell):
            raise MapError("充电桩落在障碍上")
        for wp in self.waypoints.values():
            if not self.plan(self.charger_cell, wp.cell):
                raise MapError("点位 %s 从充电桩不可达" % wp.name)
        names = [w.name for w in self.waypoints.values()]
        if len(set(names)) != len(names):
            raise MapError("点位名称重复（白名单按名称匹配，必须唯一）")

    def is_free(self, cell: Cell) -> bool:
        r, c = cell
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            return False
        if cell in self.runtime_blocks:
            return False
        return self.grid_rows[r][c] == "0"

    def block_cell(self, cell: Cell) -> None:
        self.runtime_blocks.add(cell)

    def clear_blocks(self) -> None:
        self.runtime_blocks.clear()

    def inflate(self, radius: int = 1) -> set[Cell]:
        """障碍膨胀：按机器人半径把障碍格向四周扩，避免贴墙擦角。"""
        blocked: set[Cell] = set()
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid_rows[r][c] == "1":
                    for dr in range(-radius, radius + 1):
                        for dc in range(-radius, radius + 1):
                            blocked.add((r + dr, c + dc))
        return blocked

    # ---------- 规划 ----------
    def plan(self, start: Cell, goal: Cell, inflate: int = 1) -> list[Cell]:
        """A*（octile 启发式）。返回不含起点的路径；无解返回空列表。"""
        if not self.is_free(start) or not self.is_free(goal):
            return []
        blocked = self.inflate(inflate) | self.runtime_blocks
        # 起点/终点自身允许在膨胀区内，否则窄通道会直接判死
        blocked.discard(start)
        blocked.discard(goal)
        if start == goal:
            return []

        def h(a: Cell, b: Cell) -> float:
            dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
            return (dr + dc) + (math.sqrt(2) - 2) * min(dr, dc)

        openq: list[tuple[float, Cell]] = [(h(start, goal), start)]
        g = {start: 0.0}
        came: dict[Cell, Cell] = {}
        seen = set()
        while openq:
            _, cur = heapq.heappop(openq)
            if cur in seen:
                continue
            seen.add(cur)
            if cur == goal:
                path = []
                while cur in came:
                    path.append(cur)
                    cur = came[cur]
                return list(reversed(path))
            for dr, dc, cost in NEIGHBORS:
                nxt = (cur[0] + dr, cur[1] + dc)
                if not self.is_free(nxt) or nxt in blocked:
                    continue
                ng = g[cur] + cost
                if ng < g.get(nxt, float("inf")):
                    g[nxt] = ng
                    came[nxt] = cur
                    heapq.heappush(openq, (ng + h(nxt, goal), nxt))
        return []

    def path_length_m(self, path: list[Cell], start: Cell) -> float:
        pts = [start] + path
        total = 0.0
        for a, b in zip(pts, pts[1:]):
            total += math.hypot(b[0] - a[0], b[1] - a[1]) * self.resolution
        return total

    # ---------- 命名解析 ----------
    def find_by_name(self, text: str) -> Waypoint | None:
        for wp in self.waypoints.values():
            if wp.name == text or wp.name in text:
                return wp
        return None

    def resolve_cell(self, name_or_id: str) -> Waypoint | None:
        if name_or_id in self.waypoints:
            return self.waypoints[name_or_id]
        return self.find_by_name(name_or_id)


def load_map(path: str) -> GridMap:
    with open(path, encoding="utf-8") as f:
        return GridMap(json.load(f))
