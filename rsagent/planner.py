# -*- coding: utf-8 -*-
"""自然语言 → 结构化巡检计划。

设计要点（这是整个系统最像「机器人 Agent」的一层）：

1. **模型只产出候选**：LLM 路径只负责把自然语言变成 JSON，落成 Plan 之前必须过
   三道确定性校验 —— Schema（字段/类型/枚举）、地点白名单（必须是地图里真有的点）、
   地图指纹（计划必须针对当前地图生成）。
2. **规则兜底**：没有模型也能跑（rule 路径），保证离线可测、可回归。
3. **拒绝要说明原因**：reject_reason 落到 plan 上，供上游回灌重生成或直接拒单 ——
   机器人场景宁可拒单，不可猜着执行。
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import uuid

from .mapgrid import GridMap
from .model import Plan, Step

ALLOWED_STEP_TYPES = {"navigate", "inspect", "dock", "wait"}
ALLOWED_SENSORS = {"temp", "humidity", "gas", "meter", "indicator"}
MAX_STEPS = 32

# 中文数词：巡检任务里出现「三个点位」这类表述很常见，模型也可能照抄
CN_NUM = {"一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5,
          "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}

SENSOR_WORDS = {
    "温度": "temp", "体温": "temp", "测温": "temp",
    "湿度": "humidity",
    "气体": "gas", "浓度": "gas", "泄漏": "gas",
    "仪表": "meter", "表计": "meter", "读数": "meter",
    "指示灯": "indicator", "灯": "indicator",
}


class PlanRejected(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _new_plan_id() -> str:
    return "plan_" + uuid.uuid4().hex[:10]


def _extract_waypoints(text: str, gmap: GridMap) -> list:
    """按名称命中点位，保持文本中出现的先后顺序（「先去A再去B」不能被打乱）。"""
    hits = []
    for wp in gmap.waypoints.values():
        pos = text.find(wp.name)
        if pos >= 0:
            hits.append((pos, wp))
    hits.sort(key=lambda x: x[0])
    return [wp for _, wp in hits]


def _extract_sensors(text: str) -> list[str]:
    found = []
    for word, sensor in SENSOR_WORDS.items():
        if word in text and sensor not in found:
            found.append(sensor)
    return found


def plan_by_rule(task_text: str, gmap: GridMap) -> Plan:
    """规则路径：不依赖模型，用于离线回归与基线对照。"""
    text = (task_text or "").strip()
    if not text:
        raise PlanRejected("任务描述为空")

    # 「全点位巡检」：不点名时按名称顺序走一遍（先判关键词，再抽点位）
    all_points = any(k in text for k in ("全部", "所有", "全点位", "整层", "逐点"))
    wps = sorted(gmap.waypoints.values(), key=lambda w: w.name) if all_points \
        else _extract_waypoints(text, gmap)
    if not wps:
        raise PlanRejected("未识别到任何已知点位（地点白名单未命中）")

    sensors = _extract_sensors(text)
    charge = any(k in text for k in ("回充", "充电", "回桩"))

    steps: list[Step] = []
    for wp in wps:
        steps.append(Step(index=len(steps), step_type="navigate", waypoint_id=wp.id))
        targets = [s for s in sensors if s in wp.sensors] or list(wp.sensors)
        for s in targets:
            steps.append(Step(index=len(steps), step_type="inspect",
                              waypoint_id=wp.id, sensor=s, timeout_s=20.0))
    if charge:
        steps.append(Step(index=len(steps), step_type="dock", waypoint_id=None))

    if len(steps) > MAX_STEPS:
        raise PlanRejected("步数 %d 超出上限 %d" % (len(steps), MAX_STEPS))

    return Plan(plan_id=_new_plan_id(), task_text=text, map_name=gmap.name,
                map_digest=gmap.digest, steps=steps, source="rule")


# ---------------------------------------------------------------- 校验层
def validate_plan(payload: dict, gmap: GridMap, expected_digest: str | None = None) -> Plan:
    """把（可能来自模型的）JSON 收紧成可信 Plan。任何一项不过就抛 PlanRejected。"""
    if not isinstance(payload, dict):
        raise PlanRejected("计划不是 JSON 对象")

    required = ("task_text", "map_name", "steps")
    for k in required:
        if k not in payload:
            raise PlanRejected("缺少必填字段 %s" % k)

    if payload["map_name"] != gmap.name:
        raise PlanRejected("计划地图 %r 与当前地图 %r 不一致"
                           % (payload["map_name"], gmap.name))

    digest = payload.get("map_digest", gmap.digest)
    if expected_digest and digest != expected_digest:
        raise PlanRejected("地图指纹不一致（计划基于旧版本地图）")

    raw_steps = payload["steps"]
    if not isinstance(raw_steps, list) or not raw_steps:
        raise PlanRejected("steps 必须是非空数组")
    if len(raw_steps) > MAX_STEPS:
        raise PlanRejected("步数 %d 超出上限 %d" % (len(raw_steps), MAX_STEPS))

    steps: list[Step] = []
    for i, raw in enumerate(raw_steps):
        if not isinstance(raw, dict):
            raise PlanRejected("第 %d 步不是对象" % i)
        stype = raw.get("step_type")
        if stype not in ALLOWED_STEP_TYPES:
            # 白名单：未声明的动作类型一律拒绝，不做「差不多能跑」的兜底
            raise PlanRejected("第 %d 步动作类型 %r 不在白名单" % (i, stype))

        wp_id = raw.get("waypoint_id")
        if stype in ("navigate", "inspect"):
            if not wp_id:
                raise PlanRejected("第 %d 步缺少 waypoint_id" % i)
            if wp_id not in gmap.waypoints:
                raise PlanRejected("第 %d 步点位 %r 不在白名单" % (i, wp_id))

        sensor = raw.get("sensor")
        if stype == "inspect":
            if sensor not in ALLOWED_SENSORS:
                raise PlanRejected("第 %d 步传感器 %r 不合法" % (i, sensor))
            if sensor not in gmap.waypoints[wp_id].sensors:
                raise PlanRejected("第 %d 步点位 %s 没有 %s 传感器"
                                   % (i, gmap.waypoints[wp_id].name, sensor))

        timeout = raw.get("timeout_s", 60.0)
        if not isinstance(timeout, (int, float)) or not (0 < timeout <= 900):
            raise PlanRejected("第 %d 步 timeout_s 越界" % i)

        steps.append(Step(index=i, step_type=stype, waypoint_id=wp_id,
                          sensor=sensor, timeout_s=float(timeout)))

    # 结构校验：inspect 必须落在最近的 navigate 之后，不能凭空检测
    last_nav = None
    for s in steps:
        if s.step_type == "navigate":
            last_nav = s.waypoint_id
        elif s.step_type == "inspect" and s.waypoint_id != last_nav:
            raise PlanRejected("第 %d 步检测点位 %s 与上一导航目标不一致"
                               % (s.index, s.waypoint_id))

    return Plan(plan_id=payload.get("plan_id") or _new_plan_id(),
                task_text=payload["task_text"], map_name=gmap.name,
                map_digest=digest, steps=steps,
                source=payload.get("source", "llm"))


# ---------------------------------------------------------------- 模型路径
SYSTEM_PROMPT = """你是巡检机器人任务规划器。把用户的中文指令转成 JSON 计划，不要输出任何解释。
输出格式：
{"task_text": "...", "map_name": "<给定地图名>", "map_digest": "<给定指纹>",
 "steps": [{"step_type": "navigate|inspect|dock|wait", "waypoint_id": "...",
            "sensor": "temp|humidity|gas|meter|indicator", "timeout_s": 60}]}
硬性约束：
1. 只能使用白名单里的点位 id，禁止创造新点位；
2. inspect 必须紧跟同一点位的 navigate；
3. 步数不超过 32；
4. 用户要求回充时，末尾追加一步 dock。"""


def build_prompt(task_text: str, gmap: GridMap) -> list[dict]:
    wl = "\n".join("- %s（%s）可选传感器：%s" % (w.id, w.name, "/".join(w.sensors))
                   for w in gmap.waypoints.values())
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "地图名：%s\n地图指纹：%s\n可用点位：\n%s\n\n指令：%s"
         % (gmap.name, gmap.digest, wl, task_text)},
    ]


def plan_by_llm(task_text: str, gmap: GridMap, llm_call, retries: int = 2) -> Plan:
    """模型只出候选，校验不过就把错误回灌重生成；重试上限用完即失败。

    llm_call(messages) -> str：由调用方注入（测试里用假模型，线上接真实 API）。
    """
    messages = build_prompt(task_text, gmap)
    last_err = ""
    for attempt in range(retries + 1):
        raw = llm_call(messages)
        try:
            payload = json.loads(_strip_json(raw))
        except Exception:
            last_err = "模型返回不是合法 JSON"
        else:
            try:
                return validate_plan(payload, gmap, expected_digest=gmap.digest)
            except PlanRejected as e:
                last_err = e.reason
        messages = messages + [
            {"role": "assistant", "content": raw[:2000]},
            {"role": "user", "content": "上面的计划被拒绝：%s。原始指令：%s。"
                                        "请只输出修正后的 JSON。" % (last_err, task_text)},
        ]
    raise PlanRejected("重试 %d 次仍不合规：%s" % (retries, last_err))


def _strip_json(text: str) -> str:
    """模型常把 JSON 包在 ```json 里，或前后带一句话。剥出来再解析。"""
    t = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(.+?)```", t, re.S)
    if m:
        t = m.group(1).strip()
    i, j = t.find("{"), t.rfind("}")
    if i >= 0 and j > i:
        t = t[i:j + 1]
    return t


def plan_digest(plan: Plan) -> str:
    """计划指纹：同一条指令重复下发时可据此判重，避免重复派单。"""
    body = json.dumps([s.to_dict() for s in plan.steps], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------- 离线假模型（仅测试/自测用）
def fake_llm(good: bool = True, mode: str = "ok"):
    """产出可预期的模型响应，用于验证校验层真的能拦住坏计划。"""
    def call(messages):
        # 提示词里带了完整点位清单，只从「指令：」之后取用户原话，否则会把清单里
        # 每个点位都当成用户点名的目标（第一版就是这么错的，被单测抓到）
        task = messages[-1]["content"].split("指令：")[-1].strip()
        if mode == "bad_json":
            return "好的，我来规划一下：先巡检配电柜A"
        if mode == "unknown_waypoint":
            return json.dumps({"task_text": task, "map_name": "office_floor_3f",
                               "steps": [{"step_type": "navigate", "waypoint_id": "wp_zz"},
                                         {"step_type": "inspect", "waypoint_id": "wp_zz",
                                          "sensor": "temp"}]}, ensure_ascii=False)
        if mode == "bad_action":
            return json.dumps({"task_text": task, "map_name": "office_floor_3f",
                               "steps": [{"step_type": "launch_missile", "waypoint_id": "wp_a1"}]},
                              ensure_ascii=False)
        if mode == "mismatch":
            return json.dumps({"task_text": task, "map_name": "office_floor_3f",
                               "steps": [{"step_type": "navigate", "waypoint_id": "wp_a1"},
                                         {"step_type": "inspect", "waypoint_id": "wp_b2",
                                          "sensor": "temp"}]}, ensure_ascii=False)
        # 正常：从指令里抽点位
        steps = []
        for name, wid, sensors in [("配电柜A", "wp_a1", ["temp"]), ("机柜B", "wp_b2", ["temp"]),
                                   ("水泵房", "wp_c3", ["gas"]), ("气体阀组", "wp_d4", ["gas"]),
                                   ("仪表盘", "wp_e5", ["meter"]), ("末端管廊", "wp_f6", ["temp"])]:
            if name in task:
                steps.append({"step_type": "navigate", "waypoint_id": wid})
                for s in sensors:
                    steps.append({"step_type": "inspect", "waypoint_id": wid,
                                  "sensor": s, "timeout_s": 20})
        if not steps:
            return json.dumps({"task_text": task, "map_name": "office_floor_3f", "steps": []})
        return "```json\n" + json.dumps(
            {"plan_id": "plan_" + hashlib.sha256(task.encode()).hexdigest()[:10],
             "task_text": task, "map_name": "office_floor_3f", "steps": steps},
            ensure_ascii=False, indent=2) + "\n```"
    return call
