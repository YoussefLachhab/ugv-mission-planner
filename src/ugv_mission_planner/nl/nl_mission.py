from __future__ import annotations

import re
import uuid
from collections.abc import Sequence
from typing import Any

from pydantic import ValidationError

from ugv_mission_planner.models.itinerary import Constraints
from ugv_mission_planner.models.mission_plan import MissionPlan


class ParseError(Exception):
    pass


R_POINT = re.compile(r"\(([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\)")
R_AVOID = re.compile(
    r"avoid\s*\[\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]",
    re.I,
)
R_SPEED = re.compile(r"max\s*speed\s*([-+]?\d*\.?\d+)\s*m/?s", re.I)

_MAX_SAFE_SPEED = 3.0  # conservative


def _as_rect_any(seq: Sequence[Any]) -> tuple[float, float, float, float]:
    a = float(seq[0])
    b = float(seq[1])
    c = float(seq[2])
    d = float(seq[3])
    return (a, b, c, d)


def _from_payload(payload: dict[str, Any]) -> MissionPlan:
    goals = [(float(a), float(b)) for a, b in payload.get("goals", [])]
    c = payload.get("constraints", {}) or {}
    cons = Constraints(
        max_speed_mps=(
            float(c.get("max_speed_mps", 1.2))
            if c.get("max_speed_mps", None) is not None
            else 1.2
        ),
        avoid_zones=[_as_rect_any(az) for az in c.get("avoid_zones", [])],
        battery_min_pct=(
            float(c.get("battery_min_pct", 20))
            if c.get("battery_min_pct", None) is not None
            else 20.0
        ),
    )
    if cons.max_speed_mps and cons.max_speed_mps > _MAX_SAFE_SPEED:
        raise ParseError(f"Requested speed {cons.max_speed_mps} exceeds policy {_MAX_SAFE_SPEED}")
    return MissionPlan(mission_id=uuid.uuid4().hex[:12], goals=goals, constraints=cons)


def _parse_no_llm(text: str) -> MissionPlan:
    points = [(float(x), float(y)) for x, y in R_POINT.findall(text)]
    avoid = [_as_rect_any(m) for m in R_AVOID.findall(text)]
    sp = R_SPEED.search(text)
    mx = float(sp.group(1)) if sp else 1.2
    cons = Constraints(max_speed_mps=mx, avoid_zones=avoid)
    if cons.max_speed_mps and cons.max_speed_mps > _MAX_SAFE_SPEED:
        raise ParseError("max speed too high")
    return MissionPlan(mission_id=uuid.uuid4().hex[:12], goals=points, constraints=cons)


def parse(text: str, llm: Any | None = None) -> MissionPlan:
    try:
        return _from_payload(llm.structured_mission(text)) if llm else _parse_no_llm(text)
    except ValidationError as e:
        raise ParseError(str(e)) from e
