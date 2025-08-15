from __future__ import annotations
from typing import Tuple, List
from pydantic import ValidationError
from ugv_mission_planner.genai.llm_client import LLM
from ugv_mission_planner.models import MissionPlan, Constraints

class ParseError(Exception):
    def __init__(self, reason: str, hints: Tuple[str, ...] = ()) -> None:
        super().__init__(reason)
        self.hints = hints

def _normalize_goals(raw_goals) -> List[tuple[float, float]]:
    """Accept [(x,y)], [[x,y]], or [{'x':..,'y':..}] and return [(x,y), ...]."""
    norm: List[tuple[float, float]] = []
    for g in raw_goals:
        if isinstance(g, dict) and "x" in g and "y" in g:
            norm.append((float(g["x"]), float(g["y"])))
        elif isinstance(g, (list, tuple)) and len(g) == 2:
            norm.append((float(g[0]), float(g[1])))
        else:
            raise ValueError(f"Unsupported goal format: {g!r}")
    return norm

def parse(text: str, llm: LLM) -> MissionPlan:
    try:
        data = llm.structured_mission(text)
    except Exception as e:
        raise ParseError("LLM call failed", hints=("Check API key / network.", str(e)))

    try:
        goals = _normalize_goals(data["goals"])
        constraints = Constraints(**data["constraints"])
        return MissionPlan(goals=goals, constraints=constraints, waypoints=[])
    except (KeyError, TypeError, ValidationError, ValueError) as e:
        raise ParseError(
            "Parsed output failed schema validation",
            hints=("Ensure numeric goals as (x,y).", "max_speed_mps <= 2.0.", str(e)),
        )
