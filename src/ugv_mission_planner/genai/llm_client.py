from __future__ import annotations
from typing import Protocol, Any, Dict
import os
from pydantic import BaseModel
from ugv_mission_planner.models import Constraints

class LLM(Protocol):
    def structured_mission(self, prompt: str) -> Dict[str, Any]: ...

class FakeLLM:
    def __init__(self, payload: Dict[str, Any]) -> None:
        self._payload = payload
    def structured_mission(self, prompt: str) -> Dict[str, Any]:
        return self._payload

# ---- Structured models for the LLM boundary ----
class Goal(BaseModel):
    x: float
    y: float

class MissionModel(BaseModel):
    # Use objects for goals to satisfy OpenAI structured outputs
    goals: list[Goal]
    constraints: Constraints

_SYSTEM = (
    "You are a mission planner assistant.\n"
    "Extract JSON for MissionModel(goals: list[Goal{x:float,y:float}], constraints: Constraints).\n"
    "Rules: return JSON only; no extra keys; units in meters/seconds; "
    "max_speed_mps <= 2.0; avoid_zones are [xmin,ymin,xmax,ymax]."
)

# ---- Real client ----
from langchain_openai import ChatOpenAI

class OpenAILLM:
    def __init__(self, model: str | None = None, temperature: float = 0.0) -> None:
        model = model or os.getenv("UGV_OPENAI_MODEL", "gpt-4o-mini")
        self._llm = ChatOpenAI(model=model, temperature=temperature)
        # Be explicit: use function-calling to avoid strict JSON schema issues
        self._structured = self._llm.with_structured_output(
            MissionModel,
            method="function_calling"
        )

    def structured_mission(self, prompt: str) -> Dict[str, Any]:
        result: MissionModel = self._structured.invoke(f"{_SYSTEM}\n\nUser: {prompt}")
        return result.model_dump()
