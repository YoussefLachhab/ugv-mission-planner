from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple, TypedDict

# Import at top (ruff E402)
try:  # pragma: no cover - import presence varies by environment
    from langchain_openai import ChatOpenAI
    HAS_OPENAI = True
except Exception:  # pragma: no cover
    ChatOpenAI = None  # type: ignore[assignment]
    HAS_OPENAI = False

from ugv_mission_planner.models import Constraints


class LLM:
    def structured_mission(self, text: str) -> Dict[str, Any]:
        raise NotImplementedError


# ---- Fake client for tests/offline ----
@dataclass
class FakeLLM(LLM):
    payload: Dict[str, Any]

    def structured_mission(self, text: str) -> Dict[str, Any]:
        return json.loads(json.dumps(self.payload))  # deep copy


# ---- Real client ----
class OpenAILLM(LLM):  # pragma: no cover
    def __init__(self, model: str | None = None) -> None:
        if not HAS_OPENAI:
            raise RuntimeError("langchain-openai not installed")
        self.model = model or os.getenv("UGV_OPENAI_MODEL", "gpt-4o-mini")
        self.client = ChatOpenAI(model=self.model, temperature=0)

    def structured_mission(self, text: str) -> Dict[str, Any]:
        # Return a dict compatible with MissionPlan fields
        # We use a light guidance prompt; structured output handled by downstream validation.
        prompt = (
            "Extract goals and constraints from the mission. "
            "Return JSON with keys: goals[[x,y]...], constraints{max_speed_mps, avoid_zones[[xmin,ymin,xmax,ymax]], battery_min_pct}.\n"
            f"Mission: {text}"
        )
        resp = self.client.invoke(prompt)  # type: ignore[union-attr]
        content = resp.content if hasattr(resp, "content") else str(resp)
        try:
            data = json.loads(content)
        except Exception:
            # Allow model to return text; try to find a JSON object
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                data = json.loads(content[start : end + 1])
            else:
                raise
        return data


def get_llm() -> LLM:
    """Return either FakeLLM (when UGV_FAKE_LLM=1) or the real OpenAI-backed client."""
    if os.getenv("UGV_FAKE_LLM") == "1":
        payload = os.getenv("UGV_FAKE_PAYLOAD")
        if payload:
            return FakeLLM(json.loads(payload))
        # Minimal default to keep CLI happy when someone sets UGV_FAKE_LLM but no payload
        return FakeLLM(
            {
                "goals": [[2, 2], [18, 2], [2, 2], [18, 2]],
                "constraints": {"max_speed_mps": 1.2, "avoid_zones": [[8, 0, 12, 6]], "battery_min_pct": 15},
            }
        )
    return OpenAILLM()
