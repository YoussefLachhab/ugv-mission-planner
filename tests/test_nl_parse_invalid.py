import pytest

from ugv_mission_planner.genai.llm_client import FakeLLM
from ugv_mission_planner.nl.nl_mission import ParseError, parse


def test_parse_invalid_speed_fails_closed():
    fake = FakeLLM(
        {
            "goals": [(0.0, 0.0), (1.0, 1.0)],
            "constraints": {"max_speed_mps": 9.9, "avoid_zones": [], "battery_min_pct": 10.0},
        }
    )
    with pytest.raises(ParseError):
        parse("Go fast", fake)
