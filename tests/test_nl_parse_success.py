from ugv_mission_planner.genai.llm_client import FakeLLM
from ugv_mission_planner.nl.nl_mission import parse


def test_parse_success_minimal():
    fake = FakeLLM(
        {
            "goals": [(0.0, 0.0), (5.0, 2.0)],
            "constraints": {"max_speed_mps": 1.2, "avoid_zones": [[1, 1, 2, 2]], "battery_min_pct": 20.0},
        }
    )
    plan = parse("Patrol between A and B twice, avoid zone Z", fake)
    assert plan.constraints.max_speed_mps == 1.2
    assert plan.goals == [(0.0, 0.0), (5.0, 2.0)]
