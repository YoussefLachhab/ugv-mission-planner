import json
import os
import sys
from pathlib import Path

# Ensure src is importable
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ugv_mission_planner.genai.llm_client import OpenAILLM
from ugv_mission_planner.nl.nl_mission import ParseError, parse


def run_case(mission: str) -> bool:
    print(f"\n=== Mission ===\n{mission}")
    try:
        plan = parse(mission, OpenAILLM())
    except ParseError as e:
        print("❌ ParseError:", e, "| hints:", list(e.hints))
        return False
    print("✅ Parsed MissionPlan:")
    print(json.dumps(plan.model_dump(), indent=2, default=str))
    # Basic sanity assertions
    assert plan.constraints.max_speed_mps <= 2.0
    assert len(plan.goals) >= 1
    return True

def main() -> int:
    if not os.getenv("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run live tests.")
        return 2

    ok = True
    ok &= run_case("Go from (0,0) to (10,0), avoid [3,-1,6,2], max speed 1.0 m/s")
    ok &= run_case("Patrol between (2,2) and (18,2) twice, avoid [8,0,12,6], cap speed 1.2 m/s")
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
