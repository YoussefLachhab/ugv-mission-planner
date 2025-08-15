import argparse, json, os, sys, pathlib

# Ensure src is on path when running as a script
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ugv_mission_planner.genai.llm_client import OpenAILLM, FakeLLM
from ugv_mission_planner.nl.nl_mission import parse, ParseError
from ugv_mission_planner.pipeline.plan_and_execute import run_plan
from ugv_mission_planner.utils.trace import new_trace_id

def _get_llm():
    if os.getenv("UGV_FAKE_LLM") == "1":
        return FakeLLM(json.loads(os.getenv("UGV_FAKE_PAYLOAD", "{}")))
    return OpenAILLM()

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True, help="Path to .npy occupancy grid")
    ap.add_argument("--mission", required=True, help="Natural-language mission")
    ap.add_argument("--dry", action="store_true", help="Parse+plan only (no sim)")
    args = ap.parse_args()

    trace_id = new_trace_id()
    print(f"[trace:{trace_id}] Start NL->Plan")

    try:
        plan = parse(args.mission, _get_llm())
    except ParseError as e:
        print(f"[trace:{trace_id}] ERROR: {e}; hints={list(e.hints)}")
        sys.exit(2)

    print(f"[trace:{trace_id}] MissionPlan (schema-valid):")
    print(json.dumps(plan.model_dump(), indent=2, default=str))

    result = run_plan(map_path=args.map, plan=plan, trace_id=trace_id, execute=not args.dry)
    print(f"[trace:{trace_id}] Done. Status={result.status}")

if __name__ == "__main__":
    main()
