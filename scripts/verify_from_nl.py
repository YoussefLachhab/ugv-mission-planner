import argparse, json, os, sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from ugv_mission_planner.genai.llm_client import OpenAILLM, FakeLLM
from ugv_mission_planner.nl.nl_mission import parse, ParseError
from ugv_mission_planner.pipeline.plan_and_execute import run_plan
from ugv_mission_planner.verify.compliance import verify_plan
from ugv_mission_planner.vis.anim import animate_run

def get_llm():
    if os.getenv("UGV_FAKE_LLM") == "1":
        return FakeLLM(json.loads(os.getenv("UGV_FAKE_PAYLOAD", "{}")))
    return OpenAILLM()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--mission", required=True)
    ap.add_argument("--gif", action="store_true", help="Also save a GIF to runs/sim_<id>.gif")
    args = ap.parse_args()

    try:
        plan = parse(args.mission, get_llm())
    except ParseError as e:
        print("PARSE ERROR:", e, "| hints:", list(e.hints)); sys.exit(2)

    # deterministically plan (no sim)
    rr = run_plan(map_path=args.map, plan=plan, trace_id="verify", execute=False)
    if rr.status != "ok":
        print("PLANNING ERROR"); sys.exit(3)

    grid = np.load(args.map)
    rep = verify_plan(grid, plan, dt=float(os.getenv("UGV_EXEC_DT", "0.1")))
    status = "PASS" if rep.ok else "FAIL"
    print(f"\n=== COMPLIANCE {status} ===")
    print(f"path_length:           {rep.path_length:.2f} m")
    print(f"steps:                 {rep.n_steps}")
    print(f"max_speed_used:        {rep.max_speed_used:.2f} m/s (cap {plan.constraints.max_speed_mps:.2f})")
    print(f"obstacle hits:         {rep.n_hits_obstacles}")
    print(f"avoid-zone hits:       {rep.n_hits_avoid}")
    print(f"min clearance to avoid {rep.min_clearance_to_avoid:.2f} m")

    if args.gif:
        outdir = pathlib.Path("runs"); outdir.mkdir(exist_ok=True)
        out = str(outdir / f"sim_{plan.mission_id[:8]}.gif")
        animate_run(
            grid=grid.copy(),
            waypoints=plan.waypoints,
            avoid_zones=plan.constraints.avoid_zones,
            out_path=out,
            max_speed_mps=float(plan.constraints.max_speed_mps),
            show=False,
        )
        print(f"Saved animation to: {out}")

    sys.exit(0 if rep.ok else 4)

if __name__ == "__main__":
    main()
