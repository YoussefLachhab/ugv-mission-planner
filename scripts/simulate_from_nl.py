import argparse
import json
import os
import sys
import pathlib

# ensure 'src' on path
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from ugv_mission_planner.genai.llm_client import OpenAILLM, FakeLLM
from ugv_mission_planner.nl.nl_mission import parse, ParseError
from ugv_mission_planner.pipeline.plan_and_execute import _apply_avoid_zones_inplace  # reuse
from ugv_mission_planner.vis.anim import animate_run

def get_llm():
    if os.getenv("UGV_FAKE_LLM") == "1":
        return FakeLLM(json.loads(os.getenv("UGV_FAKE_PAYLOAD", "{}")))
    return OpenAILLM()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True, help="Path to .npy occupancy grid")
    ap.add_argument("--mission", required=True, help="Natural-language mission")
    ap.add_argument("--out", default=None, help="Output GIF/MP4 path (default: runs/<trace>.gif)")
    ap.add_argument("--show", action="store_true", help="Also show a live window")
    args = ap.parse_args()

    # parse NL -> plan
    try:
        plan = parse(args.mission, get_llm())
    except ParseError as e:
        print("ERROR:", e, "| hints:", list(e.hints))
        sys.exit(2)

    # load grid and apply avoid zones
    grid = np.load(args.map).copy()
    _apply_avoid_zones_inplace(grid, plan.constraints.avoid_zones)

    # build default out path
    out = args.out
    if not out:
        out_dir = pathlib.Path("runs")
        out_dir.mkdir(parents=True, exist_ok=True)
        out = str(out_dir / f"sim_{plan.mission_id[:8]}.gif")

    # animate and save
    out_path = animate_run(
        grid=grid,
        waypoints=plan.waypoints if plan.waypoints else [  # if you call this before planning,
            # we degrade gracefully: animate the goals polyline
            *[type("W", (), {"x": g[0], "y": g[1], "speed_mps": plan.constraints.max_speed_mps}) for g in plan.goals]
        ],
        avoid_zones=plan.constraints.avoid_zones,
        out_path=out,
        max_speed_mps=float(plan.constraints.max_speed_mps),
        show=args.show,
    )
    print(f"Saved animation to: {out_path}")

if __name__ == "__main__":
    main()
