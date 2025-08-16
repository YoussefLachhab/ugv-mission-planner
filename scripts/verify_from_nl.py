#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from ugv_mission_planner.genai.llm_client import get_llm
from ugv_mission_planner.models import MissionPlan
from ugv_mission_planner.nl.nl_mission import ParseError, parse
from ugv_mission_planner.pipeline.plan_and_execute import run_plan
from ugv_mission_planner.verify.compliance import compliance_report
from ugv_mission_planner.vis.anim import animate_run


def main() -> int:
    parser = argparse.ArgumentParser(description="NL → Plan → Verify (+ optional GIF)")
    parser.add_argument("--map", required=True, help="Path to .npy occupancy grid")
    parser.add_argument("--mission", required=True, help="Natural-language mission text")
    parser.add_argument("--gif", action="store_true", help="Save GIF to runs/sim_<id>.gif")
    args = parser.parse_args()

    # 1) Parse NL → MissionPlan (using LLM or FakeLLM)
    try:
        plan: MissionPlan = parse(args.mission, get_llm())
    except ParseError as e:
        print("PARSE ERROR:", e, "| hints:", list(e.hints))
        sys.exit(2)

    # 2) Deterministic plan (no simulation)
    rr = run_plan(map_path=args.map, plan=plan, trace_id="verify", execute=False)
    if rr.status != "ok":
        print("PLANNING ERROR")
        sys.exit(3)

    # 3) Compliance check
    grid = np.load(args.map)
    report = compliance_report(
        map_grid=grid,
        avoid_zones=plan.constraints.avoid_zones or [],
        waypoints=plan.waypoints,
        max_speed=float(plan.constraints.max_speed_mps),
    )

    print("\n=== COMPLIANCE " + ("PASS" if report.pass_ok else "FAIL") + " ===")
    print(f"path_length:           {report.path_length_m:.2f} m")
    print(f"steps:                 {report.steps}")
    
    cap = float(plan.constraints.max_speed_mps)
    print(f"max_speed_used:        {report.max_speed_used_mps:.2f} m/s (cap {cap:.2f})")

    print(f"obstacle hits:         {report.obstacle_hits}")
    print(f"avoid-zone hits:       {report.avoid_hits}")
    print(f"min clearance to avoid {report.min_clearance_m:.2f} m")

    # 4) Optional GIF
    if args.gif:
        outdir = Path("runs")
        outdir.mkdir(exist_ok=True)
        out = str(outdir / f"sim_{plan.mission_id[:8]}.gif")
        animate_run(
            map_grid=grid,
            avoid_zones=plan.constraints.avoid_zones or [],
            waypoints=plan.waypoints,
            out_path=out,
        )
        print(f"Saved animation to: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
