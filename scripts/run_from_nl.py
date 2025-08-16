#!/usr/bin/env python
import argparse
import json
import os
import pathlib
import sys
from typing import Any

# ensure 'src' on path when running from repo root
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np

# LLM + NL parsing
from ugv_mission_planner.genai.llm_client import FakeLLM, OpenAILLM
from ugv_mission_planner.nl.nl_mission import ParseError, parse

# Planning pipeline
from ugv_mission_planner.pipeline.plan_and_execute import run_plan

# Verification + viz
from ugv_mission_planner.verify.compliance import verify_plan
from ugv_mission_planner.vis.anim import animate_run

# Trace id (fallback to uuid if helper not available)
try:
    from ugv_mission_planner.utils.trace import new_trace_id  # type: ignore
except Exception:
    import uuid

    def new_trace_id() -> str:
        return uuid.uuid4().hex[:12]


def _get_llm() -> Any:
    """Select real OpenAI LLM or a FakeLLM via env (UGV_FAKE_LLM/UGV_FAKE_PAYLOAD)."""
    if os.getenv("UGV_FAKE_LLM") == "1":
        payload = os.getenv("UGV_FAKE_PAYLOAD", "{}")
        try:
            data = json.loads(payload)
        except Exception:
            data = {}
        return FakeLLM(data)
    return OpenAILLM()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="NL → MissionPlan → Plan → Verify → (optional) GIF"
    )
    ap.add_argument("--map", required=True, help="Path to .npy occupancy grid")
    ap.add_argument("--mission", required=True, help="Natural-language mission")
    ap.add_argument(
        "--dry",
        action="store_true",
        help="Do not call executor; plan + verify only",
    )
    ap.add_argument(
        "--save-gif",
        nargs="?",
        const="auto",
        metavar="PATH",
        help="Save animation GIF (optional PATH; default runs/sim_<trace>.gif)",
    )
    args = ap.parse_args()

    trace_id = new_trace_id()
    print(f"[trace:{trace_id}] Start NL->Plan")

    # 1) Parse NL → MissionPlan (schema-validated)
    try:
        plan = parse(args.mission, _get_llm())
    except ParseError as e:
        print(f"[trace:{trace_id}] ERROR: LLM/NL parse failed; hints={list(e.hints)}")
        return 2

    print(f"[trace:{trace_id}] MissionPlan (schema-valid):")
    print(plan.model_dump_json(indent=2))

    # 2) Plan (deterministic). If external planner violates policy, our pipeline
    #    will replan with built-in A* to enforce avoid-zones.
    rr = run_plan(map_path=args.map, plan=plan, trace_id=trace_id, execute=not args.dry)

    # 3) Verify compliance (no obstacle / avoid-zone hits; speed cap respected)
    grid_for_verify = np.load(args.map)  # original map (no painted zones)
    rep = verify_plan(grid_for_verify, plan, dt=float(os.getenv("UGV_EXEC_DT", "0.1")))
    status = "PASS" if rep.ok else "FAIL"
    print(f"\n=== COMPLIANCE {status} ===")
    print(f"path_length:           {rep.path_length:.2f} m")
    print(f"steps:                 {rep.n_steps}")

    cap = float(plan.constraints.max_speed_mps)
    print(f"max_speed_used:        {rep.max_speed_used:.2f} m/s (cap {cap:.2f})")
    
    print(f"obstacle hits:         {rep.n_hits_obstacles}")
    print(f"avoid-zone hits:       {rep.n_hits_avoid}")
    print(f"min clearance to avoid {rep.min_clearance_to_avoid:.2f} m")

    # 4) Optional GIF
    if args.save_gif:
        out_path = (
            str(pathlib.Path("runs") / f"sim_{trace_id}.gif")
            if args.save_gif == "auto"
            else args.save_gif
        )
        pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        animate_run(
            grid=grid_for_verify.copy(),
            waypoints=plan.waypoints,
            avoid_zones=plan.constraints.avoid_zones,
            out_path=out_path,
            max_speed_mps=float(plan.constraints.max_speed_mps),
            show=False,
        )
        print(f"Saved animation to: {out_path}")

    final = rr.status if rep.ok else "error"
    print(f"[trace:{trace_id}] Done. Status={final}")
    return 0 if rep.ok else 4


if __name__ == "__main__":
    raise SystemExit(main())
