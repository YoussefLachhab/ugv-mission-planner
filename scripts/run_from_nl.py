#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Any, Dict

import numpy as np

# --- Project imports ----------------------------------------------------------
try:
    from ugv_mission_planner.planner import plan_waypoints, rasterize_avoid_zones
except Exception as e:
    raise SystemExit(f"[FATAL] Cannot import planner modules: {e}")

try:
    from ugv_mission_planner.executor import execute_waypoints  # type: ignore
except Exception:
    execute_waypoints = None  # type: ignore


# --- Data models --------------------------------------------------------------
@dataclass
class MissionSpec:
    start: Tuple[float, float]
    goal: Tuple[float, float]
    avoid_rect: Optional[Tuple[int, int, int, int]] = None  # x0,y0,x1,y1
    max_speed_mps: float = 1.0

@dataclass
class RunArtifacts:
    plan_waypoints: List[Tuple[int, int]]
    total_waypoints: int
    mission: MissionSpec
    map_path: str
    connectivity: int
    inflate: int
    outdir: str
    created_at: str


# --- Utilities ----------------------------------------------------------------
_COORD_PAIR = re.compile(r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)")
_RECT4 = re.compile(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]")
_SPEED = re.compile(r"max\s*speed\s*([0-9]+(?:\.[0-9]+)?)\s*m/?s", re.IGNORECASE)

def _parse_all_pairs(s: str) -> List[Tuple[float, float]]:
    return [(float(a), float(b)) for a, b in _COORD_PAIR.findall(s)]

def _parse_rect4(s: str) -> Optional[Tuple[int, int, int, int]]:
    m = _RECT4.search(s)
    if not m:
        return None
    x0, y0, x1, y1 = (int(float(x)) for x in m.groups())
    # normalize order
    if x0 > x1: x0, x1 = x1, x0
    if y0 > y1: y0, y1 = y1, y0
    return (x0, y0, x1, y1)

def _parse_speed(s: str, default: float = 1.0) -> float:
    m = _SPEED.search(s)
    return float(m.group(1)) if m else default

def _coerce_int_pairs(points: Iterable[Tuple[float, float]]) -> List[Tuple[int, int]]:
    return [(int(round(x)), int(round(y))) for x, y in points]

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _write_json(p: Path, obj: Any) -> None:
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def _write_text(p: Path, text: str) -> None:
    with p.open("w", encoding="utf-8") as f:
        f.write(text)

def _clip_ij(ij: Tuple[int, int], h: int, w: int) -> Tuple[int, int]:
    i, j = ij
    return (max(0, min(i, h - 1)), max(0, min(j, w - 1)))

def _rect_in_bounds(rect: Tuple[int, int, int, int], h: int, w: int) -> Optional[Tuple[int, int, int, int]]:
    x0, y0, x1, y1 = rect
    # clip to bounds
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(w - 1, x1), min(h - 1, y1)
    if x0c > x1c or y0c > y1c:
        return None
    return (x0c, y0c, x1c, y1c)


# --- NL mission parsing -------------------------------------------------------
def parse_mission(mission: str) -> MissionSpec:
    ms = mission.strip().lower()

    # Prefer explicit "from (...) to (...)" if present
    pairs = _parse_all_pairs(ms)
    if len(pairs) >= 2:
        start, goal = pairs[0], pairs[1]
    else:
        raise ValueError("Could not parse start/goal. Include two pairs like (x,y) ... to (x,y).")

    avoid_rect = _parse_rect4(ms)
    max_speed = _parse_speed(ms, default=1.0)

    return MissionSpec(
        start=(float(start[0]), float(start[1])),
        goal=(float(goal[0]), float(goal[1])),
        avoid_rect=avoid_rect,
        max_speed_mps=float(max_speed),
    )


# --- Core workflow ------------------------------------------------------------
def run(
    map_path: Path,
    mission_text: str,
    connectivity: int = 8,
    inflate: int = 0,
    outdir: Path = Path("runs") / "latest",
    save_gif: bool = False,
    auto_amend: bool = False,
    dry: bool = False,
) -> RunArtifacts:
    # Load map
    if not map_path.exists():
        raise FileNotFoundError(f"Map not found: {map_path}")
    grid = np.load(map_path)
    if grid.ndim != 2:
        raise ValueError("Map must be a 2D numpy array (occupancy grid).")
    h, w = grid.shape

    # Parse mission
    mission = parse_mission(mission_text)

    # Auto-amend trivial issue: start==goal → nudge goal
    if auto_amend and mission.start == mission.goal:
        mission = MissionSpec(
            start=mission.start,
            goal=(mission.goal[0] + 1.0, mission.goal[1]),
            avoid_rect=mission.avoid_rect,
            max_speed_mps=mission.max_speed_mps,
        )

    # Coerce → int indices
    start_ij = _coerce_int_pairs([mission.start])[0]
    goal_ij = _coerce_int_pairs([mission.goal])[0]

    # Bounds check (clip if auto_amend; else fail with a helpful error)
    def _oob_msg(lbl: str, ij: Tuple[int, int]) -> str:
        return (f"{lbl} {ij} is out of bounds for map {w}x{h}. "
                f"Valid index range is x:[0,{w-1}], y:[0,{h-1}]. "
                f"Tip: use --auto-amend or keep goal <= ({w-1},{h-1}).")

    oob = []
    if not (0 <= start_ij[0] < h and 0 <= start_ij[1] < w):
        oob.append(("start", start_ij))
    if not (0 <= goal_ij[0] < h and 0 <= goal_ij[1] < w):
        oob.append(("goal", goal_ij))

    if oob and not auto_amend:
        msgs = "; ".join(_oob_msg(lbl, ij) for lbl, ij in oob)
        raise ValueError(msgs)

    if oob and auto_amend:
        orig_start, orig_goal = start_ij, goal_ij
        start_ij = _clip_ij(start_ij, h, w)
        goal_ij = _clip_ij(goal_ij, h, w)
        print(f"[auto-amend] Clipped start {orig_start}->{start_ij}, goal {orig_goal}->{goal_ij}")

    # Avoid zone: clip to bounds if requested
    avoid_rect = None
    if mission.avoid_rect is not None:
        r = _rect_in_bounds(mission.avoid_rect, h, w)
        if r is None:
            if auto_amend:
                print(f"[auto-amend] Ignoring avoid rect {mission.avoid_rect} (out of bounds after clipping).")
            else:
                raise ValueError(
                    f"Avoid rect {mission.avoid_rect} is outside bounds of map {w}x{h}. "
                    f"Use --auto-amend or choose an in-range rectangle."
                )
        else:
            avoid_rect = r

    # Prepare working grid with avoid zone
    work_grid = grid.copy()
    if avoid_rect is not None:
        try:
            work_grid = rasterize_avoid_zones(work_grid, [avoid_rect], inflate=inflate)
        except TypeError:
            work_grid = rasterize_avoid_zones(work_grid, [avoid_rect])  # type: ignore

    # Plan
    try:
        waypoints = plan_waypoints(
            grid=work_grid,
            start=start_ij,
            goal=goal_ij,
            connectivity=connectivity,
            inflate=inflate,
            max_speed_mps=mission.max_speed_mps,  # try if supported
        )  # type: ignore
    except TypeError:
        try:
            waypoints = plan_waypoints(work_grid, start_ij, goal_ij, connectivity)  # type: ignore
        except TypeError:
            waypoints = plan_waypoints(work_grid, start_ij, goal_ij)  # type: ignore

    if not waypoints:
        raise RuntimeError("Planner returned no waypoints (no path found).")

    # Outputs
    _ensure_dir(outdir)
    ts = datetime.now(UTC).isoformat()

    _write_text(outdir / "mission.txt", mission_text.strip())
    _write_json(outdir / "plan.json", {
        "waypoints": waypoints,
        "total": len(waypoints),
        "start": start_ij,
        "goal": goal_ij,
        "connectivity": connectivity,
        "inflate": inflate,
    })

    telemetry: Optional[List[Dict[str, Any]]] = None
    if not dry and execute_waypoints is not None:
        try:
            telemetry = execute_waypoints(waypoints, max_speed_mps=mission.max_speed_mps)  # type: ignore
        except TypeError:
            try:
                telemetry = execute_waypoints(waypoints)  # type: ignore
            except Exception:
                telemetry = None
        if telemetry is not None:
            _write_json(outdir / "telemetry.json", telemetry)

    if save_gif:
        if not _try_render_gif(grid, waypoints, outdir / "mission.gif"):
            _write_text(outdir / "_gif_failed.txt", "GIF rendering not available.")

    return RunArtifacts(
        plan_waypoints=waypoints,
        total_waypoints=len(waypoints),
        mission=mission,
        map_path=str(map_path),
        connectivity=connectivity,
        inflate=inflate,
        outdir=str(outdir),
        created_at=ts,
    )

def _try_render_gif(grid: np.ndarray, waypoints: List[Tuple[int, int]], out_gif: Path) -> bool:
    try:
        from ugv_mission_planner.vis.animate import save_path_gif  # type: ignore
        save_path_gif(grid, waypoints, str(out_gif))
        return True
    except Exception:
        pass
    try:
        from ugv_mission_planner.vis.anim import save_path_gif  # type: ignore
        save_path_gif(grid, waypoints, str(out_gif))
        return True
    except Exception:
        pass
    return False


# --- CLI ----------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_from_nl",
        description="Parse an NL mission, plan a path on a grid map, optionally execute & render.",
    )
    p.add_argument("--map", required=True, type=Path, help="Path to .npy occupancy grid (2D).")
    p.add_argument("--mission", required=True, type=str,
                   help='e.g. "go from (2,2) to (18,2), avoid [8,0,12,6], max speed 1.2 m/s"')
    p.add_argument("--conn", "--connectivity", dest="connectivity", type=int, default=8,
                   help="Connectivity (4 or 8). Default: 8.")
    p.add_argument("--inflate", type=int, default=0, help="Obstacle inflation (cells). Default: 0.")
    p.add_argument("--outdir", type=Path, default=Path("runs") / "latest",
                   help="Output directory. Default: runs/latest.")
    gif = p.add_mutually_exclusive_group()
    gif.add_argument("--save-gif", dest="save_gif", action="store_true", help="Render mission.gif if possible.")
    gif.add_argument("--no-save-gif", dest="save_gif", action="store_false", help="Do not render GIF.")
    p.set_defaults(save_gif=False)
    p.add_argument("--auto-amend", action="store_true",
                   help="Automatically clip OOB coordinates/rectangles and nudge trivial issues.")
    p.add_argument("--dry", action="store_true", help="Plan only; skip execution.")
    return p

def main(argv: Optional[List[str]] = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)
    try:
        artifacts = run(
            map_path=args.map,
            mission_text=args.mission,
            connectivity=int(args.connectivity),
            inflate=int(args.inflate),
            outdir=args.outdir,
            save_gif=bool(args.save_gif),
            auto_amend=bool(args.auto_amend),
            dry=bool(args.dry),
        )
    except Exception as e:
        print(f"[ERROR] {e}")
        return 2

    print("=== UGV Mission Planner ===")
    print(f"Map:        {artifacts.map_path}")
    print(f"Start→Goal: {artifacts.mission.start} -> {artifacts.mission.goal}")
    if artifacts.mission.avoid_rect:
        print(f"Avoid:      {artifacts.mission.avoid_rect}")
    print(f"Speed:      {artifacts.mission.max_speed_mps} m/s")
    print(f"Waypoints:  {artifacts.total_waypoints}")
    print(f"Out dir:    {artifacts.outdir}")
    print(f"Created:    {artifacts.created_at}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
