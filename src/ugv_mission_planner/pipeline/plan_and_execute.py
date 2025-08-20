from __future__ import annotations

import heapq
import inspect
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ugv_mission_planner import planner  # existing module
from ugv_mission_planner.models import MissionPlan, Waypoint

try:
    from ugv_mission_planner import executor
except Exception:  # pragma: no cover
    executor = None  # type: ignore[assignment]


@dataclass
class RunResult:
    status: str  # "ok" | "blocked" | "error"


def _vprint(msg: str) -> None:
    if os.getenv("UGV_VERBOSE") == "1":
        print(msg)


def _apply_avoid_zones_inplace(grid: np.ndarray, avoid_zones: list[list[float]] | None) -> None:
    """
    Paint avoid_zones as obstacles in the occupancy grid.

    Occupancy convention is configurable via env:
      - UGV_OCC_MODE = "one_is_blocked"  (default)
      - UGV_OCC_MODE = "zero_is_blocked"

    Inflate each side by `UGV_AVOID_INFLATE_CELLS` (default 1) so paths keep a
    1-cell clearance and don't "graze" the rectangle boundary.
    """
    if not avoid_zones:
        return

    mode = os.getenv("UGV_OCC_MODE", "one_is_blocked").lower()
    inflate = int(os.getenv("UGV_AVOID_INFLATE_CELLS", "1"))
    inflate = max(0, inflate)

    is_int = np.issubdtype(grid.dtype, np.integer)
    max_val = int(np.iinfo(grid.dtype).max) if is_int else 1.0

    if mode == "one_is_blocked":
        blocked_val = max_val if is_int else 1.0
    elif mode == "zero_is_blocked":
        blocked_val = 0
    else:
        blocked_val = max_val if is_int else 1.0

    h, w = grid.shape[:2]
    for rect in avoid_zones:
        if not isinstance(rect, list | tuple) or len(rect) != 4:
            continue

        xmin, ymin, xmax, ymax = rect
        # Compute integer indices and apply inflation (note: slices are half-open)
        x0 = int(np.floor(xmin)) - inflate
        y0 = int(np.floor(ymin)) - inflate
        x1 = int(np.ceil(xmax)) + inflate
        y1 = int(np.ceil(ymax)) + inflate

        # Clamp to bounds
        x0 = max(0, min(x0, w))
        y0 = max(0, min(y0, h))
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))

        if x1 > x0 and y1 > y0:
            grid[y0:y1, x0:x1] = blocked_val

    if os.getenv("UGV_VERBOSE") == "1":
        print(f"[pipeline] painted avoid-zones as blocked={blocked_val} (mode={mode}, inflate={inflate} cell)")


def _to_waypoints(seq: Sequence[Any], max_speed_mps: float) -> list[Waypoint]:
    wps: list[Waypoint] = []
    for item in seq:
        if isinstance(item, Waypoint):
            wps.append(item)
        elif isinstance(item, dict) and "x" in item and "y" in item:
            spd = float(item.get("speed_mps", max_speed_mps))
            wps.append(Waypoint(x=float(item["x"]), y=float(item["y"]), speed_mps=spd))
        elif isinstance(item, list | tuple) and len(item) >= 2:
            spd = float(item[2]) if len(item) >= 3 else float(max_speed_mps)
            wps.append(Waypoint(x=float(item[0]), y=float(item[1]), speed_mps=spd))
        else:
            raise TypeError(f"Unrecognized waypoint format: {item!r}")
    return wps


def _plan_leg_with_grid_signature(
    grid: np.ndarray,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    max_speed_mps: float,
) -> list[Waypoint]:
    fn = getattr(planner, "plan_waypoints", None)
    if fn is None:
        raise AttributeError("planner.plan_waypoints not found")

    _vprint(f"[planner] params={list(inspect.signature(fn).parameters.keys())}")

    # Positional preferred
    try:
        out = fn(grid, start_xy, goal_xy, max_speed_mps)  # type: ignore[misc]
        return _to_waypoints(out, max_speed_mps)
    except TypeError:
        # Keyword fallback
        out = fn(  # type: ignore[misc]
            grid=grid, start_xy=start_xy, goal_xy=goal_xy, max_speed_mps=max_speed_mps
        )
        return _to_waypoints(out, max_speed_mps)


def _plan_multi_leg(
    grid: np.ndarray,
    goals: list[tuple[float, float]],
    max_speed_mps: float,
) -> list[Waypoint]:
    all_wps: list[Waypoint] = []
    for i in range(len(goals) - 1):
        start_xy = goals[i]
        goal_xy = goals[i + 1]
        leg = _plan_leg_with_grid_signature(grid, start_xy, goal_xy, max_speed_mps)
        if i > 0 and leg:
            leg = leg[1:]  # avoid duplicate vertex
        all_wps.extend(leg)
    return all_wps


def _wp_as_xy_speed(wps: list[Waypoint]) -> list[tuple[float, float, float]]:
    return [(float(w.x), float(w.y), float(w.speed_mps)) for w in wps]


def _wp_as_xy(wps: list[Waypoint]) -> list[tuple[float, float]]:
    return [(float(w.x), float(w.y)) for w in wps]


def _call_execute_waypoints(
    waypoints: list[Waypoint],
    max_speed_mps: float,
    trace_id: str,
) -> Any:
    if executor is None or not hasattr(executor, "execute_waypoints"):
        _vprint("[executor] no executor found; skipping")
        return "skipped"

    fn = executor.execute_waypoints  # type: ignore[attr-defined]
    params = list(inspect.signature(fn).parameters.keys())
    _vprint(f"[executor] params={params}")

    # Allow user to control dt via env, default 0.1 s
    dt = float(os.getenv("UGV_EXEC_DT", "0.1"))

    # Most likely shapes given your log: (waypoints, dt)
    # Try 3-tuples first (x, y, speed)
    try:
        return fn(waypoints=_wp_as_xy_speed(waypoints), dt=dt)  # type: ignore[misc]
    except Exception as e:
        _vprint(f"[executor] 3-tuple call failed: {e!r}")

    # Then try 2-tuples (x, y)
    try:
        return fn(waypoints=_wp_as_xy(waypoints), dt=dt)  # type: ignore[misc]
    except Exception as e:
        _vprint(f"[executor] 2-tuple call failed: {e!r}")

    # Positional variants
    try:
        return fn(_wp_as_xy_speed(waypoints), dt)  # type: ignore[misc]
    except Exception as e:
        _vprint(f"[executor] positional 3-tuple failed: {e!r}")

    try:
        return fn(_wp_as_xy(waypoints), dt)  # type: ignore[misc]
    except Exception as e:
        _vprint(f"[executor] positional 2-tuple failed: {e!r}")

    raise TypeError(
        "executor.execute_waypoints couldn't be called with supported shapes (tried (x,y,speed), (x,y) × kw/pos)."
    )


# ----------------------- Compliance + A* fallback -----------------------


def _point_in_rect_xy(p: tuple[float, float], rect: list[float]) -> bool:
    x, y = p
    xmin, ymin, xmax, ymax = rect
    return (xmin <= x <= xmax) and (ymin <= y <= ymax)


def _hits_avoid(waypoints: list[Waypoint], avoid_zones: list[list[float]] | None) -> int:
    if not waypoints or not avoid_zones:
        return 0
    hits = 0
    for a, b in zip(waypoints[:-1], waypoints[1:], strict=False):
        x0, y0 = float(a.x), float(a.y)
        x1, y1 = float(b.x), float(b.y)
        dist = math.hypot(x1 - x0, y1 - y0)
        steps = max(1, int(dist / 0.25))
        for k in range(steps + 1):
            t = k / steps
            x = x0 + t * (x1 - x0)
            y = y0 + t * (y1 - y0)
            for rect in avoid_zones:
                if _point_in_rect_xy((x, y), rect):
                    hits += 1
    return hits


def _passable(grid: np.ndarray, rc: tuple[int, int], occ_mode: str) -> bool:
    r, c = rc
    h, w = grid.shape[:2]
    if not (0 <= r < h and 0 <= c < w):
        return False
    v = grid[r, c]
    if occ_mode == "zero_is_blocked":
        return v != 0
    else:  # "one_is_blocked" (or anything else) => nonzero is blocked
        return v == 0


def _astar_grid(
    grid: np.ndarray,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    occ_mode: str,
) -> list[tuple[int, int]]:
    """8-neighbor A* on integer grid coordinates, returns list of (x,y) cells."""
    sx, sy = start_xy
    gx, gy = goal_xy
    start = (int(round(sy)), int(round(sx)))  # (r,c)
    goal = (int(round(gy)), int(round(gx)))

    if not _passable(grid, start, occ_mode) or not _passable(grid, goal, occ_mode):
        return []

    nbrs8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    g = {start: 0.0}
    came: dict[tuple[int, int], tuple[int, int]] = {}
    pq: list[tuple[float, tuple[int, int]]] = [(0.0, start)]

    def hfun(rc: tuple[int, int]) -> float:
        r, c = rc
        return math.hypot(r - goal[0], c - goal[1])

    while pq:
        _, cur = heapq.heappop(pq)
        if cur == goal:
            # reconstruct
            path_rc = [cur]
            while cur in came:
                cur = came[cur]
                path_rc.append(cur)
            path_rc.reverse()
            return [(rc[1], rc[0]) for rc in path_rc]  # (x,y)

        # expand
        cr, cc = cur
        for dr, dc in nbrs8:
            nr, nc = cr + dr, cc + dc
            nxt = (nr, nc)
            if not _passable(grid, nxt, occ_mode):
                continue
            cost = 1.0 if (dr == 0 or dc == 0) else math.sqrt(2)
            tentative = g[cur] + cost
            if tentative < g.get(nxt, float("inf")):
                g[nxt] = tentative
                came[nxt] = cur
                heapq.heappush(pq, (tentative + hfun(nxt), nxt))
    return []


def _wp_from_cells(cells: list[tuple[int, int]], speed: float) -> list[Waypoint]:
    return [Waypoint(x=float(x), y=float(y), speed_mps=float(speed)) for (x, y) in cells]


# ------------------------------- Pipeline -------------------------------


def run_plan(map_path: str, plan: MissionPlan, trace_id: str, execute: bool) -> RunResult:
    try:
        grid = np.load(map_path)
        grid = grid.copy()
        _apply_avoid_zones_inplace(grid, plan.constraints.avoid_zones)

        waypoints = _plan_multi_leg(
            grid=grid,
            goals=plan.goals,
            max_speed_mps=float(plan.constraints.max_speed_mps),
        )

        # Enforce avoid-zone compliance; replan with built-in A* if needed
        occ_mode = os.getenv("UGV_OCC_MODE", "one_is_blocked").lower()
        if _hits_avoid(waypoints, plan.constraints.avoid_zones) > 0 or not waypoints:
            _vprint("[pipeline] external plan violated avoid-zones or was empty; replanning with A*")
            fixed: list[Waypoint] = []
            for s, t in zip(plan.goals[:-1], plan.goals[1:], strict=False):
                cells = _astar_grid(grid, s, t, occ_mode)
                if not cells:
                    raise RuntimeError("Built-in A* failed to find a path")
                seg = _wp_from_cells(cells, float(plan.constraints.max_speed_mps))
                if fixed and seg:
                    seg = seg[1:]  # drop duplicate join
                fixed.extend(seg)
            waypoints = fixed

        plan.waypoints = waypoints
        _vprint(f"[planner] produced {len(waypoints)} waypoints")

        if execute:
            _call_execute_waypoints(
                waypoints=waypoints,
                max_speed_mps=float(plan.constraints.max_speed_mps),
                trace_id=trace_id,
            )

        return RunResult(status="ok")
    except Exception as e:
        _vprint(f"[pipeline] ERROR: {e!r}")
        return RunResult(status="error")
