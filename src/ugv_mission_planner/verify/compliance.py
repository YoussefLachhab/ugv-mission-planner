from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple
import math

from ugv_mission_planner.models import Waypoint


@dataclass
class Report:
    pass_ok: bool
    path_length_m: float
    steps: int
    max_speed_used_mps: float
    obstacle_hits: int
    avoid_hits: int
    min_clearance_m: float


def _interp_points(
    wps: Sequence[Waypoint], dt: float, max_speed: float
) -> List[Tuple[float, float, float]]:
    pts: List[Tuple[float, float, float]] = []
    if not wps:
        return pts
    for a, b in zip(wps[:-1], wps[1:]):
        x0, y0 = float(a.x), float(a.y)
        x1, y1 = float(b.x), float(b.y)
        dx, dy = x1 - x0, y1 - y0
        d = math.hypot(dx, dy)
        if d == 0:
            pts.append((x0, y0, float(a.speed_mps)))
            continue
        spd = min(float(a.speed_mps), float(b.speed_mps), max_speed)
        steps = max(1, int(math.ceil((d / max(spd, 1e-6)) / dt)))
        for i in range(steps):
            t = (i + 1) / steps
            pts.append((x0 + t * dx, y0 + t * dy, spd))
    return pts


def _hits(map_grid, pts_xy: Iterable[Tuple[float, float]]) -> int:
    h, w = map_grid.shape[:2]
    hits = 0
    for x, y in pts_xy:
        xi = int(round(x))
        yi = int(round(y))
        if 0 <= xi < w and 0 <= yi < h and map_grid[yi, xi] == 1:
            hits += 1
    return hits


def _hits_avoid(pts_xy: Iterable[Tuple[float, float]], avoid_zones: List[List[float]]) -> int:
    hits = 0
    for x, y in pts_xy:
        for (xmin, ymin, xmax, ymax) in avoid_zones or []:
            if xmin <= x <= xmax and ymin <= y <= ymax:
                hits += 1
                break
    return hits


def _min_clearance_to_avoid(pts_xy: Iterable[Tuple[float, float]], avoid_zones: List[List[float]]) -> float:
    if not avoid_zones:
        return float("inf")
    best = float("inf")
    for x, y in pts_xy:
        for (xmin, ymin, xmax, ymax) in avoid_zones:
            dx = min(abs(x - xmin), abs(x - xmax), 0 if xmin <= x <= xmax else float("inf"))
            dy = min(abs(y - ymin), abs(y - ymax), 0 if ymin <= y <= ymax else float("inf"))
            best = min(best, math.hypot(dx, dy))
    return 0.0 if best == float("inf") else best


def compliance_report(map_grid, avoid_zones, waypoints, max_speed) -> Report:
    pts = _interp_points(waypoints, dt=0.05, max_speed=max_speed)
    pts_xy = [(x, y) for (x, y, _s) in pts]
    hits_obs = _hits(map_grid, pts_xy)
    hits_avoid = _hits_avoid(pts_xy, avoid_zones)
    min_clear = _min_clearance_to_avoid(pts_xy, avoid_zones)
    path_len = sum(
        math.hypot(b.x - a.x, b.y - a.y) for a, b in zip(waypoints[:-1], waypoints[1:])
    )
    max_spd_used = max([w.speed_mps for w in waypoints], default=0.0)
    ok = hits_obs == 0 and hits_avoid == 0
    return Report(
        pass_ok=ok,
        path_length_m=path_len,
        steps=len(pts),
        max_speed_used_mps=float(max_spd_used),
        obstacle_hits=hits_obs,
        avoid_hits=hits_avoid,
        min_clearance_m=float(min_clear),
    )
