from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple
import math
import numpy as np

from ugv_mission_planner.models import MissionPlan, Waypoint


@dataclass
class ComplianceReport:
    ok: bool
    reasons: List[str] = field(default_factory=list)
    path_length: float = 0.0
    n_steps: int = 0
    n_hits_obstacles: int = 0
    n_hits_avoid: int = 0
    min_clearance_to_avoid: float | None = None
    max_speed_used: float = 0.0


def _resample(waypoints: List[Waypoint], dt: float, max_speed: float) -> List[Tuple[float, float]]:
    if len(waypoints) < 2:
        return [(float(w.x), float(w.y)) for w in waypoints]
    pts: List[Tuple[float, float]] = []
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        x0, y0 = float(a.x), float(a.y)
        x1, y1 = float(b.x), float(b.y)
        dx, dy = x1 - x0, y1 - y0
        d = math.hypot(dx, dy)
        if d == 0:
            pts.append((x0, y0)); continue
        spd = min(float(a.speed_mps), float(b.speed_mps), max_speed)
        steps = max(1, int(math.ceil((d / max(spd, 1e-6)) / dt)))
        for k in range(steps):
            t = k / steps
            pts.append((x0 + t * dx, y0 + t * dy))
    pts.append((float(waypoints[-1].x), float(waypoints[-1].y)))
    return pts


def _point_in_rect(p: Tuple[float, float], rect: List[float], radius: float = 0.0) -> bool:
    x, y = p
    xmin, ymin, xmax, ymax = rect
    return (x >= xmin - radius and x <= xmax + radius and
            y >= ymin - radius and y <= ymax + radius)


def _dist_point_rect(p: Tuple[float, float], rect: List[float]) -> float:
    x, y = p
    xmin, ymin, xmax, ymax = rect
    dx = max(xmin - x, 0.0, x - xmax)
    dy = max(ymin - y, 0.0, y - ymax)
    return math.hypot(dx, dy)


def verify_plan(map_grid: np.ndarray, plan: MissionPlan, dt: float = 0.1, robot_radius: float = 0.25) -> ComplianceReport:
    """Check: no obstacle hits, no avoid-zone hits, speeds under cap."""
    report = ComplianceReport(ok=True)

    # resample path
    pts = _resample(plan.waypoints, dt, float(plan.constraints.max_speed_mps))
    report.n_steps = len(pts)

    # path length
    report.path_length = sum(
        math.hypot(pts[i+1][0] - pts[i][0], pts[i+1][1] - pts[i][1]) for i in range(len(pts)-1)
    )

    # speed cap (on waypoints themselves)
    report.max_speed_used = max((float(w.speed_mps) for w in plan.waypoints), default=0.0)
    if report.max_speed_used > float(plan.constraints.max_speed_mps) + 1e-9:
        report.ok = False
        report.reasons.append(f"speed cap exceeded: used={report.max_speed_used:.3f} cap={float(plan.constraints.max_speed_mps):.3f}")

    # obstacle hits (grid==1)
    h, w = map_grid.shape[:2]
    hits_obs = 0
    for x, y in pts:
        xi = int(round(x)); yi = int(round(y))
        if 0 <= xi < w and 0 <= yi < h and map_grid[yi, xi] == 1:
            hits_obs += 1
    report.n_hits_obstacles = hits_obs
    if hits_obs > 0:
        report.ok = False
        report.reasons.append(f"path enters occupied cells: {hits_obs} steps")

    # avoid zones (inflate by robot_radius)
    hits_avoid = 0
    az = plan.constraints.avoid_zones or []
    min_clr = None
    for p in pts:
        for rect in az:
            if _point_in_rect(p, rect, robot_radius):
                hits_avoid += 1
            d = _dist_point_rect(p, rect)
            min_clr = d if min_clr is None else min(min_clr, d)
    report.n_hits_avoid = hits_avoid
    report.min_clearance_to_avoid = 0.0 if min_clr is None else float(min_clr)
    if hits_avoid > 0:
        report.ok = False
        report.reasons.append(f"path violates avoid zone: {hits_avoid} steps")

    return report
