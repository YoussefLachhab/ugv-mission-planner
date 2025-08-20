from __future__ import annotations

import json
import math
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from ugv_mission_planner.models.itinerary import (
    LegType,
    MissionItinerary,
)

XY = tuple[int, int]


@dataclass
class RunOptions:
    save_gif: bool = True
    dt: float = 0.05              # executor timestep (s)
    conn: int = 8                 # 4 or 8 connectivity
    inflate_cells: int = 0        # obstacle dilation radius (cells)


# ---------------------------
# Grid helpers (image coords)
# ---------------------------
def _is_inside(grid: np.ndarray, x: int, y: int) -> bool:
    h, w = grid.shape
    return 0 <= x < w and 0 <= y < h


def _is_free(grid: np.ndarray, x: int, y: int) -> bool:
    return _is_inside(grid, x, y) and grid[y, x] == 0  # row=y, col=x


def _neighbors(grid: np.ndarray, x: int, y: int, conn: int) -> Iterable[XY]:
    """Connectivity with corner-cut prevention for 8-connected moves."""
    if conn == 4:
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if _is_free(grid, nx, ny):
                yield nx, ny
    else:  # 8-connected with corner rule
        for dx, dy in (
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1),
        ):
            nx, ny = x + dx, y + dy
            if not _is_free(grid, nx, ny):
                continue
            if dx != 0 and dy != 0:
                # Diagonal: both side-adjacent orthogonals must be free
                if not (_is_free(grid, x + dx, y) and _is_free(grid, x, y + dy)):
                    continue
            yield nx, ny


def _inflate_obstacles(grid: np.ndarray, r: int) -> np.ndarray:
    """Dilate obstacles by r cells (Manhattan). r=0 returns grid."""
    if r <= 0:
        return grid
    h, w = grid.shape
    out = grid.copy()
    blocked = np.argwhere(grid == 1)
    for y, x in blocked:
        for dx in range(-r, r + 1):
            rem = r - abs(dx)
            for dy in range(-rem, rem + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    out[ny, nx] = 1
    return out


def _nearest_free(grid: np.ndarray, sx: int, sy: int) -> XY:
    """BFS to the nearest free cell (ties broken by BFS order)."""
    if _is_free(grid, sx, sy):
        return sx, sy
    q: deque[XY] = deque()
    seen = set()
    q.append((sx, sy))
    seen.add((sx, sy))
    while q:
        x, y = q.popleft()
        for nx, ny in _neighbors(grid, x, y, 8):
            if (nx, ny) in seen:
                continue
            if _is_free(grid, nx, ny):
                return nx, ny
            seen.add((nx, ny))
            q.append((nx, ny))
    # As a last resort, return input (won't be free), but keeps pipeline robust.
    return sx, sy


# -----------------
# A* (deterministic)
# -----------------
def _h(p1: XY, p2: XY) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def astar(grid: np.ndarray, start: XY, goal: XY, conn: int) -> list[XY]:
    """A* on binary grid with 4/8 connectivity and corner-cut prevention."""
    if start == goal:
        return [start]

    import heapq

    open_heap: list[tuple[float, XY]] = []
    heapq.heappush(open_heap, (0.0, start))
    g_score: dict[XY, float] = {start: 0.0}
    came_from: dict[XY, XY] = {}

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == goal:
            path: list[XY] = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        cx, cy = current
        for nx, ny in _neighbors(grid, cx, cy, conn):
            tentative = g_score[current] + _h(current, (nx, ny))
            if tentative < g_score.get((nx, ny), float("inf")):
                came_from[(nx, ny)] = current
                g_score[(nx, ny)] = tentative
                f = tentative + _h((nx, ny), goal)
                heapq.heappush(open_heap, (f, (nx, ny)))

    # No path: return start only to keep executor stable.
    return [start]


# --------------------------
# Execution + artifact write
# --------------------------
def _write_ndjson_line(path: Path, obj: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


def _clip_speed(v: float, cap: float | None) -> float:
    return min(v, cap) if cap is not None else v


def _draw_frame(
    ax,
    grid: np.ndarray,
    traj: list[tuple[float, float]],
    avoid: list[tuple[float, float, float, float]],
) -> None:
    h, w = grid.shape
    ax.clear()
    ax.imshow(grid, extent=[0, w, h, 0], interpolation="nearest")
    if traj:
        xs = [p[0] for p in traj]
        ys = [p[1] for p in traj]
        ax.plot(xs, ys)
    for (x1, y1, x2, y2) in avoid:
        ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], linestyle="--")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Mission (result)")


def run_itinerary(
    grid_raw: np.ndarray,
    itinerary: MissionItinerary,
    outdir: Path,
    opts: RunOptions,
) -> Path:
    """
    Execute itinerary on a static grid with safe A*.
    Returns the directory with artifacts (outdir).
    """
    outdir.mkdir(parents=True, exist_ok=True)
    telemetry_path = outdir / "telemetry.ndjson"
    summary_path = outdir / "summary.json"
    for p in (telemetry_path, summary_path):
        if p.exists():
            p.unlink()

    # Apply obstacle inflation (safety buffer)
    grid = _inflate_obstacles(grid_raw, opts.inflate_cells)

    # Telemetry time base
    t = 0.0

    # Initial position: first explicit point or (0,0), snapped to free
    sx, sy = 0, 0
    for leg in itinerary.legs:
        if leg.points:
            sx, sy = int(round(leg.points[0].x)), int(round(leg.points[0].y))
            break
    sx, sy = _nearest_free(grid, sx, sy)
    curx, cury = float(sx), float(sy)
    _write_ndjson_line(telemetry_path, {"t": t, "x": curx, "y": cury})

    total_distance = 0.0
    max_speed_seen = 0.0
    leg_summaries: list[dict] = []

    def step_to(nx: float, ny: float, v_cap: float | None) -> float:
        nonlocal t, curx, cury, total_distance, max_speed_seen
        dx = nx - curx
        dy = ny - cury
        dist = math.hypot(dx, dy)
        if dist == 0.0:
            t += opts.dt
            _write_ndjson_line(telemetry_path, {"t": t, "x": curx, "y": cury})
            return 0.0
        v = _clip_speed(1.0, v_cap)
        max_speed_seen = max(max_speed_seen, v)
        steps = max(1, int(math.ceil(dist / (v * opts.dt))))
        for i in range(1, steps + 1):
            x = curx + dx * (i / steps)
            y = cury + dy * (i / steps)
            t += opts.dt
            _write_ndjson_line(telemetry_path, {"t": t, "x": x, "y": y})
        curx, cury = nx, ny
        total_distance += dist
        return dist

    # Execute legs
    for idx, leg in enumerate(itinerary.legs):
        leg_info: dict = {"index": idx, "type": leg.type.value, "ok": True}
        leg_speed_cap = leg.max_speed_mps or itinerary.constraints.max_speed_mps
        leg_distance = 0.0

        if leg.type == LegType.GOTO and len(leg.points) == 1:
            gx, gy = int(round(leg.points[0].x)), int(round(leg.points[0].y))
            gx, gy = _nearest_free(grid, gx, gy)
            path = astar(grid, (int(round(curx)), int(round(cury))), (gx, gy), opts.conn)
            for (x, y) in path[1:]:
                leg_distance += step_to(float(x), float(y), leg_speed_cap)

        elif leg.type == LegType.PATROL and len(leg.points) == 2:
            p1 = (int(round(leg.points[0].x)), int(round(leg.points[0].y)))
            p2 = (int(round(leg.points[1].x)), int(round(leg.points[1].y)))
            p1 = _nearest_free(grid, *p1)
            p2 = _nearest_free(grid, *p2)
            loops = max(1, leg.count or 1)
            # Move to p1 first if needed
            s = (int(round(curx)), int(round(cury)))
            if s != p1:
                path = astar(grid, s, p1, opts.conn)
                for (x, y) in path[1:]:
                    leg_distance += step_to(float(x), float(y), leg_speed_cap)
            # Loops p1 <-> p2
            cur = p1
            tgt = p2
            for _ in range(loops):
                path = astar(grid, cur, tgt, opts.conn)
                for (x, y) in path[1:]:
                    leg_distance += step_to(float(x), float(y), leg_speed_cap)
                cur, tgt = tgt, cur

        elif leg.type == LegType.LOITER and len(leg.points) == 1:
            tx, ty = int(round(leg.points[0].x)), int(round(leg.points[0].y))
            tx, ty = _nearest_free(grid, tx, ty)
            s = (int(round(curx)), int(round(cury)))
            if s != (tx, ty):
                path = astar(grid, s, (tx, ty), opts.conn)
                for (x, y) in path[1:]:
                    leg_distance += step_to(float(x), float(y), leg_speed_cap)
            steps = max(1, int(math.ceil((leg.duration_s or 0.0) / opts.dt)))
            for _ in range(steps):
                step_to(curx, cury, leg_speed_cap)

        elif leg.type == LegType.RETURN and len(leg.points) == 1:
            gx, gy = int(round(leg.points[0].x)), int(round(leg.points[0].y))
            gx, gy = _nearest_free(grid, gx, gy)
            path = astar(grid, (int(round(curx)), int(round(cury))), (gx, gy), opts.conn)
            for (x, y) in path[1:]:
                leg_distance += step_to(float(x), float(y), leg_speed_cap)
        else:
            leg_info["ok"] = False

        leg_info["distance_m"] = leg_distance
        leg_summaries.append(leg_info)

    # Write summary.json
    summary = {
        "mission_id": itinerary.mission_id,
        "legs": leg_summaries,
        "totals": {"distance_m": total_distance, "time_s": t, "max_speed_mps": max_speed_seen},
        "constraints": {
            "max_speed_mps": itinerary.constraints.max_speed_mps,
            "avoid_zones": itinerary.constraints.avoid_zones,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Optional GIF — animate from telemetry
    if opts.save_gif:
        traj: list[tuple[float, float]] = []
        with telemetry_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                traj.append((float(rec["x"]), float(rec["y"])))

        fig, ax = plt.subplots()

        def update(i: int):
            _draw_frame(ax, grid, traj[: i + 1], itinerary.constraints.avoid_zones)
            return []

        ani = animation.FuncAnimation(fig, update, frames=len(traj), interval=int(1000 * opts.dt), blit=False)
        gif_path = outdir / "mission.gif"
        ani.save(gif_path, writer="pillow")
        plt.close(fig)

    return outdir
