from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Iterable
import math
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle

from ugv_mission_planner.models import Waypoint


@dataclass
class SimConfig:
    dt: float = 0.1             # seconds per step
    fps: int = 20               # frames per second in the saved GIF
    robot_radius: float = 0.25  # for plotting the dot size
    trail_len: int = 60         # how many recent points to keep in the trail


def _segment_speed(w0: Waypoint, w1: Waypoint, max_speed: float) -> float:
    # conservative: use the lower of the two speeds, capped by max_speed
    return float(min(max_speed, w0.speed_mps, w1.speed_mps))


def _resample_path(
    waypoints: List[Waypoint],
    dt: float,
    max_speed: float,
) -> List[Tuple[float, float]]:
    """
    Turn waypoints into a list of (x, y) robot positions sampled every dt seconds.
    Uses linear interpolation with per-segment constant speed.
    """
    if len(waypoints) < 2:
        return [(float(w.x), float(w.y)) for w in waypoints]

    poses: List[Tuple[float, float]] = []
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        x0, y0 = float(a.x), float(a.y)
        x1, y1 = float(b.x), float(b.y)
        dx, dy = x1 - x0, y1 - y0
        dist = math.hypot(dx, dy)
        if dist == 0:
            # same point: still record at least once
            poses.append((x0, y0))
            continue
        speed = max(_segment_speed(a, b, max_speed), 1e-6)
        steps = max(1, int(math.ceil((dist / speed) / dt)))
        for k in range(steps):
            t = k / steps
            poses.append((x0 + t * dx, y0 + t * dy))
    # ensure final point present
    poses.append((float(waypoints[-1].x), float(waypoints[-1].y)))
    return poses


def animate_run(
    grid: np.ndarray,
    waypoints: List[Waypoint],
    avoid_zones: List[List[float]] | None,
    out_path: str,
    max_speed_mps: float,
    show: bool = False,
    cfg: SimConfig | None = None,
) -> str:
    """
    Render a simple 2D sim:
      - background: occupancy grid (0 free, 1 obstacle)
      - red rectangles: avoid zones
      - blue path: waypoint polyline
      - moving dot: robot position
    Saves GIF to `out_path`. If `show=True`, also opens a live window.
    Returns the output path.
    """
    cfg = cfg or SimConfig(dt=float(os.getenv("UGV_EXEC_DT", "0.1")))
    poses = _resample_path(waypoints, cfg.dt, max_speed_mps)

    h, w = grid.shape[:2]
    fig, ax = plt.subplots(figsize=(6, 6 * h / w if w else 6))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)  # invert y for matrix-style origin at top-left

    # draw grid
    ax.imshow(grid, cmap="Greys", interpolation="nearest", vmin=0, vmax=1)

    # draw avoid zones
    for rect in (avoid_zones or []):
        if len(rect) != 4:
            continue
        xmin, ymin, xmax, ymax = rect
        r = Rectangle(
            (xmin, ymin),
            width=max(0.0, xmax - xmin),
            height=max(0.0, ymax - ymin),
            fill=False,
            edgecolor="red",
            linewidth=1.5,
            linestyle="--",
        )
        ax.add_patch(r)

    # draw planned polyline
    if waypoints:
        xs = [float(w.x) for w in waypoints]
        ys = [float(w.y) for w in waypoints]
        ax.plot(xs, ys, linewidth=1.5)

    # robot marker + trail
    (dot,) = ax.plot([], [], marker="o", markersize=6, linestyle="none")
    trail, = ax.plot([], [], linewidth=1)

    def init():
        dot.set_data([], [])
        trail.set_data([], [])
        return (dot, trail)

    def update(i: int):
        x, y = poses[i]
        dot.set_data([x], [y])
        j0 = max(0, i - cfg.trail_len)
        tx = [p[0] for p in poses[j0:i + 1]]
        ty = [p[1] for p in poses[j0:i + 1]]
        trail.set_data(tx, ty)
        return (dot, trail)

    anim = animation.FuncAnimation(
        fig, update, init_func=init, frames=len(poses), interval=1000 / cfg.fps, blit=True
    )

    # Save GIF via Pillow if available; otherwise try ffmpeg (mp4)
    out_path = os.fspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        from matplotlib.animation import PillowWriter
        anim.save(out_path, writer=PillowWriter(fps=cfg.fps))
    except Exception:
        try:
            from matplotlib.animation import FFMpegWriter
            mp4 = os.path.splitext(out_path)[0] + ".mp4"
            anim.save(mp4, writer=FFMpegWriter(fps=cfg.fps))
            out_path = mp4
        except Exception as e:
            plt.close(fig)
            raise RuntimeError(f"Could not save animation: {e!r}")

    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_path
