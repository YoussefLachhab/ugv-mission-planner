from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Tuple, Any

import numpy as np

# headless-safe
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _fig_to_rgb_array(fig) -> np.ndarray:
    # Matplotlib ≥3.8 friendly (RGBA buffer)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape((h, w, 4))
    return rgba[:, :, :3].copy()


def _to_xy(w: Any) -> Tuple[int, int]:
    """
    Normalize a waypoint record to (x, y) ints.
    Accepts:
      - (x,y), [x,y], np.array([x,y])
      - (x,y,anything...), [x,y,anything...]
      - {"x": x, "y": y} or {"i": i, "j": j}
    """
    if isinstance(w, dict):
        if "x" in w and "y" in w:
            return int(round(float(w["x"]))), int(round(float(w["y"])))
        if "i" in w and "j" in w:
            # allow row/col naming
            return int(round(float(w["i"]))), int(round(float(w["j"])))
        # try first two keys as fallback
        vals = list(w.values())
        if len(vals) >= 2:
            return int(round(float(vals[0]))), int(round(float(vals[1])))

    # sequence-like
    if isinstance(w, (list, tuple, np.ndarray)):
        if len(w) >= 2:
            return int(round(float(w[0]))), int(round(float(w[1])))

    # last resort: fail loud
    raise ValueError(f"Unsupported waypoint shape: {type(w)} -> {w!r}")


def _maybe_swap_xy(pts: List[Tuple[int, int]], w: int, h: int) -> List[Tuple[int, int]]:
    """
    Waypoints may be (i,j) (row,col) or (x,y). If some x exceed width or y exceed height
    but swapping fixes it, auto-swap.
    """
    def in_bounds(p): return 0 <= p[0] < w and 0 <= p[1] < h
    def in_bounds_swapped(p): return 0 <= p[1] < w and 0 <= p[0] < h

    any_oob = any(not in_bounds(p) for p in pts)
    if any_oob and all(in_bounds_swapped(p) for p in pts):
        return [(y, x) for (x, y) in pts]
    return pts


def save_path_gif(
    grid: np.ndarray,
    waypoints: Iterable[Any],
    out_gif: str | Path,
    fps: int = 8,
    dpi: int = 140,
    title: str = "UGV Mission",
) -> None:
    """
    Render an animated GIF of the planned waypoints on the given occupancy grid.

    grid: 2D occupancy (0 free, >0 obstacle). Use the *worked* grid (with avoid zone rasterized).
    waypoints: list of points; robust to (x,y), [x,y], (x,y,extra), dicts with x/y or i/j.
    """
    try:
        import imageio.v2 as imageio
    except Exception as e:
        raise RuntimeError("imageio is required. Install with: pip install imageio") from e

    if grid.ndim != 2:
        raise ValueError("grid must be a 2D array")

    h, w = grid.shape

    # normalize all waypoint records
    pts: List[Tuple[int, int]] = [_to_xy(wp) for wp in waypoints]
    if not pts:
        raise ValueError("No waypoints to animate.")

    # auto-detect coordinate ordering
    pts = _maybe_swap_xy(pts, w=w, h=h)

    out_gif = str(out_gif)
    Path(out_gif).parent.mkdir(parents=True, exist_ok=True)

    # Figure (match CLI axes: origin top-left, y increases downward)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
    ax.imshow(grid, extent=[0, w, h, 0], interpolation="nearest")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)

    # Start/goal markers
    sx, sy = pts[0]
    gx, gy = pts[-1]
    ax.scatter([sx], [sy], s=70, marker="o")
    ax.scatter([gx], [gy], s=90, marker="*", zorder=3)

    frames: List[np.ndarray] = []

    # draw incrementally with a single line object for performance
    xs: List[int] = []
    ys: List[int] = []
    (line,) = ax.plot([], [], linewidth=3)

    # initial frame
    frames.append(_fig_to_rgb_array(fig))

    for x, y in pts:
        xs.append(x)
        ys.append(y)
        line.set_data(xs, ys)
        frames.append(_fig_to_rgb_array(fig))

    imageio.mimsave(out_gif, frames, duration=1.0 / float(max(1, fps)))
    plt.close(fig)
