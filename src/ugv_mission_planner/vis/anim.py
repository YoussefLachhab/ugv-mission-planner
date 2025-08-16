from __future__ import annotations

import os

import matplotlib.pyplot as plt
from matplotlib import animation

from ugv_mission_planner.models import Waypoint


def animate_run(map_grid, avoid_zones, waypoints: list[Waypoint], out_path: str) -> None:
    h, w = map_grid.shape[:2]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)

    # map
    ax.imshow(map_grid, cmap="gray_r", vmin=0, vmax=1, origin="upper")

    # avoid zones
    for xmin, ymin, xmax, ymax in avoid_zones or []:
        ax.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], "r--")

    # path
    xs = [wpt.x for wpt in waypoints]
    ys = [wpt.y for wpt in waypoints]
    ax.plot(xs, ys, "-")

    # moving dot
    dot, = ax.plot([], [], "o")

    def init():
        dot.set_data([], [])
        return (dot,)

    def update(i: int):
        if not waypoints:
            return (dot,)
        idx = min(i, len(waypoints) - 1)
        dot.set_data([waypoints[idx].x], [waypoints[idx].y])
        return (dot,)

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=max(1, len(waypoints)),
        interval=20,
        blit=True,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    anim.save(out_path, writer="pillow")
    plt.close(fig)
