# scripts/plot_sim.py
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from ugv_mission_planner.planner import plan_waypoints, rasterize_avoid_zones
from ugv_mission_planner.executor import execute_waypoints

def load_demo():
    grid = np.load(Path("examples/maps/with_block.npy"))
    mission = json.loads(Path("examples/missions/patrol_avoid_zone.json").read_text())
    grid = rasterize_avoid_zones(grid, mission["constraints"]["avoid_zones"])
    wps = plan_waypoints(grid, (2, 2), (18, 2), mission["constraints"]["max_speed_mps"])
    tel = execute_waypoints(wps, dt=0.05)
    return grid, mission, wps, tel

def to_img(grid: np.ndarray):
    # 0 (free) -> 1.0 white, 1 (obstacle) -> 0.0 black
    return 1.0 - grid.astype(float)

def main():
    grid, mission, wps, tel = load_demo()
    h, w = grid.shape

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_title("UGV Mission — Path & Execution")
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_aspect("equal")

    # Background map
    ax.imshow(to_img(grid), origin="upper", interpolation="nearest")

    # Avoid zones (draw rectangles)
    for (x0, y0, x1, y1) in mission["constraints"]["avoid_zones"]:
        rx, ry = [y0, y1 - y0], [x0, x1 - x0]  # swap to match imshow axes
        rect = plt.Rectangle((y0, x0), y1 - y0, x1 - x0, fill=False, linewidth=1.5)
        ax.add_patch(rect)

    # Planned path (waypoints as small markers + line)
    xs = [y for (_, y, _ ) in wps]   # note: imshow uses (row=x, col=y)
    ys = [x for (x, _, _ ) in wps]
    path_line, = ax.plot(xs, ys, linewidth=1.0)
    path_pts = ax.scatter(xs, ys, s=10)

# Moving agent (marker-only Line2D)
    agent, = ax.plot([], [], marker="o", markersize=6, linestyle="")

    def init():
        agent.set_data([], [])
        return (agent,)

    def update(frame_idx: int):
        if not tel:  # safety: no frames
            return (agent,)
        p = tel[min(frame_idx, len(tel) - 1)]
        ax.set_xlabel(f"x={p['x']:.1f}, y={p['y']:.1f}, speed={p['speed_mps']:.1f} m/s")
        # set_data needs sequences, even for a single point
        agent.set_data([p["y"]], [p["x"]])  # swap for display (col,row)
        return (agent,)


    ani = animation.FuncAnimation(
        fig, update, init_func=init, frames=len(tel), interval=30, blit=True
    )

    # Save a GIF for the README/demo (requires Pillow, which Matplotlib bundles)
    out_path = Path("docs/demo.gif")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(out_path, writer="pillow", fps=30)

    print(f"Saved animation to {out_path}")
    fig.savefig("docs/demo.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    main()
