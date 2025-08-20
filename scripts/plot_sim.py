# scripts/plot_sim.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless backend for CI and servers
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D

from ugv_mission_planner.executor import execute_waypoints
from ugv_mission_planner.planner import plan_waypoints, rasterize_avoid_zones

# Types
Waypoint = tuple[float, float, float]
Telemetry = dict[str, float]


def load_demo() -> tuple[np.ndarray, dict[str, Any], list[Waypoint], list[Telemetry]]:
    """Load a demo grid, mission JSON, plan waypoints and simulate execution."""
    grid = np.load(Path("examples/maps/with_block.npy"))
    mission_path = Path("examples/missions/patrol_avoid_zone.json")
    mission: dict[str, Any] = json.loads(mission_path.read_text())

    # Rasterize avoid zones from mission
    grid = rasterize_avoid_zones(grid, mission["constraints"]["avoid_zones"])

    # Extract planner inputs from mission
    goal_xy_raw = mission.get("goal", {}).get("xy", [0.0, 0.0])
    goal_xy: tuple[float, float] = (float(goal_xy_raw[0]), float(goal_xy_raw[1]))
    max_speed: float = float(mission.get("constraints", {}).get("max_speed_mps", 1.0))

    # Fixed function calls based on expected signatures
    waypoints: list[Waypoint] = plan_waypoints(grid, goal_xy, goal_xy, max_speed_mps=max_speed)
    telemetry: list[Telemetry] = execute_waypoints(waypoints, max_speed)
    return grid, mission, waypoints, telemetry


def to_img(grid: np.ndarray) -> np.ndarray:
    """Convert occupancy grid (0 free, 1 obstacle) to grayscale image (1 free, 0 obstacle)."""
    return 1.0 - grid.astype(float)


def main() -> None:
    grid, mission, wps, tel = load_demo()
    h, w = grid.shape

    # Figure and base image
    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(figsize=(w / 20.0, h / 20.0), dpi=80)
    img: AxesImage = ax.imshow(to_img(grid), cmap="gray", origin="lower", interpolation="nearest")
    ax.set_title("UGV Mission Simulation")
    ax.set_xlabel("X (cells)")
    ax.set_ylabel("Y (cells)")
    ax.set_xlim(0, w - 1)
    ax.set_ylim(0, h - 1)

    # Planned path line and current pose marker
    (path_line,) = ax.plot([], [], lw=2)  # type: ignore[assignment]
    pose_marker: Line2D = ax.plot([], [], marker="o", markersize=6)[0]

    # Draw planned path once
    if wps:
        xs = [p[0] for p in wps]
        ys = [p[1] for p in wps]
        path_line.set_data(xs, ys)

    def init() -> list[Any]:
        pose_marker.set_data([], [])
        return [pose_marker, path_line, img]

    def update(i: int) -> list[Any]:
        # telemetry expected to have 'x' and 'y'
        x = float(tel[i]["x"]) if "x" in tel[i] else float(i)
        y = float(tel[i]["y"]) if "y" in tel[i] else float(i)
        pose_marker.set_data([x], [y])
        return [pose_marker]

    ani = animation.FuncAnimation(fig, update, init_func=init, frames=len(tel), interval=30, blit=True)

    # Save outputs
    out_path = Path("docs/demo.gif")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(out_path, writer="pillow", fps=30)

    print(f"Saved animation to {out_path}")
    fig.savefig("docs/demo.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
