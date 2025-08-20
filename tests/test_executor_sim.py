import json
from pathlib import Path

import numpy as np

from ugv_mission_planner.executor import execute_waypoints
from ugv_mission_planner.planner import plan_waypoints, rasterize_avoid_zones


def test_executor_reaches_last_waypoint():
    grid = np.load(Path("examples/maps/open_area.npy"))
    mission = json.loads(Path("examples/missions/patrol_avoid_zone.json").read_text())
    g2 = rasterize_avoid_zones(grid, mission["constraints"]["avoid_zones"])
    wps = plan_waypoints(g2, (2, 2), (18, 2), mission["constraints"]["max_speed_mps"])
    telemetry = execute_waypoints(wps, dt=0.05)
    last = telemetry[-1]
    assert abs(last["x"] - 18.0) < 1e-6
    assert abs(last["y"] - 2.0) < 1e-6
