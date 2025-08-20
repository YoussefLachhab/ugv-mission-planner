import hashlib
import json
from pathlib import Path

import numpy as np

from ugv_mission_planner.planner import plan_waypoints, rasterize_avoid_zones


def sha_path(wps):
    s = ";".join(f"{x:.0f},{y:.0f}" for (x, y, _) in wps)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def test_astar_deterministic_on_block_map():
    grid = np.load(Path("examples/maps/with_block.npy"))
    mission = json.loads(Path("examples/missions/patrol_avoid_zone.json").read_text())
    g2 = rasterize_avoid_zones(grid, mission["constraints"]["avoid_zones"])
    # first leg only: (2,2) -> (18,2)
    wps = plan_waypoints(g2, (2, 2), (18, 2), mission["constraints"]["max_speed_mps"])
    # Golden hash (stable ordering & neighbor scan makes this deterministic)
    assert sha_path(wps) == "02a02bfc7f8c7222de41dadfbedcdcf80ddc249a1d7c93230e75981c94995c87"
