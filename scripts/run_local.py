from pathlib import Path
import json
import numpy as np
from ugv_mission_planner.planner import plan_waypoints, rasterize_avoid_zones
from ugv_mission_planner.executor import execute_waypoints

if __name__ == "__main__":
    grid = np.load(Path("examples/maps/with_block.npy"))
    mission = json.loads(Path("examples/missions/patrol_avoid_zone.json").read_text())
    grid = rasterize_avoid_zones(grid, mission["constraints"]["avoid_zones"])
    wps = plan_waypoints(grid, (2,2), (18,2), mission["constraints"]["max_speed_mps"])
    tel = execute_waypoints(wps)
    print(f"Waypoints: {len(wps)}; Telemetry samples: {len(tel)}; Last: {tel[-1]}")
