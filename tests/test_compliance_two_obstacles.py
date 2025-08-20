# tests/test_compliance_two_obstacles.py
import numpy as np

from ugv_mission_planner.models import Waypoint
from ugv_mission_planner.verify.compliance import compliance_report


def test_two_obstacles_route_stays_safe():
    grid = np.zeros((20, 20), dtype=np.uint8)
    avoid = [
        (7, 4, 11, 10),  # left box
        (12, 4, 16, 10),  # right box
    ]

    # Still stay below both boxes (y<4)
    wps = [
        Waypoint(x=2, y=2, speed_mps=1.0),
        Waypoint(x=9, y=3, speed_mps=1.0),  # below left box
        Waypoint(x=14, y=3, speed_mps=1.0),  # below right box
        Waypoint(x=18, y=2, speed_mps=1.0),
    ]

    rep = compliance_report(
        map_grid=grid,
        avoid_zones=avoid,
        waypoints=wps,
        max_speed=1.2,
    )

    assert rep.pass_ok, f"Expected PASS, got report={rep}"
    assert rep.avoid_hits == 0
    assert rep.min_clearance_m >= 1.0
