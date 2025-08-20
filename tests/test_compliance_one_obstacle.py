# tests/test_compliance_one_obstacle.py
import numpy as np

from ugv_mission_planner.models import Waypoint
from ugv_mission_planner.verify.compliance import compliance_report


def test_single_obstacle_passes_with_clearance():
    grid = np.zeros((20, 20), dtype=np.uint8)
    avoid = [(8, 4, 12, 10)]  # box spanning y = 4..10

    # Stay below the box (y=2..3) the whole way
    wps = [
        Waypoint(x=2, y=2, speed_mps=1.0),
        Waypoint(x=7, y=3, speed_mps=1.0),  # still < 4
        Waypoint(x=13, y=3, speed_mps=1.0),  # pass under the box
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
    assert rep.min_clearance_m >= 1.0  # below the box → at least 1 cell
