# tests/test_constraints_battery_default.py
from ugv_mission_planner.models import Constraints


def test_battery_default_is_20_when_missing():
    cons = Constraints(max_speed_mps=1.2, avoid_zones=[(8, 0, 12, 6)])  # no battery provided
    assert cons.battery_min_pct == 20.0
