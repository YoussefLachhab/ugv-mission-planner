from ugv_mission_planner.models import Constraints


def test_battery_none_becomes_20():
    c = Constraints(max_speed_mps=1.2, avoid_zones=[(8, 0, 12, 6)], battery_min_pct=None)
    assert c.battery_min_pct == 20.0
