import json
from pathlib import Path

from ugv_mission_planner.models import MissionPlan


def test_mission_plan_example_loads() -> None:
    example = Path("examples/missions/patrol_avoid_zone.json")
    data = json.loads(example.read_text())
    plan = MissionPlan.model_validate(data)
    assert plan.mission_id == "demo_patrol_001"
    assert len(plan.waypoints) >= 2
    assert all(wp.speed_mps <= plan.constraints.max_speed_mps for wp in plan.waypoints)
