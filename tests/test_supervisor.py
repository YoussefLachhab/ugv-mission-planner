from ugv_mission_planner.supervisor.supervisor import Supervisor


def test_preplan_speed_cap():
    sup = Supervisor({"max_speed_mps": 1.2})
    plan = {"constraints": {"max_speed_mps": 2.0}}
    v = sup.verify_pre_plan(plan)
    assert not v.ok and v.amendment
    assert v.amendment.changes["constraints"]["max_speed_mps"] == 1.2


def test_postrun_avoid_hits():
    sup = Supervisor({"max_speed_mps": 1.2})
    compliance = {"metrics": {"avoid_zone_hits": 1}}
    v = sup.verify_post_run(compliance)
    assert not v.ok and v.amendment
