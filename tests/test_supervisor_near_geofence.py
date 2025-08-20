# tests/test_supervisor_near_geofence.py

from ugv_mission_planner.supervisor.supervisor import Supervisor


def test_near_geofence_speed_only_when_close():
    sup = Supervisor(
        {
            "max_speed_mps": 1.2,
            "near_geofence_cells": 2.0,
            "near_geofence_speed_mps": 0.6,
        }
    )

    # Exactly at 2 cells → no gating
    v = sup.verify_post_run({"metrics": {"avoid_zone_hits": 0, "max_speed_used": 1.2, "min_clearance": 2.0}})
    assert v.ok

    # Closer than 2 cells → gating kicks in if speed > 0.6
    v2 = sup.verify_post_run({"metrics": {"avoid_zone_hits": 0, "max_speed_used": 1.2, "min_clearance": 1.5}})
    assert not v2.ok and v2.amendment
    assert v2.amendment.changes["constraints"]["max_speed_mps"] == 0.6
