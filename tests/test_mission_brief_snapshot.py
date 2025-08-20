from pathlib import Path

from ugv_mission_planner.report.mission_brief import write_mission_brief


def test_brief_snapshot(tmp_path: Path):
    trace = "snap001"
    plan = {"goals": [[0, 0], [10, 0]], "constraints": {"max_speed_mps": 1.0}}
    compliance = {"passed": True, "metrics": {"path_length": 10, "steps": 11, "avoid_zone_hits": 0}}
    citations = [{"topic": "max_speed", "snippet": "Max speed is 1.2 m/s", "score": 0.99}]
    out = write_mission_brief(plan=plan, compliance=compliance, citations=citations, trace_id=trace, run_dir=tmp_path)
    text = out.read_text(encoding="utf-8").replace("\r\n", "\n")
    # normalize timestamps
    text = "\n".join([ln for ln in text.splitlines() if not ln.startswith("**Generated:**")])
    golden = (
        (Path(__file__).parent / "snapshots" / "brief_basic.md")
        .read_text(encoding="utf-8")
        .replace("\r\n", "\n")
        .strip()
    )
    assert golden in text  # robust to header/artifacts changes and trailing content
