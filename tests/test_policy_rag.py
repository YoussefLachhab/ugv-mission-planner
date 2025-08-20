from pathlib import Path

from ugv_mission_planner.policy.rag import PolicyRAG


def test_retrieve_topk(tmp_path: Path):
    md = tmp_path / "POLICY.md"
    md.write_text(
        """# Policy\n\n## §2.1 Max speed\nMax speed is 1.2 m/s for normal operations.\n\n## §2.2 Geofences\nAvoid zones are hard no-go and require inflation with 1 cell clearance.\n""",
        encoding="utf-8",
    )
    rag = PolicyRAG(md)
    out = rag.retrieve("max speed limit m/s", k=1)
    assert out and "1.2 m/s" in out[0].chunk
