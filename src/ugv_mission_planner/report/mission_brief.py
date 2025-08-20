from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def _fmt_metrics(metrics: dict[str, Any]) -> str:
    lines = ["| Metric | Value |", "|---|---|"]
    for k, v in metrics.items():
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines)


def _fmt_citations(citations: list[dict]) -> str:
    if not citations:
        return "_No policy citations available._"
    out = []
    for c in citations:
        out.append(f"- **{c['topic']}** (score {c['score']}):\n\n  {c['snippet']}")
    return "\n".join(out)


def _now_iso() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def write_mission_brief(
    *,
    plan: dict,
    compliance: dict,
    citations: list[dict],
    trace_id: str,
    run_dir: Path | None = None,
) -> Path:
    run_dir = run_dir or Path("runs")
    run_dir.mkdir(parents=True, exist_ok=True)
    brief_path = run_dir / f"brief_{trace_id}.md"

    title = "# Mission Brief" if compliance.get("passed", False) else "# Mission Brief (FAILED)"
    metrics = compliance.get("metrics", {})

    # Build the markdown step by step (safe for Windows; JSON-friendly plan serialization)
    lines: list[str] = []
    lines.append(title)
    lines.append("")
    lines.append(f"**Trace ID:** `{trace_id}`  ")
    lines.append(f"**Generated:** {_now_iso()}")
    lines.append("")
    lines.append("## 1) Mission Intent (from plan)")
    lines.append("```json")
    # default=str makes datetimes and other non-serializable objects safe
    lines.append(json.dumps(plan, indent=2, ensure_ascii=False, default=str))
    lines.append("```")
    lines.append("")
    lines.append("## 2) Compliance Verdict")
    lines.append(f"- **Status:** {'✅ PASS' if compliance.get('passed', False) else '❌ FAIL'}")
    lines.append("")
    lines.append("### Metrics")
    lines.append(_fmt_metrics(metrics))
    lines.append("")
    lines.append("## 3) Policy Citations (why this is compliant)")
    lines.append(_fmt_citations(citations))
    lines.append("")
    lines.append("## 4) Artifacts")
    lines.append(f"- GIF: `runs/sim_{trace_id}.gif` (if generated)")
    lines.append(f"- Logs: `runs/run_{trace_id}.log` (if present)")

    body = "\n".join(lines)
    brief_path.write_text(body, encoding="utf-8")
    return brief_path
