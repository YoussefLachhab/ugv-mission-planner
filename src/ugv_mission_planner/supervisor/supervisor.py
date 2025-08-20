from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Amendment:
    message: str
    changes: dict[str, Any]


@dataclass
class Verdict:
    ok: bool
    reasons: list[str]
    amendment: Amendment | None = None


class Supervisor:
    """Runtime guard that checks plan + post-run metrics."""

    def __init__(self, policy_limits: dict[str, float] | None = None):
        self.policy_limits = policy_limits or {}

    def verify_pre_plan(self, plan: Any) -> Verdict:
        """Check plan constraints against global speed policy; accept dict or object."""
        cons = getattr(plan, "constraints", None)
        if cons is None and isinstance(plan, dict):
            cons = plan.get("constraints")

        max_speed = None
        if isinstance(cons, dict):
            max_speed = cons.get("max_speed_mps") or cons.get("max_speed")
        elif cons is not None:
            max_speed = getattr(cons, "max_speed_mps", None) or getattr(cons, "max_speed", None)

        cap = self.policy_limits.get("max_speed_mps")
        # Only enforce the global cap here (not the near-geofence speed)
        if cap is not None and max_speed is not None and float(max_speed) > float(cap):
            msg = f"max_speed_mps {max_speed} exceeds policy {cap}; propose reduction"
            amend = Amendment(message=msg, changes={"constraints": {"max_speed_mps": cap}})
            return Verdict(ok=False, reasons=[msg], amendment=amend)
        return Verdict(ok=True, reasons=["pre-plan checks passed"])

    def verify_post_run(self, compliance: dict) -> Verdict:
        metrics = compliance.get("metrics", {})
        avoid_hits = int(metrics.get("avoid_zone_hits", 0))
        if avoid_hits > 0:
            msg = f"Detected {avoid_hits} avoid-zone hits; must replan or inflate geofence"
            amend = Amendment(message=msg, changes={"env": {"UGV_AVOID_INFLATE_CELLS": 1}})
            return Verdict(ok=False, reasons=[msg], amendment=amend)

        # Global speed cap check
        cap = self.policy_limits.get("max_speed_mps")
        max_used = metrics.get("max_speed_used")
        if cap is not None and max_used is not None and float(max_used) > float(cap) + 1e-6:
            msg = f"Execution used speed {max_used} > policy {cap}"
            amend = Amendment(message=msg, changes={"constraints": {"max_speed_mps": cap}})
            return Verdict(ok=False, reasons=[msg], amendment=amend)

        # Near-geofence speed rule applies only when we actually get closer than N cells (strictly < N)
        near_cells = self.policy_limits.get("near_geofence_cells")
        near_speed = self.policy_limits.get("near_geofence_speed_mps")
        min_clear = metrics.get("min_clearance") or metrics.get("min_clearance_cells") or metrics.get("min_clearance_m")
        try:
            if (near_cells is not None) and (near_speed is not None) and (min_clear is not None):
                if float(min_clear) < float(near_cells) and float(max_used or 0) > float(near_speed) + 1e-6:
                    msg = (
                        f"Path approached geofence closer than {near_cells} cells and used speed {max_used} > "
                        f"near-geofence cap {near_speed}"
                    )
                    amend = Amendment(message=msg, changes={"constraints": {"max_speed_mps": near_speed}})
                    return Verdict(ok=False, reasons=[msg], amendment=amend)
        except Exception:
            pass

        return Verdict(ok=True, reasons=["post-run checks passed"])
