from __future__ import annotations

from contextlib import suppress

# ---- Legacy export to keep tests & scripts working ----
with suppress(Exception):
    # Re-export MissionPlan if present (newer codebases might not need it)
    from .mission_plan import MissionPlan  # noqa: F401

# ---- Public itinerary exports (used by tests) ----
from .itinerary import (  # noqa: F401
    Constraints,
    Leg,
    LegType,
    MissionItinerary,
    Waypoint,
    mission_itinerary_json_schema,
)
