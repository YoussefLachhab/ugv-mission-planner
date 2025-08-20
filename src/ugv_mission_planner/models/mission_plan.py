from __future__ import annotations

from pydantic import BaseModel, Field

from .itinerary import Constraints, Waypoint


class MissionPlan(BaseModel):
    """
    Backwards-compatible mission plan used by tests & scripts.
    """
    mission_id: str
    goals: list[tuple[float, float]] = Field(default_factory=list)
    waypoints: list[Waypoint] = Field(default_factory=list)
    constraints: Constraints = Field(default_factory=Constraints)
