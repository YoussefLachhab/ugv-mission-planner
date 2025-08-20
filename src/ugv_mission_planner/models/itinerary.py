# src/ugv_mission_planner/models/itinerary.py
"""
High-level itinerary contract for multi-leg missions.
Does NOT replace your existing MissionPlan; it sits above it.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Waypoint(BaseModel):
    x: float
    y: float
    speed_mps: float | None = Field(default=None, ge=0)


class LegType(str, Enum):
    GOTO = "goto"
    PATROL = "patrol"
    LOITER = "loiter"
    RETURN = "return"


class Leg(BaseModel):
    type: LegType
    # GOTO/LOITER/RETURN => len(points)==1 ; PATROL => len(points)==2
    points: list[Waypoint] = Field(default_factory=list)
    count: int | None = Field(default=None, ge=1, description="Loops for PATROL")
    duration_s: float | None = Field(default=None, ge=0, description="LOITER dwell")
    max_speed_mps: float | None = Field(default=None, ge=0)


XYRect = tuple[float, float, float, float]


class Constraints(BaseModel):
    max_speed_mps: float | None = Field(default=1.2, ge=0)
    avoid_zones: list[XYRect] = Field(default_factory=list)
    battery_min_pct: float | None = Field(default=20, ge=0, le=100)


class MissionItinerary(BaseModel):
    mission_id: str
    legs: list[Leg]
    constraints: Constraints = Field(default_factory=Constraints)


def mission_itinerary_json_schema() -> dict:
    return MissionItinerary.model_json_schema()
