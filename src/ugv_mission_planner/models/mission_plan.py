from __future__ import annotations

from typing import List, Tuple
from datetime import datetime, timezone

from pydantic import BaseModel, Field, confloat, conlist, model_validator

import uuid

# Type aliases
Speed = confloat(ge=0.0, le=2.0)
Point = Tuple[float, float]


class Constraints(BaseModel):
    max_speed_mps: Speed
    avoid_zones: List[conlist(float, min_length=4, max_length=4)] = Field(
        default_factory=list,
        description="Axis-aligned rectangles: [xmin, ymin, xmax, ymax]",
    )
    battery_min_pct: confloat(ge=0.0, le=100.0) = 15.0


class Waypoint(BaseModel):
    x: float
    y: float
    speed_mps: Speed


class MissionPlan(BaseModel):
    mission_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    goals: List[Point] = Field(
        ..., min_length=1, description="Ordered goals A,B,..."
    )
    waypoints: List[Waypoint] = Field(default_factory=list)
    constraints: Constraints

    @model_validator(mode="after")
    def _validate_speeds_vs_cap(self) -> "MissionPlan":
        cap = self.constraints.max_speed_mps
        for w in self.waypoints:
            if w.speed_mps > cap:
                raise ValueError(f"Waypoint speed {w.speed_mps} exceeds cap {cap}")
        return self
