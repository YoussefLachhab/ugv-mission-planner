from __future__ import annotations

import uuid
from datetime import UTC, datetime

from pydantic import BaseModel, Field, confloat, conlist, model_validator

# Type aliases
Speed = confloat(ge=0.0, le=2.0)
Point = tuple[float, float]


class Constraints(BaseModel):
    max_speed_mps: Speed
    avoid_zones: list[conlist(float, min_length=4, max_length=4)] = Field(
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
        default_factory=lambda: datetime.now(UTC)
    )
    goals: list[Point] = Field(
        ..., min_length=1, description="Ordered goals A,B,..."
    )
    waypoints: list[Waypoint] = Field(default_factory=list)
    constraints: Constraints

    @model_validator(mode="after")
    def _validate_speeds_vs_cap(self) -> MissionPlan:
        cap = self.constraints.max_speed_mps
        for w in self.waypoints:
            if w.speed_mps > cap:
                raise ValueError(f"Waypoint speed {w.speed_mps} exceeds cap {cap}")
        return self
