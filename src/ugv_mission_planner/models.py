from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

AvoidZone = tuple[float, float, float, float]  # x_min, y_min, x_max, y_max

class Waypoint(BaseModel):
    x: float
    y: float
    speed_mps: float = Field(ge=0.0, le=2.0)

class Constraints(BaseModel):
    max_speed_mps: float = Field(default=1.5, ge=0.0, le=2.0)
    avoid_zones: list[AvoidZone] = Field(default_factory=list)
    battery_min_pct: int = Field(default=20, ge=0, le=100)

class MissionPlan(BaseModel):
    mission_id: str
    waypoints: list[Waypoint]
    constraints: Constraints = Field(default_factory=Constraints)

    @model_validator(mode="after")
    def _check_waypoints_and_speeds(self) -> MissionPlan:
        if len(self.waypoints) < 2:
            raise ValueError("MissionPlan requires at least two waypoints")
        max_v = self.constraints.max_speed_mps
        for i, wp in enumerate(self.waypoints):
            if wp.speed_mps > max_v:
                raise ValueError(
                    f"waypoint[{i}].speed_mps={wp.speed_mps} exceeds max_speed_mps={max_v}"
                )
        return self
