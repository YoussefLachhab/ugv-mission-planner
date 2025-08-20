from pydantic import BaseModel, Field
from typing import List, Optional, Tuple

class Constraints(BaseModel):
    max_speed_mps: float
    avoid_zones: List[Tuple[float, float, float, float]] = Field(default_factory=list)
    battery_min_pct: Optional[float] = None

    # ensure default of 20.0
    def __init__(self, **data):
        if data.get("battery_min_pct") is None:
            data["battery_min_pct"] = 20.0
        super().__init__(**data)
