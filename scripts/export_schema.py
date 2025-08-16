import json
from pathlib import Path

from ugv_mission_planner.models import MissionPlan

out = Path("schemas/mission_plan.schema.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(MissionPlan.model_json_schema(), indent=2))
print(f"Wrote {out}")
