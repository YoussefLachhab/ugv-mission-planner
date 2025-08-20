import json
from pathlib import Path

from jsonschema import Draft202012Validator


def test_json_schema_validates_example() -> None:
    schema_path = Path("interfaces/schemas/mission_plan.schema.json")
    schema = json.loads(schema_path.read_text())
    validator = Draft202012Validator(schema)

    example = json.loads(Path("examples/missions/patrol_avoid_zone.json").read_text())
    errors = sorted(validator.iter_errors(example), key=lambda e: e.path)
    assert not errors, f"Schema validation errors: {[e.message for e in errors]}"
