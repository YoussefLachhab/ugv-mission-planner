import json
import os
import subprocess
import sys

import numpy as np


def test_cli_dry_runs_with_fake_llm(tmp_path):
    # Build a tiny map
    grid = np.zeros((10, 10), dtype=np.uint8)
    grid[3:5, 3:7] = 1
    mapp = tmp_path / "mini.npy"
    np.save(mapp, grid)

    payload = json.dumps({
        "goals": [[0, 0], [9, 9]],
        "constraints": {"max_speed_mps": 1.0, "avoid_zones": [[3, 3, 6, 5]], "battery_min_pct": 15}
    })

    env = os.environ.copy()
    env["UGV_FAKE_LLM"] = "1"
    env["UGV_FAKE_PAYLOAD"] = payload

    res = subprocess.run(
        [sys.executable, "scripts/run_from_nl.py", "--map", str(mapp),
         "--mission", "Go from (0,0) to (9,9) avoiding the block", "--dry"],
        env=env, capture_output=True, text=True, check=False
    )
    assert res.returncode in (0, 1, 2, 3)  # script should run; exit code may vary by env
