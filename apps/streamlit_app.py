# streamlit_app.py
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# ---------- Repo root detection (works from apps/ or root) ----------
def find_repo_root(start: Path) -> Path:
    for cand in [start, *start.parents]:
        if (cand / "pyproject.toml").exists() or (cand / ".git").exists():
            return cand
    # fallback: if examples/maps exists above, use that
    for cand in [start, *start.parents]:
        if (cand / "examples" / "maps").exists():
            return cand
    return start  # last resort

HERE = Path(__file__).resolve().parent
REPO_ROOT = find_repo_root(HERE)

# ---------- Config ----------
MAPS_DIR = REPO_ROOT / "examples" / "maps"
POLICY_PATH = REPO_ROOT / "docs" / "UGV_POLICY.md"  # not passed to CLI (informational)
RUNS_DIR = REPO_ROOT / "runs" / "streamlit"  # UI’s trace folders live here
CLI_SCRIPT = REPO_ROOT / "scripts" / "run_from_nl.py"

# ---------- Preview regexes ----------
R_FROM = re.compile(r"from\s*\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)", re.I)
R_TO = re.compile(r"to\s*\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)", re.I)
R_BETWEEN = re.compile(
    r"between\s*\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)\s*and\s*\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)",
    re.I,
)
R_AVOID = re.compile(
    r"avoid\s*\[\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]", re.I
)
R_SPEED = re.compile(r"max\s*speed\s*([-+]?\d*\.?\d+)\s*m\s*/\s*s", re.I)

# ---------- Utilities ----------
def list_maps() -> dict[str, Path]:
    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    maps: dict[str, Path] = {}
    for p in sorted(MAPS_DIR.glob("*.npy")):
        maps[p.stem] = p
    return maps

def _find_float_pair(rgx: re.Pattern, text: str) -> tuple[float, float] | None:
    m = rgx.search(text)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))

def parse_mission(text: str) -> dict[str, Any]:
    start = _find_float_pair(R_FROM, text)
    goal = _find_float_pair(R_TO, text)
    if start is None or goal is None:
        m = R_BETWEEN.search(text)
        if m:
            start = (float(m.group(1)), float(m.group(2)))
            goal = (float(m.group(3)), float(m.group(4)))
    avoid = None
    m = R_AVOID.search(text)
    if m:
        avoid = tuple(float(m.group(i)) for i in range(1, 5))
    speed = None
    m = R_SPEED.search(text)
    if m:
        speed = float(m.group(1))
    return {"start": start, "goal": goal, "avoid_rect_xyxy": avoid, "max_speed_mps": speed}

def plot_map_overlay(grid, start=None, goal=None, avoid=None, path_xy=None, waypoints=None, title="Map preview"):
    H, W = grid.shape
    fig, ax = plt.subplots()
    # match CLI-style axes: origin top-left, y increases downward
    ax.imshow(grid, extent=[0, W, H, 0], interpolation="nearest")
    if avoid:
        x1, y1, x2, y2 = avoid
        ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], linestyle="--", label="avoid zone")
    if start:
        ax.scatter([start[0]], [start[1]], s=60, marker="o", label="start")
    if goal:
        ax.scatter([goal[0]], [goal[1]], s=80, marker="*", label="goal")
    if waypoints:
        xs = [w[0] for w in waypoints]; ys = [w[1] for w in waypoints]
        ax.plot(xs, ys, linestyle=":", linewidth=2, label="waypoints")
    if path_xy:
        xs = [p[0] for p in path_xy]; ys = [p[1] for p in path_xy]
        ax.plot(xs, ys, linewidth=2, label="trajectory")
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_aspect("equal", adjustable="box")
    ax.set_title(title); ax.legend(loc="best")
    return fig

def read_telemetry_json(path: Path) -> list[tuple[float, float]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        pts = [(float(p["x"]), float(p["y"])) for p in data]
        return pts
    except Exception:
        return []

def read_telemetry_ndjson(path: Path) -> list[tuple[float, float]]:
    pts: list[tuple[float, float]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                pts.append((float(rec["x"]), float(rec["y"])))
    except Exception:
        pass
    return pts

def read_waypoints_from_plan(path: Path) -> list[tuple[float, float]] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        wps = data.get("waypoints")
        if isinstance(wps, list):
            # support either dicts {"x":..,"y":..} or [x,y]
            def to_xy(w): 
                if isinstance(w, dict): return (float(w["x"]), float(w["y"]))
                if isinstance(w, (list, tuple)) and len(w) == 2: return (float(w[0]), float(w[1]))
                return None
            out = [to_xy(w) for w in wps]
            return [t for t in out if t is not None]
    except Exception:
        pass
    return None

def newest_file(patterns, search_dirs, max_age_sec=900) -> Path | None:
    newest = None; newest_mtime = 0.0; now = time.time()
    for d in search_dirs:
        d = Path(d)
        if not d.exists():
            continue
        for pat in patterns:
            for p in d.rglob(pat):
                try:
                    mtime = p.stat().st_mtime
                except FileNotFoundError:
                    continue
                if now - mtime <= max_age_sec and mtime > newest_mtime:
                    newest = p; newest_mtime = mtime
    return newest

# ---------- CLI runner ----------
def run_cli_live(map_path: Path, mission: str, save_gif: bool, auto_amend: bool) -> dict[str, Any]:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    trace_id = datetime.now(UTC).strftime("ui-%Y%m%d-%H%M%S")
    work_dir = RUNS_DIR / trace_id
    work_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    existing_py = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (str(REPO_ROOT / "src") + os.pathsep + existing_py) if existing_py else str(REPO_ROOT / "src")

    cmd = [sys.executable, str(CLI_SCRIPT), "--map", str(map_path), "--mission", mission, "--outdir", str(work_dir)]
    if save_gif: cmd.append("--save-gif")
    if auto_amend: cmd.append("--auto-amend")

    start_ts = time.time()
    try:
        proc = subprocess.Popen(
            cmd, cwd=REPO_ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
    except Exception as e:
        return {"ok": False, "error": f"Failed to launch CLI: {e}"}

    stdout_lines: list[str] = []
    while True:
        line = proc.stdout.readline()
        if line: stdout_lines.append(line)
        if proc.poll() is not None:
            leftover = proc.stdout.read()
            if leftover: stdout_lines.append(leftover)
            break

    duration = time.time() - start_ts
    stdout = "".join(stdout_lines)

    # Look for artifacts
    plan = work_dir / "plan.json"
    telem_json = work_dir / "telemetry.json"
    telem_ndjson = work_dir / "telemetry.ndjson"

    # Try to locate a GIF
    gif = None
    for name in ("mission.gif", "run.gif", "trajectory.gif"):
        p = work_dir / name
        if p.exists():
            gif = p; break
    if gif is None:
        gif = newest_file(["*.gif"], [work_dir, REPO_ROOT / "runs"], max_age_sec=1800)

    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "duration_sec": duration,
        "work_dir": str(work_dir),
        "stdout": stdout,
        "gif_path": str(gif) if gif else None,
        "plan_path": str(plan),
        "telemetry_json_path": str(telem_json),
        "telemetry_ndjson_path": str(telem_ndjson),
    }

# ---------- UI ----------
st.set_page_config(page_title="UGV Mission Planner — Demo", layout="wide")
st.title("UGV Mission Planner — GenAI-Guarded (Streamlit Demo)")
st.caption("Natural‑language mission → policy‑checked plan → deterministic execution. The UI shells the same CLI used in CI.")

with st.sidebar:
    st.header("Scenario")
    maps = list_maps()
    # quick map generator
    if st.button("Generate maps (once)"):
        try:
            subprocess.check_call([sys.executable, str(REPO_ROOT / "examples" / "maps" / "generate_maps.py")], cwd=REPO_ROOT)
            maps = list_maps()
            st.success("Maps generated.")
        except Exception as e:
            st.error(f"Failed to generate maps: {e}")

    if not maps:
        st.error(f"No maps found in {MAPS_DIR}. Click 'Generate maps (once)' above.")
    map_name = st.selectbox("Map", options=list(maps.keys()), index=0 if maps else None)

    presets = {
        "Open area patrol": "Patrol between (2,2) and (18,2) twice, avoid [8,0,12,6], max speed 1.2 m/s",
        "Go A→B simple": "Go from (2,2) to (19,3), max speed 1.2 m/s",
        "Geofence stress": "Go from (1,1) to (19,6), avoid [5,0,16,8], max speed 1.0 m/s",
    }
    preset_name = st.selectbox("Preset", options=list(presets.keys()), index=0)
    mission_text = st.text_area("Mission (editable)", presets[preset_name], height=120)

    st.divider()
    st.header("Options")
    save_gif = st.checkbox("Save GIF", value=True)
    auto_amend = st.checkbox("Auto‑amend on policy fail", value=True)
    st.divider()
    run_clicked = st.button("▶ Run mission", use_container_width=True)

left, right = st.columns([1, 1])

with left:
    st.subheader("Mission preview")
    if maps:
        grid = np.load(maps[map_name])
        parse = parse_mission(mission_text)
        c1, c2 = st.columns([1, 1])
        with c1:
            st.code(json.dumps(parse, indent=2))
        with c2:
            st.pyplot(plot_map_overlay(grid, parse["start"], parse["goal"], parse["avoid_rect_xyxy"]))
    else:
        st.info("Add maps to preview the scenario.")

with right:
    st.subheader("Results")
    if run_clicked:
        if not maps:
            st.error("No maps available.")
        elif not CLI_SCRIPT.exists():
            st.error(f"CLI not found at {CLI_SCRIPT}")
        else:
            with st.status("Running CLI…", expanded=True) as status:
                st.write(f"Launching: {CLI_SCRIPT.relative_to(REPO_ROOT)}")
                result = run_cli_live(maps[map_name], mission_text, save_gif=save_gif, auto_amend=auto_amend)

                if not result.get("ok", False):
                    st.error(f"CLI exited with code {result.get('returncode')}. See logs below.")
                else:
                    st.success(f"OK in {result['duration_sec']:.2f}s")

                # Read artifacts (plan/telemetry)
                plan_path = Path(result["plan_path"])
                wps = read_waypoints_from_plan(plan_path) if plan_path.exists() else None

                tele_json = Path(result["telemetry_json_path"])
                tele_nd = Path(result["telemetry_ndjson_path"])
                path_xy = read_telemetry_json(tele_json) if tele_json.exists() else read_telemetry_ndjson(tele_nd)

                # Overlay
                if maps:
                    grid = np.load(maps[map_name])
                    fig = plot_map_overlay(
                        grid,
                        parse["start"], parse["goal"], parse["avoid_rect_xyxy"],
                        path_xy=path_xy if path_xy else None,
                        waypoints=wps if wps else None,
                        title="Result overlay",
                    )
                    st.pyplot(fig)

                # GIF (optional)
                gif_path = result.get("gif_path")
                if gif_path and Path(gif_path).exists():
                    st.markdown("**Execution GIF**")
                    st.image(gif_path)
                else:
                    st.info("No GIF found. (Overlay above uses plan/telemetry if available.)")

                # Logs
                st.divider()
                st.markdown("**Logs (stdout)**")
                with st.expander("stdout", expanded=False):
                    st.code(result.get("stdout", ""))

                status.update(label="Done.", state="complete")
    else:
        st.info("Configure a scenario on the left and click **Run mission**.")

st.caption("Tip: the UI never calls the LLM directly; it shells the CLI for reproducibility.")
