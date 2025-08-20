import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# ---------- Config ----------
REPO_ROOT = Path(__file__).resolve().parent
MAPS_DIR = REPO_ROOT / "examples" / "maps"
POLICY_PATH = REPO_ROOT / "docs" / "UGV_POLICY.md"
RUNS_DIR = REPO_ROOT / "runs" / "streamlit"  # UI’s trace folders live here
CLI_SCRIPT = REPO_ROOT / "scripts" / "run_from_nl.py"

# ---------- Preview regexes ----------
# Supports both:
#   "Go from (2,2) to (18,2), avoid [8,0,12,6], max speed 1.2 m/s"
#   "Patrol between (2,2) and (18,2), avoid [8,0,12,6], max speed 1.2 m/s"
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
    # Try "from ... to ..." first
    start = _find_float_pair(R_FROM, text)
    goal = _find_float_pair(R_TO, text)

    # If not found, try "between ... and ..."
    if start is None or goal is None:
        m = R_BETWEEN.search(text)
        if m:
            start = (float(m.group(1)), float(m.group(2)))
            goal = (float(m.group(3)), float(m.group(4)))

    avoid = None
    m = R_AVOID.search(text)
    if m:
        avoid = tuple(float(m.group(i)) for i in range(1, 5))  # x1,y1,x2,y2

    speed = None
    m = R_SPEED.search(text)
    if m:
        speed = float(m.group(1))

    return {
        "start": start,
        "goal": goal,
        "avoid_rect_xyxy": avoid,
        "max_speed_mps": speed,
    }


def plot_map_overlay(
    grid,
    start=None,
    goal=None,
    avoid=None,
    path_xy=None,
    waypoints=None,
    title="Map preview",
    match_cli_axes=True,  # y-down like the CLI
):
    H, W = grid.shape  # rows (y), cols (x)
    fig, ax = plt.subplots()

    if match_cli_axes:
        # y-down (origin top-left), no transpose
        ax.imshow(grid, extent=[0, W, H, 0], interpolation="nearest")
        # x, y you plot below are in the same coordinate frame:
        #   x: 0..W (to the right), y: 0..H (downwards)
    else:
        # y-up (origin bottom-left)
        ax.imshow(grid, extent=[0, W, 0, H], interpolation="nearest")

    # Draw avoid rect
    if avoid:
        x1, y1, x2, y2 = avoid
        ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], linestyle="--", label="avoid zone")

    # Start/goal
    if start:
        ax.scatter([start[0]], [start[1]], s=60, marker="o", label="start")
    if goal:
        ax.scatter([goal[0]], [goal[1]], s=80, marker="*", label="goal")

    # Planned waypoints (dotted) and executed path (solid)
    if waypoints:
        xs = [w[0] for w in waypoints]
        ys = [w[1] for w in waypoints]
        ax.plot(xs, ys, linestyle=":", linewidth=2, label="waypoints")
    if path_xy:
        xs = [p[0] for p in path_xy]
        ys = [p[1] for p in path_xy]
        ax.plot(xs, ys, linewidth=2, label="trajectory")

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0) if match_cli_axes else ax.set_ylim(0, H)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.legend(loc="best")
    return fig


def read_telemetry(path: Path) -> list[tuple[float, float]]:
    """Read incremental trajectory from telemetry.ndjson if present."""
    pts: list[tuple[float, float]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                pts.append((float(rec["x"]), float(rec["y"])))
    except FileNotFoundError:
        pass
    except Exception:
        # ignore partial lines while file is being written
        pass
    return pts


def read_waypoints_from_summary(path: Path) -> list[tuple[float, float]] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        wps = [(float(w["x"]), float(w["y"])) for w in data.get("waypoints", [])]
        return wps
    except Exception:
        return None


def newest_file(patterns, search_dirs, max_age_sec=900) -> Path | None:
    """Recursive newest-file search."""
    newest = None
    newest_mtime = 0.0
    now = time.time()
    for d in search_dirs:
        d = Path(d)
        if not d.exists():
            continue
        for pat in patterns:
            for p in d.rglob(pat):  # recursive
                try:
                    mtime = p.stat().st_mtime
                except FileNotFoundError:
                    continue
                if now - mtime <= max_age_sec and mtime > newest_mtime:
                    newest = p
                    newest_mtime = mtime
    return newest


# ---------- CLI runners ----------
def run_cli_live(map_path: Path, mission: str, save_gif: bool, auto_amend: bool) -> dict[str, Any]:
    """
    Launch the CLI and, while it runs, tail telemetry/summary from UGV_OUTDIR to redraw the preview.
    Returns the final result + paths to artifacts if found.
    """
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    trace_id = datetime.now().strftime("ui-%Y%m%d-%H%M%S")
    work_dir = RUNS_DIR / trace_id
    work_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    existing_py = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (str(REPO_ROOT / "src") + os.pathsep + existing_py) if existing_py else str(REPO_ROOT / "src")
    env["UGV_OUTDIR"] = str(work_dir)  # hint to CLI where to write artifacts

    cmd = [sys.executable, str(CLI_SCRIPT), "--map", str(map_path), "--mission", mission, "--policy", str(POLICY_PATH)]
    if save_gif:
        cmd.append("--save-gif")
    if auto_amend:
        cmd.append("--auto-amend")

    start_ts = time.time()
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except Exception as e:
        return {"ok": False, "error": f"Failed to launch CLI: {e}"}

    stdout_lines: list[str] = []
    artifacts_dir = work_dir
    telemetry_file = work_dir / "telemetry.ndjson"
    summary_file = work_dir / "summary.json"

    # stream logs + watch for "ARTIFACTS:" hint if CLI prints it
    while True:
        line = proc.stdout.readline()
        if line:
            stdout_lines.append(line)
            if line.startswith("ARTIFACTS:"):
                hinted = Path(line.split("ARTIFACTS:", 1)[1].strip())
                if hinted.exists():
                    artifacts_dir = hinted
                    telemetry_file = artifacts_dir / "telemetry.ndjson"
                    summary_file = artifacts_dir / "summary.json"
        if proc.poll() is not None:
            # drain remaining lines
            leftover = proc.stdout.read()
            if leftover:
                stdout_lines.append(leftover)
            break

    duration = time.time() - start_ts
    stdout = "".join(stdout_lines)
    stderr = ""  # merged above

    # Try to locate a GIF in likely places
    gif = None
    for name in ("mission.gif", "run.gif", "trajectory.gif"):
        p = artifacts_dir / name
        if p.exists():
            gif = p
            break
    if gif is None:
        gif = newest_file(
            ["*.gif"],
            [
                artifacts_dir,
                work_dir,
                REPO_ROOT,
                REPO_ROOT / "runs",
                REPO_ROOT / "outputs",
                REPO_ROOT / "scripts",
                MAPS_DIR,
            ],
            max_age_sec=1800,
        )

    # Persist logs
    (work_dir / "stdout.txt").write_text(stdout, encoding="utf-8")

    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "duration_sec": duration,
        "work_dir": str(work_dir),
        "artifacts_dir": str(artifacts_dir),
        "stdout_path": str(work_dir / "stdout.txt"),
        "gif_path": str(gif) if gif else None,
        "telemetry_path": str(telemetry_file),
        "summary_path": str(summary_file),
        "stdout": stdout,
        "stderr": stderr,
    }


# ---------- UI ----------
st.set_page_config(page_title="UGV Mission Planner — Demo", layout="wide")
st.title("UGV Mission Planner — GenAI-Guarded (Streamlit Demo)")
st.caption(
    "Natural-language mission ➜ policy-checked plan ➜ deterministic execution. "
    "This UI wraps the CLI so the core remains deterministic and testable."
)

left, right = st.columns([1, 1])

with st.sidebar:
    st.header("Scenario")
    maps = list_maps()
    if not maps:
        st.error(f"No maps found in {MAPS_DIR}. Generate with examples/maps/generate_maps.py")
    map_name = st.selectbox("Map", options=list(maps.keys()), index=0 if maps else None)

    presets = {
        "Open area patrol": "Patrol between (2,2) and (18,2) twice, avoid [8,0,12,6], max speed 1.2 m/s",
        "Go A→B simple": "Go from (2,2) to (20,3), max speed 1.2 m/s",
        "Geofence stress": "Go from (1,1) to (22,6), avoid [5,0,16,8], max speed 1.0 m/s",
    }
    preset_name = st.selectbox("Preset", options=list(presets.keys()), index=0)
    mission_text = st.text_area("Mission (editable)", presets[preset_name], height=120)

    st.divider()
    st.header("Options")
    save_gif = st.checkbox("Save GIF", value=True)
    auto_amend = st.checkbox("Auto-amend on policy fail", value=True)

    st.divider()
    run_clicked = st.button("▶ Run mission", use_container_width=True)

# Preview panel (left)
with left:
    st.subheader("Mission preview")
    if maps:
        grid = np.load(maps[map_name])
        parse = parse_mission(mission_text)
        col1, col2 = st.columns([1, 1])
        with col1:
            st.code(json.dumps(parse, indent=2))
        with col2:
            fig = plot_map_overlay(grid, parse["start"], parse["goal"], parse["avoid_rect_xyxy"])
            st.pyplot(fig)
    else:
        st.info("Add maps to preview the scenario.")

# Run & results (right)
with right:
    st.subheader("Results")
    if run_clicked:
        if not maps:
            st.error("No maps available.")
        elif not CLI_SCRIPT.exists():
            st.error(f"CLI not found at {CLI_SCRIPT}")
        else:
            with st.status("Running CLI…", expanded=True) as status:
                st.write("Launching deterministic pipeline via `scripts/run_from_nl.py`…")

                # 1) Launch + wait (we still get the speed & simplicity of run)
                result = run_cli_live(maps[map_name], mission_text, save_gif=save_gif, auto_amend=auto_amend)

                # 2) After it finishes, try to show the final path/waypoints if CLI emitted them
                artifacts_dir = Path(result["artifacts_dir"])
                telemetry_path = Path(result["telemetry_path"])
                summary_path = Path(result["summary_path"])

                # Read whatever is available
                path_xy = read_telemetry(telemetry_path) if telemetry_path.exists() else []
                wps = read_waypoints_from_summary(summary_path) if summary_path.exists() else None

                if not result.get("ok", False):
                    st.error(f"CLI exited with code {result.get('returncode')}. See logs below.")
                else:
                    st.success(f"OK in {result['duration_sec']:.2f}s")

                st.write(f"Trace folder: `{Path(result['work_dir'])}`")
                if artifacts_dir != Path(result["work_dir"]):
                    st.write(f"Artifacts: `{artifacts_dir}`")

                # Final overlay (even if no GIF)
                if maps:
                    grid = np.load(maps[map_name])
                    fig = plot_map_overlay(
                        grid,
                        parse["start"],
                        parse["goal"],
                        parse["avoid_rect_xyxy"],
                        path_xy=path_xy if path_xy else None,
                        waypoints=wps if wps else None,
                        title="Result overlay",
                    )
                    st.pyplot(fig)

                # Show GIF if found
                gif_path = result.get("gif_path")
                if gif_path and Path(gif_path).exists():
                    st.markdown("**Execution GIF**")
                    st.image(gif_path)
                else:
                    st.info("No GIF found. (The run is OK; overlay above uses telemetry/summary when available.)")

                # Logs
                st.divider()
                st.markdown("**Logs**")
                with st.expander("stdout", expanded=False):
                    st.code(result.get("stdout", ""))
                if result.get("stderr"):
                    with st.expander("stderr", expanded=False):
                        st.code(result.get("stderr", ""))

                # Downloads
                st.divider()
                col_a, col_b = st.columns(2)
                with col_a:
                    try:
                        with open(result["stdout_path"], "rb") as f:
                            st.download_button("Download stdout.txt", f, file_name="stdout.txt")
                    except Exception:
                        pass
                with col_b:
                    try:
                        if gif_path and Path(gif_path).exists():
                            with open(gif_path, "rb") as f:
                                st.download_button("Download mission.gif", f, file_name="mission.gif")
                    except Exception:
                        pass

                status.update(label="Done.", state="complete")
    else:
        st.info("Configure a scenario on the left and click **Run mission**.")

st.caption(
    "Tip: the UI never calls the LLM directly; it shells the same CLI used in CI. "
    "If the CLI writes telemetry.ndjson + summary.json into UGV_OUTDIR, the map overlays the result."
)
