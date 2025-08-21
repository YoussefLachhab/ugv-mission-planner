# UGV Mission Planner (GenAI-Guarded)

![CI](https://github.com/YoussefLachhab/ugv-mission-planner/actions/workflows/ci.yml/badge.svg?branch=main)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

> **TL;DR** — Natural-language missions → **schema-validated MissionPlan** → **deterministic** A* planning/execution → **policy guardrails** (geofences, speed caps) with **observable artifacts** (PASS/FAIL metrics + GIF).  
> GenAI is **boxed** to the interface: it interprets text into a strict schema; planning/execution are deterministic and testable.

---

## 🎯 Objective

Showcase an interview-ready, safety-aware system architecture:

- **Contracts-first:** strict JSON Schema + Pydantic models.
- **GenAI at the edges:** LLM parses NL → structured plan; the core is deterministic.
- **Policy by design:** geofences painted onto the grid; supervisor enforces limits.
- **Traceability:** every run has a **trace id** and produces artifacts (logs, PASS/FAIL, GIF/plots).

---

## Architecture at a Glance

The UGV Mission Planner uses a GenAI edge for natural-language parsing + policy retrieval, and a deterministic core for planning, execution, and verification.

<p align="center">
  <img src="docs/diagrams/e2e-sequence.png" alt="UGV Mission Planner — End-to-End Sequence" width="980">
</p>

### Flow (high level)
1. **NL → Plan**: Parse mission text into a typed `MissionPlan` (LLM optional, regex fallback).
2. **Policy**: Retrieve caps from `docs/UGV_POLICY.md` (speed, clearance).
3. **Pre-Plan Supervision**: Enforce caps; optionally auto-amend max speed.
4. **Plan**: A* over occupancy grid (+ avoid-zone rasterization/inflation) → waypoints.
5. **Execute**: Simulator steps along waypoints to generate telemetry.
6. **Compliance + Post-Run Supervision**: Check hits, min clearance, near-geofence speed; optional amendments.
7. **Artifacts**: Console summary, mission brief, optional GIF.

> Source for the diagram: [`docs/diagrams/sequence diag E2E.png`](docs/diagrams/sequence diag E2E.png)

---

## 🧩 What GenAI does vs. what’s deterministic

| Layer | Responsibility | Tech |
|---|---|---|
| **GenAI interface** | Parse free-text mission into a schema (`goals`, `constraints`) | OpenAI (configurable), LangChain |
| **Deterministic core** | Path planning, waypoint speeds, execution | Python, A*, NumPy |
| **Guardrails** | Enforce geofences/speed; policy verdict & (optional) auto-amend | Micro-RAG over `docs/UGV_POLICY.md`, supervisor |
| **Observability** | Trace IDs, brief, artifacts | Structured logs, summary JSON, GIF |

---

## 🚀 Streamlit Demo (UI wrapper over the CLI)

The UI is a **thin wrapper** around the same CLI used in CI. It helps you test & demo quickly.

```powershell
# 1) Setup (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\activate
pip install -e ."[dev]"
pip install streamlit matplotlib numpy pillow

# 2) Generate example maps (if not present)
python examples/maps/generate_maps.py

# 3) Run the UI
streamlit run app.py

