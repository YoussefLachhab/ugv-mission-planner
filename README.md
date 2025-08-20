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
