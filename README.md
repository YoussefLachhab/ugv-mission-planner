UGV Mission Planner (GenAI‑Guarded)




Turn natural‑language missions into validated, policy‑compliant plans and execute them in a deterministic 2D sim. GenAI is boxed behind strict schemas and policy checks; planning and execution are deterministic.

📅 Project Timeline

Day 1 (✅ Complete)

Public GitHub repo with proper structure.

.gitignore and .gitattributes configured.

Core docs, schemas, models, example missions/maps.

Unit tests for schema and models.

CI workflow prepared (lint, type-check, tests).

Fixed CI by adding types-jsonschema stubs for mypy.

Day 2 (Next)

Deterministic path planner (A* on 2D grid).

Simple executor (PID or direct waypoint follow).

Golden-path test cases for reproducibility.

Future Days

Day 3: GenAI NL→MissionPlan parser (structured outputs).

Day 4: Policy micro-RAG + Supervisor.

Day 5: E2E pipeline, polish, demo GIF.

🚀 Quickstart

python -m venv .venv
.\.venv\Scripts\activate
pip install -e ."[dev]"
# Install missing type stubs for CI consistency
pip install types-jsonschema
pytest
python examples\maps\generate_maps.py

📂 Repository Structure

docs/                 # Requirements, architecture, safety case, ADRs, policy corpus
interfaces/schemas/   # MissionPlan JSON Schema
src/ugv_mission_planner/  # Pydantic models
examples/maps/        # Demo maps generator (.npy)
examples/missions/    # Example mission JSONs
tests/                # Unit tests (schema, models)
.github/workflows/    # CI configuration

📜 Documentation

Requirements

Architecture

Safety Case Lite

UGV Policy Corpus

ADRs in docs/ADRs/ (0001–0003)

📈 Why This Matters

This project demonstrates System Architect thinking:

Contracts-first design (MissionPlan schema).

Separation of deterministic core from GenAI interface.

Traceability from requirements to implementation.

CI-driven development with reproducible runs.

📅 Status: Day 1 Complete

We’ve laid the groundwork. On Day 2, we’ll bring the system to life with a reproducible planner and executor, paving the way for GenAI integration.

