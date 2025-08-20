# UGV Safety & Ops Policy (Lite)

## §1. Scope
These rules govern a small ground robot operating on 2D occupancy grids. All planning/execution must be deterministic and fail‑closed.

## §2. Motion Limits
- **§2.1 Max speed (normal)**: The robot **must not exceed 1.2 m/s** ground speed in normal operations.
- **§2.1.1 Near geofence**: If the robot is within **2 cells of an avoid zone**, the **max speed is 0.6 m/s**.
- **§2.2 Geofences**: Avoid zones are **hard no‑go**. The robot must **not enter** any avoid polygon/box.
- **§2.2.1 Clearance**: Maintain **≥ 1 cell** minimum clearance from avoid zones. Use inflation to enforce this.

## §3. Power & Health
- **§3.1 Battery reserve**: Start a mission with **≥ 20% SoC**. Abort if projected SoC at landing < 10%.

## §4. Determinism & Observability
- **§4.1 Determinism**: For a given map/mission, planning is reproducible.
- **§4.2 Artifacts**: Produce compliance metrics and a GIF for each run.
