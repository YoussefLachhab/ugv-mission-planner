from __future__ import annotations

from collections.abc import Iterable

Telemetry = dict[str, float]
Waypoint = tuple[float, float, float]  # x, y, speed_mps

def execute_waypoints(waypoints: Iterable[Waypoint], dt: float = 0.1) -> list[Telemetry]:
    """
    Very simple kinematic follower: moves in a straight line to each waypoint
    at commanded speed (no dynamics). Returns telemetry samples.
    """
    wps = list(waypoints)
    if not wps:
        return []
    x, y, _ = wps[0]
    telemetry: list[Telemetry] = []

    for (tx, ty, spd) in wps[1:]:
        # step until close to target
        while True:
            dx, dy = tx - x, ty - y
            dist = (dx*dx + dy*dy) ** 0.5
            if dist < max(1e-6, spd * dt * 0.5):
                x, y = tx, ty
                telemetry.append({"x": x, "y": y, "speed_mps": spd})
                break
            ux, uy = (dx / dist, dy / dist)
            x += ux * spd * dt
            y += uy * spd * dt
            telemetry.append({"x": x, "y": y, "speed_mps": spd})
    return telemetry
