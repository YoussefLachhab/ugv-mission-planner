from __future__ import annotations

import heapq
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

Grid = np.ndarray  # dtype uint8, 0=free, 1=obstacle
Coord = tuple[int, int]


@dataclass(frozen=True)
class PlanResult:
    path: list[Coord]  # grid cells from start..goal inclusive


def _heuristic(a: Coord, b: Coord) -> float:
    # Manhattan is deterministic on grids
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _neighbors(p: Coord, grid: Grid) -> Iterable[Coord]:
    # 4-connected (deterministic ordering N,E,S,W)
    x, y = p
    h, w = grid.shape
    cand = [(x - 1, y), (x, y + 1), (x + 1, y), (x, y - 1)]
    for nx, ny in cand:
        if 0 <= nx < h and 0 <= ny < w and grid[nx, ny] == 0:
            yield (nx, ny)


def astar(grid: Grid, start: Coord, goal: Coord) -> list[Coord] | None:
    """Deterministic A* on occupancy grid. Returns path of cells or None."""
    if grid[start] == 1 or grid[goal] == 1:
        return None

    open_heap: list[tuple[float, int, Coord]] = []
    counter = 0  # tie-breaker for stability
    g = {start: 0.0}
    came: dict[Coord, Coord] = {}
    f0 = _heuristic(start, goal)
    heapq.heappush(open_heap, (f0, counter, start))

    closed: set[Coord] = set()

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal:
            # reconstruct
            path = [current]
            while current in came:
                current = came[current]
                path.append(current)
            path.reverse()
            return path
        closed.add(current)
        for nb in _neighbors(current, grid):
            tentative = g[current] + 1.0
            if nb not in g or tentative < g[nb]:
                g[nb] = tentative
                came[nb] = current
                counter += 1
                f = tentative + _heuristic(nb, goal)
                heapq.heappush(open_heap, (f, counter, nb))
    return None


def rasterize_avoid_zones(grid: Grid, avoid_zones: list[tuple[float, float, float, float]]) -> Grid:
    """Return a copy of grid with avoid_zones marked as obstacles (1)."""
    g = grid.copy()
    h, w = g.shape
    for x_min, y_min, x_max, y_max in avoid_zones:
        # Clamp to grid and mark inclusive bounds
        xi0, yi0 = max(0, int(np.floor(x_min))), max(0, int(np.floor(y_min)))
        xi1, yi1 = min(h - 1, int(np.ceil(x_max))), min(w - 1, int(np.ceil(y_max)))
        g[xi0 : xi1 + 1, yi0 : yi1 + 1] = 1
    return g


def plan_waypoints(
    grid: Grid, start_xy: tuple[float, float], goal_xy: tuple[float, float], max_speed_mps: float
) -> list[tuple[float, float, float]]:
    """
    Plan from start to goal on grid, return waypoints [(x,y,speed_mps), ...].
    Uses cell centers as waypoint coordinates.
    """
    start = (int(round(start_xy[0])), int(round(start_xy[1])))
    goal = (int(round(goal_xy[0])), int(round(goal_xy[1])))
    path = astar(grid, start, goal)
    if path is None:
        raise ValueError("No path found")
    # Convert grid cells to (x,y,speed)
    return [(float(x), float(y), float(max_speed_mps)) for (x, y) in path]
