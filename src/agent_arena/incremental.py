"""D* Lite on a changing four-neighbor unit-cost graph, with real g/rhs repair.

The reference BFS below deliberately does not use the planner's adjacency,
priority queue, g/rhs tables, path extractor, or heuristic.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import heapq
import json
import math

from .arena import DIRECTIONS, Maze, _point

Point = tuple[int, int]
INF = math.inf


class RepairBudgetExceeded(RuntimeError):
    code = "repair_budget_exhausted"


@dataclass(frozen=True)
class CellChange:
    cell: Point
    blocked: bool

    def __post_init__(self):
        object.__setattr__(self, "cell", _point(self.cell, 31, 31))
        if type(self.blocked) is not bool:
            raise ValueError("blocked must be a boolean")


@dataclass(frozen=True)
class GridSnapshot:
    width: int
    height: int
    walls: frozenset[Point]
    goal: Point

    def __post_init__(self):
        if any(type(n) is not int or not 5 <= n <= 31 or n % 2 == 0 for n in (self.width, self.height)):
            raise ValueError("grid dimensions must be odd integers between 5 and 31")
        if not isinstance(self.walls, (set, frozenset, list, tuple)) or len(self.walls) > self.width * self.height:
            raise ValueError("walls must be a bounded coordinate collection")
        object.__setattr__(self, "walls", frozenset(_point(cell, self.width, self.height) for cell in self.walls))
        object.__setattr__(self, "goal", _point(self.goal, self.width, self.height))

    @classmethod
    def from_maze(cls, maze: Maze) -> GridSnapshot:
        return cls(maze.width, maze.height, maze.walls, maze.goal)

    def walkable(self, cell: Point) -> bool:
        return 0 <= cell[0] < self.width and 0 <= cell[1] < self.height and cell not in self.walls

    def neighbors(self, cell: Point) -> tuple[Point, ...]:
        x, y = cell
        return tuple(
            (x + dx, y + dy) for dx, dy in DIRECTIONS.values() if 0 <= x + dx < self.width and 0 <= y + dy < self.height
        )

    def changed(self, changes: list[CellChange] | tuple[CellChange, ...]) -> GridSnapshot:
        if not isinstance(changes, (list, tuple)) or len(changes) > 128:
            raise ValueError("each observation may contain at most 128 cell changes")
        cells = set()
        for change in changes:
            if not isinstance(change, CellChange):
                raise ValueError("expected CellChange records")
            _point(change.cell, self.width, self.height)
            if change.cell in cells:
                raise ValueError("duplicate cells in one observation are ambiguous")
            cells.add(change.cell)
        walls = set(self.walls)
        for change in changes:
            if change.blocked:
                walls.add(change.cell)
            else:
                walls.discard(change.cell)
        return GridSnapshot(self.width, self.height, frozenset(walls), self.goal)

    def fingerprint(self) -> str:
        payload = {"width": self.width, "height": self.height, "walls": sorted(self.walls), "goal": self.goal}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _distance(left: Point, right: Point) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def reference_bfs(grid: GridSnapshot, start: Point) -> dict:
    """Fresh exact search, implemented independently for differential checks."""
    origin = _point(start, grid.width, grid.height)
    if origin in grid.walls or grid.goal in grid.walls:
        return {"path": None, "path_length": None, "vertex_expansions": 0, "edge_inspections": 0}
    frontier = deque([origin])
    previous = {origin: None}
    expanded = inspected = 0
    while frontier:
        current = frontier.popleft()
        expanded += 1
        if current == grid.goal:
            route = []
            while current is not None:
                route.append(current)
                current = previous[current]
            route.reverse()
            return {
                "path": [list(cell) for cell in route],
                "path_length": len(route) - 1,
                "vertex_expansions": expanded,
                "edge_inspections": inspected,
            }
        x, y = current
        # Deliberately independent of GridSnapshot.neighbors and DIRECTIONS.
        for candidate in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            inspected += 1
            nx, ny = candidate
            if (
                0 <= nx < grid.width
                and 0 <= ny < grid.height
                and candidate not in grid.walls
                and candidate not in previous
            ):
                previous[candidate] = current
                frontier.append(candidate)
    return {"path": None, "path_length": None, "vertex_expansions": expanded, "edge_inspections": inspected}


class DStarLite:
    """Backward incremental shortest paths with moving-start key correction.

    Open entries are lazily invalidated by generation IDs and periodically
    compacted. Edges cost 1 between adjacent free cells, infinity otherwise.
    """

    name = "d_star_lite"

    def __init__(self, grid: GridSnapshot, start: Point):
        if not isinstance(grid, GridSnapshot):
            raise ValueError("DStarLite requires a validated GridSnapshot")
        start = _point(start, grid.width, grid.height)
        if not grid.walkable(start):
            raise ValueError("start must be walkable")
        self.grid, self.start = grid, start
        self.g: dict[Point, float] = {}
        self.rhs: dict[Point, float] = {grid.goal: 0}
        self.km = 0
        self.graph_version = self.replans = 0
        self._heap = []
        self._open = {}
        self._generation = 0
        self._observations: list[dict] = []
        self._totals = dict(
            vertex_expansions=0, queue_pops=0, edge_inspections=0, vertex_updates=0, queue_pushes=0, queue_compactions=0
        )
        self._reported = dict(self._totals)
        self._push(grid.goal)

    def _g(self, cell):
        return self.g.get(cell, INF)

    def _rhs(self, cell):
        return self.rhs.get(cell, INF)

    def _key(self, cell):
        estimate = min(self._g(cell), self._rhs(cell))
        return estimate + _distance(self.start, cell) + self.km, estimate

    def _push(self, cell):
        self._generation += 1
        key = self._key(cell)
        entry = (*key, self._generation, cell)
        self._open[cell] = entry
        heapq.heappush(self._heap, entry)
        self._totals["queue_pushes"] += 1
        if len(self._heap) > max(64, 4 * len(self._open)):
            self._heap = list(self._open.values())
            heapq.heapify(self._heap)
            self._totals["queue_compactions"] += 1

    def _top_key(self):
        while self._heap and self._open.get(self._heap[0][3]) != self._heap[0]:
            heapq.heappop(self._heap)
        return self._heap[0][:2] if self._heap else (INF, INF)

    def _pop(self):
        self._top_key()
        if not self._heap:
            raise RuntimeError("inconsistent start with empty OPEN queue")
        entry = heapq.heappop(self._heap)
        del self._open[entry[3]]
        self._totals["queue_pops"] += 1
        return entry[:2], entry[3]

    def _cost(self, left, right):
        self._totals["edge_inspections"] += 1
        return 1 if self.grid.walkable(left) and self.grid.walkable(right) else INF

    def _update_vertex(self, cell):
        self._totals["vertex_updates"] += 1
        if cell != self.grid.goal:
            self.rhs[cell] = min(
                (self._cost(cell, neighbor) + self._g(neighbor) for neighbor in self.grid.neighbors(cell)), default=INF
            )
        self._open.pop(cell, None)
        if self._g(cell) != self._rhs(cell):
            self._push(cell)

    def apply_changes(self, changes: list[CellChange] | tuple[CellChange, ...]) -> dict:
        if len(self._observations) >= 128:
            raise ValueError("repair before submitting more than 128 observations")
        updated = self.grid.changed(changes)
        if not updated.walkable(self.start):
            raise ValueError("a change cannot block the currently occupied start")
        actual = [change for change in changes if (change.cell in self.grid.walls) != change.blocked]
        self.grid = updated
        self.graph_version += 1
        affected = {neighbor for change in actual for neighbor in (change.cell, *self.grid.neighbors(change.cell))}
        for cell in sorted(affected):
            self._update_vertex(cell)
        observation = {
            "type": "cell_changes",
            "graph_version": self.graph_version,
            "changes": [{"cell": list(change.cell), "blocked": change.blocked} for change in actual],
            "knowledge": "complete application-provided change feed",
            "start": list(self.start),
        }
        self._observations.append(observation)
        return observation

    def move_start(self, destination: Point) -> dict:
        if len(self._observations) >= 128:
            raise ValueError("repair before submitting more than 128 observations")
        destination = _point(destination, self.grid.width, self.grid.height)
        if not self.grid.walkable(destination):
            raise ValueError("new start must be walkable")
        old = self.start
        self.km += _distance(old, destination)
        self.start = destination
        observation = {"type": "start_moved", "from": list(old), "to": list(destination), "km": self.km}
        self._observations.append(observation)
        return observation

    def _extract_path(self):
        if not self.grid.walkable(self.grid.goal) or not math.isfinite(self._g(self.start)):
            return None
        route = [self.start]
        while route[-1] != self.grid.goal:
            if len(route) > self.grid.width * self.grid.height:
                raise RuntimeError("nondecreasing or cyclic shortest-path labels")
            current = route[-1]
            candidates = [
                (self._cost(current, neighbor) + self._g(neighbor), neighbor)
                for neighbor in self.grid.neighbors(current)
            ]
            score, selected = min(candidates)
            if not math.isfinite(score) or score != self._g(current) or self._g(selected) >= self._g(current):
                raise RuntimeError("repaired labels do not yield a valid descending path")
            route.append(selected)
        return [list(cell) for cell in route]

    def repair(self, *, max_queue_pops: int = 100_000) -> dict:
        if type(max_queue_pops) is not int or not 1 <= max_queue_pops <= 1_000_000:
            raise ValueError("max_queue_pops must be an integer between 1 and 1000000")
        if self.replans >= 256:
            raise RepairBudgetExceeded("planner lifetime is limited to 256 repair attempts")
        self.replans += 1
        pops_before = self._totals["queue_pops"]
        trace = []
        while self._top_key() < self._key(self.start) or self._rhs(self.start) != self._g(self.start):
            if self._totals["queue_pops"] - pops_before >= max_queue_pops:
                raise RepairBudgetExceeded("repair queue-pop budget exhausted; no path result fabricated")
            old_key, cell = self._pop()
            new_key = self._key(cell)
            before_g, before_rhs = self._g(cell), self._rhs(cell)
            if old_key < new_key:
                branch = "rekey"
                self._push(cell)
            elif self._g(cell) > self._rhs(cell):
                branch = "lower_g"
                self.g[cell] = self._rhs(cell)
                self._totals["vertex_expansions"] += 1
                for predecessor in self.grid.neighbors(cell):
                    self._update_vertex(predecessor)
            else:
                branch = "invalidate_g"
                self.g[cell] = INF
                self._totals["vertex_expansions"] += 1
                for predecessor in (cell, *self.grid.neighbors(cell)):
                    self._update_vertex(predecessor)
            if len(trace) < 256:
                trace.append(
                    {
                        "cell": list(cell),
                        "branch": branch,
                        "old_key": [number if math.isfinite(number) else None for number in old_key],
                        "new_key": [number if math.isfinite(number) else None for number in new_key],
                        "before_g": before_g if math.isfinite(before_g) else None,
                        "before_rhs": before_rhs if math.isfinite(before_rhs) else None,
                        "g": self._g(cell) if math.isfinite(self._g(cell)) else None,
                        "rhs": self._rhs(cell) if math.isfinite(self._rhs(cell)) else None,
                    }
                )
        path = self._extract_path()
        delta = {key: value - self._reported[key] for key, value in self._totals.items()}
        self._reported = dict(self._totals)
        observations, self._observations = self._observations, []
        return {
            "algorithm": self.name,
            "repair_number": self.replans,
            "graph_version": self.graph_version,
            "grid_sha256": self.grid.fingerprint(),
            "start": list(self.start),
            "goal": list(self.grid.goal),
            "status": "path_found" if path is not None else "no_path",
            "path": path,
            "path_length": len(path) - 1 if path is not None else None,
            "start_g": self._g(self.start) if math.isfinite(self._g(self.start)) else None,
            "start_rhs": self._rhs(self.start) if math.isfinite(self._rhs(self.start)) else None,
            "km": self.km,
            "work": delta,
            "cumulative_work": dict(self._totals),
            "active_open_vertices": len(self._open),
            "observations": observations,
            "repair_trace": trace,
            "trace_events_omitted": delta["queue_pops"] - len(trace),
        }
