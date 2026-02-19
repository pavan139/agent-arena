"""Procedural maze generation, explicit tool permissions, and bounded episodes."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import hashlib
import math
import random
import time
from typing import Callable, Protocol

DIRECTIONS = {"north": (0, -1), "east": (1, 0), "south": (0, 1), "west": (-1, 0)}
POLICY_VERSION = "navigation-v2"


def _point(point: tuple[int, int], width: int, height: int) -> tuple[int, int]:
    if not isinstance(point, (tuple, list)) or len(point) != 2 or any(type(n) is not int for n in point):
        raise ValueError("coordinates must contain exactly two integers")
    x, y = point
    if not 0 <= x < width or not 0 <= y < height:
        raise ValueError("coordinate lies outside the maze")
    return x, y


def _allowlist(actions) -> frozenset[str]:
    if not isinstance(actions, (set, frozenset, tuple, list)) or len(actions) > 4:
        raise ValueError("allowed actions must be a bounded collection")
    if any(type(action) is not str or action not in DIRECTIONS for action in actions):
        raise ValueError("unknown allowed action")
    return frozenset(actions)


@dataclass(frozen=True)
class Maze:
    width: int
    height: int
    walls: frozenset[tuple[int, int]]
    start: tuple[int, int]
    goal: tuple[int, int]
    seed: int

    def __post_init__(self):
        if any(type(n) is not int or not 5 <= n <= 31 or n % 2 == 0 for n in (self.width, self.height)):
            raise ValueError("maze dimensions must be odd integers between 5 and 31")
        if type(self.seed) is not int or not -(2**31) <= self.seed < 2**31:
            raise ValueError("seed must be a signed 32-bit integer")
        if not isinstance(self.walls, (set, frozenset, tuple, list)) or len(self.walls) > self.width * self.height:
            raise ValueError("walls must be a bounded coordinate collection")
        frozen = frozenset(_point(point, self.width, self.height) for point in self.walls)
        object.__setattr__(self, "walls", frozen)
        object.__setattr__(self, "start", _point(self.start, self.width, self.height))
        object.__setattr__(self, "goal", _point(self.goal, self.width, self.height))
        if self.start in frozen or self.goal in frozen:
            raise ValueError("start and goal must be walkable")

    def walkable(self, point: tuple[int, int]) -> bool:
        x, y = point
        return 0 <= x < self.width and 0 <= y < self.height and point not in self.walls

    def render(self, position: tuple[int, int] | None = None) -> list[str]:
        if position is not None:
            position = _point(position, self.width, self.height)
            if not self.walkable(position):
                raise ValueError("render position must be walkable")
        return [
            "".join(
                "A"
                if (x, y) == position
                else "G"
                if (x, y) == self.goal
                else "S"
                if (x, y) == self.start
                else "#"
                if (x, y) in self.walls
                else "."
                for x in range(self.width)
            )
            for y in range(self.height)
        ]

    def fingerprint(self) -> str:
        import json

        geometry = {
            "width": self.width,
            "height": self.height,
            "walls": sorted(self.walls),
            "start": self.start,
            "goal": self.goal,
            "seed": self.seed,
        }
        return hashlib.sha256(json.dumps(geometry, sort_keys=True).encode()).hexdigest()


def generate_maze(seed: int, width: int = 11, height: int = 11) -> Maze:
    if type(seed) is not int or not -(2**31) <= seed < 2**31:
        raise ValueError("seed must be a signed 32-bit integer")
    if any(type(n) is not int or n < 5 or n > 31 or n % 2 == 0 for n in (width, height)):
        raise ValueError("maze dimensions must be odd integers between 5 and 31")
    rng = random.Random(seed)
    walls = {(x, y) for y in range(height) for x in range(width)}
    start, goal = (1, 1), (width - 2, height - 2)
    walls.remove(start)
    stack = [start]
    while stack:
        x, y = stack[-1]
        candidates = [
            (x + 2 * dx, y + 2 * dy, dx, dy)
            for dx, dy in DIRECTIONS.values()
            if 0 < x + 2 * dx < width - 1 and 0 < y + 2 * dy < height - 1 and (x + 2 * dx, y + 2 * dy) in walls
        ]
        if not candidates:
            stack.pop()
            continue
        nx, ny, dx, dy = rng.choice(candidates)
        walls.remove((x + dx, y + dy))
        walls.remove((nx, ny))
        stack.append((nx, ny))
    return Maze(width, height, frozenset(walls), start, goal, seed)


@dataclass(frozen=True)
class Observation:
    maze: Maze
    position: tuple[int, int]
    remaining_steps: int
    allowed_actions: frozenset[str] = frozenset(DIRECTIONS)

    def __post_init__(self):
        if not isinstance(self.maze, Maze):
            raise ValueError("observation requires a validated Maze")
        position = _point(self.position, self.maze.width, self.maze.height)
        if not self.maze.walkable(position):
            raise ValueError("observation position must be walkable")
        if type(self.remaining_steps) is not int or not 0 <= self.remaining_steps <= 4096:
            raise ValueError("observation remaining steps must be between 0 and 4096")
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "allowed_actions", _allowlist(self.allowed_actions))


@dataclass(frozen=True)
class LocalObservation:
    """A neighbor-only view: no Maze reference or unseen occupancy information."""

    position: tuple[int, int]
    goal: tuple[int, int]
    remaining_steps: int
    neighbors: tuple[tuple[str, bool], ...]
    allowed_actions: frozenset[str]

    def __post_init__(self):
        object.__setattr__(self, "position", _point(self.position, 31, 31))
        object.__setattr__(self, "goal", _point(self.goal, 31, 31))
        if type(self.remaining_steps) is not int or not 0 <= self.remaining_steps <= 4096:
            raise ValueError("observation remaining steps must be between 0 and 4096")
        if not isinstance(self.neighbors, (tuple, list)) or len(self.neighbors) != 4:
            raise ValueError("local observation requires four neighbor readings")
        readings = []
        for reading in self.neighbors:
            if (
                not isinstance(reading, (tuple, list))
                or len(reading) != 2
                or type(reading[0]) is not str
                or reading[0] not in DIRECTIONS
                or type(reading[1]) is not bool
            ):
                raise ValueError("invalid local neighbor reading")
            readings.append(tuple(reading))
        if {action for action, _ in readings} != DIRECTIONS.keys():
            raise ValueError("local neighbor directions must be unique")
        object.__setattr__(self, "neighbors", tuple(readings))
        object.__setattr__(self, "allowed_actions", _allowlist(self.allowed_actions))


class ExplorationExhausted(RuntimeError):
    code = "exploration_exhausted"


class LocalDFSExplorer:
    """Memory-based graph exploration using only immediate-neighbor sensing."""

    name = "local_dfs_explorer"
    observation_mode = "local_neighbors"

    def __init__(self):
        self.reset()

    def reset(self):
        self.visited: set[tuple[int, int]] = set()
        self.route: list[tuple[int, int]] = []

    def choose(self, observation: LocalObservation) -> str:
        if not isinstance(observation, LocalObservation):
            raise TypeError("local explorer requires a neighbor-only observation")
        position = observation.position
        if not self.route:
            self.route.append(position)
        if self.route[-1] != position:
            raise ExplorationExhausted("observed position differs from proposed route")
        self.visited.add(position)
        for action, walkable in observation.neighbors:
            target = destination(position, action)
            if walkable and action in observation.allowed_actions and target not in self.visited:
                self.route.append(target)
                return action
        if len(self.route) > 1:
            parent = self.route[-2]
            for action in observation.allowed_actions:
                if destination(position, action) == parent:
                    self.route.pop()
                    return action
        raise ExplorationExhausted("no unvisited neighbor or permitted backtracking move")


class Policy(Protocol):
    name: str

    def choose(self, observation: Observation | LocalObservation) -> str: ...


def destination(position: tuple[int, int], action: str) -> tuple[int, int]:
    dx, dy = DIRECTIONS[action]
    return position[0] + dx, position[1] + dy


def shortest_path(maze: Maze, start: tuple[int, int] | None = None) -> list[str] | None:
    if not isinstance(maze, Maze):
        raise ValueError("shortest_path requires a validated maze")
    origin = maze.start if start is None else _point(start, maze.width, maze.height)
    if not maze.walkable(origin) or not maze.walkable(maze.goal):
        return None
    queue = deque([origin])
    parents: dict[tuple[int, int], tuple[tuple[int, int], str] | None] = {origin: None}
    while queue:
        position = queue.popleft()
        if position == maze.goal:
            actions = []
            while parents[position] is not None:
                position, action = parents[position]
                actions.append(action)
            return list(reversed(actions))
        for action in DIRECTIONS:
            point = destination(position, action)
            if maze.walkable(point) and point not in parents:
                parents[point] = (position, action)
                queue.append(point)
    return None


class ReactivePolicy:
    """Memoryless Manhattan-distance baseline; can become trapped in cycles."""

    name = "reactive_manhattan"
    observation_mode = "full_map"

    def choose(self, observation: Observation) -> str:
        choices = [(action, destination(observation.position, action)) for action in DIRECTIONS]
        choices = [(a, p) for a, p in choices if observation.maze.walkable(p)]
        if not choices:
            return "north"
        gx, gy = observation.maze.goal
        return min(choices, key=lambda pair: abs(pair[1][0] - gx) + abs(pair[1][1] - gy))[0]


class BFSOracle:
    """Full-map shortest-path oracle. This is classical search, not an LLM."""

    name = "bfs_planning_oracle"
    observation_mode = "full_map"

    def choose(self, observation: Observation) -> str:
        path = shortest_path(observation.maze, observation.position)
        return path[0] if path else "north"


@dataclass(frozen=True)
class Limits:
    max_steps: int = 128
    repeated_action_limit: int = 3
    no_progress_limit: int = 3
    timeout_seconds: float = 2.0
    allowed_actions: frozenset[str] = frozenset(DIRECTIONS)

    def __post_init__(self):
        if type(self.max_steps) is not int or not 0 <= self.max_steps <= 4096:
            raise ValueError("max_steps must be an integer between 0 and 4096")
        if any(type(n) is not int or not 1 <= n <= 100 for n in (self.repeated_action_limit, self.no_progress_limit)):
            raise ValueError("repetition and no-progress limits must be between 1 and 100")
        if (
            type(self.timeout_seconds) not in (int, float)
            or not math.isfinite(self.timeout_seconds)
            or not 0 <= self.timeout_seconds <= 60
        ):
            raise ValueError("timeout must be between 0 and 60 seconds")
        object.__setattr__(self, "allowed_actions", _allowlist(self.allowed_actions))


def _model_call(policy, previous) -> dict | None:
    current = getattr(policy, "last_call", None)
    if not isinstance(current, dict) or current is previous:
        return None
    allowed = {
        "provider",
        "checkpoint_name",
        "checkpoint_sha256",
        "prompt_version",
        "call_number",
        "input_tokens",
        "output_tokens",
        "latency_ms",
        "finish_reason",
        "error_code",
    }
    return {
        key: value
        for key, value in current.items()
        if key in allowed
        and type(value) in (str, int, float, bool, type(None))
        and (not isinstance(value, str) or len(value) <= 256)
    }


def _observe(maze: Maze, position, remaining_steps, limits: Limits, mode: str):
    if mode == "local_neighbors":
        return LocalObservation(
            position,
            maze.goal,
            remaining_steps,
            tuple((action, maze.walkable(destination(position, action))) for action in DIRECTIONS),
            limits.allowed_actions,
        )
    return Observation(maze, position, remaining_steps, limits.allowed_actions)


def run_episode(
    maze: Maze,
    policy: Policy,
    limits: Limits | None = None,
    cancelled: Callable[[], bool] = lambda: False,
    clock: Callable[[], float] = time.monotonic,
) -> dict:
    """Run synchronous trusted policies. Deadlines checked around each policy call.

    This cannot interrupt an arbitrary blocking Python policy; isolate remote or
    untrusted adapters in a worker process before admitting them to the harness.
    """
    if not isinstance(maze, Maze) or (limits is not None and not isinstance(limits, Limits)):
        raise ValueError("episode requires a validated Maze and Limits")
    if not callable(cancelled) or not callable(clock) or not callable(getattr(policy, "choose", None)):
        raise ValueError("policy, cancellation, and clock must implement the episode contract")
    if not isinstance(getattr(policy, "name", None), str) or not 1 <= len(policy.name) <= 100:
        raise ValueError("policy name must be a bounded string")
    mode = getattr(policy, "observation_mode", "full_map")
    if mode not in {"full_map", "local_neighbors"}:
        raise ValueError("unknown policy observation mode")
    limits = limits if limits is not None else Limits()
    started = clock()
    policy_usage = getattr(policy, "usage", {})
    usage_before = dict(policy_usage) if isinstance(policy_usage, dict) else {}
    position = maze.start
    trace = []
    seen: Counter = Counter()
    no_progress = 0
    stop = "step_budget"
    reset_failed = False
    if callable(getattr(policy, "reset", None)):
        try:
            policy.reset()
        except Exception as exc:
            trace.append(
                {
                    "step": 0,
                    "before": list(position),
                    "allowed": False,
                    "valid": False,
                    "error_code": "policy_reset_error",
                    "error_type": type(exc).__name__,
                }
            )
            stop, reset_failed = "policy_error", True
    for step in range(0 if reset_failed else limits.max_steps + 1):
        if cancelled():
            stop = "cancelled"
            break
        if clock() - started >= limits.timeout_seconds:
            stop = "deadline"
            break
        if position == maze.goal:
            stop = "goal_reached"
            break
        if step == limits.max_steps:
            break
        observation = _observe(maze, position, limits.max_steps - step, limits, mode)
        previous_call = getattr(policy, "last_call", None)
        try:
            action = policy.choose(observation)
        except Exception as exc:
            event = {
                "step": step,
                "before": list(position),
                "allowed": False,
                "valid": False,
                "error_code": "policy_error",
                "error_type": type(exc).__name__,
            }
            if isinstance(getattr(exc, "code", None), str):
                event["policy_error_code"] = exc.code[:80]
            metadata = _model_call(policy, previous_call)
            if metadata:
                event["model_call"] = metadata
            trace.append(event)
            stop = "policy_error"
            break
        event = {
            "step": step,
            "action": action if type(action) is str and action in DIRECTIONS else "<invalid>",
            "before": list(position),
            "allowed": False,
            "valid": False,
        }
        metadata = _model_call(policy, previous_call)
        if metadata:
            event["model_call"] = metadata
        if clock() - started >= limits.timeout_seconds:
            event["error_code"] = "deadline"
            trace.append(event)
            stop = "deadline"
            break
        if cancelled():
            event["error_code"] = "cancelled"
            trace.append(event)
            stop = "cancelled"
            break
        if type(action) is not str or action not in DIRECTIONS:
            event["error_code"] = stop = "invalid_action"
            trace.append(event)
            break
        if action not in limits.allowed_actions:
            event["error_code"] = stop = "permission_denied"
            trace.append(event)
            break
        event["allowed"] = True
        signature = (position, action)
        seen[signature] += 1
        if seen[signature] > limits.repeated_action_limit:
            event["error_code"] = stop = "repeated_action"
            trace.append(event)
            break
        target = destination(position, action)
        moved = maze.walkable(target)
        if moved:
            position = target
            no_progress = 0
        else:
            no_progress += 1
            event["error_code"] = "blocked_move"
        event.update(
            after=list(position), valid=moved, state_hash=hashlib.sha256(repr(position).encode()).hexdigest()[:16]
        )
        trace.append(event)
        if position == maze.goal:
            stop = "goal_reached"
            break
        if no_progress >= limits.no_progress_limit:
            stop = "no_progress"
            break
    optimal = shortest_path(maze)
    usage = {"provider_calls": 0, "tokens": 0, "cost_usd": 0}
    if hasattr(policy, "usage"):
        usage = {
            key: (value - usage_before.get(key, 0) if isinstance(value, (int, float)) else value)
            for key, value in policy.usage.items()
        }
    return {
        "policy": policy.name,
        "model_identity": getattr(policy, "model_identity", None),
        "policy_version": POLICY_VERSION,
        "seed": maze.seed,
        "maze_sha256": maze.fingerprint(),
        "observation_mode": mode,
        "limits": {
            "max_steps": limits.max_steps,
            "repeated_action_limit": limits.repeated_action_limit,
            "no_progress_limit": limits.no_progress_limit,
            "timeout_seconds": limits.timeout_seconds,
            "allowed_actions": sorted(limits.allowed_actions),
        },
        "start": list(maze.start),
        "dimensions": [maze.width, maze.height],
        "success": stop == "goal_reached",
        "stop_reason": stop,
        "steps": sum("after" in t for t in trace),
        "policy_attempts": sum(t.get("error_code") != "policy_reset_error" for t in trace),
        "valid_moves": sum(t.get("valid", False) for t in trace),
        "optimal_steps": len(optimal) if optimal is not None else None,
        "final_position": list(position),
        "goal": list(maze.goal),
        "latency_ms": round((clock() - started) * 1000, 3),
        "trace": trace,
        "usage": usage,
    }
