"""Seeded obstacle-change experiments with exact independent BFS checking."""

from __future__ import annotations

import hashlib
import json
import random

from .arena import DIRECTIONS, ExplorationExhausted, LocalDFSExplorer, LocalObservation, destination, generate_maze
from .incremental import CellChange, DStarLite, GridSnapshot, reference_bfs

DYNAMIC_SEEDS = tuple(range(300, 312))
DATASET_VERSION = "incremental-navigation-v1"


def _changes(records: list[dict]) -> list[CellChange]:
    return [CellChange(tuple(record["cell"]), record["blocked"]) for record in records]


def seeded_scenario(seed: int, width: int = 15, height: int = 15) -> tuple[GridSnapshot, tuple[int, int], list[dict]]:
    """Freeze one adversarial feed; all planners receive the same observations.

    BFS is used only by the simulator to select interesting obstacle events and
    legal start moves, never by D* Lite to compute or repair its answer.
    """
    maze = generate_maze(seed, width, height)
    rng = random.Random(seed)
    candidates = [cell for cell in sorted(maze.walls) if 0 < cell[0] < width - 1 and 0 < cell[1] < height - 1]
    rng.shuffle(candidates)
    initial = GridSnapshot.from_maze(maze).changed([CellChange(cell, False) for cell in candidates[:12]])
    current, start = initial, maze.start
    events = []

    def change(tag, updates):
        nonlocal current
        current = current.changed(updates)
        events.append(
            {
                "tag": tag,
                "type": "cell_changes",
                "changes": [{"cell": list(update.cell), "blocked": update.blocked} for update in updates],
            }
        )

    def move():
        nonlocal start
        route = reference_bfs(current, start)["path"]
        if route and len(route) > 1:
            start = tuple(route[1])
        else:
            available = [cell for cell in current.neighbors(start) if current.walkable(cell)]
            if available:
                start = min(available)
        events.append({"tag": "start_move", "type": "start_moved", "to": list(start)})

    route = reference_bfs(current, start)["path"]
    middle = tuple(route[len(route) // 2])
    change("close_current_route", [CellChange(middle, True)])
    change("reopen_current_route", [CellChange(middle, False)])
    move()
    exits = [cell for cell in current.neighbors(current.goal) if current.walkable(cell)]
    change("isolate_goal", [CellChange(cell, True) for cell in exits])
    change("restore_goal_edges", [CellChange(cell, False) for cell in exits])
    change("block_goal", [CellChange(current.goal, True)])
    change("reopen_goal", [CellChange(current.goal, False)])
    change("idempotent_observation", [CellChange((0, 0), True)])
    move()
    shortcuts = [
        cell
        for cell in sorted(current.walls)
        if 0 < cell[0] < width - 1
        and 0 < cell[1] < height - 1
        and sum(current.walkable(neighbor) for neighbor in current.neighbors(cell)) >= 2
    ]
    if shortcuts:
        change("open_shortcut", [CellChange(rng.choice(shortcuts), False)])
    selected = rng.sample(
        [(x, y) for y in range(1, height - 1) for x in range(1, width - 1) if (x, y) not in {start, current.goal}], 2
    )
    change("mixed_edge_changes", [CellChange(cell, cell not in current.walls) for cell in selected])
    move()
    return initial, maze.start, [{"event_number": index + 1, **event} for index, event in enumerate(events)]


def dfs_snapshot(grid: GridSnapshot, start: tuple[int, int]) -> dict:
    """Restart the existing neighbor-only DFS policy for this world snapshot."""
    policy = LocalDFSExplorer()
    position = start
    walk = [list(start)]
    expanded = 0
    max_steps = 2 * grid.width * grid.height + 1
    for step in range(max_steps):
        if position == grid.goal and grid.walkable(grid.goal):
            return {
                "algorithm": "restarted_local_dfs",
                "status": "path_found",
                "path": walk,
                "path_length": len(walk) - 1,
                "walk_length": len(walk) - 1,
                "vertex_expansions": expanded,
                "edge_inspections": 4 * expanded,
            }
        observation = LocalObservation(
            position,
            grid.goal,
            max_steps - step,
            tuple((action, grid.walkable(destination(position, action))) for action in DIRECTIONS),
            frozenset(DIRECTIONS),
        )
        expanded += 1
        try:
            action = policy.choose(observation)
        except ExplorationExhausted:
            return {
                "algorithm": "restarted_local_dfs",
                "status": "no_path",
                "path": None,
                "path_length": None,
                "walk_length": len(walk) - 1,
                "vertex_expansions": expanded,
                "edge_inspections": 4 * expanded,
            }
        target = destination(position, action)
        if not grid.walkable(target):
            raise RuntimeError("DFS proposed an invalid transition")
        position = target
        walk.append(list(position))
    raise RuntimeError("DFS exceeded the finite exploration bound")


def _valid_route(grid: GridSnapshot, start, path) -> bool:
    if not isinstance(path, list) or not path or path[0] != list(start) or path[-1] != list(grid.goal):
        return False
    for cell in path:
        if (
            not isinstance(cell, list)
            or len(cell) != 2
            or any(type(n) is not int for n in cell)
            or not grid.walkable(tuple(cell))
        ):
            return False
    return all(abs(left[0] - right[0]) + abs(left[1] - right[1]) == 1 for left, right in zip(path, path[1:]))


def dynamic_evaluate(seeds: tuple[int, ...] = DYNAMIC_SEEDS) -> dict:
    if (
        not isinstance(seeds, tuple)
        or not 1 <= len(seeds) <= 32
        or any(type(seed) is not int for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("evaluation requires 1..32 distinct integer seeds")
    scenarios, cases = [], []
    for seed in seeds:
        initial, start, events = seeded_scenario(seed)
        scenarios.append(
            {"seed": seed, "initial_grid_sha256": initial.fingerprint(), "start": list(start), "events": events}
        )
        expected_world = initial
        # The planner owns a separate immutable snapshot; BFS observes the
        # simulator's independently updated snapshot after every event.
        planner = DStarLite(initial, start)
        for event in [{"event_number": 0, "tag": "initial_plan", "type": "initial"}, *events]:
            if event["type"] == "cell_changes":
                updates = _changes(event["changes"])
                expected_world = expected_world.changed(updates)
                planner.apply_changes(updates)
            elif event["type"] == "start_moved":
                start = tuple(event["to"])
                planner.move_start(start)
            result = planner.repair()
            bfs = reference_bfs(expected_world, start)
            dfs = dfs_snapshot(expected_world, start)
            path_matches = result["path_length"] == bfs["path_length"] and (
                result["path"] is None or _valid_route(expected_world, start, result["path"])
            )
            dfs_matches = (dfs["path"] is None) == (bfs["path"] is None) and (
                dfs["path"] is None or _valid_route(expected_world, start, dfs["path"])
            )
            cases.append(
                {
                    "id": f"seed-{seed}-event-{event['event_number']}",
                    "seed": seed,
                    "observation": event,
                    "passed": path_matches and dfs_matches and planner.grid == expected_world,
                    "d_star_bfs_cost_agreement": path_matches,
                    "dfs_reachability_agreement": dfs_matches,
                    "d_star_lite": result,
                    "repeated_bfs": bfs,
                    "restarted_dfs": dfs,
                }
            )
    metrics = {}
    for name in ("d_star_lite", "repeated_bfs", "restarted_dfs"):

        def work(row, metric):
            return row[name]["work"][metric] if name == "d_star_lite" else row[name][metric]

        reachable = [row for row in cases if row["repeated_bfs"]["path"] is not None]
        initial_rows = [row for row in cases if row["observation"]["type"] == "initial"]
        repair_rows = [row for row in cases if row["observation"]["type"] != "initial"]
        metrics[name] = {
            "planning_calls": len(cases),
            "path_found": sum(row[name]["path"] is not None for row in cases),
            "no_path": sum(row[name]["path"] is None for row in cases),
            "reachable_case_count": len(reachable),
            "shortest_path_count": sum(
                row[name]["path_length"] == row["repeated_bfs"]["path_length"] for row in reachable
            ),
            "vertex_expansions": sum(work(row, "vertex_expansions") for row in cases),
            "edge_inspections": sum(work(row, "edge_inspections") for row in cases),
            "initial_vertex_expansions": sum(work(row, "vertex_expansions") for row in initial_rows),
            "repair_vertex_expansions": sum(work(row, "vertex_expansions") for row in repair_rows),
            "initial_case_count": len(initial_rows),
            "repair_case_count": len(repair_rows),
            "mean_reachable_walk_length": sum(row[name]["path_length"] for row in reachable) / len(reachable)
            if reachable
            else None,
        }
    manifest = {"version": DATASET_VERSION, "dimensions": [15, 15], "scenarios": scenarios}
    return {
        "dataset": manifest,
        "dataset_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        "run_type": "classical_incremental_search_experiment",
        "model_calls": 0,
        "checks_passed": sum(row["passed"] for row in cases),
        "checks_total": len(cases),
        "independent_bfs_agreements": sum(row["d_star_bfs_cost_agreement"] for row in cases),
        "no_path_case_count": sum(row["repeated_bfs"]["path"] is None for row in cases),
        "information_contract": "D* Lite: initial full map plus complete changes; BFS: fresh full snapshot; DFS: neighbor sensing with episode memory reset per snapshot",
        "work_counter_caveat": "expansions count algorithm-specific state processing; D* edge inspections also include rhs updates and path extraction; counters are not wall-clock speedups",
        "metrics": metrics,
        "failures": [row["id"] for row in cases if not row["passed"]],
        "cases": cases,
    }
