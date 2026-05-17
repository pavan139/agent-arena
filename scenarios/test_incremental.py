from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from agent_arena import CellChange, DStarLite, GridSnapshot, RepairBudgetExceeded, generate_maze
from agent_arena.__main__ import main
from agent_arena.incremental import reference_bfs
from agent_arena.replanning import DYNAMIC_SEEDS, dfs_snapshot, dynamic_evaluate, seeded_scenario


class IncrementalTests(unittest.TestCase):
    def assert_optimal(self, planner):
        result = planner.repair()
        expected = reference_bfs(planner.grid, planner.start)
        self.assertEqual(result["path_length"], expected["path_length"])
        if result["path"] is not None:
            self.assertEqual(result["path"][0], list(planner.start))
            self.assertEqual(result["path"][-1], list(planner.grid.goal))
            for left, right in zip(result["path"], result["path"][1:]):
                self.assertTrue(planner.grid.walkable(tuple(left)))
                self.assertTrue(planner.grid.walkable(tuple(right)))
                self.assertEqual(abs(left[0] - right[0]) + abs(left[1] - right[1]), 1)
        return result

    def test_known_open_grid_distance_and_backward_labels(self):
        planner = DStarLite(GridSnapshot(7, 7, frozenset(), (5, 5)), (1, 1))
        result = self.assert_optimal(planner)
        self.assertEqual(result["path_length"], 8)
        self.assertEqual(planner.g[planner.grid.goal], 0)
        self.assertEqual(result["start_g"], result["start_rhs"])
        self.assertGreater(result["work"]["vertex_expansions"], 0)

    def test_warm_repair_and_moving_start_reuse_labels_without_restarting(self):
        planner = DStarLite(GridSnapshot(7, 7, frozenset(), (5, 5)), (1, 1))
        original = self.assert_optimal(planner)
        labels = planner.g
        with patch("agent_arena.incremental.reference_bfs", side_effect=AssertionError("planner must not use BFS")):
            warm = self.assert_optimal(planner)
        self.assertEqual(warm["work"]["vertex_expansions"], 0)
        planner.move_start(tuple(original["path"][1]))
        moved = self.assert_optimal(planner)
        self.assertIs(planner.g, labels)
        self.assertEqual(moved["work"]["vertex_expansions"], 0)
        self.assertEqual(moved["path_length"], 7)
        self.assertEqual(moved["km"], 1)

    def test_cost_increase_disconnection_and_decrease_reopening(self):
        free = {(1, 1), (2, 1), (3, 1), (3, 2), (3, 3)}
        walls = frozenset((x, y) for y in range(5) for x in range(5) if (x, y) not in free)
        planner = DStarLite(GridSnapshot(5, 5, walls, (3, 3)), (1, 1))
        first = self.assert_optimal(planner)
        self.assertEqual(first["path_length"], 4)
        planner.apply_changes([CellChange((2, 1), True)])
        blocked = self.assert_optimal(planner)
        self.assertEqual(blocked["status"], "no_path")
        self.assertIsNone(blocked["start_g"])
        self.assertIn("invalidate_g", {event["branch"] for event in blocked["repair_trace"]})
        planner.apply_changes([CellChange((2, 1), False)])
        reopened = self.assert_optimal(planner)
        self.assertEqual(reopened["path_length"], 4)
        self.assertIn("lower_g", {event["branch"] for event in reopened["repair_trace"]})

    def test_blocked_goal_and_reopening_are_not_fabricated_success(self):
        planner = DStarLite(GridSnapshot(7, 7, frozenset(), (5, 5)), (1, 1))
        self.assert_optimal(planner)
        planner.apply_changes([CellChange((5, 5), True)])
        self.assertEqual(self.assert_optimal(planner)["status"], "no_path")
        planner.apply_changes([CellChange((5, 5), False)])
        self.assertEqual(self.assert_optimal(planner)["path_length"], 8)

    def test_initial_no_path_then_new_edges_open_a_route(self):
        walls = frozenset((3, y) for y in range(7))
        planner = DStarLite(GridSnapshot(7, 7, walls, (5, 5)), (1, 1))
        self.assertEqual(self.assert_optimal(planner)["status"], "no_path")
        planner.apply_changes([CellChange((3, 2), False)])
        self.assertEqual(self.assert_optimal(planner)["path_length"], 8)

    def test_start_equal_to_goal_and_explicit_relocalization(self):
        planner = DStarLite(GridSnapshot(7, 7, frozenset(), (5, 5)), (5, 5))
        self.assertEqual(self.assert_optimal(planner)["path_length"], 0)
        planner.move_start((0, 0))
        self.assertEqual(self.assert_optimal(planner)["path_length"], 10)
        self.assertEqual(planner.km, 10)
        planner.move_start((5, 5))
        self.assertEqual(self.assert_optimal(planner)["path_length"], 0)
        self.assertEqual(planner.km, 20)

    def test_batched_changes_are_atomic_and_duplicate_or_occupied_cells_rejected(self):
        planner = DStarLite(GridSnapshot(7, 7, frozenset(), (5, 5)), (1, 1))
        before = planner.grid.fingerprint()
        for updates in (
            [CellChange((2, 1), True), CellChange((2, 1), False)],
            [CellChange((2, 1), True), CellChange((1, 1), True)],
            [CellChange((30, 30), True)],
            ["not a change"],
        ):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                planner.apply_changes(updates)
            self.assertEqual(planner.grid.fingerprint(), before)
            self.assertEqual(planner.graph_version, 0)

    def test_grid_and_change_validation_precede_search(self):
        for cell, blocked in (((True, 1), True), ((31, 0), True), ((1, 1), 1)):
            with self.subTest(cell=cell, blocked=blocked), self.assertRaises(ValueError):
                CellChange(cell, blocked)
        for kwargs in ({"width": 4}, {"walls": {(7, 0)}}, {"goal": (7, 0)}):
            parameters = dict(width=7, height=7, walls=frozenset(), goal=(5, 5))
            parameters.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                GridSnapshot(**parameters)

    def test_failed_repair_budget_can_resume_without_a_reference_fallback(self):
        planner = DStarLite(GridSnapshot(15, 15, frozenset(), (13, 13)), (1, 1))
        with self.assertRaises(RepairBudgetExceeded):
            planner.repair(max_queue_pops=1)
        self.assertEqual(planner.replans, 1)
        result = self.assert_optimal(planner)
        self.assertEqual(result["path_length"], 24)
        self.assertGreater(result["work"]["queue_pops"], 1)
        for budget in (0, True, 1_000_001):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                planner.repair(max_queue_pops=budget)

    def test_lifetime_and_pending_observation_budgets(self):
        planner = DStarLite(GridSnapshot(5, 5, frozenset(), (3, 3)), (1, 1))
        for _ in range(128):
            planner.move_start((1, 1))
        with self.assertRaises(ValueError):
            planner.move_start((1, 1))
        self.assert_optimal(planner)
        for _ in range(255):
            planner.repair()
        with self.assertRaises(RepairBudgetExceeded):
            planner.repair()

    def test_idempotent_changes_and_boundary_cell_edges(self):
        planner = DStarLite(GridSnapshot(5, 5, frozenset({(0, 0)}), (4, 4)), (1, 1))
        self.assert_optimal(planner)
        planner.apply_changes([CellChange((0, 0), True)])
        self.assertEqual(self.assert_optimal(planner)["work"]["vertex_expansions"], 0)
        planner.apply_changes([CellChange((0, 0), False), CellChange((4, 4), True)])
        self.assertEqual(self.assert_optimal(planner)["status"], "no_path")
        planner.apply_changes([CellChange((4, 4), False)])
        self.assert_optimal(planner)

    def test_thousand_generated_repairs_match_independent_bfs(self):
        checked = 0
        for seed in range(20):
            rng = random.Random(91000 + seed)
            grid = GridSnapshot.from_maze(generate_maze(seed, 9, 9))
            planner = DStarLite(grid, (1, 1))
            expected_walls = set(grid.walls)
            for operation in range(50):
                if operation and operation % 7 == 0:
                    choices = [cell for cell in grid.neighbors(planner.start) if grid.walkable(cell)]
                    if choices:
                        planner.move_start(rng.choice(choices))
                elif operation:
                    selected = rng.sample(
                        [(x, y) for y in range(9) for x in range(9) if (x, y) != planner.start], rng.randint(1, 3)
                    )
                    updates = [CellChange(cell, cell not in expected_walls) for cell in selected]
                    for change in updates:
                        if change.blocked:
                            expected_walls.add(change.cell)
                        else:
                            expected_walls.discard(change.cell)
                    grid = GridSnapshot(9, 9, frozenset(expected_walls), grid.goal)
                    planner.apply_changes(updates)
                with self.subTest(seed=seed, operation=operation):
                    self.assertEqual(planner.grid, grid)
                    self.assert_optimal(planner)
                    self.assertLessEqual(len(planner._open), 81)
                    self.assertLessEqual(len(planner._heap), 4 * 81 + 64)
                checked += 1
        self.assertEqual(checked, 1000)

    def test_seeded_feed_is_deterministic_and_contains_required_failures(self):
        self.assertEqual(seeded_scenario(300), seeded_scenario(300))
        report = dynamic_evaluate((300, 301))
        self.assertEqual(report["checks_passed"], report["checks_total"])
        self.assertEqual(report["independent_bfs_agreements"], report["checks_total"])
        self.assertGreaterEqual(report["no_path_case_count"], 4)
        tags = {case["observation"]["tag"] for case in report["cases"]}
        self.assertTrue({"isolate_goal", "restore_goal_edges", "block_goal", "reopen_goal", "start_move"} <= tags)
        self.assertEqual(report["model_calls"], 0)
        self.assertEqual(len(report["dataset_sha256"]), 64)

    def test_dfs_baseline_uses_existing_policy_and_does_not_claim_shortest_paths(self):
        initial, start, _ = seeded_scenario(300)
        result = dfs_snapshot(initial, start)
        expected = reference_bfs(initial, start)
        self.assertIsNotNone(result["path"])
        self.assertGreaterEqual(result["path_length"], expected["path_length"])

    def test_cli_and_full_default_experiment(self):
        report = dynamic_evaluate()
        self.assertEqual(report["checks_total"], len(DYNAMIC_SEEDS) * 13)
        self.assertEqual(report["checks_passed"], report["checks_total"])
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "report.json"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["replan-evaluate", "--seed", "300", "--output", str(output)]), 0)
            self.assertEqual(json.loads(output.read_text())["independent_bfs_agreements"], 13)
        branches = {event["branch"] for row in report["cases"] for event in row["d_star_lite"]["repair_trace"]}
        self.assertEqual(branches, {"lower_g", "invalidate_g", "rekey"})


if __name__ == "__main__":
    unittest.main()
