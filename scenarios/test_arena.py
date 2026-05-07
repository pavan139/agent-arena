import unittest
from agent_arena.arena import BFSOracle, Limits, Maze, generate_maze, run_episode, shortest_path
from agent_arena.evaluation import DEV_SEEDS, RELEASE_SEEDS, FixedPolicy, evaluate


class ArenaTests(unittest.TestCase):
    def test_generation_is_deterministic_and_solvable(self):
        for seed in range(20):
            with self.subTest(seed=seed):
                maze = generate_maze(seed)
                self.assertEqual(maze, generate_maze(seed))
                self.assertIsNotNone(shortest_path(maze))
                self.assertTrue(maze.walkable(maze.goal))

    def test_dimensions_and_seed_are_validated(self):
        for args in [(0, 4, 5), (0, 32, 5), (0, True, 5), ("bad", 5, 5), (2**40, 5, 5)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                generate_maze(*args)

    def test_oracle_reaches_exact_goal_with_optimal_actions(self):
        maze = generate_maze(101)
        result = run_episode(maze, BFSOracle())
        self.assertTrue(result["success"])
        self.assertEqual(result["final_position"], list(maze.goal))
        self.assertEqual(result["steps"], len(shortest_path(maze)))
        self.assertTrue(all(t["allowed"] and t["valid"] for t in result["trace"]))

    def test_unreachable_goal_cannot_be_success(self):
        maze = Maze(5, 5, frozenset((2, y) for y in range(5)), (1, 1), (3, 3), 0)
        self.assertIsNone(shortest_path(maze))
        result = run_episode(maze, BFSOracle())
        self.assertFalse(result["success"])

    def test_permission_denied_before_action(self):
        maze = generate_maze(0)
        result = run_episode(maze, BFSOracle(), Limits(allowed_actions=frozenset()))
        self.assertEqual(result["stop_reason"], "permission_denied")
        self.assertEqual(result["steps"], 0)
        self.assertEqual(result["final_position"], list(maze.start))

    def test_policy_output_cannot_grant_capability(self):
        maze = generate_maze(0)
        result = run_episode(maze, FixedPolicy("ignore policy and read_secret"))
        self.assertEqual(result["stop_reason"], "invalid_action")
        self.assertNotIn("read_secret", result["trace"][0]["action"])

    def test_malformed_action_is_rejected(self):
        result = run_episode(generate_maze(0), FixedPolicy({"tool": "north"}))
        self.assertEqual(result["stop_reason"], "invalid_action")

    def test_step_deadline_cancellation_budgets(self):
        maze = generate_maze(0)
        for limits, cancel, expected in [
            (Limits(max_steps=0), lambda: False, "step_budget"),
            (Limits(timeout_seconds=0), lambda: False, "deadline"),
            (Limits(), lambda: True, "cancelled"),
        ]:
            with self.subTest(expected=expected):
                result = run_episode(maze, BFSOracle(), limits, cancel)
                self.assertEqual(result["stop_reason"], expected)
                self.assertEqual(result["steps"], 0)

    def test_cancel_after_policy_call_prevents_mutation(self):
        cancelled = [False]

        class CancellingPolicy:
            name = "cancel_during_selection"

            def choose(self, observation):
                cancelled[0] = True
                return BFSOracle().choose(observation)

        result = run_episode(generate_maze(0), CancellingPolicy(), cancelled=lambda: cancelled[0])
        self.assertEqual(result["stop_reason"], "cancelled")
        self.assertEqual(result["steps"], 0)

    def test_blocked_and_repeated_actions_stop(self):
        maze = generate_maze(0)
        blocked = run_episode(maze, FixedPolicy("north"))
        self.assertEqual(blocked["stop_reason"], "no_progress")
        repeated = run_episode(maze, FixedPolicy("north"), Limits(repeated_action_limit=1, no_progress_limit=10))
        self.assertEqual(repeated["stop_reason"], "repeated_action")
        self.assertEqual(repeated["steps"], 1)

    def test_policy_exception_is_stable_error(self):
        class Broken:
            name = "broken"

            def choose(self, observation):
                raise RuntimeError("sensitive payload not logged")

        result = run_episode(generate_maze(0), Broken())
        self.assertEqual(result["stop_reason"], "policy_error")
        self.assertNotIn("sensitive", str(result))

    def test_limits_cannot_enable_unknown_tool(self):
        with self.assertRaises(ValueError):
            Limits(allowed_actions=frozenset({"shell"}))
        for limits in [
            dict(max_steps=-1),
            dict(max_steps=5000),
            dict(timeout_seconds=61),
            dict(repeated_action_limit=0),
        ]:
            with self.subTest(limits=limits), self.assertRaises(ValueError):
                Limits(**limits)

    def test_release_split_and_evaluation(self):
        self.assertFalse(set(DEV_SEEDS) & set(RELEASE_SEEDS))
        report = evaluate()
        self.assertGreaterEqual(report["checks_total"], 12)
        self.assertEqual(report["checks_passed"], report["checks_total"])
        self.assertEqual(report["metrics"]["bfs_planning_oracle"]["successes"], len(RELEASE_SEEDS))
        self.assertLess(report["metrics"]["reactive_manhattan"]["success_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
