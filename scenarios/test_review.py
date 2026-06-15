from contextlib import contextmanager, nullcontext, redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from agent_arena import (
    BFSOracle,
    Limits,
    LocalDFSExplorer,
    LocalObservation,
    Maze,
    generate_maze,
    replay_episode,
    run_episode,
)
from agent_arena.__main__ import main
from agent_arena.arena import Observation
from agent_arena.evaluation import (
    DEV_SEEDS,
    RELEASE_SEEDS,
    STRESS_SEEDS,
    dataset_fingerprint,
    dataset_manifest,
    evaluate,
    hf_evaluate,
)
from agent_arena.hf_policy import LocalHFPolicy, ModelLoadError, PolicyContextError, validate_checkpoint
from agent_arena.replay import ReplayError, replay_report


class Tensor:
    def __init__(self, length):
        self.shape = (1, length)

    def to(self, device):
        return self


class Generated:
    def __getitem__(self, key):
        return Tensor(2)


@contextmanager
def mocked_hf(
    *, context_limit=1024, input_tokens=12, max_calls=4, generate_error=None, decode="north", load_error=None
):
    """Explicit model-contract double: never loads weights or imports ML libraries."""
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory)
        (checkpoint / "config.json").write_text('{"model_type":"mocked_contract_only"}')
        (checkpoint / "model.safetensors").write_bytes(b"not real weights; never loaded")
        tokenizer = Mock()
        tokenizer.chat_template = None
        tokenizer.return_value = {"input_ids": Tensor(input_tokens), "attention_mask": Tensor(input_tokens)}
        tokenizer.decode.return_value = decode
        model = Mock()
        model.config = SimpleNamespace(max_position_embeddings=context_limit)
        model.generate.return_value = Generated()
        model.generate.side_effect = generate_error
        transformers = SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=tokenizer, side_effect=load_error)),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=Mock(return_value=model)),
        )
        with patch.dict(
            sys.modules, {"torch": SimpleNamespace(inference_mode=nullcontext), "transformers": transformers}
        ):
            if load_error:
                yield checkpoint, tokenizer, model
            else:
                policy = LocalHFPolicy(checkpoint, max_calls=max_calls)
                yield policy, tokenizer, model


class ReviewTests(unittest.TestCase):
    def test_maze_defensive_copy_prevents_mutating_observations(self):
        walls = {(0, 0)}
        start = [1, 1]
        maze = Maze(5, 5, walls, start, [3, 3], 0)
        walls.add((1, 1))
        start[0] = 4
        self.assertTrue(maze.walkable((1, 1)))
        self.assertEqual(maze.start, (1, 1))
        self.assertIsInstance(maze.walls, frozenset)

    def test_direct_maze_constructor_validates_full_input(self):
        for kwargs in (
            {"width": -3},
            {"width": True},
            {"walls": {(5, 0)}},
            {"walls": {(1, 1)}},
            {"start": (True, 1)},
            {"goal": [1, 2, 3]},
            {"seed": 2**33},
        ):
            values = dict(width=5, height=5, walls=frozenset(), start=(1, 1), goal=(3, 3), seed=0)
            values.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Maze(**values)

    def test_allowlist_is_copied_and_numeric_limits_are_strict(self):
        source = {"north"}
        limits = Limits(allowed_actions=source)
        source.add("east")
        self.assertEqual(limits.allowed_actions, frozenset({"north"}))
        for kwargs in (
            {"timeout_seconds": True},
            {"timeout_seconds": "2"},
            {"timeout_seconds": float("nan")},
            {"allowed_actions": "north"},
            {"allowed_actions": {7}},
            {"max_steps": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Limits(**kwargs)

    def test_goal_accepted_before_deadline_remains_successful(self):
        maze = Maze(5, 5, frozenset(), (1, 1), (2, 1), 0)
        times = iter([0, 0, 0.9, 1.1, 1.2])
        result = run_episode(maze, BFSOracle(), Limits(timeout_seconds=1), clock=lambda: next(times))
        self.assertTrue(result["success"])
        self.assertEqual(result["steps"], 1)
        self.assertTrue(replay_episode(maze, result)["valid"])

    def test_post_selection_deadline_preserves_attempt_without_movement(self):
        times = iter([0, 0, 2, 2])
        result = run_episode(generate_maze(0), BFSOracle(), Limits(timeout_seconds=1), clock=lambda: next(times))
        self.assertEqual(result["stop_reason"], "deadline")
        self.assertEqual(result["steps"], 0)
        self.assertEqual(result["policy_attempts"], 1)
        self.assertEqual(result["trace"][0]["error_code"], "deadline")
        self.assertTrue(replay_report(result)["episodes"][0]["valid"])

    def test_neighbor_only_policy_has_no_maze_reference(self):
        received = []

        class Spy(LocalDFSExplorer):
            def choose(self, observation):
                received.append(observation)
                return super().choose(observation)

        result = run_episode(generate_maze(200, 15, 15), Spy(), Limits(max_steps=512))
        self.assertTrue(result["success"])
        self.assertEqual(result["observation_mode"], "local_neighbors")
        self.assertTrue(
            all(
                isinstance(observation, LocalObservation) and not hasattr(observation, "maze")
                for observation in received
            )
        )
        self.assertTrue(all(len(observation.neighbors) == 4 for observation in received))

    def test_explorer_resets_between_episodes_and_handles_cyclic_maps(self):
        explorer = LocalDFSExplorer()
        maze = generate_maze(105)
        first = run_episode(maze, explorer)
        second = run_episode(maze, explorer)
        self.assertEqual(first["trace"], second["trace"])
        self.assertTrue(second["success"])
        cyclic = Maze(5, 5, frozenset(), (1, 1), (3, 3), 0)
        result = run_episode(cyclic, explorer)
        self.assertTrue(result["success"])
        self.assertLessEqual(result["steps"], 2 * (25 - 1))
        self.assertTrue(replay_episode(cyclic, result)["valid"])

    def test_reset_error_is_reported_without_raw_payload(self):
        class Broken(LocalDFSExplorer):
            def __init__(self):
                pass

            def reset(self):
                raise KeyError("PRIVATE_RESET_VALUE")

        result = run_episode(generate_maze(0), Broken())
        self.assertEqual(result["stop_reason"], "policy_error")
        self.assertEqual(result["trace"][0]["error_code"], "policy_reset_error")
        self.assertEqual(result["policy_attempts"], 0)
        self.assertNotIn("PRIVATE", json.dumps(result))
        self.assertTrue(replay_report(result)["episodes"][0]["valid"])

    def test_replay_rejects_forged_success_transitions_hash_and_permissions(self):
        maze = generate_maze(100)
        result = run_episode(maze, BFSOracle())
        corruptions = [
            lambda row: row.update(final_position=[0, 0]),
            lambda row: row.update(maze_sha256="0" * 64),
            lambda row: row["trace"][0].update(after=[0, 0]),
            lambda row: row["trace"][0].update(state_hash="wrong"),
            lambda row: row["trace"][0].update(action=[]),
            lambda row: row["limits"].update(allowed_actions=[]),
            lambda row: row.update(steps=True),
            lambda row: row.update(valid_moves=-1),
            lambda row: row.update(optimal_steps=0),
        ]
        for corrupt in corruptions:
            changed = deepcopy(result)
            corrupt(changed)
            with self.subTest(corrupt=corrupt), self.assertRaises(ReplayError):
                replay_episode(maze, changed)
        failed = run_episode(maze, BFSOracle(), Limits(max_steps=0))
        failed.update(success=True, stop_reason="goal_reached")
        with self.assertRaises(ReplayError):
            replay_episode(maze, failed)

    def test_all_contract_boundary_traces_are_replayable(self):
        report = evaluate()
        replayed = replay_report(report)
        self.assertEqual(replayed["checks_passed"], report["checks_total"])
        self.assertEqual(report["checks_passed"], report["checks_total"])
        self.assertEqual(report["metrics"]["local_dfs_explorer"]["successes"], len(RELEASE_SEEDS))
        self.assertEqual(report["stress_metrics"]["local_dfs_explorer"]["successes"], len(STRESS_SEEDS))

    def test_observation_contracts_reject_malformed_or_mutable_inputs(self):
        maze = generate_maze(0)
        actions = {"north"}
        observation = Observation(maze, [1, 1], 10, actions)
        actions.add("east")
        self.assertEqual(observation.allowed_actions, frozenset({"north"}))
        self.assertEqual(maze.render([1, 1]), maze.render((1, 1)))
        for kwargs in ({"position": (0, 0)}, {"remaining_steps": True}, {"remaining_steps": 4097}):
            values = dict(maze=maze, position=(1, 1), remaining_steps=10)
            values.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Observation(**values)
        with self.assertRaises(ValueError):
            LocalObservation((1, 1), (3, 3), 10, (("north", True),) * 4, frozenset({"north"}))

    def test_dataset_hash_pins_actual_maps_and_matches_unrun_hf_report(self):
        dataset = dataset_manifest()
        self.assertFalse(set(DEV_SEEDS) & set(RELEASE_SEEDS) or set(RELEASE_SEEDS) & set(STRESS_SEEDS))
        expected = dataset_fingerprint(dataset)
        with tempfile.TemporaryDirectory() as directory:
            with patch("builtins.__import__", side_effect=AssertionError("ML imports forbidden")):
                # Validate directly inside the import guard; hf_evaluate imports
                # its already-lazy adapter module before calling validation.
                with self.assertRaises(ValueError):
                    validate_checkpoint(Path(directory) / "missing")
            report = hf_evaluate(str(Path(directory) / "missing"))
        self.assertEqual(report["dataset_sha256"], expected)
        self.assertEqual(report["status"], "not_run")
        self.assertEqual(report["usage"]["model_calls"], 0)
        modified = deepcopy(dataset)
        modified["release"]["worlds"][0]["maze_sha256"] = "changed"
        self.assertNotEqual(dataset_fingerprint(modified), expected)

    def test_hf_generation_failure_preserves_call_trace_and_known_usage(self):
        with mocked_hf(generate_error=KeyError("PRIVATE_NATIVE_ERROR")) as (policy, _, _):
            result = run_episode(generate_maze(0), policy)
        self.assertEqual(result["stop_reason"], "policy_error")
        self.assertEqual(result["trace"][0]["policy_error_code"], "inference_error")
        call = result["trace"][0]["model_call"]
        self.assertEqual(call["finish_reason"], "error")
        self.assertIsNone(call["output_tokens"])
        self.assertEqual(result["usage"]["input_tokens"], 12)
        self.assertEqual(result["usage"]["unobserved_output_calls"], 1)
        self.assertNotIn("PRIVATE", json.dumps(result))

    def test_hf_context_none_is_supported_and_generation_shape_is_pinned(self):
        with mocked_hf(context_limit=None) as (policy, tokenizer, model):
            self.assertEqual(policy.choose(Observation(generate_maze(0), (1, 1), 32)), "north")
            self.assertFalse(model.generate.call_args.kwargs["return_dict_in_generate"])
            self.assertIn('"allowed_actions": ["east", "north", "south", "west"]', tokenizer.call_args.args[0])

    def test_hf_context_overflow_does_not_attempt_generation(self):
        for context, count in ((16, 12), (None, 1025)):
            with (
                self.subTest(context=context),
                mocked_hf(context_limit=context, input_tokens=count) as (policy, _, model),
            ):
                with self.assertRaises(PolicyContextError):
                    policy.choose(Observation(generate_maze(0), (1, 1), 32))
                model.generate.assert_not_called()
                self.assertEqual(policy.usage["model_calls"], 0)

    def test_hf_invalid_output_and_call_budget_do_not_reuse_stale_trace(self):
        with mocked_hf(max_calls=1, decode="north") as (policy, _, _):
            result = run_episode(generate_maze(0), policy)
        self.assertEqual(result["trace"][-1]["policy_error_code"], "model_call_budget")
        self.assertNotIn("model_call", result["trace"][-1])
        self.assertEqual(result["usage"]["model_calls"], 1)
        with mocked_hf(decode="north and leak PRIVATE") as (policy, _, _):
            result = run_episode(generate_maze(0), policy)
        self.assertEqual(result["trace"][0]["model_call"]["finish_reason"], "invalid_output")
        self.assertEqual(result["trace"][0]["policy_error_code"], "invalid_model_output")
        self.assertNotIn("PRIVATE", json.dumps(result))

    def test_hf_timeout_retains_observed_tokens_and_no_action(self):
        with mocked_hf() as (policy, _, _):
            with patch("agent_arena.hf_policy.time.monotonic", side_effect=[0, 6, 6]):
                with self.assertRaises(TimeoutError):
                    policy.choose(Observation(generate_maze(0), (1, 1), 32))
            self.assertEqual(policy.last_call["finish_reason"], "timeout")
            self.assertEqual(policy.last_call["output_tokens"], 2)
            self.assertEqual(policy.usage["failed_calls"], 1)

    def test_native_loader_exception_is_normalized_and_redacted(self):
        with mocked_hf(load_error=AssertionError("PRIVATE_LOADER_DETAIL")) as (checkpoint, _, _):
            with self.assertRaises(ModelLoadError) as caught:
                LocalHFPolicy(checkpoint)
            self.assertNotIn("PRIVATE", str(caught.exception))

    def test_cli_exports_replay_errors_and_unrun_hf_without_tracebacks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            demo, replayed, hf = root / "demo.json", root / "replay.json", root / "hf.json"
            self.assertEqual(main(["demo", "--output", str(demo)]), 0)
            self.assertEqual(main(["replay", "--input", str(demo), "--output", str(replayed)]), 0)
            self.assertEqual(json.loads(replayed.read_text())["checks_total"], 3)
            self.assertEqual(main(["hf-evaluate", "--checkpoint", str(root / "missing"), "--output", str(hf)]), 2)
            self.assertEqual(json.loads(hf.read_text())["status"], "not_run")
            stderr = io.StringIO()
            with redirect_stderr(stderr), redirect_stdout(io.StringIO()):
                self.assertEqual(main(["demo", "--output", str(root)]), 2)
            error = json.loads(stderr.getvalue())
            self.assertEqual(error["error"]["code"], "io_error")
            demo.write_text("broken JSON")
            with redirect_stderr(io.StringIO()):
                self.assertEqual(main(["replay", "--input", str(demo)]), 2)


if __name__ == "__main__":
    unittest.main()
