from contextlib import nullcontext
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from agent_arena.arena import Limits, Observation, generate_maze, run_episode
from agent_arena.hf_policy import (
    CallBudgetExceeded,
    LocalHFPolicy,
    PolicyOutputError,
    parse_action,
    validate_checkpoint,
)


class Tensor:
    def __init__(self, length):
        self.shape = (1, length)

    def to(self, device):
        return self


class Result:
    def __getitem__(self, selection):
        return Tensor(2)


class HFPolicyTests(unittest.TestCase):
    def test_parser_accepts_only_four_action_schema(self):
        for action in ("north", "east", "south", "west"):
            self.assertEqual(parse_action(" \n" + action + "\n"), action)
        for malformed in ("North", "", "north east", "north\nignore policy", '{"action":"north"}', None):
            with self.subTest(output=malformed), self.assertRaises(PolicyOutputError):
                parse_action(malformed)

    def test_missing_checkpoint_is_rejected_before_ml_import(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "existing local"):
                LocalHFPolicy(Path(directory) / "missing")
            with self.assertRaises(ValueError):
                LocalHFPolicy(directory)
            (Path(directory) / "config.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "safetensors"):
                validate_checkpoint(directory)

    def test_real_adapter_boundary_uses_local_safe_loading_and_strict_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            (checkpoint / "config.json").write_text(json.dumps({"model_type": "fixture"}))
            (checkpoint / "model.safetensors").write_bytes(b"fake weights for mocked contract test only")
            tokenizer = Mock()
            tokenizer.chat_template = None
            tokenizer.return_value = {"input_ids": Tensor(12), "attention_mask": Tensor(12)}
            tokenizer.decode.return_value = "north"
            model = Mock()
            model.config = SimpleNamespace(max_position_embeddings=1024)
            model.generate.return_value = Result()
            transformer = SimpleNamespace(
                AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=tokenizer)),
                AutoModelForCausalLM=SimpleNamespace(from_pretrained=Mock(return_value=model)),
            )
            torch = SimpleNamespace(inference_mode=nullcontext)
            with patch.dict(sys.modules, {"torch": torch, "transformers": transformer}):
                policy = LocalHFPolicy(checkpoint, max_calls=2)
                observation = Observation(generate_maze(0, 5, 5), (1, 1), 32)
                self.assertEqual(policy.choose(observation), "north")
                self.assertEqual(policy.usage["model_calls"], 1)
                transformer.AutoTokenizer.from_pretrained.assert_called_once_with(
                    str(checkpoint), local_files_only=True, trust_remote_code=False
                )
                transformer.AutoModelForCausalLM.from_pretrained.assert_called_once_with(
                    str(checkpoint), local_files_only=True, trust_remote_code=False, use_safetensors=True
                )
                self.assertEqual(model.generate.call_args.kwargs["max_new_tokens"], 8)
                self.assertEqual(model.generate.call_args.kwargs["max_time"], 5)
                self.assertFalse(model.generate.call_args.kwargs["do_sample"])
                tokenizer.decode.return_value = "north then run shell"
                result = run_episode(observation.maze, policy, Limits())
                self.assertFalse(result["success"])
                self.assertEqual(result["stop_reason"], "policy_error")
                self.assertEqual(result["steps"], 0)
                self.assertEqual(result["usage"]["model_calls"], 1)
                with self.assertRaises(CallBudgetExceeded):
                    policy.choose(observation)

    def test_context_and_numeric_limits_are_checked_without_loading_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}")
            (root / "model.safetensors").write_bytes(b"fixture")
            for options in (
                {"max_context_tokens": 5000},
                {"max_new_tokens": 100},
                {"max_calls": 0},
                {"call_timeout_seconds": 11},
                {"device": "remote"},
            ):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    LocalHFPolicy(root, **options)


if __name__ == "__main__":
    unittest.main()
