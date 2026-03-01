"""Content-addressed held-out navigation and independent transition verification."""

from __future__ import annotations

import hashlib
import json
from .arena import BFSOracle, Limits, LocalDFSExplorer, ReactivePolicy, generate_maze, run_episode
from .replay import replay_episode

DEV_SEEDS = tuple(range(8))
RELEASE_SEEDS = tuple(range(100, 112))
STRESS_SEEDS = tuple(range(200, 206))
RELEASE_DIMENSIONS = ((11, 11), (13, 9), (9, 13))
STRESS_DIMENSIONS = ((15, 15), (17, 13), (13, 17))


class FixedPolicy:
    name = "fixed_adversarial_policy"

    def __init__(self, action):
        self.action = action

    def choose(self, observation):
        return self.action


def worlds(seeds, dimensions):
    return [generate_maze(seed, *dimensions[index % len(dimensions)]) for index, seed in enumerate(seeds)]


def dataset_manifest() -> dict:
    def describe(maze):
        return {"seed": maze.seed, "dimensions": [maze.width, maze.height], "maze_sha256": maze.fingerprint()}

    return {
        "version": "procedural-release-v2",
        "generator": "dfs-perfect-maze-v1",
        "provenance": "synthetic procedural perfect mazes",
        "development": {"seeds": list(DEV_SEEDS), "dimensions": [[9, 9]]},
        "release": {
            "seeds": list(RELEASE_SEEDS),
            "dimensions": [list(dim) for dim in RELEASE_DIMENSIONS],
            "worlds": [describe(maze) for maze in worlds(RELEASE_SEEDS, RELEASE_DIMENSIONS)],
        },
        "exploration_stress": {
            "seeds": list(STRESS_SEEDS),
            "dimensions": [list(dim) for dim in STRESS_DIMENSIONS],
            "worlds": [describe(maze) for maze in worlds(STRESS_SEEDS, STRESS_DIMENSIONS)],
        },
    }


def dataset_fingerprint(dataset: dict) -> str:
    return hashlib.sha256(json.dumps(dataset, sort_keys=True).encode()).hexdigest()


def summarize(episodes: list[dict]) -> dict:
    metrics = {}
    for name in sorted({episode["policy"] for episode in episodes}):
        results = [episode for episode in episodes if episode["policy"] == name]
        successful = [result for result in results if result["success"]]
        efficiencies = [
            result["optimal_steps"] / max(result["steps"], result["optimal_steps"]) if result["optimal_steps"] else 1.0
            for result in successful
        ]
        metrics[name] = {
            "episodes": len(results),
            "successes": len(successful),
            "success_rate": len(successful) / len(results),
            "observation_modes": sorted({result["observation_mode"] for result in results}),
            "mean_success_steps": sum(result["steps"] for result in successful) / len(successful)
            if successful
            else None,
            "successful_episode_count": len(successful),
            "mean_path_efficiency": sum(efficiencies) / len(efficiencies) if efficiencies else None,
            "success_weighted_path_efficiency": sum(efficiencies) / len(results),
            "stop_reasons": {
                reason: sum(result["stop_reason"] == reason for result in results)
                for reason in sorted({result["stop_reason"] for result in results})
            },
        }
    return metrics


def evaluate() -> dict:
    dataset = dataset_manifest()
    cases = []
    for split, seeds, dimensions, policies, limits in (
        ("release", RELEASE_SEEDS, RELEASE_DIMENSIONS, (ReactivePolicy, BFSOracle, LocalDFSExplorer), Limits()),
        ("exploration_stress", STRESS_SEEDS, STRESS_DIMENSIONS, (BFSOracle, LocalDFSExplorer), Limits(max_steps=512)),
    ):
        for maze in worlds(seeds, dimensions):
            for factory in policies:
                policy = factory()
                result = run_episode(maze, policy, limits)
                replay = replay_episode(maze, result)
                passed = (
                    result["success"] and result["steps"] == result["optimal_steps"]
                    if isinstance(policy, BFSOracle)
                    else result["success"]
                    if isinstance(policy, LocalDFSExplorer)
                    else result["stop_reason"] in {"goal_reached", "repeated_action", "step_budget", "no_progress"}
                )
                cases.append(
                    {
                        "id": f"{split}-{maze.seed}-{policy.name}",
                        "tags": ["held_out", "navigation", split],
                        "passed": passed and replay["valid"],
                        "replay": replay,
                        "result": result,
                    }
                )
    maze = generate_maze(5, 5, 5)
    checks = [
        ("invalid-action", FixedPolicy("read_secret"), Limits(), "invalid_action", lambda: False),
        ("malformed-action", FixedPolicy({"action": "north"}), Limits(), "invalid_action", lambda: False),
        (
            "permission-denial",
            FixedPolicy("north"),
            Limits(allowed_actions=frozenset()),
            "permission_denied",
            lambda: False,
        ),
        ("zero-budget", BFSOracle(), Limits(max_steps=0), "step_budget", lambda: False),
        ("deadline", BFSOracle(), Limits(timeout_seconds=0), "deadline", lambda: False),
        ("cancel", BFSOracle(), Limits(), "cancelled", lambda: True),
        ("blocked-move", FixedPolicy("north"), Limits(), "no_progress", lambda: False),
        (
            "repetition",
            FixedPolicy("north"),
            Limits(repeated_action_limit=1, no_progress_limit=10),
            "repeated_action",
            lambda: False,
        ),
    ]
    for name, policy, limits, expected, cancel in checks:
        result = run_episode(maze, policy, limits, cancel)
        replay = replay_episode(maze, result)
        cases.append(
            {
                "id": name,
                "tags": ["boundary"],
                "passed": result["stop_reason"] == expected and replay["valid"],
                "expected_stop": expected,
                "replay": replay,
                "result": result,
            }
        )
    release = [case["result"] for case in cases if "release" in case["tags"]]
    stress = [case["result"] for case in cases if "exploration_stress" in case["tags"]]
    return {
        "dataset": dataset,
        "dataset_sha256": dataset_fingerprint(dataset),
        "run_type": "offline_algorithms",
        "implementation": "deterministic policies; no LLM or training",
        "observation_comparison": "full-map reactive/oracle versus separately labeled neighbor-only DFS; information budgets differ",
        "cases": cases,
        "checks_passed": sum(case["passed"] for case in cases),
        "checks_total": len(cases),
        "replay_checks_passed": sum(case["replay"]["valid"] for case in cases),
        "metrics": summarize(release),
        "stress_metrics": summarize(stress),
    }


def hf_evaluate(
    checkpoint: str,
    *,
    max_steps: int = 32,
    episode_timeout: float = 60,
    max_context_tokens: int = 1024,
    max_new_tokens: int = 8,
    max_calls: int = 512,
    call_timeout: float = 5,
    device: str = "cpu",
) -> dict:
    """Real opt-in local inference with same worlds, observation, and episode limits."""
    from .hf_policy import LocalHFPolicy

    dataset = dataset_manifest()
    try:
        limits = Limits(max_steps=max_steps, timeout_seconds=episode_timeout)
        policy = LocalHFPolicy(
            checkpoint,
            max_context_tokens=max_context_tokens,
            max_new_tokens=max_new_tokens,
            max_calls=max_calls,
            call_timeout_seconds=call_timeout,
            device=device,
        )
    except (ValueError, RuntimeError, OSError) as exc:
        return {
            "dataset_version": dataset["version"],
            "dataset_sha256": dataset_fingerprint(dataset),
            "evaluated_split": "release",
            "run_type": "real_local_hf_inference",
            "status": "not_run",
            "executed": False,
            "model_identity": None,
            "usage": {"model_calls": 0, "provider_calls": 0, "cost_usd": None},
            "cases": [],
            "error": {
                "code": getattr(exc, "code", "invalid_configuration"),
                "type": type(exc).__name__,
                "message": str(exc)[:200],
            },
        }
    cases = []
    for maze in worlds(RELEASE_SEEDS, RELEASE_DIMENSIONS):
        episodes = [run_episode(maze, candidate, limits) for candidate in (ReactivePolicy(), BFSOracle(), policy)]
        cases.append(
            {
                "id": f"release-{maze.seed}",
                "seed": maze.seed,
                "episodes": episodes,
                "replays": [replay_episode(maze, episode) for episode in episodes],
            }
        )
    return {
        "dataset": dataset,
        "dataset_version": dataset["version"],
        "dataset_sha256": dataset_fingerprint(dataset),
        "evaluated_split": "release",
        "run_type": "real_local_hf_inference",
        "status": "completed",
        "executed": bool(policy.usage["model_calls"]),
        "model_identity": policy.model_identity,
        "limits": {
            "max_steps": max_steps,
            "episode_timeout": episode_timeout,
            "max_calls": max_calls,
            "max_context_tokens": max_context_tokens,
            "max_new_tokens": max_new_tokens,
            "call_timeout": call_timeout,
        },
        "call_budget_scope": "one shared model-call budget across all 12 episodes; no cross-episode prompt history",
        "cases": cases,
        "metrics": summarize([episode for case in cases for episode in case["episodes"]]),
        "usage": policy.usage,
    }
