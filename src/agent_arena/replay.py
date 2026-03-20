"""Independent transition verification; does not trust a report's success flag."""

from __future__ import annotations

from collections import Counter
import hashlib

from .arena import DIRECTIONS, Limits, Maze, destination, generate_maze, shortest_path


class ReplayError(ValueError):
    code = "invalid_episode_trace"


def replay_episode(maze: Maze, result: dict) -> dict:
    def require(condition, message):
        if not condition:
            raise ReplayError(message)

    require(isinstance(result, dict), "episode must be an object")
    require(result.get("maze_sha256") == maze.fingerprint(), "maze content fingerprint mismatch")
    require(
        result.get("dimensions") == [maze.width, maze.height]
        and type(result.get("seed")) is int
        and result.get("seed") == maze.seed,
        "maze metadata mismatch",
    )
    require(result.get("start") == list(maze.start) and result.get("goal") == list(maze.goal), "start or goal mismatch")
    limits_data = result.get("limits")
    require(isinstance(limits_data, dict), "missing limits")
    try:
        limits = Limits(**limits_data)
    except (ValueError, TypeError) as exc:
        raise ReplayError("invalid episode limits") from exc
    trace = result.get("trace")
    require(isinstance(trace, list) and len(trace) <= limits.max_steps + 1, "trace exceeds episode budget")
    position = maze.start
    require(
        all(type(result.get(key)) is int for key in ("steps", "valid_moves", "policy_attempts")),
        "episode counters must be integers",
    )
    executed = valid_moves = no_progress = 0
    seen = Counter()
    for index, event in enumerate(trace):
        require(
            isinstance(event, dict) and type(event.get("step")) is int and event["step"] == index,
            "trace steps must be consecutive",
        )
        require(event.get("before") == list(position), "transition starts at the wrong coordinate")
        require(
            type(event.get("valid")) is bool and type(event.get("allowed")) is bool, "transition flags must be booleans"
        )
        action = event.get("action")
        if "after" not in event:
            require(index == len(trace) - 1 and not event["valid"], "rejected action must be final and cannot move")
            require(isinstance(event.get("error_code"), str), "rejected action requires an error code")
            if event["error_code"] == "repeated_action":
                require(
                    type(action) is str
                    and action in DIRECTIONS
                    and event["allowed"]
                    and action in limits.allowed_actions,
                    "invalid repeated-action rejection",
                )
                require(seen[(position, action)] >= limits.repeated_action_limit, "repetition limit was not reached")
            elif event["error_code"] == "permission_denied":
                require(
                    type(action) is str
                    and action in DIRECTIONS
                    and action not in limits.allowed_actions
                    and not event["allowed"],
                    "permission rejection does not match allowlist",
                )
            elif event["error_code"] == "invalid_action":
                require(
                    action == "<invalid>" and not event["allowed"], "invalid-action rejection contains a legal action"
                )
            else:
                require(
                    event["error_code"] in {"policy_error", "policy_reset_error", "deadline", "cancelled"}
                    and not event["allowed"],
                    "unknown rejected action",
                )
            if event["error_code"] == "policy_reset_error":
                require(index == 0 and len(trace) == 1, "reset failure must precede all policy attempts")
            continue
        require(
            type(action) is str and action in DIRECTIONS and action in limits.allowed_actions and event["allowed"],
            "unauthorized movement",
        )
        seen[(position, action)] += 1
        require(seen[(position, action)] <= limits.repeated_action_limit, "movement exceeds repeated-action limit")
        target = destination(position, action)
        moved = maze.walkable(target)
        expected = target if moved else position
        require(event["after"] == list(expected) and event["valid"] == moved, "movement does not match maze geometry")
        require(
            event.get("state_hash") == hashlib.sha256(repr(expected).encode()).hexdigest()[:16], "state hash mismatch"
        )
        if moved:
            valid_moves += 1
            no_progress = 0
            require("error_code" not in event, "successful move has an error code")
        else:
            no_progress += 1
            require(event.get("error_code") == "blocked_move", "blocked move lacks its error code")
        executed += 1
        position = expected
        if no_progress >= limits.no_progress_limit or position == maze.goal:
            require(index == len(trace) - 1, "trace continues after a terminal state")
    require(executed <= limits.max_steps and result.get("steps") == executed, "executed step count mismatch")
    require(result.get("valid_moves") == valid_moves, "valid movement count mismatch")
    require(result.get("final_position") == list(position), "final position mismatch")
    optimal = shortest_path(maze)
    expected_optimal = len(optimal) if optimal is not None else None
    require(
        result.get("optimal_steps") == expected_optimal
        and (expected_optimal is None or type(result.get("optimal_steps")) is int),
        "optimal path length mismatch",
    )
    attempts = sum(event.get("error_code") != "policy_reset_error" for event in trace)
    require(result.get("policy_attempts") == attempts and attempts <= limits.max_steps, "policy attempt count mismatch")
    reason = result.get("stop_reason")
    require(
        reason
        in {
            "goal_reached",
            "step_budget",
            "deadline",
            "cancelled",
            "policy_error",
            "invalid_action",
            "permission_denied",
            "repeated_action",
            "no_progress",
        },
        "unknown stop reason",
    )
    require(
        type(result.get("success")) is bool
        and result["success"] == (reason == "goal_reached" and position == maze.goal),
        "success claim does not match replay",
    )
    if reason == "goal_reached":
        require(position == maze.goal, "goal was not reached")
    elif reason == "step_budget":
        require(executed == limits.max_steps, "step budget was not exhausted")
    elif reason == "no_progress":
        require(no_progress >= limits.no_progress_limit, "no-progress budget was not exhausted")
    elif reason in {"invalid_action", "permission_denied", "repeated_action"}:
        require(bool(trace) and trace[-1].get("error_code") == reason, "terminal rejection does not match stop reason")
    elif reason == "policy_error":
        require(
            bool(trace) and trace[-1].get("error_code") in {"policy_error", "policy_reset_error"},
            "policy error lacks a failure trace",
        )
    return {
        "valid": True,
        "checked_steps": executed,
        "checked_policy_attempts": attempts,
        "maze_sha256": maze.fingerprint(),
        "verified": [
            "maze_identity",
            "permissions",
            "geometry",
            "state_hashes",
            "step_and_repetition_limits",
            "final_state",
            "goal_claim",
            "optimal_path_length",
        ],
        "not_verified": ["policy_reasoning", "model_execution", "wall_clock", "cancellation_signal"],
    }


def replay_report(report: dict) -> dict:
    """Replay a single episode or bounded demo/evaluation JSON container."""
    if not isinstance(report, dict):
        raise ReplayError("report must be an object")
    if "trace" in report:
        episodes = [report]
    elif isinstance(report.get("episodes"), list):
        episodes = report["episodes"]
    elif isinstance(report.get("cases"), list) and len(report["cases"]) <= 200:
        episodes = []
        for case in report["cases"]:
            if not isinstance(case, dict):
                raise ReplayError("case must be an object")
            if "result" in case:
                episodes.append(case["result"])
            elif isinstance(case.get("episodes"), list):
                episodes.extend(case["episodes"])
            else:
                raise ReplayError("case contains no replayable episode")
    else:
        raise ReplayError("no episode traces found")
    if not 1 <= len(episodes) <= 200:
        raise ReplayError("replay supports between 1 and 200 episodes")
    results = []
    for episode in episodes:
        if not isinstance(episode, dict):
            raise ReplayError("episode must be an object")
        try:
            dimensions = episode["dimensions"]
            if not isinstance(dimensions, list) or len(dimensions) != 2:
                raise ReplayError("episode dimensions must contain two integers")
            maze = generate_maze(episode["seed"], *dimensions)
            results.append(replay_episode(maze, episode))
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, ReplayError):
                raise
            raise ReplayError("invalid generated-maze episode metadata") from exc
    return {"checks_passed": len(results), "checks_total": len(results), "episodes": results}
