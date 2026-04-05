from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from .arena import BFSOracle, Limits, LocalDFSExplorer, ReactivePolicy, generate_maze, run_episode
from .evaluation import evaluate, hf_evaluate
from .replay import replay_report


def _emit(result: dict, output: Path | None):
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized)
    else:
        print(serialized, end="")


def _read_report(path: Path) -> dict:
    with path.open("rb") as stream:
        data = stream.read(20_000_001)
    if len(data) > 20_000_000:
        raise ValueError("replay input exceeds 20 MB")
    try:
        return json.loads(data)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError("replay input must contain valid bounded JSON") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bounded navigation, local exploration, and independently replayable traces"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("demo", "evaluate"):
        child = sub.add_parser(command)
        child.add_argument("--output", type=Path)
        if command == "demo":
            child.add_argument("--seed", type=int, default=100)
            child.add_argument("--width", type=int, default=11)
            child.add_argument("--height", type=int, default=11)
            child.add_argument("--max-steps", type=int, default=128)
    replay = sub.add_parser("replay", help="Independently check transitions in exported episode/demo/evaluation JSON")
    replay.add_argument("--input", type=Path, required=True)
    replay.add_argument("--output", type=Path)
    incremental = sub.add_parser(
        "replan-evaluate", help="Compare real D* Lite repair, repeated BFS, and restarted local DFS on seeded changes"
    )
    incremental.add_argument("--seed", type=int, help="Optional single scenario; default runs all 12 seeds")
    incremental.add_argument("--output", type=Path)
    hf = sub.add_parser("hf-evaluate", help="Opt-in real inference from an existing local checkpoint")
    hf.add_argument("--checkpoint", type=Path, required=True)
    hf.add_argument("--output", type=Path)
    hf.add_argument("--max-steps", type=int, default=32)
    hf.add_argument("--episode-timeout", type=float, default=60)
    hf.add_argument("--max-context-tokens", type=int, default=1024)
    hf.add_argument("--max-new-tokens", type=int, default=8)
    hf.add_argument("--max-calls", type=int, default=512)
    hf.add_argument("--call-timeout", type=float, default=5)
    hf.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    try:
        if args.command == "replan-evaluate":
            from .replanning import DYNAMIC_SEEDS, dynamic_evaluate

            result = dynamic_evaluate((args.seed,) if args.seed is not None else DYNAMIC_SEEDS)
        elif args.command == "hf-evaluate":
            result = hf_evaluate(
                str(args.checkpoint),
                max_steps=args.max_steps,
                episode_timeout=args.episode_timeout,
                max_context_tokens=args.max_context_tokens,
                max_new_tokens=args.max_new_tokens,
                max_calls=args.max_calls,
                call_timeout=args.call_timeout,
                device=args.device,
            )
        elif args.command == "replay":
            result = replay_report(_read_report(args.input))
        elif args.command == "evaluate":
            result = evaluate()
        else:
            maze = generate_maze(args.seed, args.width, args.height)
            result = {
                "map": maze.render(maze.start),
                "observation_contract": "Reactive/BFS use a full map; local DFS receives immediate-neighbor sensing only.",
                "episodes": [
                    run_episode(maze, policy, Limits(max_steps=args.max_steps))
                    for policy in (ReactivePolicy(), BFSOracle(), LocalDFSExplorer())
                ],
            }
        _emit(result, args.output)
        if result.get("status") == "not_run":
            return 2
        if (
            args.command in {"evaluate", "replay", "replan-evaluate"}
            and result["checks_passed"] != result["checks_total"]
        ):
            return 1
        return 0
    except (ValueError, RuntimeError, OSError) as exc:
        message = "Unable to read or write the requested local file" if isinstance(exc, OSError) else str(exc)[:200]
        error = {
            "status": "failed",
            "error": {
                "code": getattr(exc, "code", "io_error" if isinstance(exc, OSError) else "invalid_input"),
                "type": type(exc).__name__,
                "message": message,
            },
        }
        print(json.dumps(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
