# Heavy review: reproduced defects and verified changes

This document records the original 36-test hardening review. The subsequent real incremental-search implementation, current 51-test evidence, and measured replanning comparisons are documented in [repair-work-measurements.md](repair-work-measurements.md).

Scope: all code, tests, CLI paths, configuration, and documentation in this standalone project. No paid calls, real checkpoint weights, model downloads, or training jobs were used. Model-path tests use explicitly mocked tokenizer/model objects and do not establish model quality.

## Findings and fixes

| Finding reproduced before changes | Consequence | Implemented correction | Behavioral evidence |
| --- | --- | --- | --- |
| Direct `Maze(-3, ...)` construction succeeded; passing mutable wall/start collections preserved aliases. | Invalid dimensions bypassed generator checks; a caller could change the map after an observation was created. | Validate direct constructor, coordinates, wall bounds, seed, and walkable endpoints; freeze defensive copies. | `test_direct_maze_constructor_validates_full_input`, `test_maze_defensive_copy_prevents_mutating_observations`. |
| `Limits(allowed_actions=some_set)` retained that mutable set; boolean deadlines were accepted. | Executor permissions could change after validation and malformed numerics had inconsistent behavior. | Copy allowlists to frozenset; reject malformed types, Boolean numerics, NaN, and invalid actions. | `test_allowlist_is_copied_and_numeric_limits_are_strict`. |
| A valid goal move just before the deadline could become `deadline` at the next bookkeeping check. | Exact success could be erased after an already accepted transition. | Recognize the goal immediately after the authorized move. | Controlled clock test `test_goal_accepted_before_deadline_remains_successful`. |
| A model exception produced only `policy_error` and exception type; its completed call metadata was lost. Post-selection cancellation/deadline also dropped the attempt. | Usage/latency/failure evidence was incomplete, making failures difficult to audit. | Preserve a terminal non-mutating attempt and fresh allowlisted call metadata on every path. | `test_hf_generation_failure_preserves_call_trace_and_known_usage`, `test_post_selection_deadline_preserves_attempt_without_movement`, existing cancellation test. |
| `last_call` persisted when a later selection exhausted its call budget. | A later error could be attributed to the previous model invocation. | Clear call metadata at selection start and attach only fresh records. | `test_hf_invalid_output_and_call_budget_do_not_reuse_stale_trace`. |
| A real model config containing `max_position_embeddings=None` caused comparison with `None`; generation shape could follow checkpoint defaults. | Compatible checkpoints could fail before inference; structured generation settings could break tensor slicing. | Guard nullable native limits while enforcing own cap; pin tensor generation output. | `test_hf_context_none_is_supported_and_generation_shape_is_pinned`, `test_hf_context_overflow_does_not_attempt_generation`. |
| Native loader/generation exception types outside the narrow expected classes were not normalized; failed inference lost observed input usage. | CLI tracebacks and incomplete failure accounting were possible. | Stable load/context/inference/timeout errors, redacted messages, attempt/input counting before inference, explicit unobserved-output accounting. | Native loader `AssertionError`, generation `KeyError`, malformed output, timeout, and call-budget tests. |
| Default and optional-HF evaluators hashed different manifests for the same release worlds. | Fingerprints could not establish that comparisons used the same environments; geometry changes were not pinned. | One content-addressed manifest includes actual world geometry hashes across both evaluators. | `test_dataset_hash_pins_actual_maps_and_matches_unrun_hf_report`. |
| CLI output writes happened outside the error handler. | An unwritable target produced a raw traceback. | Cover reading, serialization, and export with structured failure responses and exit 2. | `test_cli_exports_replay_errors_and_unrun_hf_without_tracebacks`. |
| Reports trusted their own goal/counter/path claims without an independent checker. | A corrupted transition or forged success could pass downstream report inspection. | Independent bounded replay of geometry, permissions, hashes, counters, limits, shortest path, and final success. | `test_replay_rejects_forged_success_transitions_hash_and_permissions` and replay of every evaluation episode. |

The old unreachable-goal test placed a wall on the goal, which is now correctly rejected as malformed. Its replacement preserves the original behavior assertion using a valid disconnected maze with walkable start and goal; the test was not weakened to accept success.

## Capability and evaluation extensions

- Added `LocalDFSExplorer` with neighbor-only sensing and bounded episode memory. `LocalObservation` has no `Maze` reference; the spy test checks every received observation. Episode reset prevents state leakage across seeds and repeated runs. A separate cyclic-world test exercises visited-state/backtracking behavior beyond the perfect-maze generator.
- Added six larger exploration-stress worlds on separate seeds/dimensions. The report labels information regimes instead of presenting reduced-information DFS as a same-observation model comparison.
- Added success-weighted path efficiency over all episodes, preserving failures and denominators alongside success-only metrics.
- Added the `replay` CLI for single episodes, demos, and evaluation containers. Input is bounded to 20 MB/200 episodes, and all exported evaluation cases are replayed independently.
- Added explicit optional-model `not_run` artifacts and exit status. No model score is fabricated when a checkpoint is absent.

## Verification actually executed

Commands run from this project directory:

```bash
PYTHONPATH=src python -m unittest discover -s scenarios -v
/workspace/applied-ai-labs/.venv/bin/ruff check src tests
/workspace/applied-ai-labs/.venv/bin/ruff format --check src tests
PYTHONPATH=src python -m agent_arena demo --output artifacts/demo.json
PYTHONPATH=src python -m agent_arena evaluate --output artifacts/evaluation.json
PYTHONPATH=src python -m agent_arena replay --input artifacts/evaluation.json --output artifacts/replay.json
PYTHONPATH=src python -m agent_arena hf-evaluate --checkpoint /tmp/agent-arena-no-checkpoint --output artifacts/hf-not-run.json
```

Results: **36/36 unit tests**, **56/56 invariant checks**, **56/56 independently replayed traces**, clean lint/format, and successful demo/evaluate/replay exports. The missing-checkpoint command returned the intended exit 2 with `status=not_run`, `executed=false`, and zero model/provider calls; it did not import/load model weights.

Measured release success: reactive **3/12**, full-map BFS **12/12**, neighbor-only DFS **12/12**. Larger stress success: BFS **6/6**, DFS **6/6**. Release success-weighted efficiencies: **0.2500**, **1.0000**, and **0.6400** respectively; stress DFS efficiency **0.7335**. The checked-in summary retains precise values and denominators. Manifest SHA-256: `3d43b2c737bf291e2fbdd98ebcd1c5d867699fa2ed7d6de9ff835a9536333633`.

## Remaining limits and unverified conditions

No real checkpoint loading, model inference quality, GPU execution, training, deployment, or hard process isolation was tested. Cooperative deadlines cannot terminate arbitrary Python/native work, and weight loading/fingerprinting can exceed episode time or memory budgets. Python immutability is not a sandbox against malicious Python. Replay checks internal consistency with recorded limits; unsigned reports can be fully fabricated and cannot prove model execution, elapsed time, or cancellation. The procedural benchmark is small and synthetic, and its published seeds must not serve as a repeatedly tuned hidden holdout. Local-DFS and full-map policies have explicitly different information budgets.
