# Bring a small counterexample

A planner fix is easiest to inspect as a map, start, goal and sequence of cell changes. Put that sequence in `scenarios/`, compare each repair with fresh BFS, and retain the no-path transitions. A correct final route is insufficient if an earlier repair violated consistency or skipped a change.

Changes to navigation policies must state their observation regime. A neighbor-only explorer must not acquire a full-map reference, and a denied action must not move the agent. Keep the replay checker independent of policy decisions and success flags.

```bash
python -m pip install -e . -r requirements-dev.txt
PYTHONPATH=src python -m unittest discover -s scenarios -v
python -m agent_arena replan-evaluate --output artifacts/replanning.json
```

`python dev/check_arena.py --require-lint --wheel` checks the installed CLI as well. The optional HF boundary tests use doubles; do not describe those as checkpoint inference. When changing work counters, define the counted operation and update both traces and comparisons. Expansions, inspections and elapsed time are different measurements.

Preserve the published ruleset hashes. Add development cases rather than retuning a policy against the recorded release seeds and calling those cases unseen.
