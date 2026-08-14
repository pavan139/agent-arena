# Scenario and replay checks

The recorded full run contains 51 passing tests on Python 3.12.14. Navigation evidence includes 56 successful invariant checks and 56 independent transition replays. The repair experiment checks 156 snapshots against BFS and retains every no-path result. Raw software verification lives in [reports/verification.json](../reports/verification.json).

```bash
PYTHONPATH=src python -m unittest discover -s scenarios -v
python -m agent_arena evaluate --output artifacts/navigation.json
python -m agent_arena replay --input artifacts/navigation.json --output artifacts/replayed.json
python dev/check_arena.py --require-lint --wheel
```

`scenarios/` includes policy errors, mutable input boundaries, deadline/cancellation accounting, local-model doubles and generated D* Lite/BFS comparisons. Actual HF checkpoint loading and inference have not been executed. Keep that distinction when reading the test count.

The operation-count comparison and its limitations are in [repair work measurements](repair-work-measurements.md). Re-run the commands after changing the source; the verification artifact records the exact version it checked.
