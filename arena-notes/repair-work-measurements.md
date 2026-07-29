# D* Lite repair workload

## Planner state exercised

The D* Lite repair loop retains search labels across observed changes. `incremental.py` owns persistent `g`/`rhs` tables, a generation-tagged binary OPEN heap, a consistent Manhattan key, the moving-start `km` correction, atomic incident-edge updates, all three queue-processing branches, and descending-label route extraction. BFS is used only as an independent evaluator and simulator event selector; the D* Lite planner never calls it.

The default navigation/optional-HF commands remain available. The new coherent API accepts validated immutable grids, observed blocked/unblocked cells, explicit start moves, and bounded repairs. It emits paths or honest no-path results with operation counters and priority/label traces. See [search-and-memory.md](search-and-memory.md) for invariants, algorithm steps, queue strategy, complexity, and tradeoffs.

## Reproducible experiments actually executed

From this standalone repository:

```bash
python -m agent_arena replan-evaluate --output artifacts/replanning.json
python -m agent_arena replan-evaluate --seed 300 --output artifacts/replanning-seed-300.json
python -m unittest discover -s scenarios -v
python dev/check_arena.py --require-lint --wheel
```

No model weights, provider calls, paid services, training, or download was needed. The seeded experiment is a classical-search comparison, not an LLM or learned-policy evaluation.

Dataset `incremental-navigation-v1` uses seeds 300–311 on 15×15 grids. Each initial procedural maze opens 12 additional interior wall cells to create alternative routes/cycles. One frozen change feed per seed closes/reopens a current route, isolates/restores the goal, blocks/reopens the goal cell, repeats an idempotent update, opens a shortcut, changes two other cells, and moves the start three times. Every method is evaluated on the same 13 snapshots per seed: **12 initial plans plus 144 later repairs, 156 total**. The simulator uses its own BFS only to select interesting obstruction events and legal start moves; this is an adversarial synthetic event distribution, not a natural-world workload.

Manifest SHA-256: `e9fc780787fbdd4daeaf15be856e6addfeb94b5b06f75392efe02b820c16ca6b`. The hash pins initial geometry and the complete actual event feeds. The verification report separately pins source/configuration bytes.

## Measured correctness and work

| Metric across the same 156 snapshots | D* Lite | Repeated fresh BFS | Restarted local DFS |
| --- | ---: | ---: | ---: |
| Reachable goals found | 126/126 | 126/126 | 126/126 |
| No-path cases retained | 30/30 | 30/30 | 30/30 |
| Shortest routes among reachable cases | 126/126 | 126/126 | 0/126 |
| Total vertex expansions | 5,108 | 13,193 | 12,788 |
| Initial-plan vertex expansions | 898 | 1,093 | 740 |
| Later-repair vertex expansions | 4,210 | 12,100 | 12,048 |
| Edge inspections | 102,522 | 52,268 | 51,152 |
| Mean reachable route/exploration length | 25.2381 | 25.2381 | 59.3492 |

Every D* route passed independent transition legality checks and matched BFS optimal cost. All 156 comparisons passed, including all 30 no-path cases. DFS receives only neighbor sensing and restarts memory per snapshot, whereas D* Lite knows the initial map plus changes and BFS knows each full snapshot. Its results are an exploration baseline, not an equal-information optimal-planning comparison.

The reduction in later-repair label-changing expansions is real: 4,210 versus 12,100, or approximately 65.2% fewer counted expansions. It is **not** a measured 2.87× runtime improvement. D* Lite performed almost twice as many counted edge inspections overall because it maintains one-step lookahead values, propagates invalidations, and extracts routes. BFS also rejects a blocked goal immediately, while D* Lite continues maintaining its reusable labels: the 12 goal-block events cost BFS zero expansions and D* Lite 836 expansions. These unfavorable cases remain in the aggregate.

State reuse is directly observable: all **12 idempotent updates** and **36 start moves** required zero additional D* label-changing expansions, while repeated BFS expanded 1,093 and 3,234 vertices respectively for those slices. The benchmark exercised **3,037 lower-label branches, 2,071 invalidations, and two actual moving-key reinsertions**. The two algorithms' expansion definitions differ, and the initial comparison also includes D* Lite's heuristic advantage over unheuristic BFS. No universal performance or CPU-time claim is made.

The complete report remains in ignored `artifacts/`. [../transcripts/replanning-summary.json](../transcripts/replanning-summary.json) preserves precise metrics and per-event slices; [../transcripts/replanning-seed-300.json](../transcripts/replanning-seed-300.json) preserves an actual full scenario and repair traces.

## Differential and behavioral proof

The complete standard-library suite now has **51 tests**, including 15 new incremental-search tests. They cover:

- Known open-grid Manhattan distance and backward goal anchoring.
- Warm repairs and moving-start queries retaining the same label-table object with zero unnecessary expansions.
- Cost increases, complete disconnection, cost decreases, reopening, blocked/unblocked goals, and initially unreachable graphs.
- Start equals goal, ordinary start motion, and explicit nonadjacent relocalization with accumulated `km`.
- Atomic batch rejection, duplicate observations, occupied-start protection, invalid types/coordinates, boundary edges, and idempotent updates.
- Repair queue-pop exhaustion, resumable partial work, lifetime/pending-observation budgets, and bounded lazy-heap storage.
- **1,000 generated change/move repairs**, each checked against a fresh independently implemented BFS and an independently updated expected wall set.
- The complete 156-snapshot experiment, independent DFS reachability, deterministic feeds, actual rekey/lower/invalidate branches, and CLI JSON export.

A separate development fuzz command also executed **2,000/2,000** random repair/BFS agreements before the checked-in tests were added. That exploratory run is additional debugging evidence, not a hidden held-out accuracy benchmark. The reproducible checked-in regression evidence is the 1,000-case test plus the 156-case experiment.

The existing 56 navigation invariants and 56 transition replays remain separate and pass unchanged. The verification script now includes `replan-evaluate` in both source and cold-wheel CLI checks, preserving the existing CI workflow while exercising the new module from an ordinary installation.

## Limits and next decisions

The implemented graph is bounded, undirected, unit-cost, and four-neighbor. It has a fixed goal and a complete trusted change feed. Arbitrary weighted edges, delayed/incorrect observations, partially hidden dynamic obstacles, target changes, multi-agent collisions, and hard real-time/process isolation remain out of scope. DFS and the optimal planners have explicitly different observation budgets. These small published synthetic scenarios do not establish generalization or production autonomy.

Work counters are transparent algorithm-specific counts, not hardware-normalized instructions or latency. No fresh-A* ablation was run, so heuristic benefit and incremental reuse are not fully separated in aggregate comparisons. The test suite demonstrates state reuse directly but is not a formal proof over every possible grid/change sequence. Budgets cap work and report errors; they cannot promise completion under an arbitrarily small supplied budget. Checkpoint inference remains optional and unrun.
