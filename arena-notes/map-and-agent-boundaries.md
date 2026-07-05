# Architecture and execution contract

The incremental replanning subsystem is described in [search-and-memory.md](search-and-memory.md), with actual comparisons in [repair-work-measurements.md](repair-work-measurements.md). It preserves the navigation executor below and adds a separate D* Lite state machine and report schema.

The application chooses a validated immutable world, observation regime, and immutable movement permissions. Policy output is a proposal. The executor owns tool invocation, stopping rules, exact success, and the transition record.

```mermaid
sequenceDiagram
    participant E as Episode runner
    participant P as Policy
    participant W as Validated maze
    participant R as Independent replayer
    E->>P: Reset episode memory when supported
    loop Bounded policy attempts
        E->>E: Check cancellation, deadline, goal, step budget
        E->>P: Full-map or neighbor-only immutable observation
        P-->>E: Proposed action or typed error
        E->>E: Preserve fresh call metadata, recheck time/cancellation
        E->>E: Schema, permission, repetition checks
        E->>W: Apply one bounded move
        W-->>E: Coordinate or blocked move
        E->>E: Record transition; recognize accepted goal immediately
    end
    E->>R: Content-addressed world, limits, and transition trace
    R->>W: Recompute transitions and shortest-path reference
    R-->>E: Verified consistency or explicit replay error
```

## Input and state boundaries

`Maze` validates signed 32-bit seeds, odd 5–31 dimensions, bounded integer coordinates, wall count, and walkable start/goal. It copies mutable coordinates/walls into tuples/frozensets. Disconnected but otherwise valid worlds are supported through the Python API. `generate_maze` uses depth-first carving of an odd cell lattice; its generated perfect mazes are solvable. A content hash includes actual geometry, seed, start, and goal.

`Limits` validates types and finite numeric deadlines and copies permissions into a frozenset. `Observation` validates position, remaining steps, and permissions. `LocalObservation` contains only position, goal, remaining steps, four validated neighbor readings, and allowed actions; it has no reference to the full maze. These are immutable value-object conventions for trusted Python, not isolation from code that deliberately bypasses Python object protections.

## Policies and state lifecycle

`ReactivePolicy` and `BFSOracle` receive a full occupancy map. The reactive policy selects a walkable neighbor by Manhattan distance and can cycle. BFS supplies an exact shortest-path planning reference. Their output still passes the same movement gate.

`LocalDFSExplorer` receives only local occupancy readings. It keeps a visited set and DFS route, explores unvisited neighbors, and backtracks without a hidden map. The runner resets its memory between episodes. Static cyclic maps are supported; unreachable or restricted exploration can terminate with an explicitly reported `exploration_exhausted` policy error. It does not infer external facts or call a model.

The local explorer is an independent capability comparison under reduced information, not an equal-observation replacement for the full-map oracle. Evaluation reports retain the observation regime for every policy.

## Execution and error accounting

The runner validates action schema before action lookup, enforces the application allowlist, checks repeated state/action usage, and applies exactly one move. Blocked moves retain state and increment consecutive-no-progress. Accepted goal arrival terminates immediately. The default loop permits at most 128 policy/tool attempts; configuration caps attempts at 4,096 and deadlines at 60 seconds.

Rejected proposals, policy exceptions, and calls that return after cancellation/deadline preserve a terminal trace without changing state. Fresh allowlisted model-call metadata is attached even on an exception. Stale metadata from a previous call is not attached to a later budget/encoding failure. Raw malformed model output, provider exception messages, and arbitrary metadata fields are omitted.

`steps` counts executed movement calls, including blocked calls; `policy_attempts` includes a final rejected/failed selection. A failed reset counts as neither. Reports record start/final coordinates, limits, observation mode, actual maze hash, optimal path length, usage, and stop reason.

Deadlines are cooperative around synchronous trusted calls. The harness cannot interrupt a blocked Python call, GPU kernel, or external code. A process supervisor and operating-system resource limits are required for adversarial policies or hard termination.

## Independent replay

The replayer checks episode schemas, world identity, consecutive trace positions, legal movement, permissions, valid/blocked flags, state hashes, execution counters, repetition/no-progress boundaries, shortest-path length, final coordinate, and exact success claim. It recomputes state instead of accepting report totals. Rejected terminal actions cannot move. A trace cannot continue after goal/no-progress termination.

Replay verifies consistency with the supplied world and recorded limits. It cannot authenticate an unsigned report, prove which policy/model produced it, reconstruct wall-clock deadlines, or prove cancellation occurred. Fully fabricated but internally consistent traces are not cryptographically distinguishable from genuine runs. The Python API accepts a caller-provided custom maze; CLI replay rebuilds the seeded procedural world and rejects geometry mismatches.

## Dataset and metrics

A single manifest includes separate development, release, and larger exploration-stress splits with disjoint seeds and unseen dimensions. Actual world hashes are part of the dataset fingerprint; default and local-HF evaluators use the same manifest function. `rulesets/splits.json` mirrors that generated manifest and is documented evidence, not a runtime loader.

Every offline evaluation episode is independently replayed. BFS must reach the goal in exactly the shortest-path length; local DFS must reach the goal within its budget; the reactive policy may fail navigation while still obeying executor invariants. Boundary cases separately assert malformed/invalid actions, permissions, step exhaustion, deadline, cancellation, blocked movement, and repetition.

Metrics retain goal-success denominators, successful-path denominators, stop counts, and observation regimes. Success-weighted path efficiency averages successful optimal/actual ratios over all episodes with failures contributing zero. Synthetic rates do not establish learned intelligence or real-world agent reliability.

## Optional local Hugging Face boundary

Checkpoint validation, safe metadata parsing, and framed byte fingerprinting precede optional ML imports. Loading is local-files-only, remote-code-disabled, and safetensors-only. Errors from native loaders are normalized without raw provider details. The selected CPU/CUDA device and local hardware limits remain user responsibilities.

The prompt contains application-authored instructions plus JSON observation, including current allowed actions. Input tokens are counted without truncation; nullable native context limits are handled while enforcing the application cap. Generation fixes greedy decoding, one sequence, token/time caps, and tensor output. Parsing accepts only one permitted action word and does not repair arbitrary prose.

Every attempted inference creates call metadata before generation. Success, malformed output, timeout, and native generation failures preserve observed input/output counts, latency, finish/error codes, and shared call-budget state. Failed calls without a returned tensor mark output as unobserved. No deterministic action fallback exists. Local hardware cost remains unknown; there are no remote provider calls.

Model loading/fingerprinting occurs outside episode deadlines. Generation timing is cooperative. Real weights were not loaded or executed during verification; all model-path execution tests use explicit test doubles, and the checked-in HF artifact states `not_run`.
