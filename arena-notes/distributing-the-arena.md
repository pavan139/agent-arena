# Distributing the arena and its traces

Package the code, procedural rulesets and small transcripts together. A navigation report needs its maze fingerprint and observation regime; a replanning report also needs the change feed and moving-start observations. The legacy replay command accepts navigation episodes, not the separate replanning schema.

Before a release, run:

```bash
python dev/check_arena.py --require-lint --wheel
python -m agent_arena replan-evaluate --seed 300 --output artifacts/replanning.json
```

Inspect the returned paths and both operation counters. Keep `transcripts/hf-status.json` labeled as unrun unless you actually evaluate a local checkpoint. Do not bundle model weights with a source archive.

If publishing from an extracted archive, initialize Git only when `.git` is absent. Commit the source and selected transcripts after checking their size. With an authenticated GitHub CLI, an empty remote can then be created using:

```bash
gh repo create agent-arena --public --source=. --remote=origin --push
```

Use another repository name if needed. The command publishes your current commit, so review the files first. CI checks Python 3.11 and 3.12 after the push; local results do not establish a remote CI run.
