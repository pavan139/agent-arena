.PHONY: install test demo evaluate replay replan-evaluate
install:
	python -m pip install -e .
test:
	python -m unittest discover -s scenarios -v
demo:
	python -m agent_arena demo --output artifacts/demo.json
evaluate:
	python -m agent_arena evaluate --output artifacts/evaluation.json
replay:
	python -m agent_arena replay --input artifacts/evaluation.json --output artifacts/replay.json
replan-evaluate:
	python -m agent_arena replan-evaluate --output artifacts/replanning.json
