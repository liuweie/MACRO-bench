# Multi-Agent Travel Benchmark

This repository provides a practical benchmark pipeline for evaluating multi-agent orchestration in the travel domain.
It includes task generation, trace collection, LLM/rule-based evaluation, and stability/consistency analysis scripts.

## Scope

- Domain: travel-focused multi-agent workflows
- Inputs: task definitions or generated task sets
- Outputs: execution traces and evaluation summaries
- Target use: reproducible offline experiments for research review

## Repository Layout

- `run_benchmark.py`: main benchmark runner (trace collection)
- `benchmark.py`: benchmark core logic and judging utilities
- `datasets/`: task generation and dataset utilities
- `evaluators/`: evaluation pipeline
- `user_simulator/`: simulated user behavior for clarifications
- `trace_output/`: generated traces and analysis outputs
- `scripts/`: utility scripts for experiments

## Environment

- Python 3.10+ recommended
- Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1) Generate tasks from seed queries

```bash
python ./datasets/travel_generator.py --mode from-file --queries-file ./scripts/demo_query.txt --out ./datasets/demo_data/demo_query.json --format json
```

### 2) Run benchmark and collect traces

```bash
python run_benchmark.py --counts 1 --debug --max-clarifications 24 --max-subagent-clarification-rounds 12 --lang en --domain travel
```

### 3) Run evaluator on traces

```bash
python evaluators/evaluator.py -t ./datasets/travel/tasks.yaml -a ./config/travel_agents.json -i ./trace_output/af_trace/converted_travel_tasks_v1_20251224_1130_exp1_3reset.jsonl -v
```

### 4) Repeat runs for stability analysis

```bash
bash run_stability.sh NUM_RUNS
```

## Core Runner Arguments

`run_benchmark.py` commonly used options:

- `--domain`: benchmark domain (default `travel`)
- `--counts`: number of repeated runs
- `--config`: path to task config YAML
- `--task-ids`: run selected task IDs only
- `--max-clarifications`: max clarification rounds
- `--max-subagent-clarification-rounds`: per-subagent clarification limit
- `--lang`: request/simulator language (`en` or `zh`)
- `--report`: output report path

## Reproducibility Notes

- Keep task config and plugin config fixed across runs.
- Use the same model endpoints and generation parameters.
- Record all outputs under `trace_output/` for auditability.
- For repeated runs, report mean/std/CV instead of single-run scores.

## Output Artifacts

Typical artifacts include:

- benchmark traces (`jsonl`)
- detailed evaluation reports (`json`)
- stability tables/figures (`csv`, `png`)
- consistency comparison outputs (`csv`)

## Disclaimer

This codebase is an anonymized research release prepared for peer review.
Any organization-specific identifiers or private deployment details are intentionally removed.