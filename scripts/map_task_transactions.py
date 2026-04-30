import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

import yaml


ORCH_FILE_PATTERN = re.compile(r"^orchestrator_response_(T\d+_\d{3})_.*\.json$")


def load_tasks(tasks_path: Path) -> Dict[str, str]:
    with tasks_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    tasks_section = data.get("tasks") if isinstance(data, dict) else {}
    tasks: Dict[str, str] = {}

    for task_id, config in tasks_section.items():
        if not isinstance(config, dict):
            continue
        query = config.get("query")
        if query:
            tasks[task_id] = query

    return tasks


def extract_transaction_id(payload: Dict) -> Optional[str]:
    for key in ("transaction_id", "transactionId"):
        value = payload.get(key)
        if value:
            return str(value)

    meta = (payload.get("collected_json") or {}).get("meta") if isinstance(payload.get("collected_json"), dict) else None
    if isinstance(meta, dict):
        for key in ("transaction_id", "transactionId"):
            value = meta.get(key)
            if value:
                return str(value)

    raw_chunks = payload.get("raw_chunks")
    if isinstance(raw_chunks, list):
        for chunk in raw_chunks:
            if not isinstance(chunk, dict):
                continue
            for key in ("transaction_id", "transactionId"):
                value = chunk.get(key)
                if value:
                    return str(value)

    return None


def extract_step_id(payload: Dict) -> Optional[str]:
    for key in ("step_id", "stepId"):
        value = payload.get(key)
        if value:
            return str(value)

    meta = (payload.get("collected_json") or {}).get("meta") if isinstance(payload.get("collected_json"), dict) else None
    if isinstance(meta, dict):
        for key in ("step_id", "stepId"):
            value = meta.get(key)
            if value:
                return str(value)

    raw_chunks = payload.get("raw_chunks")
    if isinstance(raw_chunks, list):
        for chunk in raw_chunks:
            if not isinstance(chunk, dict):
                continue
            for key in ("step_id", "stepId"):
                value = chunk.get(key)
                if value:
                    return str(value)

    return None


def collect_transaction_map(collected_dir: Path) -> Dict[str, Dict[str, Optional[str]]]:
    task_to_info: Dict[str, Dict[str, Optional[str]]] = {}
    txn_to_tasks: defaultdict[str, Set[str]] = defaultdict(set)

    for path in collected_dir.glob("orchestrator_response_*.json"):
        match = ORCH_FILE_PATTERN.match(path.name)
        if not match:
            continue
        task_id = match.group(1)
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            continue

        transaction_id = extract_transaction_id(payload)
        step_id = extract_step_id(payload)

        info = task_to_info.setdefault(task_id, {"transaction_id": None, "step_id": None, "files": []})
        info["files"].append(str(path))

        if transaction_id:
            info["transaction_id"] = transaction_id
            txn_to_tasks[transaction_id].add(task_id)
        if step_id:
            info["step_id"] = step_id

    for transaction_id, tasks in txn_to_tasks.items():
        if len(tasks) > 1:
            for task in tasks:
                info = task_to_info.get(task)
                if info is not None:
                    dup_list = info.setdefault("duplicate_transactions", set())
                    dup_list.add(transaction_id)

    return task_to_info


def build_response_lookup(response_path: Path):
    query_to_transactions: defaultdict[str, Set[str]] = defaultdict(set)
    transaction_to_query: Dict[str, str] = {}
    response_missing_transaction: Set[str] = set()

    with response_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            rounds = record.get("rounds")
            if not isinstance(rounds, list):
                continue

            round_zero = next((item for item in rounds if item.get("round") == 0), None)
            if not isinstance(round_zero, dict):
                continue

            span = round_zero.get("span")
            if not isinstance(span, dict):
                continue

            input_query = span.get("inputQuery")
            if input_query is None:
                continue

            transaction_id = record.get("transactionId") or record.get("transaction_id")
            if transaction_id:
                transaction_id = str(transaction_id)
                query_to_transactions[input_query].add(transaction_id)
                transaction_to_query[transaction_id] = input_query
            else:
                response_missing_transaction.add(input_query)

    return query_to_transactions, transaction_to_query, response_missing_transaction


def analyse(tasks: Dict[str, str],
            task_info: Dict[str, Dict[str, Optional[str]]],
            query_lookup: Dict[str, Set[str]],
            response_missing_transaction: Set[str]):
    successes: Dict[str, str] = {}
    failures: Dict[str, List[str]] = {}

    task_ids = sorted(tasks.keys())

    for task_id in task_ids:
        query = tasks[task_id]
        info = task_info.get(task_id)
        reasons: List[str] = []

        transactions = query_lookup.get(query)
        if not transactions:
            if query in response_missing_transaction:
                reasons.append("response_transaction_missing")
            else:
                reasons.append("query_not_in_response")
        else:
            if len(transactions) > 1:
                reasons.append("duplicate_transactions_for_query")
            transaction_from_response = next(iter(transactions))

            if info is None:
                reasons.append("orchestrator_dump_missing")
            elif not info.get("transaction_id"):
                reasons.append("missing_transaction_id")
            else:
                transaction_from_files = info.get("transaction_id")
                if transaction_from_files != transaction_from_response:
                    reasons.append("transaction_mismatch")

                duplicates = info.get("duplicate_transactions")
                if duplicates:
                    reasons.append("duplicate_transaction_id")

        if not reasons:
            info = task_info.get(task_id) or {}
            transaction_id = info.get("transaction_id")
            if transaction_id is not None:
                successes[task_id] = str(transaction_id)
        else:
            failures[task_id] = reasons

    for task_id, info in task_info.items():
        if task_id not in tasks:
            reasons = ["task_not_in_config"]
            if info.get("transaction_id") is None:
                reasons.append("missing_transaction_id")
            failures[task_id] = reasons

    return successes, failures


def format_output(successes: Dict[str, str], failures: Dict[str, List[str]]):
    if successes:
        print("Matched task -> transaction_id:")
        for task_id in sorted(successes):
            print(f"  {task_id}: {successes[task_id]}")
        print()

    if failures:
        print("Failures:")
        for task_id in sorted(failures):
            joined = ", ".join(sorted(set(failures[task_id])))
            print(f"  {task_id}: {joined}")
    else:
        print("No failures detected. All tasks matched.")


def main():
    parser = argparse.ArgumentParser(description="Map tasks to transaction IDs and validate against response logs.")
    parser.add_argument("--tasks", required=True, help="Path to tasks.yaml")
    parser.add_argument("--responses", required=True, help="Path to response JSONL file")
    parser.add_argument("--collected", required=True, help="Directory with orchestrator_response_*.json files")
    args = parser.parse_args()

    tasks_path = Path(args.tasks)
    responses_path = Path(args.responses)
    collected_dir = Path(args.collected)

    if not tasks_path.is_file():
        raise FileNotFoundError(f"tasks file not found: {tasks_path}")
    if not responses_path.is_file():
        raise FileNotFoundError(f"response file not found: {responses_path}")
    if not collected_dir.is_dir():
        raise NotADirectoryError(f"collected directory not found: {collected_dir}")

    tasks = load_tasks(tasks_path)
    task_info = collect_transaction_map(collected_dir)
    query_lookup, _txn_to_query, response_missing_transaction = build_response_lookup(responses_path)

    successes, failures = analyse(tasks, task_info, query_lookup, response_missing_transaction)
    format_output(successes, failures)


if __name__ == "__main__":
    main()
