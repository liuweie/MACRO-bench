import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml


def load_tasks(tasks_path: Path) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Return mapping from query to task ids and reverse lookup."""
    with tasks_path.open("r", encoding="utf-8") as handle:
        content = yaml.safe_load(handle)

    tasks_section = content.get("tasks", {}) if isinstance(content, dict) else {}
    query_to_task_ids: dict[str, list[str]] = defaultdict(list)
    task_id_to_query: dict[str, str] = {}

    for task_id, task_body in tasks_section.items():
        if not isinstance(task_body, dict):
            continue
        query = task_body.get("query")
        if not query:
            continue
        query_to_task_ids[query].append(task_id)
        task_id_to_query[task_id] = query

    return query_to_task_ids, task_id_to_query


def load_round_zero_queries(response_path: Path) -> list[str]:
    """Extract the round-0 span input queries from the response log."""
    queries: list[str] = []

    with response_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            rounds = record.get("rounds")
            if not isinstance(rounds, list):
                continue

            round_zero_entry = next(
                (item for item in rounds if item.get("round") == 0),
                None,
            )
            if not round_zero_entry:
                continue

            span = round_zero_entry.get("span")
            if not isinstance(span, dict):
                continue

            input_query = span.get("inputQuery")
            if input_query is None:
                continue

            queries.append(input_query)

    return queries


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether task queries from tasks.yaml align one-to-one with "
            "round-0 input queries recorded in a response JSONL file."
        )
    )
    parser.add_argument("--tasks", required=True, help="Path to tasks.yaml")
    parser.add_argument(
        "--responses", required=True, help="Path to the response JSONL file"
    )
    args = parser.parse_args()

    tasks_path = Path(args.tasks)
    response_path = Path(args.responses)

    if not tasks_path.is_file():
        raise FileNotFoundError(f"tasks file not found: {tasks_path}")
    if not response_path.is_file():
        raise FileNotFoundError(f"response file not found: {response_path}")

    query_to_task_ids, task_id_to_query = load_tasks(tasks_path)
    round_zero_queries = load_round_zero_queries(response_path)

    # Track duplicates defined as multiple tasks sharing the same query.
    duplicate_map = {
        query: ids for query, ids in query_to_task_ids.items() if len(ids) > 1
    }

    # Prepare mutable copies for matching.
    remaining_map = {query: list(ids) for query, ids in query_to_task_ids.items()}
    unmatched_response_queries: list[str] = []
    extra_response_queries: set[str] = set()

    for query in round_zero_queries:
        available_ids = remaining_map.get(query)
        if available_ids:
            available_ids.pop(0)
        else:
            unmatched_response_queries.append(query)
            if query not in query_to_task_ids:
                extra_response_queries.add(query)

    unmatched_task_ids: list[str] = []
    for ids in remaining_map.values():
        unmatched_task_ids.extend(ids)

    duplicate_task_ids: list[str] = []
    for ids in duplicate_map.values():
        duplicate_task_ids.extend(ids)

    response_counter = Counter(round_zero_queries)
    duplicated_response_queries = [
        query for query, count in response_counter.items() if count > 1
    ]

    if (
        not unmatched_task_ids
        and not duplicate_task_ids
        and not unmatched_response_queries
        and not duplicated_response_queries
        and not extra_response_queries
    ):
        print("PASS")
        return

    if unmatched_task_ids:
        print("Missing task ids:")
        for task_id in sorted(unmatched_task_ids):
            query = task_id_to_query.get(task_id, "")
            print(f"{task_id}: {query}")

    if duplicate_task_ids:
        print("Duplicated task ids:")
        for task_id in sorted(duplicate_task_ids):
            query = task_id_to_query.get(task_id, "")
            print(f"{task_id}: {query}")

    if duplicated_response_queries:
        print("Duplicated response queries:")
        for query in duplicated_response_queries:
            print(query)

    if extra_response_queries:
        print("Response-only queries:")
        for query in sorted(extra_response_queries):
            print(query)

    if unmatched_response_queries:
        print("Unmatched response queries:")
        for query in unmatched_response_queries:
            print(query)


if __name__ == "__main__":
    main()
