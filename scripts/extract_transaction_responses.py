import json
from pathlib import Path
from typing import Dict, List, Set


def _extract_transaction_ids(node, seen: Set[str], ordered: List[str]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key in {"transaction_id", "transactionId"} and isinstance(value, str):
                if value not in seen:
                    seen.add(value)
                    ordered.append(value)
            _extract_transaction_ids(value, seen, ordered)
    elif isinstance(node, list):
        for item in node:
            _extract_transaction_ids(item, seen, ordered)


def collect_transaction_ids(collected_root: Path) -> List[str]:
    ordered: List[str] = []
    seen: Set[str] = set()

    for json_path in sorted(collected_root.glob("*.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        _extract_transaction_ids(data, seen, ordered)

    return ordered


def load_evaluator_lines(evaluator_path: Path) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    with evaluator_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            transaction_id = obj.get("transactionId")
            if isinstance(transaction_id, str):
                mapping.setdefault(transaction_id, []).append(stripped)
    return mapping


def write_filtered_aggregate(
    evaluator_path: Path,
    id_order: List[str],
    mapping: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    results: Dict[str, List[str]] = {"written": [], "missing": []}
    output_path = evaluator_path.with_name(evaluator_path.stem + "_filtered.jsonl")

    with output_path.open("w", encoding="utf-8") as out_handle:
        for transaction_id in id_order:
            lines = mapping.get(transaction_id)
            if not lines:
                results["missing"].append(transaction_id)
                continue
            for line in lines:
                out_handle.write(line + "\n")
            results["written"].append(transaction_id)

    return results


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    collected_root = project_root / "output" / "collected_jsons"
    evaluator_path = project_root / "evaluators" / "response_1765076550868.jsonl"

    if not collected_root.exists():
        raise SystemExit(f"Collected JSON directory not found: {collected_root}")
    if not evaluator_path.exists():
        raise SystemExit(f"Evaluator file not found: {evaluator_path}")

    id_order = collect_transaction_ids(collected_root)
    mapping = load_evaluator_lines(evaluator_path)
    results = write_filtered_aggregate(evaluator_path, id_order, mapping)

    print(f"Total transaction ids discovered: {len(id_order)}")
    print(f"Transaction IDs written: {len(results['written'])}")
    if results["missing"]:
        print("Missing transaction ids (no evaluator entries found):")
        for transaction_id in results["missing"]:
            print(f"  - {transaction_id}")


if __name__ == "__main__":
    main()
