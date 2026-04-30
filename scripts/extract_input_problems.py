"""Extract input_problem fields from scenario JSON into a plain-text list."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def extract_input_problems(source: Dict[str, Any]) -> list[str]:
    scenarios = source.get("scenarios") or []
    problems: list[str] = []
    for entry in scenarios:
        text = entry.get("input_problem")
        if isinstance(text, str):
            problems.append(text.strip())
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description="Write each input_problem to a txt file (one per line)")
    parser.add_argument("--input", required=True, help="Path to scenarios JSON file")
    parser.add_argument("--output", required=True, help="Path to output txt file")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as infile:
        data = json.load(infile)

    problems = extract_input_problems(data)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as outfile:
        for line in problems:
            outfile.write(line.replace("\n", " ").strip())
            outfile.write("\n")

    print(f"Wrote {len(problems)} input_problem entries to {output_path}")


if __name__ == "__main__":
    main()
