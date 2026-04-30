"""Utility to convert scenario JSON into travel_tasks-style YAML.

Usage:
    python scripts/convert_scenarios_to_tasks.py \
        --input datasets/scenarios_30.json \
        --output config/scenarios_as_travel_tasks.yaml
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

try:
    import yaml
except ImportError as exc:  # pragma: no cover - PyYAML should be installed
    raise SystemExit("PyYAML is required to run this converter. Please install it first.") from exc


_AGENT_NAME_PATTERN = re.compile(r"agent\s*[:：]\s*([\w:\-]+)", re.IGNORECASE)


def _normalise_description(text: str) -> str:
    """Collapse whitespace so long-form scenario text fits a single YAML line."""
    return " ".join(text.split())


def _split_assertions(assertions: List[Any]) -> Dict[str, List[str]]:
    """Split assertions into user/system milestone buckets based on prefix."""
    user: List[str] = []
    system: List[str] = []

    for assertion in assertions or []:
        if not isinstance(assertion, str):
            continue
        prefix, _, remainder = assertion.partition(":")
        text = remainder.strip() if remainder else assertion.strip()
        target = prefix.strip().lower()
        if target.startswith("agent"):
            system.append(text)
        else:
            user.append(text)
    return {"user": user, "system": system}


def _extract_expected_subagents(system_milestones: List[str]) -> List[str]:
    """Heuristically collect agent/tool names from system-side assertions."""
    discovered: List[str] = []
    for milestone in system_milestones:
        match = _AGENT_NAME_PATTERN.search(milestone)
        if match:
            name = match.group(1).strip()
            if name and name not in discovered:
                discovered.append(name)
    return discovered


def convert_scenarios(data: Dict[str, Any], *, default_level: str = "T3") -> Dict[str, Any]:
    """Convert the parsed JSON document into the travel_tasks schema."""
    scenarios = data.get("scenarios") or []
    tasks: Dict[str, Any] = {}

    for idx, entry in enumerate(scenarios, start=1):
        scenario_id = f"SCENARIO_{idx:03d}"
        scenario_text = entry.get("scenario", "")
        query = (entry.get("input_problem") or "").strip()
        assertions = entry.get("assertions") or []

        milestones = _split_assertions(assertions)
        system_milestones = milestones["system"]
        user_milestones = milestones["user"]

        expected_subagents = _extract_expected_subagents(system_milestones)

        tasks[scenario_id] = {
            "level": entry.get("level") or default_level,
            "query": query,
            "expected_subagents": expected_subagents,
            "expected_clarifications": entry.get("expected_clarifications", []),
            "user_side_milestones": user_milestones,
            "system_side_milestones": system_milestones,
            "complexity_factors": entry.get("complexity_factors", []),
            "description": _normalise_description(scenario_text),
        }

    return {"tasks": tasks}


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert scenario JSON into travel_tasks-style YAML")
    parser.add_argument("--input", required=True, help="Path to the scenarios JSON file")
    parser.add_argument("--output", required=True, help="Path to write the converted YAML file")
    parser.add_argument("--default-level", default="T3", help="Default task level to use when not provided")
    parser.add_argument("--version", required=False, help="Version tag prefix to include in the generated YAML (e.g. V1)")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as infile:
        data = json.load(infile)

    converted = convert_scenarios(data, default_level=args.default_level)

    # Compose version string when requested: <input>_YYYYMMDD_HHMM
    if getattr(args, "version", None):
        from datetime import datetime
        now = datetime.now()
        ts = now.strftime("%Y%m%d_%H%M")
        version_str = f"{args.version}_{ts}"
        # Insert version at top-level
        output_doc: Dict[str, Any] = {"version": version_str}
        output_doc.update(converted)
    else:
        output_doc = converted

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as outfile:
        yaml.safe_dump(output_doc, outfile, allow_unicode=True, sort_keys=False)

    task_count = len(converted.get('tasks', {}))
    if isinstance(output_doc, dict) and output_doc.get('version'):
        print(f"Converted {task_count} scenarios to travel_tasks format -> {output_path} (version={output_doc.get('version')})")
    else:
        print(f"Converted {task_count} scenarios to travel_tasks format -> {output_path}")


if __name__ == "__main__":
    main()
