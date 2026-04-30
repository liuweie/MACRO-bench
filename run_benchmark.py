#!/usr/bin/env python3
"""Enhanced CLI for running benchmarks (adapted from benchmark-mutliagent/main.py).

This script provides a Click-based interface that mirrors the options used
in the benchmark-mutliagent entrypoint so it's easier to run `travel` tests.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import click
import os
import sys

# Ensure package imports work when running this file directly as a script
# (e.g., `python run_benchmark.py`) by adding the workspace root to sys.path.
try:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
except Exception:
    pass

from plugins.manager import global_plugin_manager
from benchmark import PluginBasedCogBenchmark


def _extract_domain_setting(manager, domain: str, key: str):
    domain_settings = getattr(manager, '_domain_settings', {}) or {}
    if not isinstance(domain_settings, dict):
        return None

    if domain and isinstance(domain_settings.get(domain), dict):
        value = domain_settings[domain].get(key)
        if value:
            return value

    default_settings = domain_settings.get('default') if isinstance(domain_settings.get('default'), dict) else {}
    return default_settings.get(key)


def _resolve_chain(manager, hook_key: str, domain: str) -> List[str]:
    chain = _extract_domain_setting(manager, domain, f"{hook_key}_chain")
    if isinstance(chain, (list, tuple)):
        return [str(item) for item in chain]

    hook_chains = getattr(manager, '_hook_chains', {}) or {}
    section = hook_chains.get(hook_key)
    if isinstance(section, dict):
        if domain and isinstance(section.get(domain), (list, tuple)):
            return [str(item) for item in section.get(domain, [])]
        if isinstance(section.get('default_chain'), (list, tuple)):
            return [str(item) for item in section.get('default_chain', [])]
        for value in section.values():
            if isinstance(value, (list, tuple)):
                return [str(item) for item in value]
    elif isinstance(section, (list, tuple)):
        return [str(item) for item in section]

    return []


def _print_selected_plugins(domain: str,
                            evaluator_type: Optional[str] = None,
                            reporter_type: Optional[str] = None):
    try:
        manager = global_plugin_manager
    except Exception:
        print("Unable to access plugin manager; skipping plugin summary.")
        return

    config_data = getattr(manager, '_config_data', {}) if hasattr(manager, '_config_data') else {}
    logging_cfg = config_data.get('logging') if isinstance(config_data, dict) else {}

    def _is_enabled(value, default=True):
        if value is None:
            return default
        if isinstance(value, str):
            return value.strip().lower() not in ('0', 'false', 'no', 'off')
        return bool(value)

    if not _is_enabled((logging_cfg or {}).get('show_plugin_summary'), True):
        return

    try:
        orchestrators = _resolve_chain(manager, 'orchestrator', domain)
        evaluators = _resolve_chain(manager, 'evaluator', domain)
        reporters = _resolve_chain(manager, 'reporter', domain)
        domain_plugin = _extract_domain_setting(manager, domain, 'user_simulator_plugin')

        config_plugins: Dict[str, Dict[str, Dict[str, object]]] = getattr(manager, '_config_data', {}).get('plugins', {}) or {}

        def _plugin_enabled(name: Optional[str]) -> Optional[bool]:
            if not name or name not in config_plugins:
                return None
            entry = config_plugins.get(name) or {}
            enabled = entry.get('enabled')
            return bool(enabled) if enabled is not None else None

        loaded_plugins = []
        try:
            for plugin in manager.pm.get_plugins():
                cls = getattr(plugin, '__class__', None)
                if cls:
                    module = getattr(cls, '__module__', '')
                    name = getattr(cls, '__name__', '')
                    identifier = f"{module}.{name}" if module else name
                else:
                    module = getattr(plugin, '__module__', '')
                    identifier = module or str(plugin)
                loaded_plugins.append(identifier)
            loaded_plugins = sorted(set(filter(None, loaded_plugins)))
        except Exception:
            loaded_plugins = []

        print("Plugin configuration for this run:")
        print(f"  - Domain: {domain}")
        if domain_plugin:
            enabled_flag = _plugin_enabled(domain_plugin)
            if enabled_flag is True:
                status = "enabled"
            elif enabled_flag is False:
                status = "disabled"
            else:
                status = "unknown"
            print(f"  - Domain plugin: {domain_plugin} ({status})")
        else:
            print("  - Domain plugin: not configured")

        chain_display = ', '.join(orchestrators) if orchestrators else 'none'
        print(f"  - Orchestrator chain: {chain_display}")

        evaluator_chain = list(evaluators)
        if evaluator_type and evaluator_type not in evaluator_chain:
            evaluator_chain.append(evaluator_type)
        evaluator_display = ', '.join(filter(None, evaluator_chain)) if evaluator_chain else evaluator_type or 'none'
        print(f"  - Evaluator chain: {evaluator_display}")

        reporter_chain = list(reporters)
        if reporter_type and reporter_type not in reporter_chain:
            reporter_chain.append(reporter_type)
        reporter_display = ', '.join(filter(None, reporter_chain)) if reporter_chain else reporter_type or 'none'
        print(f"  - Reporter chain: {reporter_display}")

        if loaded_plugins:
            print("  - Loaded plugin objects:")
            for identifier in loaded_plugins:
                print(f"      * {identifier}")
    except Exception as exc:
        print(f"Failed to print plugin summary: {exc}")


def _format_chain(chain: List[Dict[str, Any]]) -> str:
    parts = []
    for entry in chain:
        plugin = entry.get('plugin') or entry.get('name') or '?'
        action = entry.get('action')
        if action:
            parts.append(f"{plugin} ({action})")
        else:
            parts.append(plugin)
    return ', '.join(parts)


def _print_pipeline_usage(domain: str, combined_report: Any):
    try:
        manager = global_plugin_manager
    except Exception:
        print("Unable to access plugin manager; skipping plugin usage summary.")
        return

    config_data = getattr(manager, '_config_data', {}) if hasattr(manager, '_config_data') else {}
    logging_cfg = config_data.get('logging') if isinstance(config_data, dict) else {}

    def _is_enabled(value, default=True):
        if value is None:
            return default
        if isinstance(value, str):
            return value.strip().lower() not in ('0', 'false', 'no', 'off')
        return bool(value)

    if not _is_enabled((logging_cfg or {}).get('show_plugin_pipeline'), True):
        return

    print("Plugin usage during run:")
    pipeline_labels = [
        ('orchestrator_payload', '  - Orchestrator payload'),
        ('orchestrator_call', '  - Orchestrator call'),
        ('orchestrator_stream', '  - Orchestrator stream'),
        ('evaluators', '  - Evaluators'),
    ]

    for pipeline_key, label in pipeline_labels:
        try:
            description = manager.describe_pipeline(pipeline_key, last_only=True)
        except Exception as exc:
            print(f"{label}: unavailable ({exc})")
            continue

        last_record = description.get('last_record') if isinstance(description, dict) else None
        chain = last_record.get('chain') if isinstance(last_record, dict) else None
        if chain:
            print(f"{label}: {_format_chain(chain)}")
        else:
            print(f"{label}: none")

    reporter_chain: List[str] = []
    if isinstance(combined_report, dict):
        pipeline_meta = combined_report.get('_report_pipeline')
        if isinstance(pipeline_meta, list):
            for entry in pipeline_meta:
                if not isinstance(entry, dict):
                    continue
                plugin = entry.get('plugin')
                if not plugin:
                    continue
                outcome = 'error' if 'error' in entry else 'ok'
                reporter_chain.append(f"{plugin} ({outcome})")

    reporter_display = ', '.join(reporter_chain) if reporter_chain else 'none'
    print(f"  - Reporters: {reporter_display}")


def _run_benchmark(domain: str,
                   user_profile: str,
                   orchestrator_url: str,
                   config: str,
                   task_ids: Optional[List[str]],
                   report: Optional[str],
                   out_format: str,
                   counts: int,
                   max_clarifications: int,
                   history_size: int,
                   debug: bool,
                   profile_path: Optional[str] = None,
                   max_subagent_clarification_rounds: Optional[int] = None,
                   lang: str = 'zh'):
    benchmark = PluginBasedCogBenchmark(
        orchestrator_url=orchestrator_url,
        llm_config={},
        config_path=config,
        debug=debug,
        history_size=history_size,
        profile_path=profile_path,
        domain=domain,
        lang=lang
    )

    try:
        resolved_evaluator = getattr(benchmark, 'evaluator_type', None)
    except Exception:
        resolved_evaluator = None

    try:
        resolved_reporter = getattr(benchmark, 'reporter_type', None)
    except Exception:
        resolved_reporter = None

    _print_selected_plugins(domain, resolved_evaluator, resolved_reporter)

    try:
        benchmark.lang = lang
    except Exception:
        pass

    try:
        benchmark.max_clarification_rounds = int(max_clarifications)
    except Exception:
        pass

    if max_subagent_clarification_rounds is not None:
        try:
            benchmark.max_subagent_clarification_rounds = int(max_subagent_clarification_rounds)
        except Exception:
            pass

    selected_task_ids: Optional[List[str]] = None
    if task_ids:
        selected_task_ids = list(task_ids)
    elif domain:
        try:
            selected_task_ids = [tid for tid, t in benchmark.tasks.items() if (t.get('domain') == domain or t.get('scenario') == domain or domain == 'travel')]
            if isinstance(selected_task_ids, list) and len(selected_task_ids) == 0:
                selected_task_ids = None
        except Exception:
            selected_task_ids = None

    # Print which tasks will be executed for user visibility
    try:
        print(f"Selected task ids: {selected_task_ids}")
    except Exception:
        pass

    all_reports = []
    for i in range(counts):
        print(f"Starting iteration {i+1}...")
        results = benchmark.run_batch_tasks(task_ids=selected_task_ids, user_profile=user_profile)
        report_i = benchmark.generate_report(results)
        all_reports.append(report_i)

    # try to use benchmark-provided combine if exists
    combined_report = None
    try:
        if hasattr(benchmark, 'generate_combined_report'):
            combined_report = benchmark.generate_combined_report(all_reports)
    except Exception:
        combined_report = None

    if combined_report is None:
        # fallback: take last report or simple wrapper
        combined_report = all_reports[-1] if all_reports else {'summary': {'total_tasks': 0}, 'tasks': []}

    _print_pipeline_usage(domain, combined_report)

    try:
        import sys
        combined_report.setdefault('summary', {})
        exe_name = Path(sys.executable).name
        combined_report['summary']['command'] = f"{exe_name} {' '.join(sys.argv)}"
    except Exception:
        pass

    if report:
        report_file = Path(report)
    else:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = Path(f"output/benchmark_report_{ts}.{out_format}")

    report_file.parent.mkdir(parents=True, exist_ok=True)

    if out_format == 'json':
        report_file.write_text(json.dumps(combined_report, ensure_ascii=False, indent=2), encoding='utf-8')
    else:
        try:
            import yaml
            report_file.write_text(yaml.safe_dump(combined_report, allow_unicode=True), encoding='utf-8')
        except Exception:
            report_file.write_text(json.dumps(combined_report, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f"\nReport saved to: {report_file}")


@click.command()
@click.option('--domain', type=str, default='travel', help='Only run tasks whose domain matches this value (e.g., travel)')
@click.option('--user-profile', type=str, default='profile_001', help='User profile id to simulate')
@click.option('--orchestrator-url', type=str, default='http://10.110.130.149:24003/built-in/v1/agent/routing', help='Orchestrator routing endpoint URL')
@click.option('--config', type=str, default=None, help='Path to tasks config (yaml). Default: datasets/{domain}/tasks.yaml')
@click.option('--task-ids', multiple=True, help='Optional list of task IDs to run (overrides config)')
@click.option('--report', type=click.Path(), help='Output report file path (default: output/benchmark_report_TIMESTAMP.json)')
@click.option('--format', 'out_format', type=click.Choice(['json', 'yaml']), default='json', help='Report output format')
@click.option('--counts', type=int, default=1, help='Number of iterations to run the benchmark')
@click.option('--max-clarifications', type=int, default=24, help='Maximum number of clarification rounds the simulated user will perform (default: 5)')
@click.option('--max-subagent-clarification-rounds', type=int, default=12, help='Maximum number of clarification rounds allowed per sub-agent (optional). If not set, falls back to --max-clarifications')
@click.option('--history-size', type=int, default=10, help='Number of most recent history entries to include in subsequent requests (default: 6)')
@click.option('--debug', is_flag=True, help='Enable debug output for simulator/evaluator processing')
@click.option('--profile-path', type=click.Path(), help='Optional path to `user_profiles.yaml` (overrides config/user_profiles.yaml)')
@click.option('--lang', type=click.Choice(['zh','en']), default='en', help='Language for simulator and requests (zh or en)')
def cli(domain: str,
    user_profile: str,
    orchestrator_url: str,
    config: str,
    task_ids: List[str],
    report: Optional[str],
    out_format: str,
    counts: int,
    max_clarifications: int,
    max_subagent_clarification_rounds: Optional[int],
    history_size: int,
    debug: bool,
    profile_path: Optional[str],
    lang: str
    ):
    if not config:
        default_path = Path(f"datasets/{domain}/tasks.yaml")
        config = str(default_path)
    config_path = Path(config)
    if not config_path.exists():
        raise click.UsageError(f"Task configuration file not found: {config_path}")

    _run_benchmark(domain, user_profile, orchestrator_url, str(config_path), list(task_ids) if task_ids else None, report, out_format, counts, max_clarifications, history_size, debug, profile_path, max_subagent_clarification_rounds, lang)


if __name__ == '__main__':
    cli()