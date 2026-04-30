import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence
import shutil

from .specs import hookimpl
from .invocation_logger import log_invocation


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str) and not value.strip():
            return default
        return float(value)
    except Exception:
        return default


def _mean(values: Sequence[float]) -> float:
    filtered = [v for v in values if v is not None]
    if not filtered:
        return 0.0
    return sum(filtered) / len(filtered)


def _make_safe_id(value: Any) -> str:
    if value in (None, ""):
        return "task"
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(value))


def _normalize_conversation_history(history: Any) -> List[Dict[str, Any]]:
    if not history:
        return []

    if isinstance(history, str):
        try:
            parsed = json.loads(history)
        except Exception:
            return [
                {
                    "role": "unknown",
                    "content": history,
                    "type": "",
                    "timestamp": "",
                }
            ]
        return _normalize_conversation_history(parsed)

    if isinstance(history, list):
        normalized: List[Dict[str, Any]] = []
        for entry in history:
            if isinstance(entry, dict):
                normalized.append(
                    {
                        "role": entry.get("role") or entry.get("speaker") or "unknown",
                        "content": entry.get("content")
                        or entry.get("message")
                        or entry.get("text")
                        or "",
                        "type": entry.get("type")
                        or entry.get("message_type")
                        or entry.get("kind")
                        or "",
                        "timestamp": entry.get("timestamp")
                        or entry.get("time")
                        or "",
                    }
                )
            else:
                normalized.append(
                    {
                        "role": "unknown",
                        "content": str(entry),
                        "type": "",
                        "timestamp": "",
                    }
                )
        return normalized

    return [
        {
            "role": "unknown",
            "content": str(history),
            "type": "",
            "timestamp": "",
        }
    ]


def _collect_metric_mean(tasks: Sequence[Dict[str, Any]], metric_name: str) -> float:
    values: List[float] = []
    for task in tasks:
        er = task.get("evaluation_results") or {}
        if er.get(metric_name) is None:
            continue
        try:
            values.append(float(er.get(metric_name)))
        except Exception:
            continue
    return _mean(values)


def _build_charts_payload(iterations: Sequence[Dict[str, Any]]) -> str:
    charts: Dict[str, Dict[str, List[float]]] = {}
    preferred_levels = ["T1", "T2", "T3", "T4"]

    for idx, iteration in enumerate(iterations, start=1):
        tasks = iteration.get("tasks") or []
        observed_levels: List[str] = []
        for task in tasks:
            level = task.get("level")
            if level and level not in observed_levels:
                observed_levels.append(level)

        labels: List[str] = [lvl for lvl in preferred_levels if lvl in observed_levels]
        for lvl in observed_levels:
            if lvl not in labels:
                labels.append(lvl)

        scorer: List[float] = []
        tsr: List[float] = []
        latency: List[float] = []

        for lvl in labels:
            level_tasks = [task for task in tasks if task.get("level") == lvl]
            if not level_tasks:
                scorer.append(0.0)
                tsr.append(0.0)
                latency.append(0.0)
                continue

            overall_scores = [
                _safe_float((task.get("evaluation_results") or {}).get("overall_orchestration_score", (task.get("evaluation_results") or {}).get("overall_score")), 0.0)
                for task in level_tasks
            ]
            latency_vals = [
                _safe_float((task.get("evaluation_results") or {}).get("orchestration_latency", task.get("execution_time")), 0.0)
                for task in level_tasks
            ]

            pass_count = 0
            for task, score in zip(level_tasks, overall_scores):
                status = str(task.get("status") or "").lower()
                success_metric = _safe_float((task.get("evaluation_results") or {}).get("user_task_success_rate", (task.get("evaluation_results") or {}).get("task_success_rate")), 0.0)
                if status == "completed" or (success_metric is not None and success_metric >= 0.5):
                    pass_count += 1

            count = len(level_tasks)
            scorer.append(_mean(overall_scores))
            tsr.append((pass_count / count) if count else 0.0)
            latency.append(_mean(latency_vals))

        charts[str(iteration.get("iteration") or idx)] = {
            "labels": labels,
            "scorer": scorer,
            "tsr": tsr,
            "latency": latency,
        }

    return json.dumps(charts, ensure_ascii=False)


class HTMLReporter:
    """HTML reporter that renders a simple report using Jinja2 when
    available; otherwise emits a basic HTML page.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    def _render_simple_html(self, report: Dict[str, Any]) -> str:
        # Very small inline HTML if Jinja2 not available
        title = f"Benchmark Report - {report.get('summary', {}).get('total_tasks', 0)} tasks"
        lines = [f"<h1>{title}</h1>", f"<p>Generated at: {report.get('summary', {}).get('generated_at', '')}</p>"]
        lines.append('<ul>')
        for tid, d in (report.get('details') or {}).items():
            lines.append(f"<li><strong>{tid}</strong>: score={d.get('evaluation_results', {}).get('overall_score')} interactions={d.get('total_interactions')}</li>")
        lines.append('</ul>')
        return '<html><body>' + '\n'.join(lines) + '</body></html>'

    def _render_full_html_fallback(self, report: Dict[str, Any]) -> str:
        # A richer fallback renderer used when Jinja2 is not available.
        # Produces a readable HTML with header, basic stats and a tasks table,
        # and links to static assets so the page still looks decent.
        summary = report.get('summary', {}) if isinstance(report, dict) else {}
        details = report.get('details') or {}

        title = f"Benchmark Report - {summary.get('total_tasks', len(details))} tasks"
        date = summary.get('generated_at', time.strftime('%Y-%m-%d %H:%M:%S'))

        parts: List[str] = []
        parts.append('<!doctype html>')
        parts.append('<html><head>')
        parts.append(f'<meta charset="utf-8"><title>{title}</title>')
        parts.append('<link rel="stylesheet" href="./static/report.css">')
        parts.append('<script src="./static/chart.umd.min.js"></script>')
        parts.append('</head><body>')
        parts.append(f'<header><h1>{title}</h1><p>Generated at: {date}</p></header>')

        # Top summary cards
        try:
            perf = summary.get('performance_metrics', {}) if isinstance(summary, dict) else {}
            task_complete_rate = _safe_float(perf.get('user_task_completion_rate', perf.get('task_completion_rate', 0.0)), 0.0)
            task_success_rate = _safe_float(perf.get('user_task_success_rate', perf.get('task_success_rate', 0.0)), 0.0)
        except Exception:
            task_complete_rate = 0.0
            task_success_rate = 0.0

        parts.append('<section>')
        parts.append('<div>')
        parts.append(f'<div><strong>User Task Completion Rate:</strong> {task_complete_rate*100:.2f}%</div>')
        parts.append(f'<div><strong>User Task Success Rate:</strong> {task_success_rate*100:.2f}%</div>')
        parts.append('</div>')
        parts.append('</section>')

        # Tasks table
        parts.append('<section><h2>Tasks</h2>')
        parts.append('<table border="1" cellpadding="6" cellspacing="0">')
        parts.append('<thead><tr><th>Task ID</th><th>Query</th><th>Level</th><th>Exec Time</th><th>Overall Score</th></tr></thead>')
        parts.append('<tbody>')
        if isinstance(details, dict):
            for tid, d in details.items():
                er = d.get('evaluation_results') or {}
                score = er.get('overall_orchestration_score', er.get('overall_score', ''))
                parts.append('<tr>')
                parts.append(f'<td>{tid}</td>')
                parts.append(f'<td>{(d.get("query") or "").replace("<","&lt;").replace(">","&gt;")}</td>')
                parts.append(f'<td>{d.get("level","")}</td>')
                parts.append(f'<td>{d.get("execution_time","")}</td>')
                parts.append(f'<td>{score}</td>')
                parts.append('</tr>')
        parts.append('</tbody></table></section>')

        parts.append('<footer style="margin-top:20px;color:#666;font-size:12px">Generated by HTMLReporter fallback (jinja2 not available)</footer>')
        parts.append('</body></html>')
        return '\n'.join(parts)

    def generate_report(self, results: Dict[str, Any], output_path: str, **kwargs) -> Dict:
        # Build a report dict similar to travel_generate_report's output if results are BenchmarkResult objects
        # debug: write what keys the incoming results contains
        try:
            dbg = Path('output') / 'report_incoming_debug.json'
            dbg.parent.mkdir(parents=True, exist_ok=True)
            dbg.write_text(json.dumps({'keys': list(results.keys()) if isinstance(results, dict) else str(type(results)), 'is_dict': isinstance(results, dict), 'has_summary': isinstance(results, dict) and 'summary' in results, 'has_details': isinstance(results, dict) and 'details' in results, 'has_tasks': isinstance(results, dict) and 'tasks' in results}, ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception:
            pass

        try:
            # If results already look like a dict report (has 'summary' and either 'details' or 'tasks'), use as-is
            if isinstance(results, dict) and 'summary' in results and ('details' in results or 'tasks' in results):
                report = results
            else:
                # Try to reuse a minimal JSON structure like minimal_reporter
                summary = {
                    'total_tasks': len(results or {}),
                    'generated_at': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                details = {}
                for tid, res in (results or {}).items():
                    details[tid] = {
                        'task_id': getattr(res, 'task_id', tid),
                        'level': getattr(res, 'level', None),
                        'query': getattr(res, 'query', None),
                        'execution_time': float(getattr(res, 'execution_time', 0.0) or 0.0),
                        'total_interactions': int(getattr(res, 'total_interactions', 0) or 0),
                        'final_output': getattr(res, 'final_output', None),
                        'evaluation_results': getattr(res, 'evaluation_results', {}) or {},
                    }
                report = {'summary': summary, 'details': details}
        except Exception:
            report = {'summary': {'total_tasks': 0, 'generated_at': time.strftime('%Y-%m-%d %H:%M:%S')}, 'details': {}}

        # Default output path handling: ensure we produce an HTML file. If caller
        # passed a .json path (benchmark default), replace suffix with .html to
        # keep both JSON and HTML alongside each other.
        outp = Path(output_path)
        if outp.is_dir():
            ts = time.strftime('%Y%m%d_%H%M%S')
            html_out = outp / f'benchmark_report_{ts}.html'
        else:
            if outp.suffix.lower() != '.html':
                html_out = outp.with_suffix('.html')
            else:
                html_out = outp
        html_out.parent.mkdir(parents=True, exist_ok=True)

        # Try to render with jinja2 if available and a template exists under templates/
        try:
            import jinja2
            # Candidate template locations, prefer package-local then workspace benchmark-mutliagent
            pkg_tpl = Path(__file__).parent.parent / 'reporters' / 'templates' / 'report_template.html'

            tpl_path = None
            tpl_loader_path = None
            if pkg_tpl.exists():
                tpl_path = pkg_tpl
                tpl_loader_path = pkg_tpl.parent
            else:
                raise FileNotFoundError("No package template found")

            if tpl_path and tpl_loader_path:
                loader = jinja2.FileSystemLoader(str(tpl_loader_path))
                env = jinja2.Environment(loader=loader)
                template = env.get_template(tpl_path.name)

                aggregated = report

                # Build template context from aggregated report
                try:
                    summary = aggregated.get('summary', {})
                    details = aggregated.get('details')
                    tasks_list: List[Dict[str, Any]] = []

                    if details and isinstance(details, dict):
                        for tid, d in (details or {}).items():
                            task = dict(d)
                            task_id = task.get('task_id') or tid
                            task['task_id'] = task_id
                            task['id'] = task_id
                            tasks_list.append(task)
                    else:
                        alt_tasks = aggregated.get('tasks') or []
                        for t in (alt_tasks or []):
                            task = dict(t or {})
                            task_id = task.get('task_id') or task.get('id') or ''
                            task['task_id'] = task_id
                            task['id'] = task_id
                            tasks_list.append(task)

                    for task in tasks_list:
                        task_id = task.get('task_id') or task.get('id')
                        task['safe_task_id'] = _make_safe_id(task_id)

                        normalized_history = _normalize_conversation_history(task.get('conversation_history'))
                        task['conversation_history'] = normalized_history
                        if task.get('conversation_history_json'):
                            if isinstance(task['conversation_history_json'], list):
                                try:
                                    task['conversation_history_json'] = json.dumps(
                                        task['conversation_history_json'], ensure_ascii=False, indent=2
                                    )
                                except Exception:
                                    task['conversation_history_json'] = json.dumps([], ensure_ascii=False)
                        else:
                            try:
                                task['conversation_history_json'] = json.dumps(
                                    normalized_history, ensure_ascii=False, indent=2
                                )
                            except Exception:
                                task['conversation_history_json'] = json.dumps([], ensure_ascii=False)

                        if not isinstance(task.get('evaluation_results'), dict):
                            task['evaluation_results'] = {}

                        task['execution_time'] = _safe_float(task.get('execution_time'), 0.0)
                        try:
                            task['total_interactions'] = int(task.get('total_interactions') or 0)
                        except Exception:
                            task['total_interactions'] = 0

                        if not task.get('status'):
                            task['status'] = 'unknown'

                        if task.get('iteration') is None:
                            task['iteration'] = 1

                    perf = summary.get('performance_metrics', {}) if isinstance(summary, dict) else {}
                    metric_descriptions = {
                        'overall_score': 'Aggregate score derived from task success, completion, and efficiency.',
                        'task_success_rate': 'Share of tasks where all user milestones were satisfied.',
                        'task_complete_rate': 'Average completion ratio across all user milestones.',
                        'execution_efficiency': 'Milestone completion normalised by the number of agents engaged.',
                    }

                    metric_definitions = [
                        ('overall_score', 'metric_overall_score'),
                        ('task_success_rate', 'metric_task_success_rate'),
                        ('task_complete_rate', 'metric_task_complete_rate'),
                        ('execution_efficiency', 'metric_execution_efficiency'),
                    ]

                    metric_weights = summary.get('metric_weights', {}) if isinstance(summary, dict) else {}
                    metric_averages = summary.get('metric_averages', {}) if isinstance(summary, dict) else {}

                    metric_cards = []
                    for key, label_key in metric_definitions:
                        value = perf.get(key)
                        if value is None:
                            value = metric_averages.get(key)
                        if value is None:
                            value = _collect_metric_mean(tasks_list, key)
                        if value is None:
                            continue
                        metric_cards.append({
                            'key': key,
                            'label_key': label_key,
                            'name': label_key,
                            'value': float(value),
                            'weight': metric_weights.get(key),
                            'description': metric_descriptions.get(key, ''),
                        })

                    overall_score_run = _collect_metric_mean(tasks_list, 'overall_orchestration_score')
                    if overall_score_run is None or overall_score_run == 0:
                        overall_score_run = next((card['value'] for card in metric_cards if card['key'] == 'overall_score'), 0.0)
                    tsr_mean = _collect_metric_mean(tasks_list, 'task_success_rate')
                    overall_score_value = next((card['value'] for card in metric_cards if card['key'] == 'overall_score'), 0.0)
                    task_success_value = next((card['value'] for card in metric_cards if card['key'] == 'task_success_rate'), 0.0)
                    average_rounds_value = _safe_float(perf.get('average_rounds_per_task', _collect_metric_mean(tasks_list, 'assistant_rounds')), 0.0)
                    average_agents_value = _safe_float(perf.get('average_agents_used', _collect_metric_mean(tasks_list, 'total_agents_used')), 0.0)

                    iteration_tasks = [dict(task) for task in tasks_list]
                    iterations = [
                        {
                            'iteration': 1,
                            'total_tasks': summary.get('total_tasks', len(tasks_list)),
                            'success_rate': task_success_value,
                            'average_time_per_task': _safe_float(perf.get('average_time_per_task', 0.0), 0.0),
                            'average_interactions_per_task': _safe_float(perf.get('average_interactions_per_task', 0.0), 0.0),
                            'average_rounds_per_task': average_rounds_value,
                            'orchestration_latency': _safe_float(_collect_metric_mean(tasks_list, 'orchestration_latency'), 0.0),
                            'tasks': iteration_tasks,
                        }
                    ]

                    charts_json = _build_charts_payload(iterations)

                    lang = summary.get('lang') or self.config.get('lang') or 'en'

                    context = {
                        'date': summary.get('generated_at') or time.strftime('%Y-%m-%d %H:%M:%S'),
                        'command': summary.get('command'),
                        'overall_score_value': overall_score_value,
                        'task_success_value': task_success_value,
                        'overall_score_run': overall_score_run,
                        'total_tasks': summary.get('total_tasks', len(tasks_list)),
                        'tsr_mean': tsr_mean,
                        'average_interactions_per_task': _safe_float(perf.get('average_interactions_per_task', 0.0), 0.0),
                        'average_time_per_task': _safe_float(perf.get('average_time_per_task', 0.0), 0.0),
                        'average_rounds_per_task': average_rounds_value,
                        'average_agents_used': average_agents_value,
                        'metric_cards': metric_cards,
                        'evaluator_metrics': metric_cards,
                        'iterations': iterations,
                        'tasks': tasks_list,
                        'charts_json': charts_json,
                        'lang': lang,
                    }

                    html = template.render(**context)
                except Exception:
                    # fallback to simple render if context construction fails
                    html = template.render(report=report)
            else:
                # no template found in either location; fallback to simple HTML
                html = self._render_simple_html(report)
        except Exception:
            # No jinja2 or template issues: produce a fuller fallback HTML
            html = self._render_full_html_fallback(report)


        # Perform writes and static copy inside a guarded block that records
        # debug information and any exceptions to an on-disk debug log as well
        # as to the invocation logger. This makes failures visible when the
        # caller might swallow exceptions.
        debug_info = {
            'html_path': str(html_out),
            'json_path': str(html_out.with_suffix('.json')),
            'static_src': None,
            'static_dest': None,
            'html_written': False,
            'json_written': False,
            'static_copied': False,
            'errors': []
        }

        try:
            try:
                html_out.write_text(html, encoding='utf-8')
                debug_info['html_written'] = True
            except Exception as e_html:
                # fallback attempt: write string representation
                try:
                    html_out.write_text(str(html), encoding='utf-8')
                    debug_info['html_written'] = True
                except Exception as e_html2:
                    debug_info['errors'].append({'stage': 'write_html', 'error': repr(e_html), 'fallback_error': repr(e_html2)})

            # also write a JSON copy for downstream consumption (same basename)
            try:
                json_path = html_out.with_suffix('.json')
                json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
                debug_info['json_written'] = True
            except Exception as e_json:
                debug_info['errors'].append({'stage': 'write_json', 'error': repr(e_json)})

            # copy static assets (js/css) from package reporters/static into the report output directory
            try:
                src_static = Path(__file__).parent.parent / 'reporters' / 'static'
                dest_static = html_out.parent / 'static'
                debug_info['static_src'] = str(src_static)
                debug_info['static_dest'] = str(dest_static)
                if src_static.exists() and src_static.is_dir():
                    try:
                        shutil.copytree(src_static, dest_static, dirs_exist_ok=True)
                        debug_info['static_copied'] = True
                    except TypeError:
                        # older Python: fallback to manual copy
                        dest_static.mkdir(parents=True, exist_ok=True)
                        for item in src_static.iterdir():
                            try:
                                if item.is_dir():
                                    shutil.copytree(item, dest_static / item.name)
                                else:
                                    shutil.copy2(item, dest_static / item.name)
                            except Exception:
                                continue
                        debug_info['static_copied'] = True
                else:
                    # no static assets found; not an error
                    debug_info['static_copied'] = False
            except Exception as e_static:
                debug_info['errors'].append({'stage': 'copy_static', 'error': repr(e_static)})

            # Always try to log invocation with collected debug info
            try:
                log_invocation({'hook': 'generate_report', 'plugin': 'html_reporter_plugin', **debug_info, 'message': 'HTMLReporter finished write/copy steps'}, task_id=None)
            except Exception:
                pass

        except Exception as e:
            # Last-resort catch-all: write debug file and log
            debug_info['errors'].append({'stage': 'guarded_block', 'error': repr(e)})
            try:
                log_invocation({'hook': 'generate_report', 'plugin': 'html_reporter_plugin', **debug_info, 'message': 'HTMLReporter encountered an exception in guarded block'}, task_id=None)
            except Exception:
                pass

        # Additionally, persist a small debug file next to the report so errors
        # are easy to inspect without diving into system logs.
        try:
            dbg_path = html_out.with_suffix('.debug.json')
            dbg_path.write_text(json.dumps(debug_info, ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception:
            pass

        # Return the report dict
        return report


class HTMLReporterPlugin:
    @hookimpl
    def get_supported_reporters(self):
        return ['html_reporter']

    @hookimpl
    def create_reporter(self, reporter_type: str, config: Dict[str, Any]):
        if reporter_type in ('html_reporter', 'minimal_html', 'html'):
            return HTMLReporter(config=config)
        return None


def register_plugin(pm):
    try:
        pm.register(HTMLReporterPlugin())
    except Exception:
        pass
