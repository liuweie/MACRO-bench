import json
from pathlib import Path
from typing import Any, Dict, List
from jinja2 import Environment, FileSystemLoader
from datetime import datetime
import re

def generate_html_report(report: dict, output_file: Path, lang: str = None):
    """
    Render the HTML report using a template.

    :param report: The benchmark report data as a dictionary.
    :param output_file: The output file path for the HTML report.
    """
    env = Environment(loader=FileSystemLoader(Path(__file__).parent / 'templates'))
    template = env.get_template('report_template.html')

    # make summary and perf safe
    summary = report.get('summary', {}) if isinstance(report, dict) else {}
    perf = summary.get('performance_metrics', {}) if isinstance(summary, dict) else {}

    # If the report already contains a date use it, otherwise use current time as report generation time
    date_str = summary.get('date') or datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Build a DOM-safe task id for use as element ids in the report (collapse rows)
    def _make_safe_id(val: str) -> str:
        if not val:
            return 'task'
        s = str(val)
        # replace any sequence of non-alnum/_-/ characters with underscore
        return re.sub(r'[^A-Za-z0-9_-]+', '_', s)

    # Normalize iteration tasks and top-level tasks to include `safe_task_id`
    iterations_list = summary.get('iterations', []) or []
    for it in iterations_list:
        for t in it.get('tasks', []) or []:
            orig = t.get('task_id') or t.get('id') or t.get('task_id') or ''
            t['safe_task_id'] = _make_safe_id(orig)

    tasks_list = report.get('tasks', []) or []
    for t in tasks_list:
        orig = t.get('task_id') or t.get('id') or t.get('task_id') or ''
        t['safe_task_id'] = _make_safe_id(orig)

    tasks_data = tasks_list

    # If tasks lack an explicit 'query' field (e.g., older JSON exports), try to recover
    # queries from the local tasks config `config/tasks.yaml` as a fallback.
    try:
        # build a mapping of task_id -> query from config file
        cfg_path = Path('config/tasks.yaml')
        if cfg_path.exists():
            try:
                import yaml as _yaml
                cfg = _yaml.safe_load(cfg_path.read_text(encoding='utf-8')) or {}
                cfg_tasks = cfg.get('tasks', {}) or {}
                # annotate iteration tasks
                for it in iterations_list:
                    for t in it.get('tasks', []) or []:
                        if not t.get('query'):
                            tid = t.get('task_id') or t.get('id')
                            if tid and tid in cfg_tasks:
                                t['query'] = cfg_tasks[tid].get('query', '')
                # annotate top-level tasks
                for t in tasks_list:
                    if not t.get('query'):
                        tid = t.get('task_id') or t.get('id')
                        if tid and tid in cfg_tasks:
                            t['query'] = cfg_tasks[tid].get('query', '')
            except Exception:
                # if yaml not available or parsing fails, ignore and continue
                pass
    except Exception:
        pass

    metric_descriptions = {
        'overall_score': 'Aggregate score derived from milestone success, completion, and efficiency.',
        'task_success_rate': 'Share of tasks where all user-side milestones passed.',
        'task_complete_rate': 'Average milestone completion ratio across tasks.',
        'execution_efficiency': 'Milestone completion normalised by the number of engaged agents.',
    }

    metric_weights = summary.get('metric_weights', {}) if isinstance(summary, dict) else {}
    metric_averages = summary.get('metric_averages', {}) if isinstance(summary, dict) else {}

    metric_definitions = [
        ('overall_score', 'metric_overall_score'),
        ('task_success_rate', 'metric_task_success_rate'),
        ('task_complete_rate', 'metric_task_complete_rate'),
        ('execution_efficiency', 'metric_execution_efficiency'),
    ]

    metric_cards: List[Dict[str, Any]] = []
    for key, label_key in metric_definitions:
        value = perf.get(key)
        if value is None:
            value = metric_averages.get(key)
        if value is None:
            value = _compute_metric_mean(tasks_data, key)
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

    overall_score_value = next((card['value'] for card in metric_cards if card['key'] == 'overall_score'), 0.0)
    task_success_value = next((card['value'] for card in metric_cards if card['key'] == 'task_success_rate'), 0.0)

    # Derived aggregates for charts/backwards compatibility (not shown if data missing)
    avg_rounds_per_task = _compute_metric_mean(tasks_data, 'assistant_rounds')
    avg_agents_per_task = _compute_metric_mean(tasks_data, 'total_agents_used')
    avg_orchestration_latency = _compute_metric_mean(tasks_data, 'orchestration_latency')

    # Determine language: explicit arg wins, otherwise prefer report.summary.lang, fallback to 'zh'
    lang = lang or summary.get('lang') or 'zh'

    # Ensure output directory exists
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    rendered_html = template.render(
        date=date_str,
        command=summary.get('command', ''),
        total_tasks=summary.get('total_tasks', 0),
        overall_score_value=overall_score_value,
        task_success_value=task_success_value,
        metric_cards=metric_cards,
        average_time_per_task=perf.get('average_time_per_task', 0.0),
        average_interactions_per_task=perf.get('average_interactions_per_task', 0.0),
        average_rounds_per_task=perf.get('average_rounds_per_task', avg_rounds_per_task or 0.0),
        average_agents_used=perf.get('average_agents_used', avg_agents_per_task or 0.0),
        overall_score_run=_compute_overall_score_run(report.get('tasks', []) or []),
        tsr_mean=_compute_tsr_mean(report.get('tasks', []) or []),
        tsr_pass_fraction=_compute_tsr_pass_fraction(
            report.get('tasks', []) or [],
            report.get('summary', {}).get('success_threshold', 0.5),
        ),
        iterations=iterations_list,
        tasks=tasks_list,
        evaluator_metrics=metric_cards,
        charts_json=_build_charts_json(iterations_list),
        lang=lang,
    )

    output_file.write_text(rendered_html, encoding='utf-8')

def generate_task_rows(tasks: List[Dict]) -> str:
    """
    Generate HTML rows for the task details table.

    :param tasks: List of task dictionaries.
    :return: HTML string for table rows.
    """
    rows = []
    for task in tasks:
        rows.append(
            f"""
            <tr>
                <td class=\"border px-4 py-2\">{task['id']}</td>
                <td class=\"border px-4 py-2\">{task['status']}</td>
                <td class=\"border px-4 py-2\">{task['execution_time']:.2f}s</td>
                <td class=\"border px-4 py-2 text-sm\">{json.dumps(task['final_output'], ensure_ascii=False)}</td>
            </tr>
            """.strip()
        )
    return '\n'.join(rows)

def generate_iteration_rows(iterations: List[Dict]) -> str:
    """
    Generate HTML rows for the iteration results.

    :param iterations: List of iteration dictionaries.
    :return: HTML string for iteration rows.
    """
    rows = []
    for i, iteration in enumerate(iterations):
        rows.append(
            f"""
            <div class=\"bg-white shadow-md rounded-lg p-4\">
                <h3 class=\"text-lg font-medium\">Iteration {i + 1}</h3>
                <p><strong>Total Tasks:</strong> {iteration['total_tasks']}</p>
                <p><strong>Success Rate:</strong> {iteration['success_rate']:.1%}</p>
                <p><strong>Avg Execution Time:</strong> {iteration['average_time_per_task']:.2f}s</p>
                <p><strong>Avg Interaction Rounds:</strong> {iteration['average_interactions_per_task']:.1f}</p>
            </div>
            """.strip()
        )
    return '\n'.join(rows)


def _build_charts_json(iterations_list: List[Dict]) -> str:
    """Build a JSON-serializable structure for charts used in the static report JS.

    Structure:
    { iteration_index: { labels: [...], scorer: [...], tsr: [...], latency: [...] }, ... }
    """
    import json as _json

    master = ['T1', 'T2', 'T3', 'T4']
    charts = {}
    for it in iterations_list:
        idx = str(it.get('iteration') or iterations_list.index(it) + 1)
        tasks = it.get('tasks') or []
        # observe which levels occur
        observed = []
        for t in tasks:
            lvl = t.get('level')
            if lvl and lvl not in observed:
                observed.append(lvl)

        labels = [lvl for lvl in master if lvl in observed]

        scorer = []
        tsr = []
        latency = []
        for lvl in labels:
            # scorer: average overall_score for this level
            scores = []
            passed = 0
            cnt = 0
            latencies = []
            for t in tasks:
                if t.get('level') == lvl:
                    cnt += 1
                    er = t.get('evaluation_results') or {}
                    try:
                        scores.append(float(er.get('overall_orchestration_score', er.get('overall_score', 0))))
                    except Exception:
                        scores.append(0.0)
                    # success criterion
                    status = t.get('status')
                    success_metric = er.get('user_task_success_rate', er.get('task_success_rate', 0))
                    try:
                        success_metric = float(success_metric)
                    except Exception:
                        success_metric = 0.0
                    if status == 'completed' or success_metric >= 0.5:
                        passed += 1
                    # latency
                    try:
                        latencies.append(float(er.get('orchestration_latency', t.get('execution_time') or 0.0) or 0.0))
                    except Exception:
                        latencies.append(0.0)
            avg_score = (sum(scores) / len(scores)) if scores else 0.0
            scorer.append(avg_score)
            tsr.append((passed / cnt) if cnt else 0.0)
            latency.append((sum(latencies) / len(latencies)) if latencies else 0.0)

        charts[idx] = {
            'labels': labels,
            'scorer': scorer,
            'tsr': tsr,
            'latency': latency
        }

    return _json.dumps(charts, ensure_ascii=False)


def _compute_overall_score_run(tasks: List[Dict]) -> float:
    """Compute average overall_score across tasks. Returns 0.0..1.0."""
    scores = []
    for t in tasks:
        er = t.get('evaluation_results') or {}
        try:
            # prefer new key used by tools/evaluator: overall_orchestration_score
            s = float(er.get('overall_orchestration_score', er.get('overall_score', 0)))
        except Exception:
            s = 0.0
        scores.append(s)
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def _compute_tsr_mean(tasks: List[Dict]) -> float:
    """Compute mean of per-task user-side success rate values (0..1)."""
    vals = []
    for t in tasks:
        er = t.get('evaluation_results') or {}
        try:
            v = float(er.get('user_task_success_rate', er.get('task_success_rate', 0)))
        except Exception:
            v = 0.0
        vals.append(v)
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def _compute_tsr_pass_fraction(tasks: List[Dict], threshold: float = 0.5) -> float:
    """Compute fraction of tasks whose user-side success rate >= threshold."""
    if not tasks:
        return 0.0
    passed = 0
    total = 0
    for t in tasks:
        er = t.get('evaluation_results') or {}
        try:
            v = float(er.get('user_task_success_rate', er.get('task_success_rate', 0)))
        except Exception:
            v = 0.0
        total += 1
        if v >= threshold:
            passed += 1
    return (passed / total) if total else 0.0


def _compute_metric_mean(tasks: List[Dict], metric_name: str) -> float:
    """Compute mean of a numeric metric across tasks' evaluation_results.

    Returns 0.0 if metric not present or no tasks.
    """
    vals = []
    for t in tasks:
        er = t.get('evaluation_results') or {}
        try:
            v = er.get(metric_name)
            if v is None:
                continue
            # accept numeric-like values
            if isinstance(v, (int, float)):
                vals.append(float(v))
            else:
                try:
                    vals.append(float(v))
                except Exception:
                    continue
        except Exception:
            continue
    if not vals:
        return 0.0
    return sum(vals) / len(vals)