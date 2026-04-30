import sys
from pathlib import Path
try:
    import jinja2
except Exception as e:
    print('JINJA_IMPORT_ERROR', repr(e))
    raise SystemExit(2)

# locate the template
workspace_root = Path(__file__).resolve().parents[2]
alt_tpl = workspace_root / 'benchmark-mutliagent' / 'core' / 'templates' / 'report_template.html'
print('ALT_TPL_EXISTS', alt_tpl.exists(), str(alt_tpl))
try:
    loader = jinja2.FileSystemLoader(str(alt_tpl.parent))
    env = jinja2.Environment(loader=loader)
    template = env.get_template(alt_tpl.name)
    print('TEMPLATE_LOADED_OK')
    # try render with minimal context
    ctx = {
        'date': 'now',
        'command': 'cmd',
        'task_complete_rate': 0.5,
        'task_success_rate': 0.5,
        'overall_score_run': 0.3,
        'total_tasks': 1,
        'tsr_mean': 0.4,
        'average_rounds_per_task': 4.0,
        'avg_rounds_per_task': 4.0,
        'orchestration_latency': 12.3,
        'avg_agent_routing_accuracy': 0.6,
        'avg_clarification_efficiency': 0.55,
        'avg_execution_efficiency': 0.7,
        'avg_orchestration_latency': 12.3,
        'avg_rounds_per_task': 4.0,
        'avg_completeness': 0.9,
        'evaluator_metrics': [],
        'iterations': [],
        'tasks': [],
        'charts_json': '{}'
    }
    r = template.render(**ctx)
    print('RENDER_OK', len(r))
except Exception as e:
    print('TEMPLATE_RENDER_ERROR', repr(e))
    raise SystemExit(3)
