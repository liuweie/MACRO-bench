import json
import time
from pathlib import Path
from typing import Dict, Any

from .specs import hookimpl


class JsonReporter:
    """Compact JSON reporter used as a universal fallback."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    def generate_report(self, results: Dict[str, Any], output_path: str, **kwargs) -> Dict:
        output_path_obj = Path(output_path)
        if output_path_obj.is_dir():
            ts = time.strftime('%Y%m%d_%H%M%S')
            output_path_obj = output_path_obj / f'benchmark_report_{ts}.json'
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        summary = {
            'total_tasks': len(results or {}),
            'generated_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        details = {}
        for task_id, result in (results or {}).items():
            try:
                details[task_id] = {
                    'task_id': getattr(result, 'task_id', task_id),
                    'level': getattr(result, 'level', None),
                    'query': getattr(result, 'query', None),
                    'execution_time': float(getattr(result, 'execution_time', 0.0) or 0.0),
                    'total_interactions': int(getattr(result, 'total_interactions', 0) or 0),
                    'final_output': getattr(result, 'final_output', None),
                    'evaluation_results': getattr(result, 'evaluation_results', {}) or {},
                }
            except Exception:
                details[task_id] = {'task_id': task_id, 'error': 'failed to serialize result'}

        report = {
            'summary': summary,
            'details': details,
            'meta': {'reporter': 'json_reporter'}
        }

        try:
            output_path_obj.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception:
            output_path_obj.write_text(str(report), encoding='utf-8')

        return report


class JsonReporterPlugin:
    @hookimpl
    def get_supported_reporters(self):
        return ['minimal_json', 'json_reporter']

    @hookimpl
    def create_reporter(self, reporter_type: str, config: Dict[str, Any]):
        reporter_key = str(reporter_type or '').lower()
        domain_key = str((config or {}).get('domain') or '').lower()
        if reporter_key in ('', 'minimal_json', 'json', 'default', 'json_reporter'):
            return JsonReporter(config=config)
        if domain_key in {'travel', 'tourism'} and reporter_key in ('html_reporter', 'html'):
            return JsonReporter(config=config)
        return None


def register_plugin(pm):
    try:
        pm.register(JsonReporterPlugin())
    except Exception:
        pass
