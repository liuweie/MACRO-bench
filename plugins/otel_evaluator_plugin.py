import time
import json
from typing import Dict, List, Any
from pathlib import Path
from .specs import hookimpl
from utils.tracer import BenchmarkOpenTelemetryResult


class OpenTelemetryEvaluator:
    """Evaluator that aggregates metrics from BenchmarkOpenTelemetryResult objects.

    Exposes `aggregate(results)` which expects
    `results` to be a mapping of task_id -> BenchmarkResult-like objects that
    include an `.otel` attribute of type `BenchmarkOpenTelemetryResult`.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    def aggregate(self, results: Dict[str, Any]) -> Dict:
        total = len(results)
        details = {}

        sum_total_rounds = 0
        sum_total_conversations = 0
        sum_routing_time = 0
        success_rates = []
        avg_round_durations = []

        for tid, res in results.items():
            otel = getattr(res, 'otel', None)
            # default values
            trounds = 0
            tconvs = 0
            trouting = 0
            avg_round = None
            sr = 0.0

            if isinstance(otel, BenchmarkOpenTelemetryResult):
                trounds = otel.total_rounds
                tconvs = otel.total_conversations
                trouting = otel.total_routing_time
                avg_round = otel.avg_round_duration
                sr = otel.success_rate

            sum_total_rounds += (trounds or 0)
            sum_total_conversations += (tconvs or 0)
            sum_routing_time += (trouting or 0)
            if avg_round is not None:
                avg_round_durations.append(avg_round)
            success_rates.append(float(sr or 0.0))

            details[tid] = {
                'task_id': getattr(res, 'task_id', tid),
                'level': getattr(res, 'level', None),
                'execution_time': getattr(res, 'execution_time', 0.0),
                'total_interactions': getattr(res, 'total_interactions', 0),
                'final_output': getattr(res, 'final_output', None),
                'evaluation_results': getattr(res, 'evaluation_results', {}) or {},
                'otel_summary': otel.to_dict().get('summary') if isinstance(otel, BenchmarkOpenTelemetryResult) else None
            }

        avg_round_duration = (sum(avg_round_durations) / len(avg_round_durations)) if avg_round_durations else None
        overall_success_rate = (sum(success_rates) / len(success_rates)) if success_rates else 0.0

        report = {
            'summary': {
                'total_tasks': total,
                'otel_aggregate': {
                    'total_rounds': sum_total_rounds,
                    'total_conversations': sum_total_conversations,
                    'total_routing_time': sum_routing_time,
                    'avg_round_duration': avg_round_duration,
                    'average_success_rate': overall_success_rate
                }
            },
            'details': details,
            'tasks': list(details.values())
        }

        return report


@hookimpl
def get_supported_evaluators():
    return ['opentelemetry_evaluator', 'otel_evaluator']


@hookimpl
def create_evaluator(evaluator_type: str, config: Dict[str, Any]):
    if evaluator_type in ('opentelemetry_evaluator', 'otel_evaluator'):
        return OpenTelemetryEvaluator(config=config)
    return None


def register_plugin(pm):
    try:
        import sys
        pm.register(sys.modules[__name__])
    except Exception:
        try:
            pm.register(OpenTelemetryEvaluator())
        except Exception:
            pass
