import copy
import json
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .specs import hookimpl


class TravelEvaluator:
    """Evaluator that scores individual tasks and aggregates run summaries."""

    SUPPORTED_TYPES = {"travel_generate_report", "travel_aggregator", "travel"}

    _DEFAULT_WEIGHTS: Dict[str, float] = {
        "task_success_rate": 1 / 3,
        "task_complete_rate": 1 / 3,
        "execution_efficiency": 1 / 3,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        custom_weights = self.config.get("metric_weights") or {}
        weights = dict(self._DEFAULT_WEIGHTS)
        for key, value in custom_weights.items():
            try:
                weights[key] = float(value)
            except Exception:
                continue
        total = sum(weights.values()) or 1.0
        normalized = {k: v / total for k, v in weights.items()}
        allowed_keys = {"task_success_rate", "task_complete_rate", "execution_efficiency"}
        self.metric_weights = {k: normalized[k] for k in allowed_keys if k in normalized}
        if len(self.metric_weights) < len(allowed_keys):
            fallback_weight = 1.0 / len(allowed_keys)
            for key in allowed_keys:
                self.metric_weights.setdefault(key, fallback_weight)

    _STOPWORDS: Set[str] = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "this",
        "that",
        "will",
        "have",
        "make",
        "sure",
        "need",
        "about",
        "please",
        "should",
        "ensure",
        "provide",
        "include",
        "用户",
        "需要",
        "提供",
        "确保",
        "一个",
        "一个",
        "以及",
        "已经",
        "需要",
        "完成",
    }

    # --------------------------- helpers ---------------------------
    @staticmethod
    def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
        try:
            return max(lower, min(upper, float(value)))
        except Exception:
            return lower

    @staticmethod
    def _safe_len(items: Optional[Sequence[Any]]) -> int:
        try:
            return len(items) if items is not None else 0
        except Exception:
            return 0

    def _extract_agents(self, internal_logs: Iterable[Any]) -> Set[str]:
        agents: Set[str] = set()
        for entry in internal_logs or []:
            try:
                if isinstance(entry, dict):
                    agent_call = entry.get("agent_call") or entry.get("agentCall")
                    if isinstance(agent_call, dict):
                        name = agent_call.get("name") or agent_call.get("agent")
                        if name:
                            agents.add(str(name).lower())
                    else:
                        name = entry.get("agent") or entry.get("name")
                        if name:
                            agents.add(str(name).lower())
            except Exception:
                continue
        return agents

    def _tokenize_text(self, text: str) -> List[str]:
        if not text:
            return []
        tokens = re.findall(r"[\w\u4e00-\u9fff]+", str(text).lower())
        filtered: List[str] = []
        for token in tokens:
            if len(token) < 2:
                continue
            if token in self._STOPWORDS:
                continue
            filtered.append(token)
        return filtered

    def _evaluate_milestones(
        self,
        milestones: Sequence[Any],
        final_output: Any,
        history: Sequence[Dict[str, Any]],
    ) -> Tuple[int, int, List[Dict[str, Any]]]:
        milestones_list = [m for m in milestones if isinstance(m, str) and m.strip()]
        total = len(milestones_list)
        if total == 0:
            return 0, 0, []

        corpus_parts: List[str] = []
        if isinstance(final_output, str) and final_output.strip():
            corpus_parts.append(final_output)

        for entry in history or []:
            if not isinstance(entry, dict):
                continue
            role = str(entry.get("role") or "").lower()
            if role in {"assistant", "system"}:
                content = entry.get("content") or entry.get("message") or entry.get("text")
                if content:
                    corpus_parts.append(str(content))

        corpus_tokens = set()
        for part in corpus_parts:
            corpus_tokens.update(self._tokenize_text(part))

        milestone_details: List[Dict[str, Any]] = []
        passed = 0

        for milestone in milestones_list:
            milestone_tokens = self._tokenize_text(milestone)
            if not milestone_tokens:
                milestone_details.append(
                    {
                        "milestone": milestone,
                        "passed": False,
                        "matched_tokens": [],
                        "total_tokens": 0,
                    }
                )
                continue

            matched_tokens = [tok for tok in milestone_tokens if tok in corpus_tokens]
            # Require at least 60% of the tokens to appear in the corpus to treat as completed.
            required = max(1, math.ceil(len(milestone_tokens) * 0.6))
            is_passed = len(matched_tokens) >= required
            if is_passed:
                passed += 1

            milestone_details.append(
                {
                    "milestone": milestone,
                    "passed": is_passed,
                    "matched_tokens": matched_tokens,
                    "total_tokens": len(milestone_tokens),
                    "required_tokens": required,
                }
            )

        return passed, total, milestone_details

    def _clarification_score(self, history: Sequence[Dict[str, Any]], max_rounds: int) -> float:
        if not history:
            return 1.0
        clar_rounds = 0
        for item in history:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "").lower()
            status = str(item.get("status") or "").lower()
            content = str(item.get("content") or "")
            if role == "assistant" and ("clarification" in status or content.endswith(("?", "？"))):
                clar_rounds += 1
        allowance = max(1, max_rounds)
        if clar_rounds <= allowance:
            return 1.0
        penalty = (clar_rounds - allowance) / allowance
        return self._clamp(1.0 - penalty)

    def _agent_usage_score(self, expected: Sequence[Any], actual_agents: Set[str]) -> float:
        if not expected:
            # encourage using at least one agent even when expectation missing
            return self._clamp(len(actual_agents) / 3.0)
        expected_set = {str(agent).lower() for agent in expected if agent}
        if not expected_set:
            return self._clamp(len(actual_agents) / 3.0)
        matched = expected_set.intersection(actual_agents)
        return self._clamp(len(matched) / len(expected_set))

    def _decomposition_score(self, task_config: Dict[str, Any], agents_used: Set[str]) -> float:
        milestones = task_config.get("user_side_milestones") or []
        if not milestones:
            return self._clamp(len(agents_used) / 3.0)
        matched = 0
        for milestone in milestones:
            try:
                text = str(milestone).lower()
                if any(agent in text for agent in agents_used):
                    matched += 1
            except Exception:
                continue
        return self._clamp(matched / len(milestones))

    def _execution_efficiency_score(self, total_interactions: int) -> float:
        if total_interactions <= 0:
            return 1.0
        ideal = float(self.config.get("ideal_interactions", 6))
        return self._clamp(ideal / float(total_interactions))

    @staticmethod
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
            return TravelEvaluator._normalize_conversation_history(parsed)

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

    # --------------------------- evaluation ---------------------------
    def evaluate(
        self,
        task_config: Dict[str, Any],
        orchestrator_response: Dict[str, Any],
        world_state: Dict[str, Any],
        debug: bool = False,
    ) -> Dict[str, Any]:
        task_config = task_config or {}
        response = orchestrator_response or {}

        status = str(response.get("status") or "").lower()
        final_output = response.get("final_output")
        history = response.get("clarification_history") or []
        total_interactions = int(response.get("total_interactions") or self._safe_len(history))
        internal_logs = response.get("internal_logs") or []

        agents_used = self._extract_agents(internal_logs)
        expected_agents = task_config.get("expected_subagents") or []
        user_milestones = task_config.get("user_side_milestones") or []

        passed_milestones, total_milestones, milestone_details = self._evaluate_milestones(
            user_milestones,
            final_output,
            history,
        )

        if total_milestones == 0:
            task_complete_rate = 0.0
            task_success_rate = 0.0
        else:
            task_complete_rate = self._clamp(passed_milestones / total_milestones)
            task_success_rate = 1.0 if passed_milestones == total_milestones else 0.0

        agent_count = len(agents_used)
        execution_efficiency = task_complete_rate / max(agent_count, 1)
        execution_efficiency = self._clamp(execution_efficiency)

        overall_components = {
            "task_success_rate": task_success_rate,
            "task_complete_rate": task_complete_rate,
            "execution_efficiency": execution_efficiency,
        }
        weighted_overall = sum(overall_components[key] * self.metric_weights.get(key, 0.0) for key in overall_components)
        if sum(self.metric_weights.values()) == 0:
            weighted_overall = (task_success_rate + task_complete_rate + execution_efficiency) / 3.0
        weighted_overall = self._clamp(weighted_overall)

        assistant_turns = 0
        for entry in history or []:
            if isinstance(entry, dict) and str(entry.get("role") or "").lower() == "assistant":
                assistant_turns += 1

        evaluation: Dict[str, Any] = {
            "overall_score": weighted_overall,
            "overall_orchestration_score": weighted_overall,
            "task_success_rate": task_success_rate,
            "task_complete_rate": task_complete_rate,
            "execution_efficiency": execution_efficiency,
            "passed_milestones": passed_milestones,
            "total_milestones": total_milestones,
            "milestone_details": milestone_details,
            "agents_used": sorted(agents_used),
            "total_agents_used": agent_count,
            "assistant_rounds": assistant_turns,
            "total_interactions": total_interactions,
            "status": status,
        }

        if expected_agents:
            evaluation["expected_agents"] = expected_agents

        evaluation["task_success_passed"] = bool(task_success_rate >= 1.0)

        if debug:
            evaluation["debug_details"] = {
                "agents_used": sorted(agents_used),
                "expected_agents": expected_agents,
                "status": status,
                "total_interactions": total_interactions,
                "metric_weights": self.metric_weights,
                "passed_milestones": passed_milestones,
                "total_milestones": total_milestones,
            }

        return evaluation

    # --------------------------- aggregation ---------------------------
    def aggregate(
        self,
        results: Dict[str, Any],
    ) -> Dict[str, Any]:
        total = len(results)
        times: List[float] = []
        interactions: List[int] = []
        completion_flags: List[int] = []
        per_task_success_rates: List[float] = []
        per_task_complete_rates: List[float] = []
        per_task_efficiency: List[float] = []
        overall_scores: List[float] = []
        level_scores: Dict[str, Dict[str, List[float]]] = {}
        details: Dict[str, Dict[str, Any]] = {}

        metric_collectors: Dict[str, List[float]] = {}

        metric_aliases: Dict[str, List[str]] = {
            "overall_score": ["overall_score", "overall_orchestration_score"],
            "task_success_rate": ["task_success_rate"],
            "task_complete_rate": ["task_complete_rate"],
            "execution_efficiency": ["execution_efficiency", "execution_efficiency_score"],
        }

        metric_order: Tuple[str, ...] = (
            "overall_score",
            "task_success_rate",
            "task_complete_rate",
            "execution_efficiency",
        )

        def _to_float(value: Any) -> Optional[float]:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        def _record_metric(
            level_bucket: Dict[str, Dict[str, List[float]]],
            level_key: str,
            canonical_key: str,
            raw_value: Any,
        ) -> Optional[float]:
            numeric_value = _to_float(raw_value)
            if numeric_value is None:
                return None
            metric_collectors.setdefault(canonical_key, []).append(numeric_value)
            level_bucket.setdefault(level_key, {}).setdefault(canonical_key, []).append(numeric_value)
            return numeric_value

        round_counts: List[int] = []
        agent_counts: List[int] = []
        milestone_pass_counts: List[int] = []
        milestone_total_counts: List[int] = []

        for task_id, res in results.items():
            times.append(getattr(res, "execution_time", 0.0) or 0.0)
            interactions.append(getattr(res, "total_interactions", 0) or 0)

            status = None
            try:
                history = getattr(res, "conversation_history", None)
                if isinstance(history, list) and history:
                    last_entry = history[-1]
                    if isinstance(last_entry, dict):
                        status = last_entry.get("status")
            except Exception:
                status = None

            final_output = getattr(res, "final_output", None)
            success_marker = bool((res.evaluation_results or {}).get("task_success_passed"))
            completion_flags.append(1 if success_marker else 0)

            level = getattr(res, "level", None) or "UNK"

            evals = (res.evaluation_results or {})
            metric_values: Dict[str, float] = {}

            for canonical, candidates in metric_aliases.items():
                for candidate in candidates:
                    if candidate in evals:
                        recorded = _record_metric(level_scores, level, canonical, evals.get(candidate))
                        if recorded is not None:
                            metric_values[canonical] = recorded
                            break

            overall_score = metric_values.get("overall_score", 0.0)
            overall_scores.append(overall_score)

            task_success_rate = metric_values.get("task_success_rate")
            if task_success_rate is None:
                task_success_rate = 0.0
            per_task_success_rates.append(task_success_rate)

            task_complete_rate = metric_values.get("task_complete_rate", 0.0)
            per_task_complete_rates.append(task_complete_rate)

            efficiency_value = metric_values.get("execution_efficiency")
            if efficiency_value is None:
                efficiency_value = 0.0
            per_task_efficiency.append(efficiency_value)

            success_flag = bool(evals.get("task_success_passed") or task_success_rate >= 1.0)

            item_details: Dict[str, Any] = {
                "task_id": getattr(res, "task_id", task_id),
                "id": getattr(res, "task_id", task_id),
                "query": getattr(res, "query", None),
                "level": getattr(res, "level", None),
                "execution_time": getattr(res, "execution_time", 0.0),
                "total_interactions": getattr(res, "total_interactions", 0),
                "final_output": final_output,
                "evaluation_results": evals,
                "status": status,
                "failure_reason": None,
            }

            if "milestone_details" in evals:
                item_details["milestone_details"] = evals.get("milestone_details")
            if "agents_used" in evals:
                item_details["agents_used"] = evals.get("agents_used")
            if "assistant_rounds" in evals:
                item_details["assistant_rounds"] = evals.get("assistant_rounds")

            raw_history = getattr(res, "conversation_history", None)
            item_details["conversation_history"] = self._normalize_conversation_history(raw_history)
            try:
                item_details["conversation_history_json"] = json.dumps(
                    item_details["conversation_history"], ensure_ascii=False, indent=2
                )
            except Exception:
                item_details["conversation_history_json"] = "[]"

            raw_logs = getattr(res, "orchestrator_logs", None)
            if raw_logs is None:
                item_details["orchestrator_logs"] = []
            else:
                try:
                    item_details["orchestrator_logs"] = copy.deepcopy(list(raw_logs))
                except Exception:
                    try:
                        item_details["orchestrator_logs"] = list(raw_logs)
                    except Exception:
                        item_details["orchestrator_logs"] = []

            try:
                item_details["clarification_pairs"] = copy.deepcopy(
                    getattr(res, "clarification_pairs", []) or []
                )
            except Exception:
                item_details["clarification_pairs"] = []

            try:
                iteration_idx = getattr(res, "iteration", None)
                if iteration_idx is None:
                    iteration_idx = getattr(res, "iteration_index", None)
                if iteration_idx is not None:
                    item_details["iteration"] = iteration_idx
            except Exception:
                pass

            key_scores: Dict[str, float] = {}
            for canonical in metric_order:
                value = metric_values.get(canonical)
                if value is not None:
                    key_scores[canonical] = value
            if "overall_score" in metric_values:
                key_scores.setdefault("overall_orchestration_score", metric_values["overall_score"])
            item_details["key_scores"] = key_scores

            if not success_flag:
                reason = None
                if total_milestones := evals.get("total_milestones"):
                    passed_count = evals.get("passed_milestones", 0)
                    reason = f"milestones {passed_count}/{total_milestones}"
                elif status and status != "completed":
                    reason = f"status: {status}"
                elif not final_output:
                    reason = "no final output"
                else:
                    reason = "failed"
                item_details["failure_reason"] = reason

            details[task_id] = item_details

            try:
                agent_counts.append(int(evals.get("total_agents_used", len(evals.get("agents_used", [])) or 0)))
            except Exception:
                agent_counts.append(len(evals.get("agents_used") or []))

            milestone_pass_counts.append(int(evals.get("passed_milestones", 0) or 0))
            milestone_total_counts.append(int(evals.get("total_milestones", 0) or 0))

            assistant_rounds = evals.get("assistant_rounds")
            if assistant_rounds is None:
                assistant_rounds = sum(1 for entry in (raw_history or []) if isinstance(entry, dict) and str(entry.get("role") or "").lower() == "assistant")
            round_counts.append(int(assistant_rounds or 0))

        avg_time = sum(times) / total if total else 0.0
        avg_interactions = sum(interactions) / total if total else 0.0
        task_completion_rate = (sum(completion_flags) / total) if total else 0.0
        avg_task_success_rate = (
            sum(per_task_success_rates) / len(per_task_success_rates) if per_task_success_rates else 0.0
        )
        avg_task_complete_rate = (
            sum(per_task_complete_rates) / len(per_task_complete_rates) if per_task_complete_rates else 0.0
        )
        avg_execution_efficiency = (
            sum(per_task_efficiency) / len(per_task_efficiency) if per_task_efficiency else 0.0
        )

        score_distribution = {
            "excellent": 0,
            "good": 0,
            "fair": 0,
            "poor": 0,
        }
        for overall_score in overall_scores:
            if overall_score >= 0.9:
                score_distribution["excellent"] += 1
            elif overall_score >= 0.7:
                score_distribution["good"] += 1
            elif overall_score >= 0.5:
                score_distribution["fair"] += 1
            else:
                score_distribution["poor"] += 1

        metric_means: Dict[str, float] = {
            key: (sum(values) / len(values)) for key, values in metric_collectors.items() if values
        }
        if "overall_score" in metric_means and "overall_orchestration_score" not in metric_means:
            metric_means["overall_orchestration_score"] = metric_means["overall_score"]

        average_scores: Dict[str, Dict[str, float]] = {}
        for level, metrics in level_scores.items():
            averages_for_level: Dict[str, float] = {}
            for metric_key, values in metrics.items():
                if values:
                    averages_for_level[metric_key] = sum(values) / len(values)
            average_scores[level] = averages_for_level

        performance_metrics: Dict[str, float] = {}
        for key in metric_order:
            if key in metric_means:
                performance_metrics[key] = metric_means[key]
        performance_metrics["task_success_rate"] = performance_metrics.get(
            "task_success_rate", avg_task_success_rate
        )
        performance_metrics["task_complete_rate"] = performance_metrics.get(
            "task_complete_rate", avg_task_complete_rate
        )
        performance_metrics["execution_efficiency"] = performance_metrics.get(
            "execution_efficiency", avg_execution_efficiency
        )
        performance_metrics["overall_score"] = performance_metrics.get("overall_score", 0.0)
        performance_metrics["average_time_per_task"] = avg_time
        performance_metrics["average_interactions_per_task"] = avg_interactions
        performance_metrics["average_rounds_per_task"] = (
            sum(round_counts) / len(round_counts) if round_counts else 0.0
        )
        performance_metrics["average_agents_used"] = (
            sum(agent_counts) / len(agent_counts) if agent_counts else 0.0
        )

        summary = {
            "total_tasks": total,
            "performance_metrics": performance_metrics,
            "average_scores": average_scores,
            "metric_weights": dict(self.metric_weights),
            "metric_averages": metric_means,
            "success_rate": (score_distribution["excellent"] + score_distribution["good"]) / total if total else 0.0,
            "score_distribution": score_distribution,
            "task_completion_rate": task_completion_rate,
            "task_success_rate": avg_task_success_rate,
            "task_complete_rate": avg_task_complete_rate,
            "execution_efficiency": avg_execution_efficiency,
            "milestone_summary": {
                "passed": sum(milestone_pass_counts),
                "total": sum(milestone_total_counts),
            },
        }

        return {
            "summary": summary,
            "details": details,
            "tasks": list(details.values()),
        }


class TravelEvaluatorPlugin:
    """Plug-in exposing travel evaluator and aggregation hooks."""

    SUPPORTED_TYPES = TravelEvaluator.SUPPORTED_TYPES

    @hookimpl
    def get_supported_evaluators(self):
        return list(self.SUPPORTED_TYPES)

    @hookimpl
    def create_evaluator(self, evaluator_type: str, config: Dict[str, Any]):
        evaluator_key = str(evaluator_type or '').lower()
        domain_key = str((config or {}).get('domain') or '').lower()
        if evaluator_type in self.SUPPORTED_TYPES:
            return TravelEvaluator(config=config)
        fallback_types = {'', 'default', 'core', 'opentelemetry_evaluator', 'otel_evaluator'}
        if domain_key in {'travel', 'tourism'} and evaluator_key in fallback_types:
            return TravelEvaluator(config=config)
        return None

    @hookimpl
    def evaluate_task(
        self,
        evaluator: Any,
        task_config: Dict[str, Any],
        orchestrator_response: Dict[str, Any],
        world_state: Dict[str, Any],
        debug: bool,
    ):
        if hasattr(evaluator, "evaluate"):
            return evaluator.evaluate(task_config, orchestrator_response, world_state, debug)
        return None


def register_plugin(pm):
    pm.register(TravelEvaluatorPlugin())
