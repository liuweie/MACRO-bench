import ast
import json
import time
import uuid
from typing import Dict, List, Any, Tuple, Optional, Generator
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from datasets.llm_client import LLMClient
import re
from plugins.manager import global_plugin_manager
from utils.logger import BenchmarkLogger
from plugins.invocation_logger import log_invocation
from utils.tracer import BenchmarkOpenTelemetryResult, RoundResult, Span, Conversation
import utils.console as console_utils
from utils.console import color_text, FG_GREEN, FG_YELLOW, FG_RED, FG_MAGENTA, print_info, print_warning, print_error, print_conv_entry


def _safe_serialize(obj, _depth=3):
    if _depth <= 0:
        try:
            return str(obj)
        except Exception:
            return '<unserializable>'

    if obj is None or isinstance(obj, (bool, int, float, str)):
        if isinstance(obj, str) and len(obj) > 2000:
            return obj[:2000] + '...'
        return obj

    if isinstance(obj, bytes):
        try:
            return obj.decode('utf-8', errors='replace')[:2000]
        except Exception:
            return repr(obj)[:2000]

    if isinstance(obj, dict):
        out = {}
        for k, v in list(obj.items())[:200]:
            try:
                ks = k if isinstance(k, str) else str(k)
            except Exception:
                ks = '<key>'
            out[ks] = _safe_serialize(v, _depth=_depth-1)
        return out

    if isinstance(obj, (list, tuple, set)):
        out = []
        for i, v in enumerate(list(obj)):
            if i >= 200:
                break
            out.append(_safe_serialize(v, _depth=_depth-1))
        return out

    try:
        return str(obj)[:2000]
    except Exception:
        return '<unserializable>'


class TaskCompletionJudge:
    def __init__(self, llm_config: Dict):
        self.llm_config = llm_config
        self.judge_prompts = self._initialize_judge_prompts()
        try:
            self._llm_client = LLMClient(self.llm_config)
        except Exception:
            self._llm_client = LLMClient.from_env()
    
    def _initialize_judge_prompts(self) -> Dict[str, str]:
        return {
            "completion_judge": """Evaluate task completion:
Original task: {original_query}
Task level: {task_level}
Conversation history: {conversation_history}
Final output: {final_output}

Return JSON:
{{
    "is_genuinely_completed": true/false,
    "completion_score": 0-100,
    "reasoning": "evaluation rationale",
    "missing_elements": ["missing elements"],
    "quality_assessment": {{
        "requirement_satisfaction": 0-100,
        "completeness": 0-100,
        "accuracy": 0-100,
        "actionability": 0-100
    }}
}}""",

            "clarification_need_judge": """Decide whether further clarification is needed:
Original task: {original_query}
Current conversation history: {conversation_history}
Latest response: {orchestrator_response}

Return JSON:
{{
    "needs_more_clarification": true/false,
    "missing_information": ["information items still needing clarification"],
    "readiness_assessment": "readiness assessment",
    "suggested_clarification_questions": ["suggested clarification questions"]
}}"""
        }
    
    def judge_completion(self, task_config: Dict, conversation_history: List[Dict], final_output: Any) -> Dict:
        prompt = self.judge_prompts["completion_judge"].format(
                original_query=task_config['query'],
            task_level=task_config['level'],
            conversation_history=json.dumps(conversation_history[-6:], ensure_ascii=False, indent=2),
            final_output=str(final_output)[:2000]
        )
        
        try:
            judgment = self._call_judge_llm(prompt)
            return self._parse_judgment(judgment)
        except Exception as e:
            return self._fallback_judgment(task_config, final_output)
    
    def assess_clarification_needs(self, task_config: Dict, conversation_history: List[Dict], orchestrator_response: Dict) -> Dict:
        prompt = self.judge_prompts["clarification_need_judge"].format(
                original_query=task_config['query'],
            conversation_history=json.dumps(conversation_history[-6:], ensure_ascii=False, indent=2),
            orchestrator_response=json.dumps({
                'status': orchestrator_response.get('status'),
                'final_output': str(orchestrator_response.get('final_output'))[:1000],
                'clarification_question': orchestrator_response.get('clarification_question')
            }, ensure_ascii=False)
        )
        
        try:
            judgment = self._call_judge_llm(prompt)
            return self._parse_judgment(judgment)
        except Exception:
            return {"needs_more_clarification": True}
    
    def _call_judge_llm(self, prompt: str) -> str:
        try:
            return self._llm_client.call_llm(prompt)
        except Exception as e:
            raise
    
    def _parse_judgment(self, judgment_text: str) -> Dict:
        try:
            parsed = self._llm_client.parse_llm_response(judgment_text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        try:
            start_idx = judgment_text.find('{')
            end_idx = judgment_text.rfind('}') + 1
            if start_idx == -1 or end_idx == 0 or end_idx <= start_idx:
                raise ValueError("No JSON found in judgment")
            json_str = judgment_text[start_idx:end_idx]
            return json.loads(json_str)
        except Exception:
            try:
                return self._llm_client.parse_fallback(judgment_text)
            except Exception:
                return {"error": "Failed to parse judgment"}
    
    def _fallback_judgment(self, task_config: Dict, final_output: Any) -> Dict:
        if not final_output:
            return {
                "is_genuinely_completed": False,
                "completion_score": 0,
                "reasoning": "No output provided"
            }
        
        output_str = str(final_output).lower()
        task_level = task_config['level']
        
        if task_level == "T1":
            if any(tool.replace('_', ' ') in output_str for tool in task_config.get('expected_subagents', [])):
                return {"is_genuinely_completed": True, "completion_score": 70}
        
        elif task_level == "T2":
            if any(seq_indicator in output_str for seq_indicator in ['then', 'next', 'after that']):
                return {"is_genuinely_completed": True, "completion_score": 60}
        
        elif task_level in ["T3", "T4"]:
            required_elements = ['itinerary', 'arrangement', 'plan', 'recommendation', 'budget']
            found_elements = sum(1 for elem in required_elements if elem in output_str)
            if found_elements >= 2:
                return {"is_genuinely_completed": True, "completion_score": 50}
        
        return {"is_genuinely_completed": False, "completion_score": 30}


class RuleBasedJudge:
    def __init__(self, llm_config: Dict = None):
        try:
            self._delegate = TaskCompletionJudge(llm_config or {})
        except Exception:
            self._delegate = None

    def judge_completion(self, task_config: Dict, conversation_history: List[Dict], final_output: Any) -> Dict:
        try:
            if self._delegate:
                return self._delegate._fallback_judgment(task_config, final_output)
        except Exception:
            pass
        if final_output and isinstance(final_output, str):
            out = final_output.lower()
            keywords = ['itinerary', 'arrangement', 'recommendation', 'budget', 'hotel', 'flight']
            if any(k in out for k in keywords):
                return {"is_genuinely_completed": True, "completion_score": 50}
        return {"is_genuinely_completed": False, "completion_score": 20}

    def assess_clarification_needs(self, task_config: Dict, conversation_history: List[Dict], orchestrator_response: Dict) -> Dict:
        try:
            final_output = orchestrator_response.get('final_output') if isinstance(orchestrator_response, dict) else None
            if not final_output:
                return {"needs_more_clarification": True, "missing_information": ["final_output"]}
            return {"needs_more_clarification": False, "missing_information": []}
        except Exception:
            return {"needs_more_clarification": True}


@dataclass
class ConversationState:
    user_id: str
    session_id: str
    transaction_id: Optional[str] = None
    step_id: Optional[str] = None
    root_transaction_id: Optional[str] = None
    root_step_id: Optional[str] = None
    latest_transaction_id: Optional[str] = None
    latest_step_id: Optional[str] = None
    history: List[Dict] = None
    lang: str = 'zh'

    def __post_init__(self):
        if self.history is None:
            self.history = []


@dataclass
class BenchmarkResult:
    task_id: str
    level: str
    query: str
    orchestrator_logs: List[Dict]
    evaluation_results: Dict[str, float]
    execution_time: float
    final_output: Any
    conversation_history: List[Dict]
    total_interactions: int
    user_profile: str
    otel: Optional[BenchmarkOpenTelemetryResult] = None
    clarification_pairs: Optional[List[Dict]] = None


class BaseCogBenchmark:
    def _generate_uuids(self, task_id: str) -> Tuple[str, str, str]:
        user_id = f"benchmark_user_{task_id}_{uuid.uuid4().hex[:8]}"
        session_id = f"benchmark_session_{task_id}_{uuid.uuid4().hex[:8]}"
        message_id = f"msg_{uuid.uuid4().hex}"
        return user_id, session_id, message_id

    def _create_request_payload(self, conversation_state: ConversationState, current_query: str, is_initial: bool = False) -> Dict:
        raise NotImplementedError

    def _call_orchestrator_stream(self, payload: Dict) -> Generator:
        raise NotImplementedError

    def _process_stream_response(self, stream_generator: Generator) -> Dict:
        full_response = {
            "status": "unknown",
            "final_output": None,
            "internal_logs": [],
            "transaction_id": None,
            "step_id": None,
            "clarification_question": None
        }
        for chunk in stream_generator:
            if "status" in chunk:
                full_response["status"] = chunk["status"]

            if "final_output" in chunk and chunk["final_output"] is not None:
                if full_response["final_output"] is None:
                    full_response["final_output"] = chunk["final_output"]
                elif isinstance(full_response["final_output"], str):
                    full_response["final_output"] += chunk["final_output"]
                elif isinstance(full_response["final_output"], dict) and isinstance(chunk["final_output"], dict):
                    full_response["final_output"].update(chunk["final_output"])
                else:
                    full_response["final_output"] = chunk["final_output"]

            if "internal_logs" in chunk and chunk["internal_logs"]:
                full_response["internal_logs"].extend(chunk["internal_logs"])

            if "transactionId" in chunk:
                full_response["transaction_id"] = chunk["transactionId"]
            elif "transaction_id" in chunk:
                full_response["transaction_id"] = chunk["transaction_id"]

            if "stepId" in chunk:
                full_response["step_id"] = chunk["stepId"]
            elif "step_id" in chunk:
                full_response["step_id"] = chunk["step_id"]

            if "clarification_question" in chunk:
                full_response["clarification_question"] = chunk["clarification_question"]
            elif full_response["status"] in ("user_query", "input-required") and full_response["final_output"]:
                full_response["clarification_question"] = str(full_response["final_output"])

            if 'collected_json' in chunk and chunk.get('collected_json') is not None:
                if full_response.get('collected_json') is None:
                    full_response['collected_json'] = chunk.get('collected_json')
                    try:
                        meta = (chunk.get('collected_json') or {}).get('meta') or {}
                        sub = meta.get('subAgent') or meta.get('subagent')
                        if sub:
                            synth = {
                                'agent_call': {
                                    'name': sub,
                                    'parameters': {
                                        'merged_text': (chunk.get('collected_json') or {}).get('merged_text')
                                    },
                                    'success': True
                                }
                            }
                            if 'internal_logs' not in full_response or full_response['internal_logs'] is None:
                                full_response['internal_logs'] = []
                            full_response['internal_logs'].append(synth)
                    except Exception:
                        pass

        return full_response


class PluginBasedCogBenchmark(BaseCogBenchmark):
    def __init__(self, orchestrator_url: str = None, llm_config: Dict = None,
                 config_path: str = "config/tasks.yaml", debug: bool = False,
                 history_size: int = 6,
                 profile_path: Optional[str] = None, domain: str = 'travel',
                 orchestrator_type: str = 'default', evaluator_type: str = 'default',
                 reporter_type: str = 'default', lang: str = 'zh'):
        
        self.plugin_manager = global_plugin_manager

        resolved_domain = domain or 'travel'
        resolved_orchestrator_type = orchestrator_type

        try:
            domain_settings = self.plugin_manager.get_domain_settings(resolved_domain)
            if not domain_settings and resolved_domain != 'default':
                fallback_settings = self.plugin_manager.get_domain_settings('default')
            else:
                fallback_settings = {}
        except Exception:
            domain_settings = {}
            fallback_settings = {}

        if not resolved_orchestrator_type or str(resolved_orchestrator_type).lower() == 'default':
            candidate = None
            for settings in (domain_settings, fallback_settings):
                if isinstance(settings, dict) and settings:
                    candidate = settings.get('orchestrator_type') or settings.get('orchestrator')
                    if isinstance(candidate, dict):
                        candidate = candidate.get('type') or candidate.get('name')
                    if candidate:
                        break
            if not candidate:
                try:
                    supported_orchestrators = self.plugin_manager.list_supported_orchestrators()
                except Exception:
                    supported_orchestrators = []
                if isinstance(supported_orchestrators, (list, tuple)):
                    for item in supported_orchestrators:
                        if item:
                            candidate = item
                            break
            if candidate:
                resolved_orchestrator_type = candidate

        resolved_evaluator_type = evaluator_type
        if not resolved_evaluator_type or str(resolved_evaluator_type).lower() == 'default':
            candidate = None
            for settings in (domain_settings, fallback_settings):
                if isinstance(settings, dict) and settings:
                    candidate = settings.get('evaluator_type') or settings.get('evaluator')
                    if isinstance(candidate, dict):
                        candidate = candidate.get('type') or candidate.get('name')
                    if candidate:
                        break
            if not candidate:
                try:
                    supported = self.plugin_manager.list_supported_evaluators()
                except Exception:
                    supported = []
                if isinstance(supported, (list, tuple)) and supported:
                    if 'opentelemetry_evaluator' in supported:
                        candidate = 'opentelemetry_evaluator'
                    else:
                        candidate = supported[0]
            if candidate:
                resolved_evaluator_type = candidate

        resolved_reporter_type = reporter_type
        if not resolved_reporter_type or str(resolved_reporter_type).lower() == 'default':
            candidate = None
            for settings in (domain_settings, fallback_settings):
                if isinstance(settings, dict) and settings:
                    candidate = settings.get('reporter_type') or settings.get('reporter')
                    if isinstance(candidate, dict):
                        candidate = candidate.get('type') or candidate.get('name')
                    if candidate:
                        break
            if not candidate:
                try:
                    supported_reporters = self.plugin_manager.list_supported_reporters()
                except Exception:
                    supported_reporters = []
                if isinstance(supported_reporters, (list, tuple)) and supported_reporters:
                    preferred = next((item for item in supported_reporters if str(item).lower().startswith('html')), None)
                    candidate = preferred or supported_reporters[0]
            if candidate:
                resolved_reporter_type = candidate

        try:
            config_data = getattr(self.plugin_manager, '_config_data', {}) if hasattr(self.plugin_manager, '_config_data') else {}
            logging_cfg = config_data.get('logging') if isinstance(config_data, dict) else {}

            def _is_truthy(value, default=None):
                if value is None:
                    return default
                if isinstance(value, str):
                    lower = value.strip().lower()
                    if lower in ('', 'auto'):
                        return default
                    return lower not in ('0', 'false', 'no', 'off')
                return bool(value)

            show_full_conv = _is_truthy((logging_cfg or {}).get('show_full_conversation'), None)
            if show_full_conv is None:
                show_full_conv = bool(debug)
            console_utils.VERBOSE_LLM_INTERACTION = bool(show_full_conv)
        except Exception:
            try:
                console_utils.VERBOSE_LLM_INTERACTION = bool(debug)
            except Exception:
                pass

        self.domain = resolved_domain
        if resolved_orchestrator_type:
            resolved_orchestrator_type = str(resolved_orchestrator_type)
        self.orchestrator_type = resolved_orchestrator_type or 'default'
        if resolved_evaluator_type:
            resolved_evaluator_type = str(resolved_evaluator_type)
        if resolved_reporter_type:
            resolved_reporter_type = str(resolved_reporter_type)
        self.evaluator_type = resolved_evaluator_type or 'default'
        self.reporter_type = resolved_reporter_type or 'default'
        self.orchestrator_url = orchestrator_url
        self.debug = bool(debug)
        self.history_size = int(history_size) if history_size is not None else 6
        self.profile_path = profile_path
        self.llm_config = llm_config or {}
        self.lang = str(lang).lower() if lang else 'zh'

        self.simulator = self._create_simulator()
        self.evaluator = self._create_evaluator()
        self.orchestrator_client = self._create_orchestrator_client()
        self.judge = self._create_judge()
        self.reporter = self._create_reporter()

        self.logger = BenchmarkLogger(console_full=bool(debug))
        # store the path used to load tasks so we can write patched output next to it
        self._tasks_config_path = config_path
        self.tasks = self._load_tasks(config_path)
        self.max_clarification_rounds = 5
    
    def _create_simulator(self):
        config = {
            'user_profile': "profile_001",
            'llm_config': self.llm_config,
            'lang': self.lang,
            'plugin_call_timeout': 0.5,
            'profile_path': self.profile_path,
            'domain': self.domain,
        }
        
        simulator = self.plugin_manager.create_user_simulator(
            domain=self.domain, config=config
        )

        if not simulator:
            raise RuntimeError(f"No user simulator plugin found for domain '{self.domain}'.")

        # Propagate history_size to simulator instance if it does not accept it in constructor
        try:
            if not hasattr(simulator, 'history_size'):
                try:
                    simulator.history_size = int(getattr(self, 'history_size', 6) or 6)
                except Exception:
                    simulator.history_size = 6
        except Exception:
            pass

        return simulator
    
    def _create_evaluator(self):
        config = {'debug': self.debug, 'domain': self.domain}
        
        evaluator = self.plugin_manager.create_evaluator(
            evaluator_type=self.evaluator_type, config=config
        )
        if not evaluator:
            raise RuntimeError(f"No evaluator plugin found for type '{self.evaluator_type}'.")

        return evaluator
    
    def _create_orchestrator_client(self):
        config = {
            'url': self.orchestrator_url,
            'debug': self.debug,
            'domain': self.domain
        }
        
        client = self.plugin_manager.create_orchestrator_client(
            orchestrator_type=self.orchestrator_type, config=config
        )
        if not client:
            raise RuntimeError(f"No orchestrator plugin found for type '{self.orchestrator_type}'.")

        return client


    def _create_request_payload(self, conversation_state: ConversationState, current_query: str, is_initial: bool = False) -> Dict:
        try:
            if not self.orchestrator_client:
                raise RuntimeError('No orchestrator plugin client available')
            payload = self.plugin_manager.create_orchestrator_payload(self.orchestrator_client, conversation_state, current_query, is_initial)
            if not payload:
                raise RuntimeError('Orchestrator plugin did not return a payload')
            return payload
        except Exception as e:
            raise RuntimeError(f"Failed to create orchestrator payload via plugin: {e}")
    
    def _create_judge(self):
        return TaskCompletionJudge(self.llm_config)
    
    def _create_reporter(self):
        config = {'debug': self.debug, 'domain': self.domain}
        
        reporter = self.plugin_manager.create_reporter(
            reporter_type=self.reporter_type, config=config
        )
        if not reporter:
            raise RuntimeError(f"No reporter plugin found for type '{self.reporter_type}'.")

        return reporter

    def _load_tasks(self, config_path: str):
        try:
            p = Path(config_path)
            if not p.exists():
                p = Path(__file__).parent.parent / config_path
                if not p.exists():
                    return {}
            import yaml
            with open(p, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f)
            tasks = cfg.get('tasks') if isinstance(cfg, dict) else {}
            return tasks or {}
        except Exception:
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f).get('tasks', {})
            except Exception:
                return {}
    
    def _call_orchestrator_stream(self, payload: Dict):
        if not self.orchestrator_client:
            raise RuntimeError('No orchestrator plugin client available to call stream')

        try:
            plugin_ret = self.plugin_manager.call_orchestrator_stream(self.orchestrator_client, payload)
            if not plugin_ret:
                raise RuntimeError('Orchestrator plugin did not return a stream generator')
            return plugin_ret
        except Exception as e:
            raise RuntimeError(f"Orchestrator stream call via plugin failed: {e}")
    
    def _process_stream_response(self, stream_generator):
        if not self.orchestrator_client:
            raise RuntimeError('No orchestrator plugin client available to process stream')

        try:
            plugin_ret = self.plugin_manager.process_stream_response(self.orchestrator_client, stream_generator)
            if not plugin_ret:
                raise RuntimeError('Orchestrator plugin did not return a processed response')
            return plugin_ret
        except Exception as e:
            raise RuntimeError(f"Processing orchestrator stream via plugin failed: {e}")

    @staticmethod
    def _merge_response_text(resp: Dict[str, Any]) -> Optional[str]:
        if not isinstance(resp, dict):
            return None

        collected = resp.get('collected_json') or {}
        artifacts = collected.get('artifacts') if isinstance(collected, dict) else None
        merged_parts: List[str] = []

        # Prefer any pre-merged string the orchestrator already prepared to avoid token-level noise.
        if isinstance(collected, dict):
            for key in ('merged_text', 'mergedText'):
                candidate = collected.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    merged_parts.append(candidate)
                    break

        if not merged_parts and isinstance(artifacts, dict):
            for art in artifacts.values():
                if not isinstance(art, dict):
                    continue
                candidate = art.get('merged_text') or art.get('text') or art.get('body')
                if isinstance(candidate, str) and candidate.strip():
                    merged_parts.append(candidate)
                    continue
                text_parts = art.get('text_parts') or art.get('textParts') or art.get('parts')
                if isinstance(text_parts, (list, tuple)) and text_parts:
                    combined = ''.join(str(item) for item in text_parts if item)
                    if combined.strip():
                        merged_parts.append(combined)

        final_output = resp.get('final_output')
        if not merged_parts and isinstance(final_output, str):
            merged_parts.append(final_output)

        if not merged_parts:
            return None

        try:
            merged_text = re.sub(r'\s+', ' ', ''.join(merged_parts)).strip()
            return merged_text or None
        except Exception:
            try:
                combined = ''.join(merged_parts)
                return combined.strip() or None
            except Exception:
                return None

    @staticmethod
    def _extract_clarification_question(resp: Dict[str, Any]) -> Optional[str]:
        if not isinstance(resp, dict):
            return None

        candidates: List[str] = []

        def _add_candidate(text: Any):
            if not text:
                return
            if isinstance(text, dict):
                for key in ('clarification_question', 'text', 'message', 'content'):
                    val = text.get(key)
                    if isinstance(val, str) and val.strip():
                        candidates.append(val.strip())
                return
            if isinstance(text, (list, tuple)):
                joined = ''.join(str(item) for item in text if item)
                if joined.strip():
                    candidates.append(joined.strip())
                return
            if isinstance(text, str):
                stripped = text.strip()
                if stripped:
                    candidates.append(stripped)

        collected = resp.get('collected_json') or {}
        artifacts = collected.get('artifacts') if isinstance(collected, dict) else None

        _add_candidate(resp.get('clarification_question'))
        _add_candidate(resp.get('final_output'))
        if isinstance(collected, dict):
            _add_candidate(collected.get('merged_text') or collected.get('mergedText'))

        if isinstance(artifacts, dict):
            for art in artifacts.values():
                if not isinstance(art, dict):
                    continue
                parts = art.get('text_parts') or art.get('textParts') or art.get('parts')
                parsed_parts = None
                if isinstance(parts, list):
                    parsed_parts = parts
                elif isinstance(parts, str):
                    try:
                        parsed_parts = ast.literal_eval(parts)
                    except Exception:
                        parsed_parts = None
                if parsed_parts:
                    segment = ''.join(str(item) for item in parsed_parts if item)
                    if segment.strip():
                        candidates.append(segment.strip())

        def _clean_text(value: str) -> str:
            text = re.sub(r'\s+', ' ', value).strip()
            if '###' in text or '- ' in text:
                parts = [p.strip() for p in re.split(r'[\n\r]+', value) if p.strip()]
                question_lines = [p for p in parts if p.endswith(('?', '？'))]
                for q_line in reversed(question_lines):
                    cleaned = re.sub(r'\s+', ' ', q_line).strip()
                    if cleaned:
                        return cleaned
            return text

        question_candidates = []
        for cand in candidates:
            if not isinstance(cand, str):
                continue
            cleaned = _clean_text(cand)
            if cleaned:
                meaningful = re.sub(r'[\s\?？!！。,.，、…-]', '', cleaned)
                if len(meaningful) < 2:
                    continue
                question_candidates.append(cleaned)

        prioritized = [c for c in question_candidates if c.endswith(('?', '？')) and len(c) <= 200]
        prioritized = sorted(prioritized, key=len, reverse=True)
        if not prioritized:
            prioritized = sorted(question_candidates, key=len, reverse=True)

        if prioritized:
            return prioritized[0]
        if question_candidates:
            return question_candidates[0]
        return None

    @staticmethod
    def _interpret_response_flags(resp: Dict[str, Any]) -> Dict[str, Any]:
        flags = {
            'status_lower': '',
            'is_completed': False,
            'needs_user_input': False,
            'clarification_question': None,
            'clarification_type': None,
            'meta': {},
            'transaction_id': None,
            'step_id': None,
        }

        if not isinstance(resp, dict):
            return flags

        collected = resp.get('collected_json') or {}
        meta = collected.get('meta', {}) if isinstance(collected, dict) else {}
        flags['meta'] = meta if isinstance(meta, dict) else {}

        raw_status_candidates: List[Any] = []
        for key in ('status', 'state'):
            raw_status_candidates.append(resp.get(key))
        if isinstance(collected, dict):
            raw_status_candidates.append(collected.get('final_status'))
        raw_status_candidates.append(flags['meta'].get('status'))
        raw_status_candidates.append(flags['meta'].get('state'))

        final_output = resp.get('final_output')
        if isinstance(final_output, dict):
            raw_status_candidates.append(final_output.get('status'))
            raw_status_candidates.append(final_output.get('state'))
        elif isinstance(final_output, list):
            for item in final_output:
                if isinstance(item, dict):
                    raw_status_candidates.append(item.get('status'))
                    raw_status_candidates.append(item.get('state'))

        normalized_status_values: List[str] = []
        for candidate in raw_status_candidates:
            if not candidate:
                continue
            if isinstance(candidate, dict):
                for key in ('state', 'status', 'value'):
                    val = candidate.get(key)
                    if val:
                        normalized_status_values.append(str(val).lower())
            else:
                normalized_status_values.append(str(candidate).lower())

        flags['status_lower'] = normalized_status_values[0] if normalized_status_values else ''

        completed_tokens = {'completed', 'success', 'done', 'finished'}
        clar_status_tokens = {'input-required', 'input_required'}

        flags['is_completed'] = any(value in completed_tokens for value in normalized_status_values)

        flags['transaction_id'] = resp.get('transaction_id') or resp.get('transactionId') or flags['meta'].get('transaction_id') or flags['meta'].get('transactionId')
        flags['step_id'] = resp.get('step_id') or resp.get('stepId') or flags['meta'].get('step_id') or flags['meta'].get('stepId')

        clar_question = PluginBasedCogBenchmark._extract_clarification_question(resp)

        clar_type = resp.get('clarification_type') or flags['meta'].get('clarification_type')
        flags['clarification_type'] = str(clar_type).lower() if clar_type else None

        if isinstance(clar_question, str):
            clar_question = clar_question.strip()
        flags['clarification_question'] = clar_question or None

        has_input_required_status = any(value in clar_status_tokens for value in normalized_status_values)

        if flags['is_completed']:
            flags['needs_user_input'] = False
        else:
            if flags['clarification_type'] and flags['clarification_type'] in clar_status_tokens:
                flags['needs_user_input'] = bool(flags['clarification_question'])
            elif has_input_required_status:
                flags['needs_user_input'] = bool(flags['clarification_question'])
            elif resp.get('clarification_requested'):
                flags['needs_user_input'] = bool(flags['clarification_question'])

        if not flags['needs_user_input']:
            flags['clarification_question'] = None

        return flags
    
    def _enhanced_evaluation(self, task_config: Dict, result: BenchmarkResult, completion_judgment: Dict) -> Dict:
        return {'overall_score': 0.0}
        # if self.evaluator and hasattr(self.plugin_manager, 'evaluate_task'):
        #     scores = self.plugin_manager.evaluate_task(
        #         self.evaluator, task_config, {
        #             'final_output': result.final_output,
        #             'internal_logs': result.orchestrator_logs,
        #             'clarification_history': result.conversation_history,
        #             'total_interactions': result.total_interactions,
        #             'status': 'completed' if result.final_output else 'incomplete'
        #         },
        #         getattr(self.simulator, 'world_state', {}),
        #         self.debug
        #     )
            
        #     if scores:
        #         raw_completion = completion_judgment.get('completion_score', 0) or 0
        #         raw_confidence = completion_judgment.get('quality_assessment', {}).get('requirement_satisfaction', 0) or 0
                
        #         genuine_norm = float(raw_completion) / 100.0 if isinstance(raw_completion, (int, float)) else 0.0
        #         judge_conf_norm = float(raw_confidence) / 100.0 if isinstance(raw_confidence, (int, float)) else 0.0
                
        #         judge_component = (genuine_norm * 0.75) + (judge_conf_norm * 0.25)
        #         base_score = float(scores.get('overall_score', 0) or 0)
        #         scores['overall_score'] = max(0.0, min(1.0, (base_score * 0.6) + (judge_component * 0.4)))
                
        #         return scores
        
        # return super()._enhanced_evaluation(task_config, result, completion_judgment)

    def run_batch_tasks(self, task_ids: List[str] = None, user_profile: str = "profile_001") -> Dict[str, BenchmarkResult]:
        if task_ids is None:
            task_ids = list(self.tasks.keys())

        results: Dict[str, BenchmarkResult] = {}
        for task_id in task_ids:
            try:
                res = self.run_single_task(task_id, user_profile=user_profile)
                if res:
                    results[task_id] = res
                else:
                    task_config = self.tasks.get(task_id) if isinstance(self.tasks, dict) else None
                    result = BenchmarkResult(
                            task_id=task_id,
                            level=task_config.get('level', 'UNK') if task_config else 'UNK',
                            query=task_config.get('query', '') if task_config else '',
                            orchestrator_logs=[],
                            evaluation_results={},
                            execution_time=0.0,
                            final_output=None,
                            conversation_history=[],
                            total_interactions=0,
                            user_profile=user_profile,
                            otel=None
                        )
                    result.failure_reason = 'no_orchestrator_or_plugin_error'
                    results[task_id] = result
            except Exception as e:
                try:
                    self.logger.error(f"run_batch_tasks error for {task_id}: {e}")
                except Exception:
                    pass
                task_config = self.tasks.get(task_id) if isinstance(self.tasks, dict) else None
                result = BenchmarkResult(
                    task_id=task_id,
                    level=task_config.get('level', 'UNK') if task_config else 'UNK',
                    query=task_config.get('query', '') if task_config else '',
                    orchestrator_logs=[],
                    evaluation_results={},
                    execution_time=0.0,
                    final_output=None,
                    conversation_history=[],
                    total_interactions=0,
                    user_profile=user_profile,
                    otel=None
                )
                result.failure_reason = str(e)
                results[task_id] = result

        return results

    def run_single_task(self, task_id: str, user_profile: str = "profile_001") -> Optional[BenchmarkResult]:
        task_config = self.tasks.get(task_id) if isinstance(self.tasks, dict) else None
        if not task_config:
            return None

        if hasattr(self.simulator, 'clear_history'):
            try:
                self.simulator.clear_history()
            except Exception:
                pass

        print(f"\n===== RUNNING TASK {task_id} =====")

        start_ts = time.time()
        user_id, session_id, _ = self._generate_uuids(task_id)
        conv_state = ConversationState(
            user_id=user_id,
            session_id=session_id,
            lang=str(getattr(self, 'lang', 'zh') or 'zh')
        )
        conv_state.history = []

        orchestrator_logs = []
        _per_round_records = []
        _clarification_pairs = []
        total_interactions = 0
        final_output = None
        # human-readable conversation log (list of dicts: round, role, agent, status, text, timestamp)
        human_conversation: List[Dict] = []

        current_query = task_config.get('query', '')

        # per-sub-agent clarification counters
        subagent_rounds: Dict[str, int] = {}
        # track which clarification requests (transaction+step) we've already clarified once
        seen_clarification_requests = set()
        # effective per-subagent limit: explicit per-subagent value or fallback to overall max rounds
        try:
            if getattr(self, 'max_subagent_clarification_rounds', None) is not None:
                effective_subagent_limit = int(self.max_subagent_clarification_rounds)
            else:
                effective_subagent_limit = int(getattr(self, 'max_clarification_rounds', 5) or 5)
        except Exception:
            effective_subagent_limit = int(getattr(self, 'max_clarification_rounds', 5) or 5)

        max_rounds = int(getattr(self, 'max_clarification_rounds', 5) or 5)
        for round_idx in range(max_rounds):
            round_input_query = current_query

            import uuid as _uuid
            message_id = f"msg_{_uuid.uuid4().hex}"
            user_entry = {
                'role': 'user',
                'content': current_query,
                'userId': conv_state.user_id,
                'sessionId': conv_state.session_id,
                'messageId': message_id,
                'timestamp': datetime.now().isoformat()
            }

            conv_state.history.append(user_entry)
            ts_user = datetime.now().isoformat()
            print_conv_entry(len(conv_state.history), ts_user, 'user', str(current_query), round_no=round_idx, status='user_query', origin=None)
            try:
                human_conversation.append({'round': int(round_idx), 'role': 'user', 'agent': None, 'status': 'user_query', 'text': str(current_query), 'timestamp': ts_user})
            except Exception:
                pass

            round_start_iso = datetime.now().isoformat()

            try:
                txn_for_log = conv_state.transaction_id or 'None'
                step_for_log = conv_state.step_id or 'None'
                print(f"[REQUEST_IDS] session={conv_state.session_id} user={conv_state.user_id} transaction={txn_for_log} step={step_for_log}")
            except Exception:
                pass

            payload = self._create_request_payload(conv_state, current_query, is_initial=(round_idx == 0))
            try:
                stream = self._call_orchestrator_stream(payload)
                if stream is None:
                    break
                resp = self._process_stream_response(stream)
                try:
                    resp_for_log = resp.get('collected_json') if isinstance(resp, dict) and resp.get('collected_json') else resp
                    self.logger.log_orchestrator_call(task_id, payload, resp_for_log)
                except Exception as log_exc:
                    try:
                        self.logger.debug(f"log_orchestrator_call failed for task {task_id}: {log_exc}")
                    except Exception:
                        pass
            except Exception as e:
                try:
                    self.logger.error(f"Orchestrator call failed for {task_id}: {e}")
                except Exception:
                    pass
                resp = {'status': 'error', 'final_output': None, 'internal_logs': [], 'clarification_question': None}

            merged_text = self._merge_response_text(resp)
            flags = self._interpret_response_flags(resp)

            sub_agent = flags['meta'].get('subAgent') if isinstance(flags['meta'], dict) else None
            if not sub_agent:
                sub_agent = flags['meta'].get('subagent') if isinstance(flags['meta'], dict) else None
            if not sub_agent:
                try:
                    pm_last = getattr(self.plugin_manager, '_last_plugin_invocation', None)
                    sub_agent = pm_last.get('plugin') if isinstance(pm_last, dict) else None
                except Exception:
                    sub_agent = None
            sub_agent = sub_agent or 'orchestrator'

            txn_from_resp = flags.get('transaction_id')
            if txn_from_resp:
                if not conv_state.root_transaction_id:
                    conv_state.root_transaction_id = txn_from_resp
                if conv_state.root_transaction_id:
                    conv_state.transaction_id = conv_state.root_transaction_id
                conv_state.latest_transaction_id = txn_from_resp

            step_from_resp = flags.get('step_id')
            if step_from_resp:
                if not conv_state.root_step_id:
                    conv_state.root_step_id = step_from_resp
                conv_state.step_id = step_from_resp
                conv_state.latest_step_id = step_from_resp

            if flags['needs_user_input']:
                assistant_text = flags['clarification_question'] or merged_text
            else:
                assistant_text = merged_text
            if assistant_text:
                try:
                    message_id_assistant = f"msg_{uuid.uuid4().hex}"
                except Exception:
                    message_id_assistant = f"msg_{int(time.time()*1000)}"
                assistant_entry = {
                    'role': 'assistant',
                    'content': assistant_text,
                    'messageId': message_id_assistant,
                    'timestamp': datetime.now().isoformat()
                }
                if flags['needs_user_input']:
                    assistant_entry['clarification'] = True
                    if flags['clarification_type']:
                        assistant_entry['clarification_type'] = flags['clarification_type']
                conv_state.history.append(assistant_entry)
                ts_assistant = assistant_entry['timestamp']
                try:
                    status_for_log = resp.get('status') if isinstance(resp, dict) else None
                    print_conv_entry(len(conv_state.history), ts_assistant, 'assistant', str(assistant_text), round_no=round_idx, status=status_for_log, sub_agent=sub_agent, origin=sub_agent)
                except Exception:
                    try:
                        print_info(color_text(f"[ASSISTANT] {str(assistant_text)[:1000]}", FG_GREEN))
                    except Exception:
                        pass
                try:
                    human_conversation.append({'round': int(round_idx), 'role': 'assistant', 'agent': sub_agent, 'status': str(resp.get('status') or ''), 'text': str(assistant_text), 'timestamp': ts_assistant})
                except Exception:
                    pass

            orchestrator_logs.append(resp)
            try:
                _per_round_records.append({
                    'round': int(round_idx),
                    'input_query': round_input_query,
                    'start_time': round_start_iso,
                    'end_time': datetime.now().isoformat(),
                    'resp': resp
                })
            except Exception:
                pass

            try:
                outdir_dbg = Path('output/collected_jsons')
                outdir_dbg.mkdir(parents=True, exist_ok=True)
                ts_dump = int(time.time())
                safe = _safe_serialize(resp, _depth=4)
                fname = outdir_dbg / f"orchestrator_response_{task_id}_{ts_dump}.json"
                try:
                    fname.write_text(json.dumps(safe, ensure_ascii=False, indent=2), encoding='utf-8')
                except Exception:
                    fname.write_text(str(safe), encoding='utf-8')
            except Exception:
                pass

            # Only treat final_output as definitive when orchestrator is not asking for more input.
            if isinstance(resp, dict) and resp.get('final_output') is not None and not flags['needs_user_input']:
                final_output = resp.get('final_output')

            if flags['is_completed']:
                break

            if not flags['needs_user_input']:
                break

            clar_question = flags['clarification_question']
            if not clar_question:
                break

            transaction = flags['transaction_id'] or ''
            step_id = flags['step_id'] or ''
            clar_key = f"{transaction}_{step_id}_{clar_question[:200]}"
            if clar_key in seen_clarification_requests:
                print_info(color_text(f"[DEBUG] Duplicate clarification request detected for key: {clar_key}", FG_YELLOW))
                break
            seen_clarification_requests.add(clar_key)

            try:
                used = int(subagent_rounds.get(sub_agent, 0) or 0)
            except Exception:
                used = 0
            if effective_subagent_limit is not None and used >= int(effective_subagent_limit):
                print_info(color_text(f"[CLARIFICATION][round:{round_idx}] sub-agent '{sub_agent}' reached per-subagent limit ({effective_subagent_limit}), skipping further clarifications.", FG_YELLOW, bold=True))
                break

            try:
                log_invocation({
                    'hook': 'clarification_requested',
                    'plugin': sub_agent,
                    'task': task_id,
                    'query_summary': str(current_query)[:200],
                    'round': int(round_idx),
                    'question_sample': str(clar_question)[:300],
                    'message': 'clarification requested'
                }, task_id=task_id)
            except Exception:
                pass

            preview = clar_question if len(str(clar_question)) <= 400 else str(clar_question)[:397] + '...'
            try:
                print_info(color_text(f"[CLARIFICATION][round:{round_idx}] {preview}", FG_YELLOW, bold=True))
            except Exception:
                pass

            sim_ctx = {
                'task_id': task_id,
                'conversation_history': conv_state.history,
                'round': round_idx,
                'orchestrator_transaction_id': transaction,
                'orchestrator_step_id': step_id,
                'collected_json_meta': flags['meta']
            }

            user_reply = None
            if hasattr(self.simulator, 'generate_clarification_response'):
                try:
                    user_reply = self.simulator.generate_clarification_response(clar_question, sim_ctx)
                except Exception as e:
                    print_info(color_text(f"[ERROR] Failed to generate clarification response: {e}", FG_RED))
                    user_reply = "Sorry, I couldn't process your request. Could you clarify further?"
            else:
                print_info(color_text("[WARNING] Simulator does not support clarification response generation.", FG_YELLOW))
                user_reply = "Sorry, I couldn't process your request. Could you clarify further?"

            if not user_reply:
                print_info(color_text("[DEBUG] No user reply generated. Using default fallback response.", FG_YELLOW))
                user_reply = "Sorry, I couldn't process your request. Could you clarify further?"

            orig_candidate = None
            accepted_reply = None
            rejection_reason = None
            if isinstance(user_reply, dict):
                accepted_reply = user_reply.get('reply') or user_reply.get('final_reply') or user_reply.get('reply')
                orig_candidate = user_reply.get('original_candidate')
                rejection_reason = user_reply.get('rejection_reason')
                user_reply = accepted_reply or user_reply.get('reply') or ''

            try:
                normalized_question = re.sub(r'\s+', ' ', clar_question).strip()
                normalized_question = re.sub(r"\b(\S+)(?:\s+\1\b)+", r"\1", normalized_question)
            except Exception:
                normalized_question = str(clar_question)

            ts_reply = datetime.now().isoformat()
            try:
                print_conv_entry(total_interactions + 1, ts_reply, 'user', str(user_reply), round_no=round_idx, status='simulated', sub_agent=None, origin=None)
            except Exception:
                try:
                    print_info(f"Simulated user reply: {str(user_reply)}")
                except Exception:
                    pass

            try:
                message_id_user = f"msg_{uuid.uuid4().hex}"
            except Exception:
                message_id_user = f"msg_{int(time.time()*1000)}"
            conv_state.history.append({'role': 'user', 'content': user_reply, 'messageId': message_id_user, 'timestamp': ts_reply, 'userId': conv_state.user_id, 'sessionId': conv_state.session_id})
            try:
                human_conversation.append({'round': int(round_idx), 'role': 'user', 'agent': sub_agent, 'status': 'simulated', 'text': str(user_reply), 'timestamp': ts_reply})
            except Exception:
                pass

            subagent_rounds[sub_agent] = int(subagent_rounds.get(sub_agent, 0) or 0) + 1

            _clarification_pairs.append({
                'round': int(round_idx),
                'assistant_question': str(normalized_question),
                'user_reply': str(user_reply),
                'timestamp': ts_reply,
                'original_candidate': orig_candidate,
                'rejection_reason': rejection_reason
            })

            total_interactions += 1
            current_query = user_reply
            continue

        exec_time = max(0.0, time.time() - start_ts)

        try:
            rounds_objs = []
            for r in _per_round_records:
                resp = r.get('resp') or {}
                meta = (resp.get('collected_json') or {}).get('meta', {}) if isinstance(resp, dict) else {}
                sub_agent = meta.get('subAgent') or meta.get('subagent') or 'orchestrator'
                routing_time = int(meta.get('routingTime', 0) or 0)
                conv = Conversation(
                    sub_agent_name=sub_agent,
                    response=str(resp.get('final_output') or ''),
                    status=str(resp.get('status') or ''),
                    query=str(r.get('input_query') or '')
                )

                steps_list = []
                try:
                    collected = resp.get('collected_json') or {}
                    artifacts = collected.get('artifacts') or {}
                    if isinstance(artifacts, dict):
                        for aid, art in artifacts.items():
                            try:
                                tparts = art.get('text_parts') or art.get('textParts') or []
                                if isinstance(tparts, (list, tuple)) and tparts:
                                    for tp in tparts:
                                        try:
                                            steps_list.append({'string_value': str(tp)})
                                        except Exception:
                                            continue
                                else:
                                    nm = art.get('name') or art.get('artifactId') or None
                                    if nm:
                                        steps_list.append({'string_value': str(nm)})
                            except Exception:
                                continue
                    meta_steps = meta.get('steps') or meta.get('step_list') or None
                    if meta_steps:
                        if isinstance(meta_steps, (list, tuple)):
                            for ms in meta_steps:
                                try:
                                    steps_list.append({'string_value': str(ms)})
                                except Exception:
                                    continue
                        else:
                            steps_list.append({'string_value': str(meta_steps)})
                except Exception:
                    steps_list = []

                span = Span(
                    input_query=r.get('input_query') or '',
                    input_history=None,
                    steps=steps_list,
                    routing_time=routing_time,
                    conversations=[conv],
                    internal_rerouting=[]
                )
                rr = RoundResult(round=int(r.get('round', 0)), span=span, start_time=r.get('start_time'), end_time=r.get('end_time'))
                rounds_objs.append(rr)

            otel_result = BenchmarkOpenTelemetryResult(
                transaction_id=conv_state.session_id or '',
                rounds=rounds_objs,
                test_name=task_id,
                metadata={'task_id': task_id, 'level': task_config.get('level')}
            )
        except Exception:
            otel_result = None

        try:
            completion_judgment = self.judge.judge_completion(task_config, conv_state.history, final_output)
        except Exception:
            completion_judgment = {'is_genuinely_completed': False, 'completion_score': 0}

        # Normalize judgment to include 'success_score' for downstream usage
        try:
            success_judgment = dict(completion_judgment) if isinstance(completion_judgment, dict) else {}
            if 'success_score' not in success_judgment and 'completion_score' in success_judgment:
                try:
                    success_judgment['success_score'] = success_judgment.get('completion_score')
                except Exception:
                    success_judgment['success_score'] = 0
        except Exception:
            success_judgment = {'is_genuinely_completed': False, 'success_score': 0}

        # If final_output is empty, try to fall back to the last assistant message
        try:
            if not final_output:
                for h in reversed(conv_state.history or []):
                    try:
                        if isinstance(h, dict) and h.get('role') == 'assistant' and h.get('content'):
                            final_output = h.get('content')
                            break
                    except Exception:
                        continue
        except Exception:
            pass

        try:
            scores = self._enhanced_evaluation(
                task_config,
                BenchmarkResult(
                    task_id=task_id,
                    level=task_config.get('level', 'UNK'),
                    query=task_config.get('query', ''),
                    orchestrator_logs=orchestrator_logs,
                    evaluation_results={},
                    execution_time=exec_time,
                    final_output=final_output,
                    conversation_history=conv_state.history,
                    total_interactions=total_interactions,
                    user_profile=user_profile,
                    otel=otel_result,
                    clarification_pairs=_clarification_pairs,
                ),
                completion_judgment,
            )
            if not isinstance(scores, dict):
                scores = {'overall_score': 0.0}
        except Exception:
            scores = {'overall_score': 0.0}

        result = BenchmarkResult(
            task_id=task_id,
            level=task_config.get('level', 'UNK'),
            query=task_config.get('query', ''),
            orchestrator_logs=orchestrator_logs,
            evaluation_results=scores or {},
            execution_time=exec_time,
            final_output=final_output,
            conversation_history=conv_state.history,
            total_interactions=total_interactions,
            user_profile=user_profile,
            otel=otel_result,
            clarification_pairs=_clarification_pairs
        )

        try:
            rounds_used = len(_per_round_records) if isinstance(_per_round_records, (list, tuple)) else 0
            overall_score = 0.0
            try:
                overall_score = float((scores or {}).get('overall_score', 0.0) or 0.0)
            except Exception:
                overall_score = 0.0

            end_reason = 'incomplete'
            evaluator_status = ''
            try:
                evaluator_status = str(((scores or {}).get('debug_details') or {}).get('status') or '').lower()
            except Exception:
                evaluator_status = ''

            if evaluator_status in ('completed', 'complete', 'success', 'satisfied', 'done'):
                end_reason = 'completed'
            elif evaluator_status in ('error', 'failed', 'failure', 'orchestrator_error'):
                end_reason = 'orchestrator_error'
            elif evaluator_status in ('exceeded_clarification_limits', 'incomplete', 'partial', 'pending'):
                end_reason = evaluator_status

            if end_reason == 'incomplete':
                try:
                    # If any orchestrator log explicitly indicates completion (either 'status' or 'state'), respect it.
                    orchestrator_completed = False
                    try:
                        for l in orchestrator_logs:
                            if not isinstance(l, dict):
                                continue
                            st = str(l.get('status') or l.get('state') or '').lower()
                            if st == 'completed':
                                orchestrator_completed = True
                                break
                    except Exception:
                        orchestrator_completed = False

                    if orchestrator_completed:
                        end_reason = 'completed'
                    elif completion_judgment and isinstance(completion_judgment, dict) and completion_judgment.get('is_genuinely_completed'):
                        end_reason = 'completed'
                    else:
                        try:
                            max_rounds = int(getattr(self, 'max_clarification_rounds', 5) or 5)
                        except Exception:
                            max_rounds = 5
                        if rounds_used >= max_rounds and not (completion_judgment or {}).get('is_genuinely_completed'):
                            end_reason = 'exceeded_clarification_limits'
                        else:
                            has_err = False
                            try:
                                for l in orchestrator_logs:
                                    if not isinstance(l, dict):
                                        continue
                                    st = str(l.get('status') or l.get('state') or '').lower()
                                    if st == 'error' or l.get('error'):
                                        has_err = True
                                        break
                            except Exception:
                                has_err = False
                            if has_err:
                                end_reason = 'orchestrator_error'
                except Exception:
                    end_reason = 'incomplete'

            if end_reason == 'completed':
                reason_col = FG_GREEN
            elif end_reason == 'exceeded_clarification_limits':
                reason_col = FG_YELLOW
            elif end_reason == 'orchestrator_error':
                reason_col = FG_RED
            else:
                reason_col = FG_MAGENTA

            msg = f"Task {task_id} ended -> {end_reason} | rounds_used={rounds_used} | total_interactions={total_interactions} | exec_time={exec_time:.2f}s | score={overall_score}"
            if end_reason == 'completed':
                print_info(color_text(msg, reason_col, bold=True))
            elif end_reason == 'exceeded_clarification_limits':
                print_warning(color_text(msg, reason_col, bold=True))
            elif end_reason == 'orchestrator_error':
                print_error(color_text(msg, reason_col, bold=True))
            else:
                print_warning(color_text(msg, reason_col, bold=False))
            try:
                metrics = scores or {}
                if isinstance(metrics, dict) and metrics:
                    print_info(color_text("Detailed evaluation metrics:", FG_GREEN, bold=True))
                    for k, v in metrics.items():
                        try:
                            if isinstance(v, float):
                                print_info(f" - {k}: {v:.4f}")
                            else:
                                print_info(f" - {k}: {v}")
                        except Exception:
                            try:
                                print_info(f" - {k}: {str(v)}")
                            except Exception:
                                pass
            except Exception:
                pass

            if self.debug:
                try:
                    if isinstance(success_judgment, dict):
                        print_info(color_text("Judge summary:", FG_YELLOW, bold=True))
                        try:
                            print_info(f" - is_genuinely_completed: {success_judgment.get('is_genuinely_completed')}")
                        except Exception:
                            pass
                        try:
                            print_info(f" - success_score: {success_judgment.get('success_score')}")
                        except Exception:
                            pass
                        # print user_side_milestone if provided or infer simple milestone
                        try:
                            usm = success_judgment.get('user_side_milestone') if isinstance(success_judgment, dict) else None
                            if usm is None:
                                usm = 'achieved' if success_judgment.get('is_genuinely_completed') else 'not_achieved'
                            print_info(f" - user_side_milestone: {usm}")
                        except Exception:
                            pass
                except Exception:
                    pass
            # --- write simplified logs: details.log (human-readable conversation) and summary.log (key results) ---
            try:
                out_dir = Path('output')
                out_dir.mkdir(parents=True, exist_ok=True)
                details_path = out_dir / 'details.log'
                summary_path = out_dir / 'summary.log'

                # write details.log: one human-readable session per task (append)
                try:
                    with open(details_path, 'a', encoding='utf-8') as df:
                        df.write(f"--- Task {task_id} start: {datetime.now().isoformat()} ---\n")
                        for h in (human_conversation or []):
                            try:
                                rnd = h.get('round')
                                ts = h.get('timestamp') or ''
                                role = h.get('role') or ''
                                agent = h.get('agent') or ''
                                status_h = h.get('status') or ''
                                text = h.get('text') or ''
                                if agent:
                                    df.write(f"[round:{rnd}] [{ts}] {role.upper()} (agent:{agent}) [status:{status_h}]: {text}\n")
                                else:
                                    df.write(f"[round:{rnd}] [{ts}] {role.upper()} [status:{status_h}]: {text}\n")
                            except Exception:
                                try:
                                    df.write(str(h) + '\n')
                                except Exception:
                                    pass
                        df.write(f"--- Task {task_id} end: {datetime.now().isoformat()} (end_reason={end_reason}) ---\n\n")
                except Exception:
                    pass

                # write summary.log: one JSON line per task
                try:
                    summary_obj = {
                        'task_id': task_id,
                        'end_reason': end_reason,
                        'rounds_used': rounds_used,
                        'total_interactions': total_interactions,
                        'execution_time': float(exec_time),
                        'overall_score': float(overall_score),
                        'success_judgment': success_judgment if isinstance(success_judgment, dict) else {},
                        'timestamp': datetime.now().isoformat()
                    }
                    with open(summary_path, 'a', encoding='utf-8') as sf:
                        sf.write(json.dumps(summary_obj, ensure_ascii=False) + '\n')
                except Exception:
                    pass
            except Exception:
                pass
        except Exception:
            pass

        # After each task run, persist current tasks mapping with any added transaction id(s)
        try:
            # try to obtain a sensible transaction id from conversation state if available
            tx_val = None
            try:
                tx_val = conv_state.latest_transaction_id or conv_state.transaction_id or conv_state.root_transaction_id
            except Exception:
                tx_val = None

            try:
                if isinstance(self.tasks, dict) and task_id in self.tasks:
                    # follow user's requested field name 'transcantion_id' (keep their spelling)
                    self.tasks[task_id]['transcantion_id'] = tx_val
            except Exception:
                pass

            # Decide where to write the patched tasks file
            try:
                src = Path(getattr(self, '_tasks_config_path', 'tasks.yaml'))
                if not src.exists():
                    alt = Path(__file__).parent.parent / str(getattr(self, '_tasks_config_path', 'tasks.yaml'))
                    if alt.exists():
                        src = alt
            except Exception:
                src = None

            if src and src.parent:
                dest_dir = src.parent
            else:
                dest_dir = Path('datasets') / (getattr(self, 'domain', '') or '')
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / 'tasks.patched.yaml'

            try:
                import yaml
                with open(dest, 'w', encoding='utf-8') as pf:
                    yaml.safe_dump({'tasks': self.tasks}, pf, allow_unicode=True)
            except Exception:
                try:
                    with open(dest, 'w', encoding='utf-8') as pf:
                        pf.write(json.dumps({'tasks': self.tasks}, ensure_ascii=False, indent=2))
                except Exception:
                    pass
        except Exception:
            pass

        return result
    
    def generate_report(self, results: Dict[str, BenchmarkResult], 
                       output_path: str = None) -> Dict:
        reporter = getattr(self, 'reporter', None)
        if not reporter:
            raise RuntimeError(f"No reporter available for type '{getattr(self, 'reporter_type', None)}'.")

        if output_path:
            out_path = Path(output_path)
        else:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            out_path = Path('output') / f'benchmark_report_{ts}.json'

        out_path.parent.mkdir(parents=True, exist_ok=True)

        report_input = None
        try:
            if hasattr(self, 'evaluator') and getattr(self.evaluator, 'aggregate', None):
                report_input = self.evaluator.aggregate(
                    results
                )
        except Exception:
            report_input = None

        if not isinstance(report_input, dict) or not report_input:
            report_input = results

        report_obj = self.plugin_manager.generate_report(
            reporter,
            report_input,
            str(out_path),
            domain=self.domain,
        )

        if not isinstance(report_obj, dict):
            try:
                report_obj = reporter.generate_report(report_input, str(out_path))
            except Exception:
                raise RuntimeError("Reporter plugin did not return a dict report.")

        if not isinstance(report_obj, dict):
            raise RuntimeError("Reporter plugin did not return a dict report.")

        try:
            try:
                lb_reporter = self.plugin_manager.create_reporter(reporter_type='html_leaderboard', config={'debug': self.debug})
            except Exception:
                lb_reporter = None
            if lb_reporter:
                try:
                    lb_reporter.generate_report(results, str(out_path))
                except Exception:
                    try:
                        self.logger and getattr(self.logger, 'warning', None) and self.logger.warning('Leaderboard reporter failed')
                    except Exception:
                        pass
        except Exception:
            pass

        return report_obj