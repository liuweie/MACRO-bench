import json
import re
import os
import builtins
import logging
import argparse
import sys
import time
from datetime import datetime
import random
import yaml
from abc import ABC, abstractmethod
import argparse
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path
import requests
try:
    import openai  # or other LLM APIs
except Exception:
    openai = None

# Control verbose debug/progress output to stdout
# Set EVALUATOR_VERBOSE=0 to silence all print output during evaluation,
# which prevents Celery/task-system logs from being flooded.
_EVALUATOR_VERBOSE = os.environ.get('EVALUATOR_VERBOSE', '1')
try:
    _EVALUATOR_VERBOSE = bool(int(_EVALUATOR_VERBOSE))
except Exception:
    _EVALUATOR_VERBOSE = _EVALUATOR_VERBOSE.lower() in ('1', 'true', 'yes', 'on')


def _configure_text_stream(stream):
    """Force UTF-8 with replacement on Windows console to avoid UnicodeEncodeError."""
    try:
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
        else:
            import io
            # Re-wrap the buffer with safe encoding; keep line buffering/write-through when possible.
            stream = io.TextIOWrapper(
                stream.buffer,
                encoding='utf-8',
                errors='replace',
                line_buffering=True,
                write_through=True,
            )
    except Exception:
        return stream
    return stream


try:
    sys.stdout = _configure_text_stream(sys.stdout)
    sys.stderr = _configure_text_stream(sys.stderr)
except Exception:
    # Continue even if console reconfiguration fails (e.g. when stdout redirected).
    pass

if not _EVALUATOR_VERBOSE:
    # Replace built-in print with no-op to reduce stdout writes
    def _noop_print(*a, **k):
        return
    builtins.print = _noop_print

# Initialize base logging for error/critical output
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

_LOG_ROOT = Path(os.getenv('EVALUATOR_LOG_DIR', 'output/logs'))
try:
    _LOG_ROOT.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

_LLM_EVAL_LOG_PATH = _LOG_ROOT / 'llm_enhanced_evaluator.log'
_llm_eval_logger = logging.getLogger('llm_enhanced_evaluator')
if not _llm_eval_logger.handlers:
    try:
        _handler = logging.FileHandler(_LLM_EVAL_LOG_PATH, encoding='utf-8')
        _handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        _llm_eval_logger.addHandler(_handler)
    except Exception:
        _llm_eval_logger.addHandler(logging.NullHandler())
    _llm_eval_logger.setLevel(logging.INFO)


def _truncate_for_log(text: Optional[str], limit: int = 480) -> str:
    """Condense long strings to keep log lines readable."""
    if text is None:
        return ''
    compact = ' '.join(str(text).split())
    if len(compact) <= limit:
        return compact
    return compact[:limit] + '...'

LLM_ENDPOINT = os.getenv(
    'LLM_ENDPOINT',
    '',
)
LLM_API_KEY = os.getenv('LLM_API_KEY') or os.getenv('OPENAI_API_KEY') or "sk-b7YG979VWIH3yNwDur8UVV"
LLM_MODEL = os.getenv('LLM_MODEL', 'gpt-5.1')


class TraceProcessor:
    """Trace data processor"""
    
    @staticmethod
    def parse_trace_txt_to_jsonl(input_file: str, output_file: str = None) -> str:
        """
        Convert TXT trace data to JSONL format
        
        Args:
            input_file: Input TXT file path
            output_file: Output JSONL path; auto-generated when None
            
        Returns:
            Output JSONL file path
        """
        if output_file is None:
            output_file = input_file.replace('.txt', '.jsonl')
        
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract all data:{...} traces via regex
        pattern = r'data:\s*(\{.*?\})(?=\s*data:|\s*$)'
        matches = re.findall(pattern, content, re.DOTALL)
        
        traces = []
        for i, match in enumerate(matches):
            try:
                trace_data = json.loads(match)
                traces.append(trace_data)
                print(f"Successfully parsed trace {i+1}: {trace_data.get('transactionId', 'Unknown')}")
            except json.JSONDecodeError as e:
                print(f"Failed to parse trace {i+1}: {e}")
                continue
        
        # Save as JSONL format
        with open(output_file, 'w', encoding='utf-8') as f:
            for trace in traces:
                f.write(json.dumps(trace, ensure_ascii=False) + '\n')
        
        print(f"Successfully converted {len(traces)} traces to {output_file}")
        return output_file
    
    @staticmethod
    def load_jsonl_traces(jsonl_file: str) -> List[Dict]:
        """Load trace data from JSONL file"""
        traces = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    trace_data = json.loads(line.strip())
                    traces.append(trace_data)
                except json.JSONDecodeError as e:
                    print(f"{line_num} JSON parse error: {e}")
                    continue
        return traces


class DomainManager:
    """Domain manager - manage configurations for different business scenarios"""
    
    def __init__(self):
        # Define configurations for different domains
        self.domain_configs = {
            'travel': {
                'name': '',
                'description': 'Travel-related services including flights, hotels, weather, restaurants, etc.',
                'agents': [
                    'flight search agent',
                    'weather forecast check agent', 
                    'hotel accommodation recommendation agent',
                    'restaurant recommendation agent',
                    'travel planning agent',
                    'movie recommendation agent',
                    'news search agent',
                    'chat agent'
                ],
                'keywords': ['flight', 'hotel', 'travel', 'trip', 'vacation', 'booking', 'weather', 'restaurant','news', 'movie',
                           'flight', 'hotel', '', '', '', '', 'weather', 'restaurant','news', 'movie'],
                'evaluation_weights': {
                    's_task_success_rate': 0.2,
                    's_task_complete_rate': 0.2,
                    'u_task_success_rate': 0.2,
                    'u_task_complete_rate': 0.2,
                    'execution_efficiency': 0.2,
                    'clarification_efficiency': 0.0,
                    'agent_routing_accuracy': 0.0,
                    'orchestration_latency': 0.0,
                    'average_rounds_per_task': 0.0,
                    'completeness': 0.0
                }
            },
            'hr': {
                'name': '',
                'description': 'HR management related services',
                'agents': [
                    'recruitment agent',
                    'employee onboarding agent',
                    'performance review agent',
                    'payroll management agent',
                    'training coordination agent'
                ],
                'keywords': ['recruitment', 'hire', 'employee', 'payroll', 'performance', 'training',
                           '', '', '', '', '', ''],
                'evaluation_weights': {
                    's_task_success_rate': 0.2,
                    's_task_complete_rate': 0.2,
                    'u_task_success_rate': 0.2,
                    'u_task_complete_rate': 0.2,
                    'execution_efficiency': 0.2,
                    'clarification_efficiency': 0.0,
                    'agent_routing_accuracy': 0.0,
                    'orchestration_latency': 0.0,
                    'average_rounds_per_task': 0.0,
                    'completeness': 0.0
                }
            },
            'it': {
                'name': '',
                'description': 'IT technical support and services',
                'agents': [
                    'alert_query',
                    'general_response',
                    'operations_data_query',
                    'operations_report_generation',
                    'root_cause_analysis',
                    'solution',
                    'translate_language'
                ],
                'keywords': ['alert', 'monitoring', 'operations', 'report', 'analysis', 'solution', 'translation', 'data',
                           'system', 'software', 'network', 'security', 'technical',
                           '', '', '', '', '', '', '', '', '', '', '', '', ''],
                'evaluation_weights': {
                    's_task_success_rate': 0.2,
                    's_task_complete_rate': 0.2,
                    'u_task_success_rate': 0.2,
                    'u_task_complete_rate': 0.2,
                    'execution_efficiency': 0.2,
                    'clarification_efficiency': 0.0,
                    'agent_routing_accuracy': 0.0,
                    'orchestration_latency': 0.0,
                    'average_rounds_per_task': 0.0,
                    'completeness': 0.0
                }
            },
            'customer_service': {
                'name': '',
                'description': 'Customer support and issue resolution',
                'agents': [
                    'complaint handling agent',
                    'product inquiry agent',
                    'billing support agent',
                    'technical support agent',
                    'feedback collection agent'
                ],
                'keywords': ['complaint', 'support', 'help', 'issue', 'problem', 'customer',
                           '', '', '', '', ''],
                'evaluation_weights': {
                    's_task_success_rate': 0.2,
                    's_task_complete_rate': 0.2,
                    'u_task_success_rate': 0.2,
                    'u_task_complete_rate': 0.2,
                    'execution_efficiency': 0.2,
                    'clarification_efficiency': 0.0,
                    'agent_routing_accuracy': 0.0,
                    'orchestration_latency': 0.0,
                    'average_rounds_per_task': 0.0,
                    'completeness': 0.0
                }
            }
        }
        
        # Default domain configuration
        self.default_domain = 'travel'
    
    def get_domain_config(self, domain: str) -> Dict:
        """Get configuration for a specific domain"""
        return self.domain_configs.get(domain, self.domain_configs[self.default_domain])
    
    def get_all_domains(self) -> List[str]:
        """Get all supported domains"""
        return list(self.domain_configs.keys())
    
    def detect_domain(self, text: str) -> str:
        """Detect which domain the text belongs to"""
        if not text:
            return self.default_domain
        
        text_lower = text.lower()
        domain_scores = {}
        
        for domain, config in self.domain_configs.items():
            score = sum(1 for keyword in config['keywords'] if keyword in text_lower)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores.items(), key=lambda x: x[1])[0]
        else:
            return self.default_domain
    
    def get_domain_agents(self, domain: str) -> List[str]:
        """Get all agents under a specific domain"""
        config = self.get_domain_config(domain)
        return config.get('agents', [])
    
    def get_evaluation_weights(self, domain: str) -> Dict[str, float]:
        """Get evaluation weights for a specific domain"""
        config = self.get_domain_config(domain)
        return config.get('evaluation_weights', {})
    
    def is_agent_in_domain(self, agent_name: str, domain: str) -> bool:
        """Check whether an agent belongs to a specific domain"""
        domain_agents = self.get_domain_agents(domain)
        return agent_name in domain_agents


class LanguageDomainDetector:
    """Language and domain recognizer - based on the new Domain architecture"""
    
    def __init__(self, domain_manager: DomainManager):
        self.domain_manager = domain_manager
        
        # Language detection keywords
        self.language_keywords = {
            'zh': ['', '', '', '', '', '', '', '', '', '', '', '', '', ''],
            'en': ['the', 'and', 'is', 'in', 'to', 'of', 'a', 'that', 'it', 'with', 'for', 'on', 'are', 'as']
        }
    
    def detect_language(self, text: str) -> str:
        """Detect text language"""
        if not text:
            return 'unknown'
        
        text_lower = text.lower()
        zh_count = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
        en_count = sum(1 for keyword in self.language_keywords['en'] if keyword in text_lower)
        
        if zh_count > len(text) * 0.3:  # Chinese character ratio exceeds 30%
            return 'zh'
        elif en_count > 2:  # Contains multiple English keywords
            return 'en'
        else:
            return 'unknown'
    
    def detect_domain(self, text: str) -> str:
        """Detect text domain using DomainManager"""
        return self.domain_manager.detect_domain(text)
    
    def analyze_trace(self, trace_data: Dict) -> Dict:
        """Analyze language and domain of the trace"""
        rounds = trace_data.get('rounds', [])
        if not rounds:
            return {
                'language': 'unknown', 
                'domain': self.domain_manager.default_domain
            }
        
        # Analyze initial query
        initial_query = rounds[0]['span'].get('inputQuery', '')
        language = self.detect_language(initial_query)
        domain = self.detect_domain(initial_query)
        
        # If initial query is inconclusive, analyze all conversations
        if language == 'unknown' or domain == self.domain_manager.default_domain:
            all_text = initial_query or ''
            for round_data in rounds:
                span = round_data.get('span', {}) or {}
                all_text += ' ' + (span.get('inputQuery') or '')
                for conv in span.get('conversations', []) or []:
                    all_text += ' ' + (conv.get('query') or '') + ' ' + (conv.get('response') or '')
            
            if language == 'unknown':
                language = self.detect_language(all_text)
            if domain == self.domain_manager.default_domain:
                domain = self.detect_domain(all_text)
        
        return {
            'language': language,
            'domain': domain
        }
    

class EvaluationStrategy:
    """Evaluation strategy config - based on DomainManager"""
    
    def __init__(self, domain_manager: DomainManager):
        self.domain_manager = domain_manager
        
        # Language-specific tuning on top of domain weights
        self.language_adjustments = {
            'zh': {
                'response_quality': 0.02,  # +2%
                'clarification_efficiency': 0.02
            },
            'en': {
                'agent_routing_accuracy': 0.02,  # +2%
            }
        }
    
    def get_weights(self, language: str, domain: str) -> Dict[str, float]:
        """Get weight config for specific language/domain"""
        # Get base domain weights
        weights = self.domain_manager.get_evaluation_weights(domain).copy()
        
        # Apply language-specific tuning
        if language in self.language_adjustments:
            for dim, adjustment in self.language_adjustments[language].items():
                if dim in weights:
                    weights[dim] += adjustment
        
        # Ensure total weights sum to 1
        total = sum(weights.values())
        if abs(total - 1.0) > 0.01:  # Allow small floating-point tolerance
            for dim in weights:
                weights[dim] /= total
        
        return weights


class BaseEvaluator(ABC):
    """Evaluator abstract base class - based on new Domain architecture"""
    
    def __init__(self, tasks_config=None, agents_config=None, user_profiles=None):
        self.tasks_config = tasks_config
        self.agents_config = agents_config
        self.user_profiles = user_profiles
        
        # Domain
        self.domain_manager = DomainManager()
        
        # 
        self.detector = LanguageDomainDetector(self.domain_manager)

        # 
        self.strategy = EvaluationStrategy(self.domain_manager)
        
        # Task configuration mapping (new)
        self.task_mapping = self._build_task_mapping()
        
        # Generic agent capability mapping (fallback)
        self.agent_capabilities = {
            'flight search agent': ['flight', 'air ticket', 'flight', 'airline', 'ticket'],
            'weather forecast check agent': ['weather', 'weather', 'forecast', 'temperature'],
            'hotel accommodation recommendation agent': ['hotel', 'hotel', 'accommodation', 'accommodation'],
            'restaurant recommendation agent': ['restaurant', 'restaurant', 'dining', 'food', 'cuisine'],
            'travel planning agent': ['itinerary', 'itinerary', 'plan', 'planning', 'schedule'],
            'movie recommendation agent': ['movie', 'movie', 'recommendation', 'film', 'cinema'],
            'news search agent': ['news', 'news', 'headlines', 'articles'],
            'chat agent': ['chat', 'conversation', 'conversation', 'help']
        }

        # agent config () -> /
        self.agent_aliases = {}
        self.agent_descriptions: Dict[str, str] = {
            'flight search agent': 'Looks up flight options, schedules, and fares based on route and dates.',
            'weather forecast check agent': 'Retrieves weather forecasts for specified locations and dates.',
            'hotel accommodation recommendation agent': 'Finds and recommends hotels or lodging that match user criteria.',
            'restaurant recommendation agent': 'Suggests restaurants or dining options given cuisine, location, or budget.',
            'travel planning agent': 'Builds itineraries by sequencing activities, transport, and timing.',
            'movie recommendation agent': 'Recommends movies or related media given preferences.',
            'news search agent': 'Fetches recent news articles or summaries matching a topic.',
            'chat agent': 'Provides general conversation support and basic guidance.'
        }
        if self.agents_config:
            try:
                agents_src = {}
                if isinstance(self.agents_config, dict):
                    if 'en' in self.agents_config and isinstance(self.agents_config['en'], dict):
                        agents_src.update(self.agents_config.get('en', {}))
                    if 'cn' in self.agents_config and isinstance(self.agents_config['cn'], dict):
                        for k, v in self.agents_config.get('cn', {}).items():
                            if k not in agents_src:
                                agents_src[k] = v
                    for k, v in self.agents_config.items():
                        if isinstance(v, str) and k not in ('en', 'cn'):
                            agents_src[k] = v

                for canonical, desc in agents_src.items():
                    keywords = set()
                    kc = str(canonical).lower()
                    keywords.add(kc)
                    keywords.add(kc.replace('_', ' '))
                    keywords.add(kc.replace('_', ' ') + ' agent')
                    if isinstance(desc, str):
                        for token in re.split(r'[^\w\u4e00-\u9fff]+', desc.lower()):
                            if len(token) >= 2:
                                keywords.add(token)

                    for default_name, default_keywords in list(self.agent_capabilities.items()):
                        dname = default_name.lower()
                        if canonical.replace('_', ' ') in dname or dname in canonical:
                            for kw in default_keywords:
                                keywords.add(kw)

                    self.agent_capabilities[canonical] = list(keywords)
                    if isinstance(desc, str) and desc.strip():
                        self.agent_descriptions[canonical] = desc.strip()

                    for alias in list(keywords):
                        self.agent_aliases[alias] = canonical

            except Exception:
                pass
    
    def _build_task_mapping(self) -> Dict:
        """Build mapping from task ID to config"""
        if not self.tasks_config or 'tasks' not in self.tasks_config:
            return {}
        
        task_mapping = {}
        for task_id, task_config in self.tasks_config['tasks'].items():
            # Create mapping based on query content
            query = task_config.get('query', '')
            if query:
                task_mapping[query] = {
                    'task_id': task_id,
                    'config': task_config
                }
                
                # Also add key query fragments as backup matching signals
                key_words = query.split()[:3]  # Take first 3 words as keywords
                if key_words:
                    key_phrase = ' '.join(key_words)
                    task_mapping[key_phrase] = {
                        'task_id': task_id,
                        'config': task_config
                    }
        return task_mapping
    
    def _find_matching_task(self, initial_query: str) -> Optional[Dict]:
        """Find matching task config from initial query"""
        # Exact match (preferred)
        if not initial_query:
            return None

        if initial_query in self.task_mapping:
            return self.task_mapping[initial_query]

        # ：intersection，
        query_lower = (initial_query or '').lower()

        def _tokenize(s: str) -> List[str]:
            tokens = [t for t in re.split(r"[^\w\u4e00-\u9fff0-9-]+", s.lower()) if t]
            # （），' a//and/trip' 
            stopwords = set(['a', 'an', 'the', 'i', 'you', 'to', 'for', 'in', 'on', 'with', 'and', 'or', 'my', 'me', 'want', 'need', 'please', 'trip', 'travel'])
            filtered = [t for t in tokens if t and t not in stopwords and len(t) > 1]
            return filtered

        query_tokens = _tokenize(query_lower)
        if not query_tokens:
            return None

        # ，intersection，
        for key_phrase, task_info in self.task_mapping.items():
            key_tokens = _tokenize(key_phrase)
            if not key_tokens:
                continue
            # intersection
            intersect = set(key_tokens) & set(query_tokens)
            if len(intersect) >= 2:
                return task_info

        return None

    def _find_task_by_transaction(self, transaction_id: str) -> Optional[Dict]:
        """ transaction id ， tasks.yaml  'transcantion_id' 。

        ：'transcantion_id'（）、'transaction_id'  'transactionId'
        """
        if not transaction_id:
            return None

        try:
            tasks = (self.tasks_config or {}).get('tasks', {})
            if not isinstance(tasks, dict):
                return None
            for task_id, task_cfg in tasks.items():
                if not isinstance(task_cfg, dict):
                    continue
                # check several possible keys
                for key in ('transcantion_id', 'transaction_id', 'transactionId'):
                    val = task_cfg.get(key)
                    if val and str(val) == str(transaction_id):
                        return {'task_id': task_id, 'config': task_cfg}
        except Exception:
            return None
        return None

    @abstractmethod
    def evaluate_trace(self, trace_data: Dict) -> Dict:
        """trace - """
        pass
    
    def _analyze_language_domain(self, trace_data: Dict) -> Dict:
        """Analyze language and domain of the trace"""
        return self.detector.analyze_trace(trace_data)
    
    def _get_evaluation_weights(self, language: str, domain: str) -> Dict[str, float]:
        """"""
        return self.strategy.get_weights(language, domain)

    def _calculate_agent_routing_accuracy(self, rounds: List[dict], expected_subagents: List[dict]) -> float:
        """ - domainagent"""
        "Implemented in subclass"
        pass
    
    def _calculate_execution_efficiency(
        self,
        rounds: List,
        user_complete_rate: float,
        system_complete_rate: float,
    ) -> float:
        """。

        ：((0.5 * user_complete_rate) + (0.5 * system_complete_rate)) /
        max(total_internal_reroutes, 1) * 100%， 0-1 。
        """

        base_completion = 0.5 * max(0.0, min(user_complete_rate, 1.0))
        base_completion += 0.5 * max(0.0, min(system_complete_rate, 1.0))

        total_reroutes = 0
        for round_data in rounds:
            span = round_data.get('span', {}) or {}
            rerouting = span.get('internalRerouting') or []
            total_reroutes += len(rerouting)

        denominator = max(total_reroutes, 1)
        efficiency_percent = (base_completion / denominator) * 100.0
        normalized_efficiency = efficiency_percent / 100.0

        return max(0.0, min(normalized_efficiency, 1.0))

    def _calculate_completeness(self, rounds: List) -> float:
        """，。"""
        if not rounds:
            return 0.0

        total_segments = 0
        completed_segments = 0

        for round_data in rounds:
            conversations = round_data.get('span', {}).get('conversations', []) or []
            for conv in conversations:
                total_segments += 1
                status = (conv.get('status') or '').lower()
                if status in ('completed', 'succeeded', 'done'):
                    completed_segments += 1

        if total_segments == 0:
            return 0.0

        return completed_segments / total_segments
    
    def _calculate_orchestration_latency(self, rounds: List) -> float:
        """（ routingTime）。

         trace  span.routingTime （：）。
         trace （average of sum(routingTime)）。
        """
        if not rounds:
            return 0.0

        total_routing_time = 0
        total_internal_routing_time = 0
        for round_data in rounds:
            span = round_data.get('span', {})
            # routingTime  0，
            routing_time = span.get('routingTime', 0) or 0
            try:
                total_routing_time += float(routing_time)
            except Exception:
                continue
            inter_rerouting_items = span.get('internalRerouting', [])
            for reroute in inter_rerouting_items:
                inter_routing_time = reroute.get('expired', 0) or 0
                try:
                    total_internal_routing_time += float(inter_routing_time)
                except Exception:
                    continue

        # routingTime ，
        total_seconds = (total_routing_time + total_internal_routing_time) / 1e9
        return float(total_seconds)
    
    def _calculate_clarification_efficiency(self, rounds: List) -> float:
        """ - """
        if len(rounds) <= 1:
            return 1.0  # Single-round completion, highest efficiency
            
        clarification_rounds = 0
        effective_clarifications = 0
        
        for i, round_data in enumerate(rounds[1:], 1):  # Starting from the second round
            previous_round = rounds[i-1]
            current_round = round_data
            
            # 
            prev_conversations = previous_round['span'].get('conversations', [])
            had_input_required = any(
                conv.get('status') == 'input-required' 
                for conv in prev_conversations
            )
            
            if had_input_required:
                clarification_rounds += 1
                
                # 
                current_input = current_round['span'].get('inputQuery', '')
                if current_input and len(current_input.strip()) > 5:
                    effective_clarifications += 1
        
        if clarification_rounds == 0:
            return 1.0
            
        return effective_clarifications / clarification_rounds

    

class LLMAssistedEvaluator:
    """（，LLMEnhancedEvaluator）。

    ： >  >  > 。
    """

    _AGENT_NORMALIZATION_MAP: Dict[str, str] = {
        'flight_search': 'flight search agent',
        'flight search': 'flight search agent',
        'flight search agent': 'flight search agent',
        'hotel_accommodation_recommendation': 'hotel accommodation recommendation agent',
        'hotel accommodation recommendation': 'hotel accommodation recommendation agent',
        'hotel accommodation recommendation agent': 'hotel accommodation recommendation agent',
        'movie_recommendation': 'movie recommendation agent',
        'movie recommendation': 'movie recommendation agent',
        'movie recommendation agent': 'movie recommendation agent',
        'news_search': 'news search agent',
        'news search': 'news search agent',
        'news search agent': 'news search agent',
        'restaurant_recommendation': 'restaurant recommendation agent',
        'restaurant recommendation': 'restaurant recommendation agent',
        'restaurant recommendation agent': 'restaurant recommendation agent',
        'travel_planning': 'travel planning agent',
        'travel planning': 'travel planning agent',
        'travel planning agent': 'travel planning agent',
        'weather_forecast_check': 'weather forecast check agent',
        'weather forecast check': 'weather forecast check agent',
        'weather forecast check agent': 'weather forecast check agent',
        'chat_agent': 'chat agent',
        'chat agent': 'chat agent',
    }

    _CONFIG_SEARCH_PATHS: List[Path] = [
        Path('config') / 'plugins.yaml',
        Path(__file__).resolve().parent.parent / 'config' / 'plugins.yaml',
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        *,
        agent_descriptions: Optional[Dict[str, str]] = None,
        agent_aliases: Optional[Dict[str, str]] = None,
        config: Optional[Dict[str, Any]] = None,
        endpoint: Optional[str] = None,
    ):
        self._raw_config = config or {}
        file_defaults = self._load_llm_config_from_files()
        runtime_defaults = self._resolve_llm_config(self._raw_config)

        merged_defaults: Dict[str, Any] = {}
        merged_defaults.update(file_defaults)
        merged_defaults.update(runtime_defaults)
        self.llm_config = merged_defaults

        resolved_api_key = api_key or self.llm_config.get('api_key') or LLM_API_KEY
        resolved_endpoint = endpoint or self.llm_config.get('endpoint') or os.getenv('AZURE_OPENAI_ENDPOINT')
        resolved_model = model or self.llm_config.get('model')

        self.azure_endpoint = resolved_endpoint or LLM_ENDPOINT
        self.azure_key = resolved_api_key
        self.model = resolved_model or LLM_MODEL

        # （ api key  requests ）
        self.llm_available = bool(self.azure_key and requests)
        if not requests:
            print('Warning: requests package is not available; Azure HTTP calls will fail.')
        if not self.azure_key:
            logging.warning('LLM API key is not configured; LLM-assisted scoring is disabled.')

        # ，
        self._alias_map: Dict[str, str] = dict(self._AGENT_NORMALIZATION_MAP)
        if agent_aliases:
            for alias, canonical in agent_aliases.items():
                try:
                    normalized_alias = str(alias).lower().strip()
                    if not normalized_alias:
                        continue
                    canonical_name = str(canonical).strip()
                    if not canonical_name:
                        continue
                    self._alias_map[normalized_alias] = canonical_name
                    canonical_lower = canonical_name.lower()
                    if canonical_lower and canonical_lower not in self._alias_map:
                        self._alias_map[canonical_lower] = canonical_name
                    if canonical_lower.endswith(' agent'):
                        base_canonical = canonical_lower[:-6].strip()
                        if base_canonical and base_canonical not in self._alias_map:
                            self._alias_map[base_canonical] = canonical_name
                    else:
                        agent_form = f"{canonical_lower} agent"
                        if agent_form not in self._alias_map:
                            self._alias_map[agent_form] = canonical_name
                    if normalized_alias.endswith(' agent'):
                        base_alias = normalized_alias[:-6].strip()
                        if base_alias and base_alias not in self._alias_map:
                            self._alias_map[base_alias] = canonical_name
                except Exception:
                    continue

        self.agent_descriptions: Dict[str, str] = {}
        if agent_descriptions:
            for name, desc in agent_descriptions.items():
                try:
                    canonical = self._normalize_agent_identifier(name)
                except Exception:
                    canonical = None
                if not canonical:
                    continue
                if isinstance(desc, str) and desc.strip():
                    self.agent_descriptions[canonical] = desc.strip()

    @staticmethod
    def _expand_env_value(value: Any) -> Any:
        if isinstance(value, str):
            expanded = os.path.expandvars(value).strip()
            if expanded.startswith('${') and expanded.endswith('}'):
                env_name = expanded[2:-1].strip()
                return os.getenv(env_name, '').strip()
            if expanded.startswith('%') and expanded.endswith('%') and len(expanded) > 2:
                env_name = expanded[1:-1].strip()
                return os.getenv(env_name, '').strip()
            return expanded or value
        return value

    @staticmethod
    def _safe_read_yaml(path: Path) -> Dict[str, Any]:
        try:
            with path.open('r', encoding='utf-8') as handle:
                data = yaml.safe_load(handle)
                return data if isinstance(data, dict) else {}
        except Exception as exc:
            logging.debug('Failed to read YAML config %s: %s', path, exc)
            return {}

    @classmethod
    def _resolve_llm_config(cls, data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(data, dict):
            return {}

        merged: Dict[str, Any] = {}

        provider_key = data.get('provider') or data.get('default_provider') or data.get('llm_provider')
        provider_cfg: Optional[Dict[str, Any]] = None
        if provider_key:
            if isinstance(data.get(provider_key), dict):
                provider_cfg = data.get(provider_key)
            providers = data.get('providers')
            if not provider_cfg and isinstance(providers, dict):
                candidate = providers.get(provider_key)
                if isinstance(candidate, dict):
                    provider_cfg = candidate
        if not provider_cfg:
            for value in data.values():
                if isinstance(value, dict) and any(k in value for k in ('api_key', 'endpoint', 'model')):
                    provider_cfg = value
                    break

        candidates: List[Dict[str, Any]] = []
        if provider_cfg:
            candidates.append(provider_cfg)

        direct_keys = {k: data.get(k) for k in ('api_key', 'endpoint', 'model') if data.get(k)}
        if direct_keys:
            candidates.append(direct_keys)

        for candidate in candidates:
            for key in ('api_key', 'endpoint', 'model'):
                val = candidate.get(key)
                if not val:
                    continue
                merged[key] = cls._expand_env_value(val)

        return merged

    @classmethod
    def _load_llm_config_from_files(cls) -> Dict[str, Any]:
        for path in cls._CONFIG_SEARCH_PATHS:
            try:
                candidate_path = path if path.is_absolute() else Path.cwd() / path
                if not candidate_path.exists():
                    continue
                data = cls._safe_read_yaml(candidate_path)
                if not data:
                    continue

                for section_key in ('llm', 'llm_config', 'defaults'):
                    section = data.get(section_key)
                    resolved = cls._resolve_llm_config(section)
                    if resolved:
                        return resolved

                plugins = data.get('plugins')
                if isinstance(plugins, dict):
                    for plugin_data in plugins.values():
                        if not isinstance(plugin_data, dict):
                            continue
                        plugin_config = plugin_data.get('config')
                        resolved = cls._resolve_llm_config(plugin_config)
                        if resolved:
                            return resolved
            except Exception as exc:
                logging.debug('Failed to process LLM config at %s: %s', path, exc)
                continue
        return {}

    def evaluate_milestone_success_with_llm(
        self,
        trace_data: Dict,
        expected_milestones: List[str],
        milestone_type: str = "system",
        expected_subagents: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """，。"""

        if not expected_milestones:
            return {
                'scores': [],
                'passed_flags': [],
                'overall_score': 0.0,
                'normalized_overall': 0.0,
                'raw_response': '',
                'notes': 'no_milestones',
            }

        if not self.llm_available:
            raise RuntimeError("LLM evaluator unavailable for milestone scoring")

        try:
            prompt = self._build_msr_prompt(
                trace_data,
                expected_milestones,
                milestone_type,
                expected_subagents=expected_subagents,
            )
            response = self._call_llm(prompt)
            return self._parse_llm_milestone_response(
                response,
                len(expected_milestones),
                expected_milestones,
            )
        except Exception as e:
            raise RuntimeError(
                f"LLM milestone evaluation failed: {e}"
            ) from e
    
    def _build_tsr_prompt(
        self,
        trace_data: Dict,
        task_description: str,
        expected_milestones: Optional[List[str]] = None,
    ) -> str:
        """ Task Success Rate  prompt（1-5 ）。"""

        rounds = trace_data.get('rounds', []) or []
        initial_query = (
            rounds[0].get('span', {}).get('inputQuery') if rounds and isinstance(rounds[0], dict) else None
        ) or ""
        task_desc = task_description or self._extract_default_task_description(trace_data)

        responses: List[str] = []
        for round_data in rounds:
            for conv in round_data.get('span', {}).get('conversations', []) or []:
                response = (conv.get('response') or '').strip()
                if response:
                    responses.append(response)

        responses_text = "\n".join(
            f"{idx + 1}. {resp}" for idx, resp in enumerate(responses)
        ) or "No available responses"

        milestone_lines = "\n".join(
            f"{idx + 1}. {milestone}" for idx, milestone in enumerate(expected_milestones or [])
        ) or "No user-side milestone definitions"

        score_guideline_cn = (
            "Score 1 — ：。\n"
            "Score 2 — ：(50%)。\n"
            "Score 3 — ：50%。\n"
            "Score 4 — ：(75%)，。\n"
            "Score 5 — ：。"
        )
        score_guideline_en = (
            "Score 1 — Not at all satisfied: Completely fails to cover any necessary information.\n"
            "Score 2 — Barely satisfied: Covers only a small portion (less than 50%) of necessary information.\n"
            "Score 3 — Partially satisfied: Approximately 50% of necessary information is covered.\n"
            "Score 4 — Mostly satisfied: Covers the majority (more than 75%) of necessary information with only minor omissions.\n"
            "Score 5 — Fully satisfied: Completely covers all necessary information and user requirements."
        )

        prompt_cn = f"""
        ## 
        ，“” (user_side_milestones)。：
            1. ** milestone 。**
            2. ** evidence-based reasoning（）。**
            3. **、、、。**
        
        ：
        - ："{initial_query}"
        - ：{task_desc}
        - ：
        {milestone_lines}
        - （）：
        {responses_text}
        - ：。：
        {score_guideline_cn}
        ：**、** milestone 。

        ## 
        ：
        1. （Evidence-based）： responses ，****，， “”
        2. ：  responses ，****

        ## 
        （ JSON，， markdown  ```json ）：
        {{
        "milestones": [
            {{"index": 1, "reason": "，", "score": <1-5 >}},
            ...
        ],
        "overall_score": <1-5 ，>,
        "notes": ""
        }}

        ：
        1. ， 5 ， 1 ；。
        2. score ≥ 3 ；score < 3 。
        3. overall_score （ 1-5 ）。
            """
        
        promp_en = f"""
        ## Task Background
        You are a meticulous multi-agent system evaluation expert responsible for assessing whether the system's responses meet specific user-side milestones from a "user perspective". Your core capabilities include:
            1. **Judging whether each milestone is fully covered by the system's responses.**
            2. **Strictly outputting scores based on evidence-based reasoning.**
            3. **Maintaining a structured, explainable, repeatable, and verifiable analysis approach.**
        Your inputs include:
        - User's initial query: "{initial_query}"
        - Task description: {task_desc}
        - Expected user-side milestones:
        {milestone_lines}
        - System response outputs (chronologically concatenated responses):
        {responses_text}
        - Evaluation guidelines: Analyze whether the output covers each user-side milestone item by item. Scoring criteria:
        {score_guideline_en}
        You need to determine: whether the system has **completely, accurately, and clearly** met the requirements of each milestone in these responses.
        
        ## Evaluation Principles
        During your evaluation process, you must adhere to two core principles:
        1. Evidence-based: Your conclusions must strictly come from the information presented in the responses. **You are NOT allowed** to assume unmentioned information, infer possible but unexpressed actions, or "charitably fill in" for the system.
        2. Semantic logical matching: Your conclusions must be based on the semantic understanding of the responses. **You are NOT allowed** to judge whether a milestone is met solely based on keyword matching or surface text similarity.
        
        ## Output Requirements
        Output format (be sure to return plain text JSON directly, using double quotes, without markdown code blocks or ```json wrappers):
        {{
        "milestones": [
            {{"index": 1, "reason": "Brief reason, strictly following evaluation principles", "score": <integer 1-5>}},
            ...
        ],
        "overall_score": <number 1-5, can be decimal>,
        "notes": "Optional overall remarks"
        }}
        Scoring rules:
        1. Score each milestone, with a maximum of 5 points and a minimum of 1 point; scores must be integers.
        2. score ≥ 3 indicates that the milestone is sufficiently met; score < 3 indicates not met or only partially met.
        3. overall_score provides a comprehensive evaluation of all milestones (also scored 1-5).
        """

        prompt = promp_en
        return prompt

    def _build_msr_prompt(
        self,
        trace_data: Dict,
        expected_milestones: List[str],
        milestone_type: str,
        expected_subagents: Optional[List[str]] = None,
    ) -> str:
        """ prompt（，1-5 ）。"""

        rounds = trace_data.get('rounds', []) or []
        milestone_lines = "\n".join(
            f"{idx + 1}. {milestone}" for idx, milestone in enumerate(expected_milestones or [])
        ) or "No milestone definitions"

        score_guideline_cn = (
            "Score 1 — ：。\n"
            "Score 2 — ：(50%)。\n"
            "Score 3 — ：50%。\n"
            "Score 4 — ：(75%)，。\n"
            "Score 5 — ：。"
        )
        score_guideline_en = (
            "Score 1 — Not at all satisfied: Completely fails to cover any necessary information.\n"
            "Score 2 — Barely satisfied: Covers only a small portion (less than 50%) of necessary information.\n"
            "Score 3 — Partially satisfied: Approximately 50% of necessary information is covered.\n"
            "Score 4 — Mostly satisfied: Covers the majority (more than 75%) of necessary information with only minor omissions.\n"
            "Score 5 — Fully satisfied: Completely covers all necessary information and user requirements."
        )

        if milestone_type == 'user':
            prompt = self._build_tsr_prompt(
                trace_data,
                self._extract_default_task_description(trace_data),
                expected_milestones,
            )
            return prompt

        conversations_summary = self._summarize_conversations_with_queries(rounds)

        steps_context = self._summarize_initial_steps(trace_data)
        subagent_lines = ""
        if expected_subagents:
            agent_detail_lines: List[str] = []
            for agent in expected_subagents:
                desc = self.agent_descriptions.get(agent) if hasattr(self, 'agent_descriptions') else None
                if not desc:
                    desc = "（Not provided，）"
                display_name = agent
                if '_' in display_name and ' ' not in display_name:
                    display_name = display_name.replace('_', ' ')
                if not display_name.lower().endswith('agent'):
                    display_name = f"{display_name} agent"
                agent_detail_lines.append(f"    - {display_name}: {desc}")
            details_block = "\n".join(agent_detail_lines)
            subagent_lines_cn = (
                "- ：\n"
                f"{details_block}\n"
            )
            subagent_lines_en = (
                "- Key Sub-Agent Capabilities Reference:\n"
                f"{details_block}\n"
            )
        else:
            subagent_lines_cn = "- ：\n"
            subagent_lines_en = "- Key Sub-Agent Capabilities Reference: Not specified\n"


        prompt_en = f"""
        ## Task Background
        You are a meticulous multi-agent system evaluation expert responsible for assessing whether the orchestrator's issued instructions and scheduled sub-agent actions meet specific system-side milestones from a "system orchestration perspective". Your core capabilities include:
            1. **Judging whether each milestone is fully covered by the system-generated action trajectory.**
            2. **Strictly outputting scores based on evidence-based reasoning.**
            3. **Maintaining a structured, explainable, repeatable, and verifiable analysis approach.**
        Your inputs include:
        - Initial task steps (from the steps field of the first round's inputQuery):
        {steps_context}
        - Expected system-side milestones:
        {milestone_lines}
        - Key sub-agent capability reference:
        {subagent_lines_en}
        - System multi-round orchestration action trajectory (including query, subAgentName):
        {conversations_summary}
        - Scoring criteria:
        {score_guideline_en}
        You need to determine: whether the system has **completely, accurately, and clearly** executed the actions stipulated by each milestone in these action trajectory sequences.
        
        ## Evaluation Principles
        You need to base your evaluation on the following key verification points (perform the following checks for each milestone):
        1. Identify the specific sub-agent (subAgentName) that completes the milestone. If the corresponding sub-agent does not appear, or the wrong sub-agent is called (inconsistent with expectations), the milestone should be judged as not met (score=1).
        2. For the call to that sub-agent, verify whether its triggering query is closely related to the current task progress, can naturally transition from previous actions logically, and clearly advances the milestone goal; hints such as "for flight info" and "for trip dates" in the milestone description are only for emphasis and do not require verification of whether specific information is successfully returned.
        3. Under the premise of correct sub-agent invocation and reasonable query, observe whether the response at least attempts to advance the goal (e.g., returning candidates, explaining unavailability, etc.). If the call chain is complete but there is no data due to external limitations, a score of 2-3 can be given at discretion; only when the call chain is complete and the response covers key points can a score of 4-5 be given; any missing basic conditions should be downgraded to 1 point.
        
        ## Output Requirements
        Output format (be sure to return plain text JSON directly, using double quotes, without markdown code blocks or ```json wrappers):
        {{
            "milestones": [
                {{"index": 1, "agent": "relevant sub-agent", "reason": "Brief reason explaining your thought process, specifying the sub-agent used for the milestone, the corresponding query, and evaluation basis", "score": <integer 1-5>}},
                ...
            ],
            "overall_score": <number 1-5, can be decimal>,
            "notes": "Optional overall remarks"
        }}
        Scoring rules:
        1. Score each milestone, with a maximum of 5 points and a minimum of 1 point; scores must be integers.
        2. score ≥ 3 indicates that the milestone is sufficiently met; score < 3 indicates not met or only partially met.
        3. overall_score provides a comprehensive evaluation of all milestones (also scored 1-5).
        """
        prompt = prompt_en
        return prompt


    def _call_llm(self, prompt: str) -> str:
        """ Azure OpenAI HTTP （）。

         `requests`  HTTP ，。
        """
        if not requests:
            raise RuntimeError('requests is required for Azure HTTP calls but is not installed')

        #  URL（Azure OpenAI ）
        url = f"{self.azure_endpoint}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.azure_key}",
        }

        #  messages（ prompt）
        if isinstance(prompt, list):
            messages = prompt
        else:
            messages = [
                {"role": "system", "content": "。"},
                {"role": "user", "content": prompt}
            ]

        payload = {
            'model': self.model,
            'messages': messages,
            'temperature': 0.0,
            'max_completion_tokens': 4096,
            'stream': False
        }

        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=60, verify=True)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f'Azure LLM HTTP request failed: {exc}') from exc

        try:
            parsed = resp.json()
        except ValueError as exc:
            raise RuntimeError(f'LLM response was not valid JSON: {exc}') from exc

        if not isinstance(parsed, dict):
            raise RuntimeError('LLM response payload is not a JSON object')

        choices = parsed.get('choices') or parsed.get('outputs')
        if not isinstance(choices, list) or not choices:
            preview = json.dumps(parsed, ensure_ascii=False)[:500]
            raise RuntimeError(f'LLM response missing choices: {preview}')

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise RuntimeError('LLM response first choice is not a JSON object')

        message = first_choice.get('message') or first_choice.get('delta') or {}
        if isinstance(message, dict):
            content = message.get('content')
            if isinstance(content, str) and content.strip():
                return content
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        text_value = part.get('text')
                        if text_value:
                            return text_value
                    elif isinstance(part, str) and part.strip():
                        return part

        text_fallback = first_choice.get('text')
        if isinstance(text_fallback, str) and text_fallback.strip():
            return text_fallback

        raise RuntimeError('LLM response did not contain any message content')
    
    def _parse_llm_tsr_response(self, response: str) -> float:
        """"""
        if not response:
            return 0.5
        try:
            payload = json.loads(response)
            if isinstance(payload, dict) and 'overall_score' in payload:
                overall = float(payload['overall_score'])
                return max(0.0, min(overall, 5.0)) / 5.0
        except Exception:
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                try:
                    payload = json.loads(match.group(0))
                    if isinstance(payload, dict) and 'overall_score' in payload:
                        overall = float(payload['overall_score'])
                        return max(0.0, min(overall, 5.0)) / 5.0
                except Exception:
                    pass
        # 
        score_match = re.search(r'(\d+\.\d+|\d+)', response)
        if score_match:
            score = float(score_match.group(1))
            if score > 1.0:
                normalized = min(max(score, 1.0), 5.0) / 5.0
            else:
                normalized = min(max(score, 0.0), 1.0)
            return normalized
        return 0.5  # 

    def _parse_llm_milestone_response(
        self,
        response: str,
        expected_count: int,
        expected_milestones: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """ LLM  JSON ， 1-5 。"""

        raw = response or ""
        json_payload = None
        try:
            json_payload = json.loads(raw)
        except Exception:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                try:
                    json_payload = json.loads(match.group(0))
                except Exception:
                    json_payload = None
        if not isinstance(json_payload, dict):
            raise ValueError("LLM response is not valid JSON for milestone evaluation")

        milestones_data = json_payload.get('milestones')
        if not isinstance(milestones_data, list):
            raise ValueError("LLM JSON missing 'milestones' list")

        scores: List[int] = []
        passed: List[bool] = []
        normalized_scores: List[float] = []
        reasons: List[str] = []

        for idx, entry in enumerate(milestones_data):
            if not isinstance(entry, dict):
                continue
            score_raw = entry.get('score')
            try:
                score_int = int(round(float(score_raw)))
            except Exception:
                score_int = 1
            score_int = max(1, min(5, score_int))
            scores.append(score_int)
            passed.append(score_int >= 3)
            normalized_scores.append(score_int / 5.0)
            reasons.append(entry.get('reason') or '')

        if expected_count and len(scores) < expected_count:
            # ，
            missing = expected_count - len(scores)
            scores.extend([1] * missing)
            passed.extend([False] * missing)
            normalized_scores.extend([0.2] * missing)

        average_score = sum(scores) / max(len(scores), 1)
        normalized_average = average_score / 5.0

        return {
            'scores': scores,
            'passed_flags': passed,
            'normalized_scores': normalized_scores,
            'overall_score': average_score,
            'normalized_overall': normalized_average,
            'raw_response': raw,
            'reasons': reasons,
            'expected_milestones': expected_milestones or [],
            'json': json_payload,
            'notes': json_payload.get('notes'),
        }
    
    def _parse_llm_quality_response(self, response: str) -> float:
        """"""
        return self._parse_llm_tsr_response(response)  # 
    
    def _summarize_conversations(self, rounds: List) -> str:
        """"""
        summary = ""
        for i, round_data in enumerate(rounds):
            span = round_data['span']
            summary += f"{i+1}:\n"
            summary += f": {span.get('inputQuery', '')}\n"
            
            for j, conv in enumerate(span.get('conversations', [])):
                summary += f"  {j+1}({conv.get('subAgentName', 'Unknown')}): {conv.get('response', '')[:200]}...\n"
            
            summary += "\n"
        
        return summary

    def _summarize_conversations_with_queries(self, rounds: List) -> str:
        """ subAgentName、query  response 。"""
        segments: List[str] = []
        for i, round_data in enumerate(rounds):
            span = round_data.get('span', {}) or {}
            for j, conv in enumerate(span.get('conversations', []) or []):
                agent = conv.get('subAgentName') or conv.get('agent') or 'Unknown agent'
                agent = self._normalize_agent_identifier(agent) or agent
                query = (conv.get('query') or '').strip() or '（ query）'
                # response = (conv.get('response') or '').strip() or '（）'
                # segments.append(
                #     f" {i + 1}-{j + 1}: {agent}\n  Query: {query}\n  Response: {response}"
                # )
                # response query
                segments.append(
                    f" {i + 1}-{j + 1}: {agent}\n  Query: {query}\n "
                )
        return "\n".join(segments) or "No session records"

    def _summarize_initial_steps(self, trace_data: Dict) -> str:
        """ steps ，。"""
        rounds = trace_data.get('rounds', []) or []
        if not rounds:
            return "No available steps"

        steps_raw = rounds[0].get('span', {}).get('steps')
        if not steps_raw:
            return "No available steps"

        try:
            steps_json = json.loads(steps_raw) if isinstance(steps_raw, str) else steps_raw
            values = steps_json.get('values') if isinstance(steps_json, dict) else None
            if isinstance(values, list) and values:
                lines = []
                for idx, item in enumerate(values, 1):
                    text = item.get('string_value') if isinstance(item, dict) else str(item)
                    if text:
                        lines.append(f"{idx}. {text}")
                if lines:
                    return "\n".join(lines)
        except Exception:
            pass

        if isinstance(steps_raw, str):
            return steps_raw
        return json.dumps(steps_raw, ensure_ascii=False)

    def _extract_default_task_description(self, trace_data: Dict) -> str:
        """。"""
        rounds = trace_data.get('rounds', []) or []
        if not rounds:
            return "Not provided"

        initial_query = (rounds[0].get('span', {}) or {}).get('inputQuery') or ""
        steps_summary = self._summarize_initial_steps(trace_data)
        if steps_summary and steps_summary != "No available steps":
            return f"：{initial_query}；：{steps_summary}"
        return initial_query or "Not provided"
    
    def evaluate_milestone_statuses(
        self,
        trace_data: Dict,
        expected_milestones: Optional[List[str]],
        expected_subagents: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Provide per-milestone heuristic status leveraged for transparency."""

        details: List[Dict[str, Any]] = []
        milestones = [milestone for milestone in (expected_milestones or []) if milestone]

        rounds = trace_data.get('rounds', []) or []
        collected_texts: List[str] = []
        for round_data in rounds:
            span = round_data.get('span', {}) or {}
            input_query = span.get('inputQuery')
            if input_query:
                collected_texts.append(str(input_query).lower())
            for conv in span.get('conversations', []) or []:
                collected_texts.append(str(conv.get('query') or '').lower())
                collected_texts.append(str(conv.get('response') or '').lower())
                collected_texts.append(str(conv.get('subAgentName') or '').lower())
                collected_texts.append(str(conv.get('status') or '').lower())

        corpus = "\n".join(text for text in collected_texts if text)

        for milestone in milestones:
            tokens = [
                token for token in re.split(r"[^\w\u4e00-\u9fff]+", (milestone or '').lower())
                if len(token) >= 2
            ]
            total_tokens = len(tokens) or 1
            matches = sum(1 for token in tokens if token and token in corpus)
            coverage = matches / total_tokens
            passed = coverage >= 0.6 and bool(tokens)
            details.append({
                'type': 'milestone',
                'milestone': milestone,
                'coverage': round(coverage, 3),
                'passed': passed,
                'matched_tokens': matches,
                'total_tokens': total_tokens,
            })

        if expected_subagents:
            expected_norm = {
                self._normalize_agent_identifier(agent)
                for agent in expected_subagents
                if agent
            }
            actual_agents: Set[str] = set()
            for round_data in rounds:
                for conv in round_data.get('span', {}).get('conversations', []) or []:
                    name = conv.get('subAgentName') or conv.get('agent')
                    if name:
                        actual_agents.add(self._normalize_agent_identifier(name))

            coverage = (
                len(expected_norm & actual_agents) / len(expected_norm)
                if expected_norm
                else 1.0
            )
            details.append({
                'type': 'expected_subagents',
                'milestone': 'expected_subagents',
                'coverage': round(coverage, 3),
                'passed': coverage >= 1.0 if expected_norm else True,
                'expected_agents': sorted(expected_norm),
                'observed_agents': sorted(actual_agents),
            })

        return details

    def _normalize_agent_identifier(self, name: Optional[str]) -> str:
        """Normalize agent identifiers (slug or descriptive) to canonical form."""
        if not name:
            return ''

        base = str(name).lower().strip()
        if not base:
            return ''

        # Remove any descriptive suffix after ' - '
        if ' - ' in base:
            base = base.split(' - ', 1)[0].strip()

        # Replace underscores and repeated whitespace with single spaces
        base = re.sub(r'[_]+', ' ', base)
        base = re.sub(r'\s+', ' ', base).strip()

        if not base:
            return ''

        alias_map = getattr(self, '_alias_map', self._AGENT_NORMALIZATION_MAP)

        canonical = alias_map.get(base)
        if canonical:
            return canonical

        if not base.endswith('agent'):
            base = f"{base} agent" if not base.endswith(' agent') else base

        canonical = alias_map.get(base)
        if canonical:
            return canonical

        return base

    def classify_clarification(self, clarified_queries: List[str], expected_clarifications: List[str], lang='en') -> List[str]:
        """clarified_queriesclarifications， """
        
        matched_results = []
        raw_result = []
        for query in clarified_queries:

            prompt_en = f"""
            You are a rigorous and explainable expert in text classification and semantic matching.

            Note: The "current clarification query" here refers to a **system-initiated clarification question**,
            whose purpose is to ask the user for missing or ambiguous information, rather than to provide an answer.

            Your task is to **determine whether the clarification question is explicitly and intentionally asking for
            one or more pieces of information described in the expected clarification list**, or whether it does not
            clearly correspond to any expected clarification item.

            ## Input
            Expected clarification items (each item describes a specific information point that the system intends to clarify):
            {expected_clarifications}

            Current clarification query:
            {query}

            ## Judgment Rules (Very Important)
            Please strictly follow the steps below:

            1. For **each item** in the expected clarification list, determine whether the clarification question
            clearly asks the user for the information described by that item in terms of **semantic intent and information target**;
            2. The judgment should be based on the **information-seeking intent and target** of the question,
            not on surface-level or exact wording similarity;
            3. Only when the question clearly attempts to obtain the information corresponding to a specific expected clarification item
            should it be considered a match;
            4. If the question is too broad, vague, or cannot be clearly mapped to any specific expected clarification item,
            do not treat it as matching any item;
            5. Do not force a match. It is acceptable — and common in real systems — for a clarification question
            to match none of the expected clarification items.

            ## Output Requirements
            Return **only a pure JSON object** (do not use markdown and do not add any extra text) in the following format:

            {{
                "reason": "Briefly explain your reasoning, including what information the clarification question is attempting to obtain, and why it corresponds (or does not correspond) to specific expected clarification items",
                "result": true or false,
                "matched_item": ["<exact text of the expected clarification item explicitly targeted by the question>", "..."]
            }}

            ## Output Constraints
            - result must be true if and only if matched_item is a non-empty list; if matched_item is an empty list, result must be false;
            - Each element in matched_item **must be selected verbatim from the expected clarification list**;
            - Multiple matched items are allowed **only when the clarification question clearly and explicitly asks for multiple distinct information points**;
            - If the question broadly touches on multiple aspects but lacks clear and specific information targets, return an empty list;
            - Do not invent or paraphrase expected clarification items that are not present in the input;
            - Do not output anything outside the JSON object.
            """

            prompt = prompt_en
            response = self._call_llm(prompt)
        
            try:
                result_json = json.loads(response)
                if (
                    isinstance(result_json, dict)
                    and result_json.get('result') == True
                    and isinstance(result_json.get('matched_item'), list)
                ):
                    for item in result_json['matched_item']:
                        if item in expected_clarifications and item not in matched_results:
                            matched_results.append(item)
                raw_result.append(result_json)
            except Exception:
                continue

        return matched_results, raw_result


class LLMEnhancedEvaluator(BaseEvaluator):
    """"""
    
    def __init__(
        self,
        tasks_config=None,
        agents_config=None,
        user_profiles=None,
        llm_api_key: Optional[str] = None,
        llm_model: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        llm_endpoint: Optional[str] = None,
    ):
        super().__init__(tasks_config, agents_config, user_profiles)
        self.runtime_config = config or {}
        agent_context = getattr(self, 'agent_descriptions', {}) if hasattr(self, 'agent_descriptions') else {}
        agent_aliases = getattr(self, 'agent_aliases', {}) if hasattr(self, 'agent_aliases') else {}
        self.llm_evaluator = LLMAssistedEvaluator(
            api_key=llm_api_key,
            model=llm_model,
            config=self.runtime_config,
            endpoint=llm_endpoint,
            agent_descriptions=agent_context,
            agent_aliases=agent_aliases,
        )
        self.evaluator_type = "llm_enhanced"
        if not self.llm_evaluator.llm_available:
            raise RuntimeError(
                "LLMEnhancedEvaluator requires an available LLM but none was detected."
            )

    def _calculate_clarification_efficiency(self, rounds: List, expected_clarifications: List) -> float:
        """F1"""
        if not expected_clarifications:
            return 1.0

        clarified_queries = []
        for r in rounds:
            for conv in r["span"].get("conversations", []):
                if conv.get("status") == "input-required":
                    clarified_queries.append(conv.get("query"))

        matched_items, raw_results = self.llm_evaluator.classify_clarification(clarified_queries, expected_clarifications)

        recall = len(matched_items) / len(expected_clarifications) if expected_clarifications else 1.0
        precision = len(matched_items) / len(clarified_queries) if clarified_queries else 1.0

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        return round(f1, 4)

    def _calculate_agent_routing_accuracy(self, rounds: List[dict], expected_subagents: List[dict]) -> float:
        if not expected_subagents:
            return 1.0
        # expected_subagents_agent
        expected_subagents = [item+"_agent" if not item.endswith("_agent") else item for item in expected_subagents]

        chat_agent_used = False
        called_subagents = []
        for r in rounds:
            for conv in r["span"].get("conversations", []):
                agent = conv.get("subAgentName") or conv.get("agent")
                # agent, OneflowHotel Recommendation Agent，expected_subagents(hotel_recommendation_agent)
                agent = agent.lower().replace(" ", "_")
                if agent == "chat_agent":
                    chat_agent_used = True
                called_subagents.append(agent)

        matched = [item for item in called_subagents if item in expected_subagents]
        # print(f"Expected subagents: {expected_subagents}, Called subagents: {called_subagents}, Matched: {matched}")
        # matched
        matched = list(set(matched))
        recall = len(matched) / len(expected_subagents)
        precision = len(matched) / len(called_subagents) if called_subagents else 1.0

        # expected_subagentschat_agent
        if "chat_agent" not in expected_subagents and chat_agent_used:
            penalty = 0.8
        else:
            penalty = 1.0

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        score = f1 * penalty
        return round(score, 4)
    
    def evaluate_trace(self, trace_data: Dict) -> Dict:
        """LLMtrace"""
        # 
        language_domain = self._analyze_language_domain(trace_data)
        language = language_domain['language']
        domain = language_domain['domain']
        transaction_id = trace_data.get('transactionId') or trace_data.get('id') or 'unknown'
        rounds = trace_data.get('rounds', [])
        total_rounds = len(rounds)
        input_query = ''
        if rounds and isinstance(rounds[0], dict):
            input_query = (rounds[0].get('span', {}) or {}).get('inputQuery', '')

        _llm_eval_logger.info(
            "trace_start transaction_id=%s domain=%s language=%s rounds=%s initial_query=%s",
            transaction_id,
            domain,
            language,
            total_rounds,
            _truncate_for_log(input_query),
        )
        
        evaluation = {
            'transaction_id': trace_data.get('transactionId'),
            'total_rounds': total_rounds,
            'language': language,
            'domain': domain,
            'dimensions': {},
            'task_config_used': False,  # ：
            'evaluator_type': self.evaluator_type
        }
        
        initial_query = input_query or ""
        evaluation['initial_query'] = initial_query
        
        #  transaction id  tasks.yaml  transcantion_id/transaction_id 
        transaction_key = trace_data.get('transactionId') or trace_data.get('transaction_id')
        task_info = None
        match_method = None
        if transaction_key:
            try:
                task_info = self._find_task_by_transaction(transaction_key)
                if task_info:
                    match_method = 'transaction'
            except Exception:
                task_info = None

        # ， query 
        if not task_info:
            task_info = self._find_matching_task(initial_query)
            if task_info:
                match_method = 'query'

        task_config_used = task_info is not None
        evaluation['task_config_used'] = task_config_used

        if task_config_used:
            matched_task_id = task_info['task_id']
            evaluation['matched_task_id'] = matched_task_id
            print(f": {matched_task_id}")
            # （transaction）
            match_method = match_method or 'query'
            _llm_eval_logger.info(
                "task_config_matched transaction_id=%s task_id=%s match_via=%s", transaction_id, matched_task_id, match_method
            )
        else:
            _llm_eval_logger.info("task_config_not_found transaction_id=%s", transaction_id)
        
        # （）
        task_config = task_info.get('config', {}) if task_info else {}
        expected_clarifications = task_config.get('expected_clarifications', []) or []
        expected_subagents = task_config.get('expected_subagents', []) or []
        base_dimensions: Dict[str, float] = {
            'agent_routing_accuracy': self._calculate_agent_routing_accuracy(rounds, expected_subagents),
            'average_rounds_per_task': len(rounds),
            'orchestration_latency': self._calculate_orchestration_latency(rounds),
            'clarification_efficiency': self._calculate_clarification_efficiency(rounds, expected_clarifications),
            'completeness': self._calculate_completeness(rounds),
        }
        
        raw_expected_subagents = task_config.get('expected_subagents', []) or []
        expected_subagents: List[str] = []
        seen_subagents: Set[str] = set()
        if self.llm_evaluator:
            for agent in raw_expected_subagents:
                canonical = self.llm_evaluator._normalize_agent_identifier(agent)
                if canonical and canonical not in seen_subagents:
                    expected_subagents.append(canonical)
                    seen_subagents.add(canonical)
        user_milestones = task_config.get('user_side_milestones', []) or []
        system_milestones = task_config.get('system_side_milestones', []) or []
        task_description = self._extract_task_description(trace_data)

        user_complete_rate, user_success_rate, user_used_llm, user_milestone_details, user_llm_payload = self._evaluate_side_metrics_with_llm(
            trace_data=trace_data,
            milestones=user_milestones,
            milestone_type='user',
            expected_subagents=expected_subagents,
            task_description=task_description,
        )
        system_complete_rate, system_success_rate, system_used_llm, system_milestone_details, system_llm_payload = self._evaluate_side_metrics_with_llm(
            trace_data=trace_data,
            milestones=system_milestones,
            milestone_type='system',
            expected_subagents=expected_subagents,
            task_description=task_description,
        )

        base_dimensions['u_task_complete_rate'] = user_complete_rate
        base_dimensions['u_task_success_rate'] = user_success_rate
        base_dimensions['s_task_complete_rate'] = system_complete_rate
        base_dimensions['s_task_success_rate'] = system_success_rate
        base_dimensions['execution_efficiency'] = self._calculate_execution_efficiency(
            rounds,
            user_complete_rate,
            system_complete_rate,
        )

        llm_scores = {
            'u_task_complete_rate': user_complete_rate,
            'u_task_success_rate': user_success_rate,
            's_task_complete_rate': system_complete_rate,
            's_task_success_rate': system_success_rate,
            'completeness': base_dimensions.get('completeness', 0.0),
        }
        
        # （ + agent_routing +）
        base_dimensions['overall_orchestration_score'] = self._calculate_overall_score(base_dimensions)

        evaluation['dimensions'] = base_dimensions
        evaluation['llm_assisted_scores'] = llm_scores
        evaluation['task_milestones'] = {
            'user_side': user_milestones,
            'system_side': system_milestones,
            'expected_subagents': expected_subagents,
        }
        evaluation['milestone_details'] = {
            'user_side': user_milestone_details,
            'system_side': system_milestone_details,
        }
        evaluation['llm_usage'] = {
            'user_side': user_used_llm,
            'system_side': system_used_llm,
        }
        evaluation['llm_milestone_scores'] = {
            'user_side': user_llm_payload,
            'system_side': system_llm_payload,
        }

        _llm_eval_logger.info(
            "trace_complete transaction_id=%s overall_score=%.3f user_complete=%.3f system_complete=%.3f"
            " user_success=%.3f system_success=%.3f execution_efficiency=%.3f clarification_efficiency=%.3f agent_routing_accuracy=%.3f",
            transaction_id,
            base_dimensions.get('overall_orchestration_score', 0.0),
            base_dimensions.get('u_task_complete_rate', 0.0),
            base_dimensions.get('s_task_complete_rate', 0.0),
            base_dimensions.get('u_task_success_rate', 0.0),
            base_dimensions.get('s_task_success_rate', 0.0),
            base_dimensions.get('execution_efficiency', 0.0),
            base_dimensions.get('clarification_efficiency', 0.0),
            base_dimensions.get('agent_routing_accuracy', 0.0)
        )

        
        return evaluation
    
    def _extract_task_description(self, trace_data: Dict) -> str:
        """trace"""
        rounds = trace_data.get('rounds', [])
        if not rounds:
            return "Unknown task"
        
        initial_query = rounds[0]['span'].get('inputQuery', '')
        steps = rounds[0]['span'].get('steps', '')
        
        description = f": {initial_query}\n"
        
        try:
            steps_data = json.loads(steps)
            steps_list = steps_data.get('values', [])
            if steps_list:
                description += ":\n"
                for step in steps_list:
                    description += f"- {step.get('string_value', '')}\n"
        except:
            pass
        
        return description

    def _evaluate_side_metrics_with_llm(
        self,
        trace_data: Dict,
        milestones: List[str],
        milestone_type: str,
        expected_subagents: List[str],
        task_description: Optional[str],
    ) -> Tuple[float, float, bool, List[Dict[str, Any]], Dict[str, Any]]:
        """Use LLM milestone reasoning to derive completion/success metrics."""

        transaction_id = trace_data.get('transactionId') or trace_data.get('id') or 'unknown'
        milestone_count = len([m for m in milestones if m])
        _llm_eval_logger.info(
            "milestone_eval_start transaction_id=%s milestone_type=%s milestone_count=%s expected_agents=%s",
            transaction_id,
            milestone_type,
            milestone_count,
            ','.join(expected_subagents) if expected_subagents else '',
        )

        effective_milestones = [m for m in milestones if m]
        if not effective_milestones and milestone_type == 'system' and expected_subagents:
            effective_milestones = [
                f"Ensure agent '{agent}' contributes appropriately to the task outcome."
                for agent in expected_subagents
                if agent
            ]

        label = 'User-side' if milestone_type == 'user' else 'System-side'

        milestone_details: List[Dict[str, Any]] = []
        if self.llm_evaluator:
            milestone_details = self.llm_evaluator.evaluate_milestone_statuses(
                trace_data,
                milestones,
                expected_subagents=expected_subagents if milestone_type == 'system' else None,
            )

        if not effective_milestones:
            print(f"{label} complete/success evaluation skipped: no milestones defined.")
            empty_payload = {
                'scores': [],
                'passed_flags': [],
                'normalized_scores': [],
                'reasons': [],
                'overall_score': 0.0,
                'normalized_overall': 0.0,
                'milestones': [],
                'notes': 'no_milestones',
            }
            _llm_eval_logger.info(
                "milestone_eval_skipped transaction_id=%s milestone_type=%s reason=no_milestones",
                transaction_id,
                milestone_type,
            )
            return 0.0, 0.0, False, milestone_details, empty_payload

        if not self.llm_evaluator or not self.llm_evaluator.llm_available:
            raise RuntimeError("LLM evaluator is unavailable; cannot compute milestone metrics.")

        expected_subagents_for_llm = expected_subagents if milestone_type == 'system' else []

        try:
            _llm_eval_logger.info(
                "llm_request_start transaction_id=%s milestone_type=%s total_milestones=%s",
                transaction_id,
                milestone_type,
                len(effective_milestones),
            )
            llm_result = self.llm_evaluator.evaluate_milestone_success_with_llm(
                trace_data,
                effective_milestones,
                milestone_type=milestone_type,
                expected_subagents=expected_subagents_for_llm,
            )
        except Exception as exc:
            _llm_eval_logger.exception(
                "llm_request_error transaction_id=%s milestone_type=%s error=%s",
                transaction_id,
                milestone_type,
                exc,
            )
            raise RuntimeError(
                f"LLM evaluation failed for {milestone_type} milestones: {exc}"
            ) from exc

        scores = llm_result.get('scores') or []
        passed_flags = llm_result.get('passed_flags') or [score >= 3 for score in scores]
        normalized_scores = llm_result.get('normalized_scores') or [score / 5.0 for score in scores]
        reasons = llm_result.get('reasons') or []

        total_milestones = len(effective_milestones)
        if total_milestones == 0:
            _llm_eval_logger.info(
                "milestone_eval_empty transaction_id=%s milestone_type=%s",
                transaction_id,
                milestone_type,
            )
            return 0.0, 0.0, False, milestone_details, {
                'scores': scores,
                'passed_flags': passed_flags,
                'normalized_scores': normalized_scores,
                'reasons': reasons,
                'overall_score': llm_result.get('overall_score', 0.0),
                'normalized_overall': llm_result.get('normalized_overall', 0.0),
                'milestones': [],
                'notes': llm_result.get('notes'),
            }

        def _safe_get(items: List[Any], index: int, default: Any) -> Any:
            return items[index] if index < len(items) else default

        considered_pass_flags: List[bool] = []
        considered_scores: List[int] = []
        considered_normalized: List[float] = []
        considered_reasons: List[str] = []

        for idx in range(total_milestones):
            raw_score = _safe_get(scores, idx, 1)
            try:
                score_value = float(raw_score)
            except Exception:
                score_value = 1.0
            score_value = max(1.0, min(score_value, 5.0))
            score_int = int(round(score_value))
            considered_scores.append(score_int)

            raw_pass = _safe_get(passed_flags, idx, score_int >= 3)
            considered_pass_flags.append(bool(raw_pass))

            raw_normalized = _safe_get(normalized_scores, idx, score_int / 5.0)
            try:
                normalized_value = float(raw_normalized)
            except Exception:
                normalized_value = score_int / 5.0
            normalized_value = max(0.0, min(normalized_value, 1.0))
            considered_normalized.append(normalized_value)

            considered_reasons.append(str(_safe_get(reasons, idx, '')))

        passed_count = sum(1 for flag in considered_pass_flags if flag)
        complete_rate = passed_count / total_milestones
        success_rate = 1.0 if passed_count == total_milestones else 0.0

        raw_average_score = llm_result.get('overall_score')
        if raw_average_score is None:
            raw_average_score = sum(considered_scores) / total_milestones
        normalized_average = llm_result.get('normalized_overall')
        if normalized_average is None:
            normalized_average = raw_average_score / 5.0

        complete_rate = max(0.0, min(float(complete_rate), 1.0))
        normalized_average = max(0.0, min(float(normalized_average), 1.0))

        _llm_eval_logger.info(
            "milestone_eval_result transaction_id=%s milestone_type=%s completion_rate=%.3f success_rate=%.3f"
            " average_score=%.3f normalized_average=%.3f notes=%s",
            transaction_id,
            milestone_type,
            complete_rate,
            success_rate,
            raw_average_score,
            normalized_average,
            _truncate_for_log(llm_result.get('notes')),
        )

        print(f"{label} LLM milestone scores:")
        for idx, milestone_text in enumerate(effective_milestones):
            score = considered_scores[idx]
            reason = considered_reasons[idx]
            status = 'PASS' if considered_pass_flags[idx] else 'FAIL'
            print(f"  - [{status}] Score={score} Milestone: {milestone_text} Reason: {reason}")

        for idx, milestone_text in enumerate(effective_milestones):
            _llm_eval_logger.info(
                "milestone_detail transaction_id=%s milestone_type=%s index=%s status=%s score=%s normalized=%.3f"
                " milestone=%s reason=%s",
                transaction_id,
                milestone_type,
                idx,
                'PASS' if considered_pass_flags[idx] else 'FAIL',
                considered_scores[idx],
                considered_normalized[idx],
                _truncate_for_log(milestone_text, limit=320),
                _truncate_for_log(considered_reasons[idx], limit=320),
            )

        if milestone_details:
            print(f"{label} heuristic coverage review:")
            for detail in milestone_details:
                status = 'PASS' if detail.get('passed') else 'FAIL'
                if detail.get('type') == 'milestone':
                    print(
                        f"  - [{status}] {detail.get('milestone')} (coverage={detail.get('coverage', 0.0):.3f})"
                    )
                elif detail.get('type') == 'expected_subagents':
                    expected_agents = detail.get('expected_agents') or []
                    observed_agents = detail.get('observed_agents') or []
                    coverage_value = detail.get('coverage', 0.0)
                    print(
                        f"  - [{status}] expected_subagents coverage={coverage_value:.3f} "
                        f"expected={expected_agents} observed={observed_agents}"
                    )

        if milestone_details:
            detail_summary = []
            for detail in milestone_details:
                detail_summary.append(
                    {
                        'type': detail.get('type'),
                        'passed': bool(detail.get('passed')),
                        'coverage': detail.get('coverage'),
                    }
                )
            _llm_eval_logger.info(
                "milestone_heuristics transaction_id=%s milestone_type=%s details=%s",
                transaction_id,
                milestone_type,
                detail_summary,
            )

        print(
            f"{label} LLM overall score: {raw_average_score:.2f} (normalized={normalized_average:.3f}); "
            f"completion_rate={complete_rate:.3f}; success={'YES' if success_rate >= 1.0 else 'NO'}"
        )

        milestone_breakdown = [
            {
                'milestone': effective_milestones[idx],
                'score': considered_scores[idx],
                'passed': considered_pass_flags[idx],
                'normalized_score': considered_normalized[idx],
                'reason': considered_reasons[idx],
            }
            for idx in range(total_milestones)
        ]

        llm_detail_payload = {
            'scores': considered_scores,
            'passed_flags': considered_pass_flags,
            'normalized_scores': considered_normalized,
            'reasons': considered_reasons,
            'overall_score': raw_average_score,
            'normalized_overall': normalized_average,
            'milestones': effective_milestones,
            'milestone_breakdown': milestone_breakdown,
            'raw_response': llm_result.get('raw_response'),
            'json': llm_result.get('json'),
            'notes': llm_result.get('notes'),
        }

        raw_response_text = llm_result.get('raw_response')
        structured_payload = llm_result.get('json')
        if structured_payload is not None and not isinstance(structured_payload, str):
            try:
                structured_for_log = json.dumps(structured_payload)
            except Exception:
                structured_for_log = str(structured_payload)
        else:
            structured_for_log = structured_payload or ''

        if raw_response_text or structured_for_log:
            _llm_eval_logger.info(
                "milestone_llm_payload transaction_id=%s milestone_type=%s raw_response=%s structured=%s",
                transaction_id,
                milestone_type,
                _truncate_for_log(raw_response_text),
                _truncate_for_log(structured_for_log),
            )

        return complete_rate, success_rate, True, milestone_details, llm_detail_payload
    
    def _calculate_completeness(self, rounds: List) -> float:
        """， evaluator.py """
        if not rounds:
            return 0.0

        last_round = rounds[-1]
        conversations = last_round.get('span', {}).get('conversations', []) or []
        if not conversations:
            return 0.0

        completed = 0
        for conv in conversations:
            status = (conv.get('status') or '').lower()
            if status in ('completed', 'succeeded', 'done'):
                completed += 1

        return completed / max(len(conversations), 1)

    def _calculate_overall_score(self, dimensions: Dict) -> float:
        """LLM"""
        weights = {
            's_task_success_rate': 0.16,
            's_task_complete_rate': 0.16,
            'u_task_success_rate': 0.16,
            'u_task_complete_rate': 0.16,
            'execution_efficiency': 0.16,
            'clarification_efficiency': 0.1,
            'agent_routing_accuracy': 0.1,
            'orchestration_latency': 0.0,
            'average_rounds_per_task': 0.0,
            'completeness': 0.0
        }
        
        overall_score = 0
        for dimension, weight in weights.items():
            overall_score += dimensions.get(dimension, 0) * weight
        
        return round(overall_score, 3)


class BatchMultiAgentEvaluator:
    """ - domain"""
    
    def __init__(self, evaluator: BaseEvaluator):
        self.evaluator = evaluator
        self.results = []
    
    def evaluate_batch(self, traces: List[Dict], output_file: str = None) -> Dict:
        """
        trace - domain
        """
        batch_results = {
            'summary': {},
            'detailed_results': [],
            'dimension_stats': {},
            'language_stats': {},
            'domain_stats': {},
            'domain_details': {},
            'task_config_analysis': {},
            'evaluator_type': self.evaluator.evaluator_type if hasattr(self.evaluator, 'evaluator_type') else 'unknown',
            'failure_records': [],
            'unevaluated_traces': [],
        }

        total_traces = len(traces)
        print(f" {total_traces} trace...")
        print(f": {batch_results['evaluator_type']}")
        print(f": {self.evaluator.tasks_config is not None}")

        #  evaluator  tasks ， tasks.yaml  task ，
        #  task  traces （transaction_id ，query ，）。
        max_attempts = 16
        attempt = 1
        successful_results: List[Dict] = []
        failure_records: List[Dict[str, Any]] = []

        #  trace ，
        trace_by_tx: Dict[str, Dict] = {}
        trace_by_query: Dict[str, Dict] = {}
        for tr in traces:
            tx = tr.get('transactionId') or tr.get('transaction_id') or ''
            rounds = tr.get('rounds', []) or []
            initial_query = ''
            if rounds and isinstance(rounds[0], dict):
                initial_query = (rounds[0].get('span', {}) or {}).get('inputQuery', '') or ''
            if tx:
                trace_by_tx[str(tx)] = tr
            if initial_query:
                trace_by_query[str(initial_query)] = tr

        pending_items: List[Dict[str, Any]] = []

        tasks_cfg = (self.evaluator.tasks_config or {}).get('tasks') if self.evaluator.tasks_config else None
        if isinstance(tasks_cfg, dict) and tasks_cfg:
            #  tasks.yaml ： task  trace（）
            for task_id, task_cfg in tasks_cfg.items():
                matched_trace = None
                #  transaction id
                for key in ('transcantion_id', 'transaction_id', 'transactionId'):
                    tid = task_cfg.get(key)
                    if tid and str(tid) in trace_by_tx:
                        matched_trace = trace_by_tx.get(str(tid))
                        break

                #  transaction ， task  query 
                if not matched_trace:
                    q = task_cfg.get('query') or ''
                    if q and q in trace_by_query:
                        matched_trace = trace_by_query.get(q)

                # ， traces  evaluator （ initial_query）
                if not matched_trace:
                    for tr in traces:
                        rounds = tr.get('rounds', []) or []
                        initial_query = ''
                        if rounds and isinstance(rounds[0], dict):
                            initial_query = (rounds[0].get('span', {}) or {}).get('inputQuery', '') or ''
                        if not initial_query:
                            continue
                        # _find_matching_task  task_info， task_id
                        try:
                            found = self.evaluator._find_matching_task(initial_query)
                        except Exception:
                            found = None
                        if found and found.get('task_id') == task_id:
                            matched_trace = tr
                            break

                if matched_trace:
                    pending_items.append({'task_id': task_id, 'task_cfg': task_cfg, 'trace': matched_trace})
                else:
                    #  trace  task
                    batch_results['unevaluated_traces'].append({
                        'task_id': task_id,
                        'status': 'no_matching_trace',
                        'task_query': task_cfg.get('query') if isinstance(task_cfg, dict) else None,
                    })

        else:
            #  tasks ，： traces 
            for tr in traces:
                pending_items.append({'task_id': None, 'task_cfg': None, 'trace': tr})

        failure_log_path = Path('output') / 'evaluation_failures.jsonl'
        failure_log_path.parent.mkdir(parents=True, exist_ok=True)
        with failure_log_path.open('w', encoding='utf-8') as failure_log:
            #  task ： pending_item （ trace）
            while pending_items and attempt <= max_attempts:
                print(f"\n {attempt} ，: {len(pending_items)}")
                next_pending: List[Dict[str, Any]] = []

                for item in pending_items:
                    task_id = item.get('task_id')
                    trace = item.get('trace')
                    transaction_id = trace.get('transactionId', 'Unknown')
                    rounds = trace.get('rounds', []) or []
                    initial_query = ''
                    if rounds and isinstance(rounds[0], dict):
                        initial_query = (rounds[0].get('span', {}) or {}).get('inputQuery', '')

                    display_label = task_id or transaction_id or initial_query or 'Unknown'
                    print(f": attempt {attempt} - task {display_label}")
                    try:
                        result = self.evaluator.evaluate_trace(trace)
                        #  matched_task_id（） task_id（ tasks.yaml）， tasks 
                        if task_id and isinstance(result, dict):
                            result['matched_task_id'] = task_id
                            result['task_config_used'] = True
                        successful_results.append(result)
                    except Exception as exc:
                        error_record = {
                            'attempt': attempt,
                            'task_id': task_id,
                            'transaction_id': transaction_id,
                            'initial_query': initial_query,
                            'error': repr(exc),
                        }
                        print(f" task {display_label} : {exc}")
                        failure_records.append(error_record)
                        failure_log.write(json.dumps(error_record, ensure_ascii=False) + "\n")
                        failure_log.flush()
                        next_pending.append(item)

                pending_items = next_pending
                attempt += 1

        if pending_items:
            # /trace ，
            for item in pending_items:
                trace = item.get('trace') or {}
                task_id = item.get('task_id')
                transaction_id = trace.get('transactionId', 'Unknown')
                rounds = trace.get('rounds', []) or []
                initial_query = ''
                if rounds and isinstance(rounds[0], dict):
                    initial_query = (rounds[0].get('span', {}) or {}).get('inputQuery', '')
                final_record = {
                    'transaction_id': transaction_id,
                    'task_id': task_id,
                    'initial_query': initial_query,
                    'status': 'unevaluated_after_retries'
                }
                batch_results['unevaluated_traces'].append(final_record)

        batch_results['failure_records'] = failure_records

        deduped_results, duplicate_count = self._deduplicate_results(successful_results)
        if duplicate_count:
            print(f":  {duplicate_count} trace（ transaction_id  initial_query）。")

        batch_results['detailed_results'] = deduped_results

        if deduped_results:
            batch_results['summary'] = self._calculate_batch_summary(deduped_results)
            if duplicate_count:
                batch_results['summary']['duplicates_removed'] = duplicate_count
            batch_results['dimension_stats'] = self._calculate_dimension_statistics(deduped_results)
            batch_results['language_stats'] = self._calculate_language_statistics(deduped_results)
            batch_results['domain_stats'] = self._calculate_domain_statistics(deduped_results)
            batch_results['domain_details'] = self._calculate_domain_details(deduped_results)
            batch_results['task_config_analysis'] = self._calculate_task_config_analysis(deduped_results)

        batch_results['evaluation_summary'] = {
            'total_traces': total_traces,
            'successful_traces': len(deduped_results),
            'failed_traces': total_traces - len(successful_results),
            'duplicates_removed': duplicate_count,
            'attempts_used': min(max_attempts, attempt - 1),
        }

        if output_file:
            self._save_results(batch_results, output_file)

        return batch_results
    
    def _calculate_task_config_analysis(self, results: List[Dict]) -> Dict:
        """"""
        task_config_used_count = sum(1 for r in results if r.get('task_config_used', False))
        total_count = len(results)
        
        analysis = {
            'total_traces': total_count,
            'traces_with_task_config': task_config_used_count,
            'task_config_usage_rate': task_config_used_count / total_count if total_count > 0 else 0,
            'task_level_breakdown': {}
        }
        
        # trace
        task_config_results = [r for r in results if r.get('task_config_used', False)]
        non_task_config_results = [r for r in results if not r.get('task_config_used', False)]
        
        if task_config_results:
            task_config_scores = [r['dimensions'].get('overall_orchestration_score', 0) for r in task_config_results]
            analysis['avg_score_with_task_config'] = sum(task_config_scores) / len(task_config_scores)
            
            # 
            for result in task_config_results:
                task_id = result.get('matched_task_id', 'Unknown')
                level = task_id.split('_')[0] if '_' in task_id else 'Unknown'
                
                if level not in analysis['task_level_breakdown']:
                    analysis['task_level_breakdown'][level] = {
                        'count': 0,
                        'scores': []
                    }
                
                analysis['task_level_breakdown'][level]['count'] += 1
                analysis['task_level_breakdown'][level]['scores'].append(
                    result['dimensions'].get('overall_orchestration_score', 0)
                )
        
        if non_task_config_results:
            non_task_config_scores = [r['dimensions'].get('overall_orchestration_score', 0) for r in non_task_config_results]
            analysis['avg_score_without_task_config'] = sum(non_task_config_scores) / len(non_task_config_scores)
        
        # 
        for level, data in analysis['task_level_breakdown'].items():
            if data['scores']:
                data['avg_score'] = sum(data['scores']) / len(data['scores'])
        
        return analysis
    
    def _calculate_domain_details(self, results: List[Dict]) -> Dict:
        """domain"""
        domain_details = {}
        
        for result in results:
            domain = result.get('domain', 'unknown')
            if domain not in domain_details:
                domain_details[domain] = {
                    'count': 0,
                    'scores': [],
                    'languages': {},
                    'avg_score': 0.0
                }
            
            domain_details[domain]['count'] += 1
            score = result['dimensions'].get('overall_orchestration_score', 0)
            domain_details[domain]['scores'].append(score)
            
            # 
            language = result.get('language', 'unknown')
            domain_details[domain]['languages'][language] = domain_details[domain]['languages'].get(language, 0) + 1
        
        # 
        for domain, details in domain_details.items():
            if details['scores']:
                details['avg_score'] = sum(details['scores']) / len(details['scores'])
        
        return domain_details
    
    def _calculate_language_statistics(self, results: List[Dict]) -> Dict:
        """"""
        language_data = {}
        
        for result in results:
            language = result.get('language', 'unknown')
            if language not in language_data:
                language_data[language] = []
            language_data[language].append(result['dimensions'].get('overall_orchestration_score', 0))
        
        stats = {}
        for language, scores in language_data.items():
            stats[language] = {
                'count': len(scores),
                'mean': sum(scores) / len(scores),
                'min': min(scores),
                'max': max(scores),
                'std': (sum((x - (sum(scores) / len(scores))) ** 2 for x in scores) / len(scores)) ** 0.5
            }
        
        return stats
    
    def _calculate_domain_statistics(self, results: List[Dict]) -> Dict:
        """"""
        domain_data = {}
        
        for result in results:
            domain = result.get('domain', 'general')
            if domain not in domain_data:
                domain_data[domain] = []
            domain_data[domain].append(result['dimensions'].get('overall_orchestration_score', 0))
        
        stats = {}
        for domain, scores in domain_data.items():
            stats[domain] = {
                'count': len(scores),
                'mean': sum(scores) / len(scores),
                'min': min(scores),
                'max': max(scores),
                'std': (sum((x - (sum(scores) / len(scores))) ** 2 for x in scores) / len(scores)) ** 0.5
            }
        
        return stats
    
    def _calculate_batch_summary(self, results: List[Dict]) -> Dict:
        """"""
        total_traces = len(results)
        
        # 
        score_distribution = {
            'excellent': 0,  # 0.9-1.0
            'good': 0,       # 0.7-0.89
            'fair': 0,       # 0.5-0.69
            'poor': 0        # <0.5
        }
        
        dimension_scores = {}
        
        for result in results:
            overall_score = result['dimensions'].get('overall_orchestration_score', 0)
            
            if overall_score >= 0.9:
                score_distribution['excellent'] += 1
            elif overall_score >= 0.7:
                score_distribution['good'] += 1
            elif overall_score >= 0.5:
                score_distribution['fair'] += 1
            else:
                score_distribution['poor'] += 1
            
            # 
            for dim, score in result['dimensions'].items():
                if dim not in dimension_scores:
                    dimension_scores[dim] = []
                dimension_scores[dim].append(score)
        
        # 
        avg_scores = {}
        for dim, scores in dimension_scores.items():
            avg_scores[f'avg_{dim}'] = sum(scores) / len(scores)
        
        summary = {
            'total_traces_evaluated': total_traces,
            'score_distribution': score_distribution,
            'average_scores': avg_scores,
            'success_rate': (score_distribution['excellent'] + score_distribution['good']) / total_traces
        }
        
        return summary

    def _deduplicate_results(self, results: List[Dict]) -> Tuple[List[Dict], int]:
        """Remove duplicate trace evaluations using transaction id or initial query."""
        seen_keys: Set[Tuple[str, str]] = set()
        deduped: List[Dict] = []
        duplicates = 0

        for index, result in enumerate(results):
            transaction_id = str(result.get('transaction_id') or '').strip()
            initial_query = str(result.get('initial_query') or '').strip().lower()

            if transaction_id:
                key = ('tx', transaction_id.lower())
            elif initial_query:
                key = ('query', initial_query)
            else:
                key = ('index', str(index))

            if key in seen_keys:
                duplicates += 1
                continue

            seen_keys.add(key)
            deduped.append(result)

        return deduped, duplicates
    
    def _calculate_dimension_statistics(self, results: List[Dict]) -> Dict:
        """"""
        dimension_data = {}
        
        for result in results:
            for dim, score in result['dimensions'].items():
                if dim not in dimension_data:
                    dimension_data[dim] = []
                dimension_data[dim].append(score)
        
        stats = {}
        for dim, scores in dimension_data.items():
            stats[dim] = {
                'mean': sum(scores) / len(scores),
                'min': min(scores),
                'max': max(scores),
                'std': (sum((x - (sum(scores) / len(scores))) ** 2 for x in scores) / len(scores)) ** 0.5
            }
        
        return stats
    
    def _save_results(self, batch_results: Dict, output_file: str):
        """"""
        # 
        detailed_file = output_file.replace('.json', '_detailed.json')
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(batch_results, f, ensure_ascii=False, indent=2)
        
        # （）
        summary_file = output_file.replace('.json', '_summary.json')
        summary_data = {
            'summary': batch_results['summary'],
            'dimension_statistics': batch_results['dimension_stats'],
            'task_config_analysis': batch_results['task_config_analysis'],
            'evaluator_type': batch_results['evaluator_type']
        }
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        
        print(f": {detailed_file}")
        print(f": {summary_file}")


def main():
    """ - """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input', '-i', required=True, help='（txtjsonl）')
    parser.add_argument('--output', '-o', help='')
    parser.add_argument('--task-config', '-t', help='（tasks.yaml）')
    parser.add_argument('--agent-config', '-a', help='（jsonyaml）， subAgentName ')
    parser.add_argument('--llm-api-key', help='API（LLM）')
    parser.add_argument('--llm-model', default='gpt-5.1', help='，gpt-5.1')
    parser.add_argument('--max-traces', type=int, help='trace（）')
    parser.add_argument('--verbose', '-v', action='store_true', help='')
    parser.add_argument('--language-analysis', action='store_true', help='')
    parser.add_argument('--domain-analysis', action='store_true', help='')
    parser.add_argument('--list-domains', action='store_true', help='')
    parser.add_argument('--filter-domain', help='trace（travel、hr、it）')
    
    args = parser.parse_args()
    evaluator_type = 'llm_enhanced'
    
    # ，
    if args.list_domains:
        try:
            domain_manager = DomainManager()
            domains = domain_manager.get_all_domains()
            print(":")
            for domain in domains:
                config = domain_manager.get_domain_config(domain)
                print(f"  {domain}: {config['name']} - {config['description']}")
                print(f"    Agent: {', '.join(config['agents'])}")
        except ImportError:
            print(": DomainManager，")
        return
    
    # 
    if args.output is None:
        base_name = os.path.splitext(args.input)[0]  # 
        domain_suffix = f"_{args.filter_domain}" if args.filter_domain else ""
        args.output = f"{base_name}_{evaluator_type}{domain_suffix}_results.json"
    
    # ：
    task_config = None
    if args.task_config:
        try:
            with open(args.task_config, 'r', encoding='utf-8') as f:
                if args.task_config.endswith('.yaml') or args.task_config.endswith('.yml'):
                    task_config = yaml.safe_load(f)
                else:
                    task_config = json.load(f)
            print(f": {args.task_config}")
        except Exception as e:
            print(f": {e}")
            return
    
    # 
    input_file = args.input
    if input_file.endswith('.txt'):
        print("txt，jsonl...")
        jsonl_file = TraceProcessor.parse_trace_txt_to_jsonl(input_file)
        traces = TraceProcessor.load_jsonl_traces(jsonl_file)
    elif input_file.endswith('.jsonl'):
        traces = TraceProcessor.load_jsonl_traces(input_file)
    else:
        raise ValueError("txtjsonl")
    
    if not traces:
        print("trace")
        return
    
    # ，
    if args.filter_domain:
        try:
            domain_manager = DomainManager()
            if args.filter_domain not in domain_manager.get_all_domains():
                print(f":  '{args.filter_domain}'")
                print(f": {', '.join(domain_manager.get_all_domains())}")
                return
            
            print(f": {args.filter_domain}")
            detector = LanguageDomainDetector(domain_manager)
            filtered_traces = []
            
            for trace in traces:
                domain_info = detector.analyze_trace(trace)
                if domain_info['domain'] == args.filter_domain:
                    filtered_traces.append(trace)
            
            print(f"trace: {len(filtered_traces)}/{len(traces)}")
            traces = filtered_traces
        except ImportError:
            print(": ，")
    
    # trace（）
    if args.max_traces and args.max_traces < len(traces):
        print(f"trace: {args.max_traces}")
        traces = traces[:args.max_traces]
    
    if not traces:
        print("trace")
        return
    
    print(f" {len(traces)} trace")
    
    # （）
    agents_config = None
    user_profiles = None
    config_data = None
    # ： agent 
    if args.agent_config:
        try:
            ac_path = args.agent_config
            with open(ac_path, 'r', encoding='utf-8') as f:
                if ac_path.endswith('.json'):
                    agents_config = json.load(f)
                else:
                    agents_config = yaml.safe_load(f)
            print(f" agent : {ac_path}")
        except Exception as e:
            print(f":  agent  {args.agent_config}: {e}")

    
    # （ LLM ）
    if not args.llm_api_key:
        args.llm_api_key = os.getenv('OPENAI_API_KEY') or os.getenv('LLM_API_KEY')

    try:
        evaluator = LLMEnhancedEvaluator(
            tasks_config=task_config,
            agents_config=agents_config,
            user_profiles=user_profiles,
            config=config_data,
            llm_api_key=args.llm_api_key,
            llm_model=args.llm_model,
        )
        print(f"LLM (: {evaluator.llm_evaluator.model})")
    except RuntimeError as exc:
        print(f"LLM: {exc}")
        return
    except ImportError:
        print(": LLMEnhancedEvaluator")
        return
    
    # 
    batch_start_time = time.time()
    try:
        batch_evaluator = BatchMultiAgentEvaluator(evaluator)
        results = batch_evaluator.evaluate_batch(traces, args.output)
    except ImportError:
        print(": BatchMultiAgentEvaluator")
        return
    
    batch_end_time = time.time()
    
    # 
    summary = results.get('summary') or {}
    dimension_stats = results['dimension_stats']
    language_stats = results.get('language_stats', {})
    domain_stats = results.get('domain_stats', {})
    domain_details = results.get('domain_details', {})
    task_config_analysis = results.get('task_config_analysis', {})
    evaluator_type = results.get('evaluator_type', 'unknown')
    
    print("\n" + "="*60)
    print("")
    print("="*60)
    print(f": {evaluator_type}")
    if args.filter_domain:
        print(f": {args.filter_domain}")
    if not summary:
        print("（trace）。")
        return

    total_traces_evaluated = summary.get('total_traces_evaluated', len(results.get('detailed_results', [])))
    print(f"trace: {total_traces_evaluated}")
    if 'success_rate' in summary:
        print(f": {summary['success_rate']:.2%}")
    else:
        print(": ")
    print(f": {batch_end_time - batch_start_time:.2f}")
    print(f"trace: {(batch_end_time - batch_start_time) / max(total_traces_evaluated, 1):.2f}")
    
    # 
    if task_config_analysis:
        print(f"\n:")
        print(f"  trace: {task_config_analysis['traces_with_task_config']}/{task_config_analysis['total_traces']}")
        print(f"  : {task_config_analysis['task_config_usage_rate']:.1%}")
        
        if 'avg_score_with_task_config' in task_config_analysis:
            print(f"  : {task_config_analysis['avg_score_with_task_config']:.3f}")
        
        if 'avg_score_without_task_config' in task_config_analysis:
            print(f"  : {task_config_analysis['avg_score_without_task_config']:.3f}")
        
        level_breakdown = task_config_analysis.get('task_level_breakdown', {})
        if level_breakdown:
            print(f"\n  :")
            for level, data in level_breakdown.items():
                print(f"    {level}: {data['count']} traces, : {data.get('avg_score', 0):.3f}")
    
    print(f"\n:")
    dist = summary.get('score_distribution', {})
    print(f"   (0.9-1.0): {dist.get('excellent', 0)} traces")
    print(f"   (0.7-0.89): {dist.get('good', 0)} traces") 
    print(f"   (0.5-0.69): {dist.get('fair', 0)} traces")
    print(f"   (<0.5): {dist.get('poor', 0)} traces")
    
    # 
    if language_stats and (args.language_analysis or args.verbose):
        print(f"\n:")
        for lang, stats in language_stats.items():
            print(f"  {lang.upper()}: {stats['count']} traces, : {stats['mean']:.3f}")
    
    # 
    if domain_stats and (args.domain_analysis or args.verbose):
        print(f"\n:")
        for domain, stats in domain_stats.items():
            try:
                domain_config = evaluator.domain_manager.get_domain_config(domain)
                domain_name = domain_config.get('name', domain)
                print(f"  {domain_name}: {stats['count']} traces, : {stats['mean']:.3f}")
            except:
                print(f"  {domain}: {stats['count']} traces, : {stats['mean']:.3f}")
    
    # 
    if domain_details and args.verbose:
        print(f"\n:")
        for domain, details in domain_details.items():
            try:
                domain_config = evaluator.domain_manager.get_domain_config(domain)
                domain_name = domain_config.get('name', domain)
                print(f"  {domain_name}:")
                print(f"    : {details['count']}")
                print(f"    : {details['avg_score']:.3f}")
                if details['languages']:
                    print(f"    : {details['languages']}")
            except:
                print(f"  {domain}:")
                print(f"    : {details['count']}")
                print(f"    : {details['avg_score']:.3f}")
                if details['languages']:
                    print(f"    : {details['languages']}")
    
    print(f"\n:")
    for dim, score in summary.get('average_scores', {}).items():
        dim_name = dim.replace('avg_', '').replace('_', ' ').title()
        print(f"  {dim_name}: {score:.3f}")
    
    # 
    if args.verbose:
        print(f"\n:")
        for dim, stats in dimension_stats.items():
            print(f"  {dim.replace('_', ' ').title()}:")
            print(f"    : {stats['mean']:.3f}")
            print(f"    : {stats['min']:.3f}")
            print(f"    : {stats['max']:.3f}")
            print(f"    : {stats['std']:.3f}")
    
    # 
    print(f"\n:")
    overall_scores = [r['dimensions'].get('overall_orchestration_score', 0) for r in results['detailed_results']]
    avg_score = sum(overall_scores) / len(overall_scores) if overall_scores else 0
    print(f"  : {avg_score:.3f}")
    print(f"  : {max(overall_scores) if overall_scores else 0:.3f}")
    print(f"  : {min(overall_scores) if overall_scores else 0:.3f}")
    
    # trace
    if results['detailed_results']:
        best_trace = max(results['detailed_results'], 
                        key=lambda x: x['dimensions'].get('overall_orchestration_score', 0))
        worst_trace = min(results['detailed_results'], 
                         key=lambda x: x['dimensions'].get('overall_orchestration_score', 0))
        
        print(f"\ntrace: {best_trace['transaction_id']}")
        print(f"  : {best_trace['dimensions'].get('overall_orchestration_score', 0):.3f}")
        print(f"  : {best_trace.get('language', 'unknown')}, : {best_trace.get('domain', 'general')}")
        if best_trace.get('task_config_used'):
            print(f"  : {best_trace.get('matched_task_id', 'Unknown')}")
        
        print(f"trace: {worst_trace['transaction_id']}")
        print(f"  : {worst_trace['dimensions'].get('overall_orchestration_score', 0):.3f}")
        print(f"  : {worst_trace.get('language', 'unknown')}, : {worst_trace.get('domain', 'general')}")
        if worst_trace.get('task_config_used'):
            print(f"  : {worst_trace.get('matched_task_id', 'Unknown')}")


if __name__ == "__main__":
    main()