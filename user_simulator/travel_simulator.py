import random
from datetime import date
import re
from typing import Dict, List, Optional
import json
import time
from datetime import datetime, timedelta
from datasets.llm_client import LLMClient
import importlib
import os
import traceback
from plugins import manager as plugin_manager
from .base_simulator import BaseUserSimulator


class TravelUserSimulator(BaseUserSimulator):
    """LLM-based travel-domain user simulator (travel domain)

    This class is derived from the original llm_user_simulator.py with minimal travel-specific adaptation.
    Generic methods should be kept in `BaseUserSimulator` for reuse across domains.
    """

    def __init__(self, user_profile: str = "profile_001", llm_config: Dict = None, domain: str = 'travel', rule_based_simulator=None, rule_confidence_threshold: float = 0.8, lang: str = 'zh', plugin_call_timeout: float = 0.5, profile_path: Optional[str] = None):
        # initialize base which sets conversation history and basic utilities
        super().__init__()
        # language for prompts and followups ('zh' or 'en')
        self.lang = str(lang).lower() if lang else 'zh'
        self.domain = domain

        # plugin manager discovery early so plugins can supply domain-specific profile
        try:
            self.domain_plugin = plugin_manager.get_plugin_for_domain(self.domain)
        except Exception:
            self.domain_plugin = None

        # If domain plugin can provide a profile, prefer it; otherwise load from config
        prof = None
        try:
            if self.domain_plugin and hasattr(self.domain_plugin, 'get_domain_profile'):
                prof = self.domain_plugin.get_domain_profile(user_profile)
        except Exception:
            prof = None

        if isinstance(prof, dict):
            self.profile = prof
        else:
            self.profile_path = profile_path
            self.profile = self._load_profile(user_profile)

        # allow injection of a rule-based simulator (domain-specific); otherwise try to auto-load
        self.rule_based_simulator = rule_based_simulator
        if self.rule_based_simulator is None:
            try:
                mod_name = f"user_simulator.{domain}_simulator"
                mod = importlib.import_module(mod_name)
                cls_name = f"{domain.capitalize()}UserSimulator"
                if hasattr(mod, cls_name):
                    SimCls = getattr(mod, cls_name)
                    self.rule_based_simulator = SimCls(user_profile)
            except Exception:
                self.rule_based_simulator = None

        # confidence threshold to accept a rule-based match
        self.rule_confidence_threshold = float(rule_confidence_threshold)
        # If caller provides an explicit llm_config, use it; otherwise
        # prefer centralized env/file loading via LLMClient.from_env()
        if llm_config:
            self.llm_config = llm_config
            # create a client that uses the provided config dict
            self._llm_client = LLMClient(self.llm_config)
        else:
            # let the centralized LLMClient resolve .env and config files
            self._llm_client = LLMClient.from_env()
            # keep the resolved config for callers/inspection
            self.llm_config = self._llm_client.config or {}

        # Last rule match info (for diagnostics)
        self.last_rule_match = None
        # Last used strategy for clarification: 'llm', 'rule', 'fallback-rule', etc.
        self.last_used_strategy = None
        # plugin call timeout (seconds). Passed to plugins.manager.safe_invoke
        self.plugin_call_timeout = float(plugin_call_timeout or 0.5)


    def _load_profile(self, profile_name: str) -> Dict:
        """Load user profile config: prefer `config/user_profiles.yaml`, fallback to built-in samples on failure."""
        try:
            import yaml
            path = getattr(self, 'profile_path', None) or 'config/user_profiles.yaml'
            with open(path, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f) or {}
                users = cfg.get('user_profiles') or cfg.get('profiles') or {}
                def normalize_profile(p: Dict) -> Dict:
                    if not isinstance(p, dict):
                        return p
                    out = {}
                    name_field = p.get('name') or p.get('display_name') or p.get('id') or 'user'
                    if isinstance(name_field, dict):
                        out['name'] = name_field.get(getattr(self, 'lang', 'zh'), next(iter(name_field.values())))
                    else:
                        out['name'] = name_field
                    if 'traits' in p and isinstance(p.get('traits'), (list, tuple)):
                        out['traits'] = list(p.get('traits'))
                    elif 'characteristics' in p and isinstance(p.get('characteristics'), (list, tuple)):
                        out['traits'] = list(p.get('characteristics'))
                    else:
                        out['traits'] = []
                    comm = p.get('communication_style') or p.get('style') or p.get('tone') or 'natural'
                    if isinstance(comm, dict):
                        out['communication_style'] = comm.get(getattr(self, 'lang', 'zh'), next(iter(comm.values())))
                    else:
                        out['communication_style'] = comm
                    raw_tr = None
                    if 'typical_responses' in p and isinstance(p.get('typical_responses'), dict):
                        raw_tr = p.get('typical_responses')
                    elif 'response_patterns' in p and isinstance(p.get('response_patterns'), dict):
                        raw_tr = p.get('response_patterns')
                    else:
                        raw_tr = {}
                    tr = {}
                    for k, v in raw_tr.items():
                        if isinstance(v, dict):
                            tr[k] = v.get(getattr(self, 'lang', 'zh'), next(iter(v.values())))
                        else:
                            tr[k] = v
                    out['typical_responses'] = tr
                    return out

                if profile_name in users:
                    return normalize_profile(users[profile_name])
                if 'profile_001' in users:
                    return normalize_profile(users['profile_001'])
        except Exception:
            pass

        profiles = {
            "profile_001": {
                "name": "Business Traveler",
                "traits": ["efficiency-focused", "mid-range budget", "prefers convenient transportation", "needs business facilities"],
                "communication_style": "concise and professional",
                "typical_responses": {
                    "budget": "Budget is around 5,000 RMB and can be adjusted.",
                    "dates": "From next Wednesday to Friday, about three days.",
                    "preferences": "Need a hotel near the conference center."
                }
            },
            "profile_002": {
                "name": "Family Traveler",
                "traits": ["values family-friendly facilities", "limited budget", "enjoys attractions", "needs family rooms"],
                "communication_style": "friendly and detailed",
                "typical_responses": {
                    "budget": "Family budget is within 8,000 RMB.",
                    "dates": "During summer vacation, from July 10 to July 15.",
                    "preferences": "Need kids facilities and child-friendly attractions."
                }
            }
        }
        return profiles.get(profile_name, profiles["profile_001"])

    def _call_llm(self, messages: List[Dict], override_config: Optional[Dict] = None) -> str:
        sanitized = []
        for m in messages:
            if not isinstance(m, dict):
                continue
            role = m.get('role', 'user')
            content = m.get('content', '')
            sanitized.append({'role': role, 'content': content})

        def _should_retry(exc: Exception) -> bool:
            try:
                msg = str(exc).lower()
            except Exception:
                msg = ''
            if not msg:
                return False
            transient_keywords = ['timed out', 'timeout', 'connection', 'temporarily unavailable', 'gateway', 'unreachable']
            return any(kw in msg for kw in transient_keywords)

        max_attempts = 2
        last_exc: Optional[Exception] = None

        for attempt in range(1, max_attempts + 1):
            try:
                try:
                    response = self._llm_client.call_llm(sanitized, override_config=override_config)
                except TypeError:
                    response = self._llm_client.call_llm(sanitized)
                try:
                    self._log_llm_interaction(
                        sanitized,
                        response,
                        metadata={'source': '_call_llm', 'override_config': override_config, 'attempt': attempt}
                    )
                except Exception:
                    pass
                return response
            except Exception as exc:
                last_exc = exc
                try:
                    self._log_llm_interaction(
                        sanitized,
                        None,
                        metadata={'source': '_call_llm', 'error': traceback.format_exc(), 'attempt': attempt}
                    )
                except Exception:
                    pass
                if attempt < max_attempts and _should_retry(exc):
                    try:
                        time.sleep(5)
                    except Exception:
                        pass
                    continue
                break

        if last_exc is not None:
            raise last_exc
        raise RuntimeError('LLM call failed without exception context')

    def _invoke_plugin_hook(self, hook_name: str, *args, **kwargs):
        try:
            return plugin_manager.safe_invoke(getattr(self, 'domain_plugin', None), hook_name, *args, timeout=getattr(self, 'plugin_call_timeout', None), **kwargs)
        except Exception:
            return None

    def _log_llm_interaction(self, messages, response, metadata: Dict = None):
        try:
            out = {
                'timestamp': self.get_timestamp(),
                'metadata': metadata or {},
                'messages': messages,
                'response': response
            }
            path = os.path.join('output', 'llm_calls_debug.jsonl')
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(out, ensure_ascii=False) + '\n')
        except Exception:
            try:
                print('LLM log failed:', traceback.format_exc())
            except Exception:
                pass

    def _extract_json_from_text(self, text: str) -> Optional[Dict]:
        if not text or not isinstance(text, str):
            return None
        start = text.find('{')
        if start == -1:
            return None
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    candidate = text[start:i+1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        s = candidate.replace("'", '"')
                        s = s.replace(',}', '}').replace(',]', ']')
                        try:
                            return json.loads(s)
                        except Exception:
                            return None
        return None

    def _trim_context_history(self, context: Dict, rounds: int = 3) -> Dict:
        ctx = dict(context or {})
        convo = ctx.get('conversation_history', []) or []
        try:
            trimmed = convo[-(rounds * 2):] if len(convo) > (rounds * 2) else convo[:]
            ctx['conversation_history'] = trimmed
        except Exception:
            ctx['conversation_history'] = convo
        return ctx

    def _extract_sub_agent_name(self, entry: Dict, context_meta: Optional[Dict] = None, default: Optional[str] = None) -> Optional[str]:
        """Best-effort extraction of subAgentName/agent identifiers from a history entry."""
        keys = (
            'subAgentName', 'sub_agent_name', 'subAgent', 'subagent',
            'agent', 'agentName', 'agent_name', 'plugin', 'pluginName', 'plugin_name'
        )

        def _check_dict(data: Dict) -> Optional[str]:
            for key in keys:
                try:
                    val = data.get(key)
                except Exception:
                    val = None
                if val:
                    try:
                        text = str(val).strip()
                        if text:
                            return text
                    except Exception:
                        return str(val)
            return None

        def _extract_from_nested_dict(container: Dict) -> Optional[str]:
            if not isinstance(container, dict):
                return None
            candidate = _check_dict(container)
            if candidate:
                return candidate
            for meta_key in ('metadata', 'meta', 'extra', 'info', 'details'):
                nested = container.get(meta_key)
                if isinstance(nested, dict):
                    candidate = _check_dict(nested)
                    if candidate:
                        return candidate
            return None

        def _extract_from_stream_text(text: str) -> Optional[str]:
            if not text or not isinstance(text, str):
                return None
            try:
                lines = text.splitlines()
            except Exception:
                lines = [text]
            for line in lines:
                try:
                    stripped = line.strip()
                except Exception:
                    continue
                if not stripped:
                    continue
                if stripped.startswith('data:'):
                    payload = stripped[len('data:'):].strip()
                    if not payload:
                        continue
                    parsed = self._extract_json_from_text(payload) if '{' in payload else None
                    if isinstance(parsed, dict):
                        candidate = _extract_from_nested_dict(parsed)
                        if candidate:
                            return candidate
                else:
                    parsed = self._extract_json_from_text(stripped) if '{' in stripped else None
                    if isinstance(parsed, dict):
                        candidate = _extract_from_nested_dict(parsed)
                        if candidate:
                            return candidate
            try:
                match = re.search(r'"subAgent"\s*:\s*"([^"]+)"', text, re.IGNORECASE)
                if match:
                    candidate = match.group(1).strip()
                    if candidate:
                        return candidate
            except Exception:
                pass
            return None

        try:
            if isinstance(entry, dict):
                direct = _check_dict(entry)
                if direct:
                    return direct

                for meta_key in ('meta', 'metadata', 'info', 'extra', 'collected_json', 'collectedJson', 'collected'):
                    nested = entry.get(meta_key)
                    if isinstance(nested, dict):
                        candidate = _check_dict(nested)
                        if candidate:
                            return candidate

            source_fields = ('response', 'content', 'text', 'message', 'final_output', 'finalOutput')
            for field in source_fields:
                target = entry.get(field) if isinstance(entry, dict) else None
                if isinstance(target, dict):
                    candidate = _extract_from_nested_dict(target)
                    if candidate:
                        return candidate
                elif isinstance(target, str):
                    parsed = self._extract_json_from_text(target)
                    if isinstance(parsed, dict):
                        candidate = _extract_from_nested_dict(parsed)
                        if candidate:
                            return candidate
                    stream_candidate = _extract_from_stream_text(target)
                    if stream_candidate:
                        return stream_candidate

            if isinstance(context_meta, dict):
                candidate = _check_dict(context_meta)
                if candidate:
                    return candidate
        except Exception:
            return default

        return default

    def _prepare_history_with_agents(self, history: List[Dict], context: Dict) -> List[Dict[str, Optional[str]]]:
        """Normalize conversation history entries and tag them with sub-agent information if available."""
        enriched: List[Dict[str, Optional[str]]] = []
        try:
            context_meta = context.get('collected_json_meta') if isinstance(context, dict) else {}
            if not isinstance(context_meta, dict):
                context_meta = {}
        except Exception:
            context_meta = {}

        last_agent: Optional[str] = None

        for entry in history:
            if not isinstance(entry, dict):
                continue
            try:
                role = (entry.get('role') or entry.get('speaker') or '').strip().lower()
            except Exception:
                role = ''
            if role not in ('assistant', 'user'):
                continue

            content = entry.get('content')
            if content is None:
                content = entry.get('text')

            if isinstance(content, dict):
                try:
                    content = content.get('content') or content.get('text')
                except Exception:
                    content = None
                if content is None:
                    try:
                        content = json.dumps(entry.get('content'), ensure_ascii=False)
                    except Exception:
                        content = str(entry.get('content')) if entry.get('content') is not None else None

            if content is None:
                response_candidate = entry.get('response')
                if isinstance(response_candidate, dict):
                    picked = None
                    for key in ('message', 'text', 'content', 'final_output', 'response', 'finalOutput'):
                        val = response_candidate.get(key)
                        if val:
                            picked = val
                            break
                    if picked is None:
                        try:
                            picked = json.dumps(response_candidate, ensure_ascii=False)
                        except Exception:
                            picked = str(response_candidate)
                    content = picked
                elif isinstance(response_candidate, str):
                    content = response_candidate

            if content is None:
                continue

            try:
                content_str = str(content).strip()
            except Exception:
                content_str = str(content)
            if not content_str:
                continue

            agent = self._extract_sub_agent_name(entry, context_meta=context_meta)
            if not agent and role == 'assistant':
                agent = self._extract_sub_agent_name({'meta': context_meta}, context_meta=context_meta, default=last_agent)
            if role == 'assistant' and agent:
                last_agent = agent
            elif role == 'assistant' and not agent:
                agent = last_agent

            if agent:
                try:
                    agent = str(agent).strip() or None
                except Exception:
                    agent = str(agent)

            enriched.append({'role': role, 'content': content_str, 'agent': agent})

        return enriched

    def _build_conversation_context(self, current_question: str, context: Dict, enforce_future_dates: bool = False,
                                    enforce_precise: bool = False, prefer_within_days: Optional[int] = None,
                                    requires_round_trip: bool = False) -> List[Dict]:
        try:
            plugin_msgs = None
            plugin = getattr(self, 'domain_plugin', None)
            if plugin and hasattr(plugin, 'build_conversation_context'):
                try:
                    plugin_msgs = plugin.build_conversation_context(
                        current_question, context,
                        enforce_future_dates=enforce_future_dates,
                        enforce_precise=enforce_precise,
                        prefer_within_days=prefer_within_days,
                        requires_round_trip=requires_round_trip
                    )
                except Exception:
                    plugin_msgs = None
            if isinstance(plugin_msgs, list):
                return plugin_msgs
        except Exception:
            pass
        
        today_date = str(date.today())
        # Support both Chinese and English system prompts; choose by self.lang
        cn_prompt = f"""，。
        
：
- ID: {context.get('task_id', '')}
- : {context.get('round', 0)}
- : {context.get('current_status', 'clarification')}

###  (Behavioral Requirements)

1. **。**
    - ：date、city、、budget。
2. **，。**
3. **：**
    - Yes, 。、、destination、、budget、、。
4. **date：**
    - date **{today_date}**。datedatedate。
5. **。**
    - Yes, 、。。
6. **：**
    - date：**{today_date}**
    - location：**China**
    - date，datedate。
    - location，**China**citylocation；returncity（Japan、Korea、）。
    - Chinacity，return："Chinadestination"，returncity。

：
1. ，。
2. 、，。
"""

        en_prompt = f"""You are simulating a human being traveler answering clarification questions from an assistant.

Current conversation context:
- Task ID: {context.get('task_id', 'unknown')}
- Clarification round: {context.get('round', 0)}
- Current state: {context.get('current_status', 'clarification')}

### Behavioral Requirements

1. **Always answer directly and naturally.**
    - Provide concrete user content such as dates, cities, preferences, budget, etc.
2. **Human style, concise, realistic.**
3. **Travel domain context:**
    - You are a human traveler. Questions may involve flights, hotels, destinations, restaurants, budgets, transportation, visas, or local recommendations.
4. **Reference date:**
    - The reference date is **{today_date}**. When providing dates, prefer realistic future dates after this reference.
5. **Do not break role under any condition.**
    - Never acknowledge being an agent, simulator, or test harness. Always remain in-character as the user.
6. **Current environment information:**
    - Current date: **{today_date}**
    - Current location: **China**
    - When giving dates, ensure they are realistic future dates after this environment date.
    - When giving locations, you MUST pick places located within **China** and MUST NOT suggest or return locations outside of China (for example: Japan, Korea, USA).
    - If no suitable Chinese city can be provided, respond exactly with: "No suitable location within China" (do not provide an international city or vague location).

Guidelines:
1. Read the assistant's question carefully and answer the core ask directly.
2. Keep answers concise, relevant, and avoid unnecessary extra information.
"""

        # choose prompt language by self.lang (default to Chinese)
        try:
            if getattr(self, 'lang', 'zh') and str(getattr(self, 'lang', 'zh')).lower().startswith('en'):
                system_prompt = en_prompt
            else:
                system_prompt = cn_prompt
        except Exception:
            system_prompt = cn_prompt

        messages = [
            {"role": "system", "content": system_prompt}
        ]

        # ：，，return
        # Insert a bilingual (ZH/EN) no-echo system instruction selected by `self.lang`
        no_echo_cn = (
            "：。，"
            "。return。"
        )
        no_echo_en = (
            "Note: Answer only the assistant's current clarification question. Do not repeat the original user request"
            " or the assistant's question, and do not include extra contextual explanation. Return only the required direct value."
        )
        try:
            lang = getattr(self, 'lang', 'zh') or 'zh'
        except Exception:
            lang = 'zh'
        if str(lang).lower().startswith('en'):
            messages.insert(1, {"role": "system", "content": no_echo_en})
        else:
            messages.insert(1, {"role": "system", "content": no_echo_cn})
        raw_history = context.get('conversation_history', [])
        enriched_history = self._prepare_history_with_agents(raw_history, context)
        try:
            n = int(getattr(self, 'history_size', 6) or 6)
        except Exception:
            n = 6
        lang_code = str(lang).lower()
        for entry in enriched_history[-n:]:
            role = entry.get('role') or 'assistant'
            content = entry.get('content') or ''
            if not content:
                continue
            agent = entry.get('agent')
            if agent:
                if lang_code.startswith('en'):
                    if role == 'assistant':
                        prefix = f"[Agent: {agent}] "
                    else:
                        prefix = f"[User reply for {agent}] "
                else:
                    if role == 'assistant':
                        prefix = f"[: {agent}] "
                    else:
                        prefix = f"[ {agent}] "
                content = prefix + content
            messages.append({"role": role, "content": content})
        messages.append({"role": "assistant", "content": current_question})
        if str(lang).lower().startswith('en'):
            user_prompt = "Please answer directly based on your traveler role and provide only the necessary details."
        else:
            user_prompt = "，。"
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def _generate_specific_dates(self) -> str:
        try:
            cand = self._invoke_plugin_hook('generate_specific_dates', self._build_plugin_context({}, 'travel_dates'), max_candidates=3)
            if isinstance(cand, (list, tuple)) and len(cand) > 0:
                first = cand[0]
                if isinstance(first, str):
                    return first
                if isinstance(first, dict):
                    for k in ('dates', 'value'):
                        if first.get(k):
                            return first.get(k)
        except Exception:
            pass
        now = datetime.now()
        start = now + timedelta(days=7)
        end = start + timedelta(days=3)
        return f"{start.year}-{start.month:02d}-{start.day:02d} to {end.year}-{end.month:02d}-{end.day:02d}"

    def _generate_round_trip_dates(self) -> str:
        try:
            cand = self._invoke_plugin_hook('generate_round_trip_dates', self._build_plugin_context({}, 'travel_dates'), max_candidates=3)
            if isinstance(cand, (list, tuple)) and len(cand) > 0:
                first = cand[0]
                if isinstance(first, str):
                    return first
        except Exception:
            pass
        now = datetime.now()
        go = now + timedelta(days=7)
        ret = go + timedelta(days=3)
        return f"{go.year}-{go.month:02d}-{go.day:02d} to {ret.year}-{ret.month:02d}-{ret.day:02d}"

    def _contains_specific_dates(self, text: str) -> bool:
        try:
            res = self._invoke_plugin_hook('contains_specific_dates', text)
            if isinstance(res, bool):
                return res
        except Exception:
            pass
        return False

    def _is_reply_matching_question(self, question_type: str, reply: str) -> bool:
        if not reply or not isinstance(reply, str):
            return False
        r = reply.strip()
        low = r.lower()
        if low in ('ok', 'okay', 'yes', 'y', 'no', 'thanks', 'thank you', 'thx', '', '', '', '', '', ''):
            return False
        if len(r) < 4 and not any(ch.isdigit() for ch in r) and not any(tok in r for tok in ['', '', '', '', ' city', ' district', '', '', '', '', '']):
            return False
        try:
            res = self._invoke_plugin_hook('is_reply_matching_question', question_type, reply)
            if isinstance(res, bool):
                return res
        except Exception:
            pass
        if any(ch.isdigit() for ch in r):
            return True
        if len(r) >= 6:
            return True
        return False

    def _classify_question(self, question: str) -> str:
        try:
            plugin = getattr(self, 'domain_plugin', None)
            if plugin and hasattr(plugin, 'classify_question'):
                try:
                    res = plugin.classify_question(question, getattr(self, 'lang', 'zh'))
                    if isinstance(res, str) and res:
                        return res
                except Exception:
                    pass
        except Exception:
            pass
        try:
            return 'general'
        except Exception:
            return 'general'

    def generate_clarification_response(self, question: str, context: Dict) -> str:
        """History-first LLM clarification pipeline:
        1) 1) Query LLM with recent history (including subAgentName labels);
        2) 2) If unresolved, fallback to quick current-question-only attempt;
        3) 3) If still unresolved, call domain plugin `generate_candidate_list` as fallback;
        4) 4) Finally fallback to rule-based or profile quick answer.
        """
        qtype = self._classify_question(question)
        # treat question as date-targeting if heuristic detects date keywords
        # q_is_date = self._is_date_question(qtype) or self._question_targets_date(question)
        # q_is_city = self._question_targets_city(question)
        # q_is_destination = self._is_destination_question(qtype) or self._question_targets_destination(question) or q_is_city

        q_is_date = None
        q_is_city = None
        q_is_destination = None

        # Directly defer to LLM-driven clarifications; heuristic shortcuts were removed due to poor response quality.

        # 1) history-informed LLM attempt (trimmed context with sub-agent hints)
        try:
            trimmed_ctx = self._trim_context_history(context, rounds=3)
            msgs = self._build_conversation_context(question, trimmed_ctx)
            history_resp = None
            try:
                history_resp = self._call_llm(msgs)
                print("LLM history-informed response:", history_resp)
            except Exception:
                history_resp = None
                print("LLM history-informed call failed.")

            if history_resp:
                norm = self.normalize_user_response(history_resp, question)
                # Detect LLM echoing the original user request: if normalized reply contains
                # the earliest user utterance from context (likely the original query), treat as invalid.
                try:
                    initial_user = None
                    convo = context.get('conversation_history', []) or []
                    for entry in convo:
                        if entry.get('role') == 'user':
                            initial_user = (entry.get('content') or '').strip()
                            break
                    if initial_user and isinstance(norm, str) and initial_user and initial_user.lower() in norm.lower() and len(norm) > 20:
                        # consider this a failed LLM reply and fall through to fallback attempts
                        raise ValueError('llm_echo_detected')
                except Exception:
                    # fall through to plugin / rule fallbacks
                    print("Fallback through to plugin / rule fallbacks ")
                    pass
                try:
                    if q_is_date and not q_is_city:
                        if not (isinstance(norm, str) and self._dates_within_window(norm, days=30)):
                            round_trip_flag = True if ('round trip' in (question or '') or 'return' in (question or '').lower()) else False
                            return self._handle_rejected_candidate(norm, 'date_out_of_window', context, round_trip=round_trip_flag)
                        try:
                            norm = self._ensure_future_dates_with_year(norm, days=30)
                        except Exception:
                            pass
                        try:
                            norm = self._ensure_future_dates_with_year(norm, days=30)
                        except Exception:
                            pass
                    # If question explicitly requests city, ensure reply includes it; otherwise synthesize
                    if q_is_city and not self._is_in_china(norm):
                        dates_found = re.findall(r"(\d{4}-\d{1,2}-\d{1,2})", str(norm))
                        date_reply = None
                        if dates_found:
                            if len(dates_found) >= 2:
                                date_reply = f"Departure date: {dates_found[0]} Return date: {dates_found[1]}"
                            else:
                                try:
                                    d0 = self._parse_iso_date(dates_found[0])
                                    r0 = d0 + timedelta(days=3)
                                    date_reply = f"Departure date: {d0.strftime('%Y-%m-%d')} Return date: {r0.strftime('%Y-%m-%d')}"
                                except Exception:
                                    date_reply = ''

                        city_candidate = None
                        try:
                            cand = self._invoke_plugin_hook('generate_candidate_list', question, self._build_plugin_context(context, question), max_candidates=3)
                            if isinstance(cand, (list, tuple)) and len(cand) > 0:
                                first = cand[0]
                                if isinstance(first, dict):
                                    for k in ('departure_city', 'departure', 'place', 'location', 'value'):
                                        if first.get(k):
                                            city_candidate = first.get(k)
                                            break
                                elif isinstance(first, str):
                                    city_candidate = first
                        except Exception:
                            city_candidate = None
                        if not city_candidate:
                            try:
                                city_candidate = self.profile.get('home_city') or self.profile.get('city') or None
                            except Exception:
                                city_candidate = None
                        if not city_candidate:
                            city_candidate = ''

                        combined = None
                        if date_reply:
                            combined = f"Departure city: {city_candidate} {date_reply}"
                        else:
                            combined = f"Departure city: {city_candidate}"

                        try:
                            combined = self._ensure_future_dates_with_year(combined, days=30)
                        except Exception:
                            pass

                        try:
                            self.record_interaction('user', combined, context)
                        except Exception:
                            pass
                        self.last_used_strategy = 'llm-history-synth'
                        return combined

                    if q_is_destination:
                        if not (isinstance(norm, str) and self._is_in_china(norm)):
                            raise ValueError('destination_not_in_china')

                    if self._is_reply_matching_question(qtype, norm):
                        if q_is_city and not self._is_in_china(norm):
                            raise ValueError('llm_history_invalid_city')
                        try:
                            self.record_interaction('user', norm, context)
                        except Exception:
                            pass
                        self.last_used_strategy = 'llm-history'
                        return norm
                except Exception:
                    # fall through to next fallback
                    pass
        except Exception:
            pass

        # 2) quick fallback LLM attempt (minimal context only)
        try:
            print("Fallback to quick LLM attempt")
            quick = None
            try:
                # reuse base helper when possible
                quick = getattr(self, '_llm_quick_clarify', None)
                if callable(quick):
                    quick = quick(question, qtype)
                else:
                    # fallback: call LLM with minimal messages including 'return only' hint
                    minimal_ctx = dict(context or {})
                    minimal_ctx['conversation_history'] = []
                    msgs = self._build_conversation_context(question, minimal_ctx)
                    msgs.append({'role': 'user', 'content': 'Please return only the direct value without extra explanation.'})
                    quick = self._call_llm(msgs, override_config={'temperature': 0.0, 'max_tokens': 150})
            except Exception:
                quick = None

            if quick:
                norm = self.normalize_user_response(quick, question)
                # Detect LLM echoing the original user request: if normalized reply contains
                # the earliest user utterance from context (likely the original query), treat as invalid.
                try:
                    initial_user = None
                    convo = context.get('conversation_history', []) or []
                    for entry in convo:
                        if entry.get('role') == 'user':
                            initial_user = (entry.get('content') or '').strip()
                            break
                    if initial_user and isinstance(norm, str) and initial_user and initial_user.lower() in norm.lower() and len(norm) > 20:
                        # treat as rejected echo
                        quick = None
                    else:
                        pass
                except Exception:
                    pass
                # If question expects a destination but LLM returned a date-like value, do not accept it
                try:
                    # enforce simulator constraints: dates must be within next 30 days; destinations must be in China
                    if q_is_date and not q_is_city:
                        if not (isinstance(norm, str) and self._dates_within_window(norm, days=30)):
                            # not acceptable -> generate replacement and return metadata dict
                            round_trip_flag = True if ('round trip' in (question or '') or 'return' in (question or '').lower()) else False
                            return self._handle_rejected_candidate(norm, 'date_out_of_window', context, round_trip=round_trip_flag)

                    # If question explicitly asks for city but LLM returned only dates, try to synthesize a combined reply
                    if q_is_city and not self._is_in_china(norm):
                        # norm doesn't contain a valid city; try to extract dates from norm
                        dates_found = re.findall(r"(\d{4}-\d{1,2}-\d{1,2})", str(norm))
                        date_reply = None
                        if dates_found:
                            if len(dates_found) >= 2:
                                date_reply = f"Departure date: {dates_found[0]} Return date: {dates_found[1]}"
                            else:
                                try:
                                    d0 = self._parse_iso_date(dates_found[0])
                                    r0 = d0 + timedelta(days=3)
                                    date_reply = f"Departure date: {d0.strftime('%Y-%m-%d')} Return date: {r0.strftime('%Y-%m-%d')}"
                                except Exception:
                                    date_reply = ''

                        # obtain a city candidate from plugin/profile/fallback
                        city_candidate = None
                        try:
                            cand = self._invoke_plugin_hook('generate_candidate_list', question, self._build_plugin_context(context, question), max_candidates=3)
                            if isinstance(cand, (list, tuple)) and len(cand) > 0:
                                first = cand[0]
                                if isinstance(first, dict):
                                    for k in ('departure_city', 'departure', 'place', 'location', 'value'):
                                        if first.get(k):
                                            city_candidate = first.get(k)
                                            break
                                elif isinstance(first, str):
                                    city_candidate = first
                        except Exception:
                            city_candidate = None
                        if not city_candidate:
                            # try profile common fields
                            try:
                                city_candidate = self.profile.get('home_city') or self.profile.get('city') or None
                            except Exception:
                                city_candidate = None
                        if not city_candidate:
                            city_candidate = ''

                        combined = None
                        if date_reply:
                            combined = f"Departure city: {city_candidate} {date_reply}"
                        else:
                            combined = f"Departure city: {city_candidate}"

                        try:
                            combined = self._ensure_future_dates_with_year(combined, days=30)
                        except Exception:
                            pass

                        try:
                            self.record_interaction('user', combined, context)
                        except Exception:
                            pass
                        self.last_used_strategy = 'llm-quick-synth'
                        return combined

                    if self._is_destination_question(qtype):
                        if not (isinstance(norm, str) and self._is_in_china(norm)):
                            raise ValueError('destination_not_in_china')

                    if self._is_reply_matching_question(qtype, norm):
                        if q_is_destination and not self._is_in_china(norm):
                            raise ValueError('llm_quick_invalid_city')
                        try:
                            self.record_interaction('user', norm, context)
                        except Exception:
                            pass
                        self.last_used_strategy = 'llm-quick'
                        return norm
                except Exception:
                    pass
        except Exception:
            pass

        # 3) plugin-based fallback: generate_candidate_list
        try:
            print("Fallback to plugin-based")
            cand = self._invoke_plugin_hook('generate_candidate_list', question, self._build_plugin_context(context, question), max_candidates=5)
            if isinstance(cand, str) and cand:
                # validate plugin candidate
                try:
                    val = self.normalize_user_response(cand, question)
                    if q_is_date and not q_is_city and not self._dates_within_window(val, days=30):
                        round_trip_flag = True if ('round trip' in (question or '') or 'return' in (question or '').lower()) else False
                        return self._handle_rejected_candidate(val, 'date_out_of_window', context, round_trip=round_trip_flag)
                        if q_is_date and not q_is_city:
                            try:
                                val = self._ensure_future_dates_with_year(val, days=30)
                            except Exception:
                                pass
                    if q_is_destination and not self._is_in_china(val):
                        raise ValueError('destination_not_in_china')
                    try:
                        self.record_interaction('user', val, context)
                    except Exception:
                        pass
                    self.last_used_strategy = 'plugin-candidate'
                    return val
                except Exception:
                    pass
            if isinstance(cand, (list, tuple)) and len(cand) > 0:
                first = cand[0]
                if isinstance(first, str):
                    try:
                        val = self.normalize_user_response(first, question)
                        if q_is_date and not q_is_city and not self._dates_within_window(val, days=30):
                            round_trip_flag = True if ('round trip' in (question or '') or 'return' in (question or '').lower()) else False
                            return self._handle_rejected_candidate(val, 'date_out_of_window', context, round_trip=round_trip_flag)
                        if q_is_date and not q_is_city:
                            try:
                                val = self._ensure_future_dates_with_year(val, days=30)
                            except Exception:
                                pass
                        if q_is_destination and not self._is_in_china(val):
                            raise ValueError('destination_not_in_china')
                        try:
                            self.record_interaction('user', val, context)
                        except Exception:
                            pass
                        self.last_used_strategy = 'plugin-candidate'
                        return val
                    except Exception:
                        pass
                if isinstance(first, dict):
                    for k in ('destination', 'place', 'location', 'value'):
                        if first.get(k):
                            try:
                                val = self.normalize_user_response(first.get(k), question)
                                if q_is_date and not q_is_city and not self._dates_within_window(val, days=30):
                                    raise ValueError('date_out_of_window')
                                if q_is_date and not q_is_city:
                                    try:
                                        val = self._ensure_future_dates_with_year(val, days=30)
                                    except Exception:
                                        pass
                                if q_is_destination and not self._is_in_china(val):
                                    raise ValueError('destination_not_in_china')
                                try:
                                    self.record_interaction('user', val, context)
                                except Exception:
                                    pass
                                self.last_used_strategy = 'plugin-candidate'
                                return val
                            except Exception:
                                pass
        except Exception:
            pass

        # 4) rule-based simulator fallback if available
        try:
            print("Fallback to rule-based")
            if self.rule_based_simulator:
                det = self.rule_based_simulator.generate_clarification_response(question, context)
                if det:
                    try:
                        self.record_interaction('user', det, context)
                    except Exception:
                        pass
                    self.last_used_strategy = 'rule-fallback'
                    return det
        except Exception:
            pass

        # 5) final generic quick answer from profile or neutral message
        try:
            last_user = context.get('conversation_history', [])[-1]['content'] if context.get('conversation_history') else None
        except Exception:
            last_user = None
        fallback_qtype = 'destination' if (q_is_destination and qtype == 'general') else qtype
        final = self._generate_quick_answer(fallback_qtype, last_user, question)
        try:
            self.record_interaction('user', final, context)
        except Exception:
            pass
        self.last_used_strategy = 'quick-profile'
        return final
    
    def _contains_relative_keyword(self, text: str) -> bool:
        kws = ['next week', 'next month', 'tomorrow', 'the day after tomorrow', 'next week']
        lower = text
        for k in kws:
            if k in lower:
                return True
        return False

    def _is_date_question(self, qtype: str) -> bool:
        if not qtype or not isinstance(qtype, str):
            return False
        q = qtype.lower()
        return 'date' in q or 'time' in q or 'trip' in q or 'travel' in q or 'departure' in q or 'return' in q

    def _question_targets_date(self, question_text: str) -> bool:
        """Heuristic: inspect the clarification question text to see if it is asking for date/time info.

        Earlier logic treated any occurrence of "departure" or "return" as a date request, which caused
        departure city clarifications to be mislabelled as date prompts. The updated heuristic limits
        matches to explicit date/time cues and only treats mixed questions as date-related when those
        cues are present alongside the city wording.
        """
        if not question_text or not isinstance(question_text, str):
            return False

        lowered = question_text.lower()
        cleaned = re.sub(r'[\*`_~]+', ' ', lowered)

        def _contains_en_token(text: str, token: str) -> bool:
            pattern = r'\b' + re.escape(token) + r'\b'
            return re.search(pattern, text) is not None

        # If the question is clearly about a city/airport and lacks explicit date hints, skip
        if self._question_targets_city(question_text):
            explicit_cn_tokens = ['date', 'time', 'which day', 'which day', 'what date', 'what date', 'exact time']
            explicit_en_tokens = ['which day', 'what day', 'what date', 'time frame', 'date range']
            numeric_date_pattern = r'(\d{1,2}\s*(||))'
            has_cn = any(token in cleaned for token in explicit_cn_tokens)
            has_en = any(_contains_en_token(cleaned, token) for token in ['date', 'dates'])
            has_phrase = any(_contains_en_token(cleaned, token) for token in explicit_en_tokens)
            if not (has_cn or has_en or has_phrase or re.search(numeric_date_pattern, question_text)):
                return False

        cn_date_keywords = ['date', 'time', 'which day', 'which day', 'what date', 'what date', 'when', 'exact time', 'date range', 'time range']
        if any(keyword in cleaned for keyword in cn_date_keywords):
            return True

        en_date_keywords = ['travel dates', 'trip dates', 'date range', 'time frame']
        if any(_contains_en_token(cleaned, keyword) for keyword in en_date_keywords):
            return True

        if _contains_en_token(cleaned, 'schedule'):
            return True

        if _contains_en_token(cleaned, 'date') or _contains_en_token(cleaned, 'dates'):
            return True

        paired_keywords = [
            ('departure', 'date'),
            ('departure', 'time'),
            ('return', 'date'),
            ('return', 'time'),
            ('departure', 'date'),
            ('return', 'date'),
            ('departure', 'time'),
            ('return', 'time'),
        ]
        for first, second in paired_keywords:
            if first in cleaned and second in cleaned:
                return True

        if re.search(r'(what|which)\s+(day|date)', cleaned):
            return True

        if re.search(r'(\d{1,2})[-/](\d{1,2})', question_text):
            return True

        return False

    def _question_targets_city(self, question_text: str) -> bool:
        """Heuristic: detect whether the clarification question explicitly asks for a departure city/place."""
        if not question_text or not isinstance(question_text, str):
            return False
        # remove lightweight markdown/punctuation noise that can break substring checks
        lowered = question_text.lower()
        cleaned = re.sub(r'[\*`_~]+', ' ', lowered)
        normalized = cleaned.replace('/', ' ')

        city_keywords = [
            'departurecity', 'departure', 'departurelocation', 'city', 'city',
            'destination city', 'destination', 'travel destination',
            'departure city', 'departure airport', 'departure location',
            'departing city', 'departing airport', 'depart from',
            'flying from', 'leave from', 'leaving from',
            'from city', 'from which city', 'from which airport',
            'city name', 'airport code', 'destination'
        ]
        for keyword in city_keywords:
            if keyword in normalized:
                return True

        patterns = [
            r'(?:flying|departing|leaving)\s+from\b',
            r'from\s+(?:what|which)\s+(?:city|airport)\b',
            r'provide\s+(?:the\s+)?(?:city|airport)\b',
        ]
        for pattern in patterns:
            if re.search(pattern, normalized):
                return True

        return False

    def _question_targets_destination(self, question_text: str) -> bool:
        if not question_text or not isinstance(question_text, str):
            return False
        lowered = question_text.lower()
        cleaned = re.sub(r'[\*`_~]+', ' ', lowered)
        normalized = cleaned.replace('/', ' ')

        destination_keywords = [
            'destination', 'warm-weather destination', 'holiday spot', 'resort',
            'vacation spot', 'travel destination', 'visit where', 'where to go',
            'where to go', 'destination', 'travel destination', 'vacation destination', 'destination city'
        ]
        for keyword in destination_keywords:
            if keyword in normalized:
                return True

        patterns = [
            r'(?:which|what)\s+(?:destination|city|place)\b',
            r'\s*',
            r''
        ]
        for pattern in patterns:
            if re.search(pattern, normalized):
                return True

        return False

    def _extract_forced_choice_options(self, question_text: str) -> List[str]:
        if not question_text or not isinstance(question_text, str):
            return []
        lowered = question_text.lower()
        if not any(marker in lowered for marker in (' or ', 'or', ' ', 'or')):
            return []
        candidates: List[str] = []
        for token in self._extract_date_phrases(question_text):
            cleaned = token.strip().strip(',.?;:')
            if cleaned and cleaned not in candidates:
                candidates.append(cleaned)
        if len(candidates) >= 2:
            return candidates
        return []

    def _extract_date_phrases(self, text: str) -> List[str]:
        if not text or not isinstance(text, str):
            return []
        patterns = [
            re.compile(r'\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{4})?', re.IGNORECASE),
            re.compile(r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b'),
            re.compile(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}'),
            re.compile(r'\d{1,2}[-/]\d{1,2}')
        ]
        results: List[str] = []
        for pattern in patterns:
            for match in pattern.finditer(text):
                token = match.group(0)
                if token and token not in results:
                    results.append(token)
        return results

    def _select_forced_choice_option(self, options: List[str], context: Dict) -> Optional[str]:
        if not options:
            return None
        history_text = ''
        try:
            history = (context or {}).get('conversation_history', []) or []
            parts = []
            for entry in history:
                if isinstance(entry, dict):
                    content = entry.get('content')
                    if isinstance(content, str) and content:
                        parts.append(content)
            history_text = ' '.join(parts).lower()
        except Exception:
            history_text = ''

        best_option = None
        best_score = -1
        for opt in options:
            if not isinstance(opt, str):
                continue
            score = history_text.count(opt.lower()) if history_text else 0
            if score > best_score:
                best_score = score
                best_option = opt
        if best_option is None:
            return options[0]
        return best_option

    def _format_forced_choice_reply(self, option: str) -> str:
        if not option:
            return ''
        choice = option.strip()
        lang = str(getattr(self, 'lang', 'zh') or 'zh').lower()
        if lang.startswith('en'):
            return choice if choice.endswith('.') else choice
        if choice.startswith('Yes, '):
            return choice if choice.endswith(('。', '！', '？')) else f"{choice}。"
        suffix = '' if choice.endswith(('。', '！', '？')) else '。'
        return f"Yes, {choice}{suffix}"

    def _extract_candidate_destinations(self, question_text: str) -> List[str]:
        if not question_text or not isinstance(question_text, str):
            return []
        lowered = question_text.lower()
        candidates: List[str] = []
        for canonical, variants in self._destination_variant_map().items():
            for variant in variants:
                if variant.isascii():
                    pattern = r'\b' + re.escape(variant) + r'\b'
                    if re.search(pattern, lowered):
                        candidates.append(canonical)
                        break
                else:
                    if variant in question_text:
                        candidates.append(canonical)
                        break
        return list(dict.fromkeys(candidates))

    def _select_destination_option(self, options: List[str], context: Dict) -> Optional[str]:
        if not options:
            return None
        profile = getattr(self, 'profile', {}) or {}
        preferred_values: List[str] = []
        for key in ('preferred_destinations', 'favorite_destinations', 'go_to_destination', 'preferred_destination', 'travel_goal'):
            val = profile.get(key)
            if isinstance(val, (list, tuple)):
                preferred_values.extend([str(item) for item in val if isinstance(item, (str, int, float))])
            elif isinstance(val, (str, int, float)):
                preferred_values.append(str(val))

        def _normalize(value: str) -> Optional[str]:
            if not value:
                return None
            lowered = str(value).strip().lower()
            for canonical, variants in self._destination_variant_map().items():
                if any(lowered == variant for variant in variants):
                    return canonical
            return lowered

        normalized_options = [(opt, _normalize(opt)) for opt in options]
        normalized_preferences = [_normalize(pref) for pref in preferred_values]

        for opt, norm in normalized_options:
            if norm and norm in normalized_preferences:
                return norm

        for opt, norm in normalized_options:
            if norm and self._is_destination_warm(norm):
                return norm

        return normalized_options[0][1] or normalized_options[0][0]

    def _format_destination_choice_reply(self, canonical_name: str) -> str:
        display_en = self._destination_display_map().get(canonical_name, canonical_name.title())
        display_cn = self._destination_chinese_map().get(canonical_name, canonical_name)
        lang = str(getattr(self, 'lang', 'zh') or 'zh').lower()
        if lang.startswith('en'):
            return f"I'd prefer to focus on {display_en} for this trip."
        return f"I'd prefer to go to {display_cn}, it feels more suitable."

    def _destination_variant_map(self) -> Dict[str, List[str]]:
        return {
            'shanghai': ['shanghai', 'shanghai city'],
            'hangzhou': ['hangzhou', 'hangzhou city'],
            'suzhou': ['suzhou'],
            'sanya': ['sanya'],
            'xiamen': ['xiamen'],
            'haikou': ['haikou'],
            'guangzhou': ['guangzhou'],
            'shenzhen': ['shenzhen'],
            'beijing': ['beijing'],
            'chengdu': ['chengdu'],
            'qingdao': ['qingdao'],
            'guilin': ['guilin'],
            'chongqing': ['chongqing'],
            'nanjing': ['nanjing'],
            'lijiang': ['lijiang'],
            'zhuhai': ['zhuhai'],
            'wuhan': ['wuhan'],
            'dalian': ['dalian'],
        }

    def _destination_display_map(self) -> Dict[str, str]:
        return {
            'shanghai': 'Shanghai',
            'hangzhou': 'Hangzhou',
            'suzhou': 'Suzhou',
            'sanya': 'Sanya',
            'xiamen': 'Xiamen',
            'haikou': 'Haikou',
            'guangzhou': 'Guangzhou',
            'shenzhen': 'Shenzhen',
            'beijing': 'Beijing',
            'chengdu': 'Chengdu',
            'qingdao': 'Qingdao',
            'guilin': 'Guilin',
            'chongqing': 'Chongqing',
            'nanjing': 'Nanjing',
            'lijiang': 'Lijiang',
            'zhuhai': 'Zhuhai',
            'wuhan': 'Wuhan',
            'dalian': 'Dalian',
        }

    def _destination_chinese_map(self) -> Dict[str, str]:
        return {
            'shanghai': '',
            'hangzhou': '',
            'suzhou': '',
            'sanya': '',
            'xiamen': '',
            'haikou': '',
            'guangzhou': '',
            'shenzhen': '',
            'beijing': '',
            'chengdu': '',
            'qingdao': '',
            'guilin': '',
            'chongqing': '',
            'nanjing': '',
            'lijiang': '',
            'zhuhai': '',
            'wuhan': '',
            'dalian': '',
        }

    def _is_destination_warm(self, canonical_name: Optional[str]) -> bool:
        warm_set = {
            'sanya', 'xiamen', 'haikou', 'zhuhai', 'guilin', 'guangzhou', 'shenzhen', 'hangzhou'
        }
        if not canonical_name:
            return False
        return canonical_name in warm_set

    def _is_destination_question(self, qtype: str) -> bool:
        if not qtype or not isinstance(qtype, str):
            return False
        q = qtype.lower()
        return 'dest' in q or 'destination' in q or 'place' in q or 'location' in q

    def _is_in_china(self, text: str) -> bool:
        """Check whether a place text refers to a location within China.

        Strategy:
        - Ask domain plugin via hook `is_place_in_china` if available.
        - Fallback heuristics: contains 'China' or 'China' or contains CJK characters or matches common Chinese city names.
        """
        try:
            res = self._invoke_plugin_hook('is_place_in_china', text)
            if isinstance(res, bool):
                return res
        except Exception:
            pass
        if not text or not isinstance(text, str):
            return False
        low = text.lower()
        # explicit indicators
        if 'china' in low or 'China' in text:
            return True

        # explicit negative indicators for obvious foreign places (reduce false positives)
        negative_indicators = ['Japan', 'Tokyo', '', 'Osaka', 'osaka', 'tokyo', 'Hokkaido', 'Sapporo', 'Korea', 'korea', 'seoul', '서울', '미국', 'usa', 'united states']
        for neg in negative_indicators:
            if neg in text or neg in low:
                return False

        # Require stronger evidence that text refers to a Chinese location:
        # 1) contains Chinese administrative suffixes like ' city',' province',' autonomous region',' district',' county'
        chinese_suffixes = [' city', ' province', ' autonomous region', ' district', ' county', ' town']
        for suf in chinese_suffixes:
            if suf in text:
                return True

        # 2) match against a known list of major Chinese city names (both english and chinese)
        city_keywords = ['beijing','','shanghai','','guangzhou','','shenzhen','','chengdu','','hangzhou','','xian','','wuhan','','nanjing','','tianjin','','chongqing','','suzhou','','qingdao','','dalian','','sanya','','xiamen','','haikou','']
        for c in city_keywords:
            if c in low or c in text:
                return True

        # Fallback: conservative false for any other short/ambiguous names to avoid accepting e.g. Japanese/Korean
        return False

    def _parse_iso_date(self, text: str):
        """Try to parse first yyyy-mm-dd or yyyy/mm/dd found in text. Return datetime.date or None."""
        try:
            import re
            m = re.search(r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})', text)
            if m:
                y = int(m.group(1)); mo = int(m.group(2)); d = int(m.group(3))
                return datetime(year=y, month=mo, day=d)
        except Exception:
            pass
        return None

    def _dates_within_window(self, text: str, days: int = 30) -> bool:
        """Override: try plugin hook first, otherwise simple ISO-date heuristic within `days` days from now."""
        try:
            res = self._invoke_plugin_hook('dates_within_window', text, days)
            if isinstance(res, bool):
                return res
        except Exception:
            pass
        if not text or not isinstance(text, str):
            return False
        dt = self._parse_iso_date(text)
        if dt is None:
            # try to find a year-less month-day like '12-20' or '1220' (basic)
            import re
            m = re.search(r'(\d{1,2})[-/](\d{1,2})', text)
            if m:
                mo = int(m.group(1)); d = int(m.group(2))
                now = datetime.now()
                # pick this year, but if that date already passed, try next year
                try:
                    cand = datetime(year=now.year, month=mo, day=d)
                except Exception:
                    return False
                if cand < now:
                    try:
                        cand = datetime(year=now.year + 1, month=mo, day=d)
                    except Exception:
                        return False
                dt = cand
            else:
                # detect Chinese '1220' pattern
                m2 = re.search(r'(\d{1,2})[-/](\d{1,2})', text)
                if m2:
                    mo = int(m2.group(1)); d = int(m2.group(2))
                    now = datetime.now()
                    try:
                        cand = datetime(year=now.year, month=mo, day=d)
                    except Exception:
                        return False
                    if cand < now:
                        try:
                            cand = datetime(year=now.year + 1, month=mo, day=d)
                        except Exception:
                            return False
                    dt = cand
                else:
                    return False
        # now have dt datetime
        now = datetime.now()
        delta = dt - now
        return 0 <= delta.days <= int(days)

    def _contains_venue_indicator(self, text: str) -> bool:
        try:
            res = self._invoke_plugin_hook('contains_venue_indicator', text)
            if isinstance(res, bool):
                return res
        except Exception:
            pass
        return False

    def _has_explicit_year(self, text: str) -> bool:
        try:
            res = self._invoke_plugin_hook('has_explicit_year', text)
            if isinstance(res, bool):
                return res
        except Exception:
            pass
        return False

    def _has_month_day(self, text: str) -> bool:
        try:
            res = self._invoke_plugin_hook('has_month_day', text)
            if isinstance(res, bool):
                return res
        except Exception:
            pass
        return False

    def _ensure_years_in_dates(self, text: str) -> str:
        try:
            res = self._invoke_plugin_hook('ensure_years_in_dates', text)
            if isinstance(res, str) and res:
                return res
        except Exception:
            pass
        return text

    def _generate_near_future_dates(self, round_trip: bool = False, prefer_days: int = 14, max_days: int = 30) -> str:
        try:
            res = self._invoke_plugin_hook('generate_near_future_dates', round_trip=round_trip, prefer_days=prefer_days, max_days=max_days)
            if isinstance(res, str) and res:
                return res
        except Exception:
            pass
        now = datetime.now()
        start = now + timedelta(days=7)
        end = start + timedelta(days=3)
        return f"{start.year}-{start.month:02d}-{start.day:02d} to {end.year}-{end.month:02d}-{end.day:02d}"

    def _generate_quick_answer(
        self,
        question_type: Optional[str] = None,
        last_user: Optional[str] = None,
        question_text: Optional[str] = None,
    ) -> str:
        """Provide a lightweight fallback answer when all other strategies fail."""
        try:
            context = self._build_plugin_context({}, question=question_type)
            plugin_reply = self._invoke_plugin_hook('generate_quick_answer', question_type, context, last_user=last_user)
            if isinstance(plugin_reply, str) and plugin_reply.strip():
                return plugin_reply.strip()
        except Exception:
            pass

        q = (question_type or '').lower()
        question_hint = question_text or ''
        lang = str(getattr(self, 'lang', 'zh') or 'zh').lower()

        try:
            round_trip = 'round' in q or 'return' in q or (isinstance(last_user, str) and 'round trip' in last_user)
        except Exception:
            round_trip = False

        def _is_budget_question() -> bool:
            return any(key in q for key in ('budget', 'price', 'cost', 'cost', 'budget'))

        def _is_destination_question() -> bool:
            return any(key in q for key in ('dest', 'city', 'place', 'location', 'destination', 'city', 'location'))

        def _is_date_question() -> bool:
            if not q:
                return False
            return any(key in q for key in ('date', 'time', 'travel', 'departure', 'return', 'date', 'time', 'departure', 'return'))

        if _is_date_question():
            dates = self._generate_near_future_dates(round_trip=round_trip)
            if not isinstance(dates, str):
                dates = ''
            try:
                dates = self._ensure_future_dates_with_year(dates, days=30)
            except Exception:
                pass
            if lang.startswith('en'):
                return f"Travel dates: {dates}" if dates else "Travel dates: December 20, 2025 to December 24, 2025"
            return f"Travel dates: {dates}" if dates else "Travel dates: 2025-12-20 to 2025-12-24"

        if _is_destination_question() or self._question_targets_destination(question_hint) or self._question_targets_city(question_hint):
            city = None
            try:
                profile = getattr(self, 'profile', {}) or {}
                pref_list = profile.get('preferred_destinations') or profile.get('favorite_destinations')
                if isinstance(pref_list, (list, tuple)) and pref_list:
                    city = pref_list[0]
                if city is None:
                    city = profile.get('preferred_destination')
                if city is None:
                    city = profile.get('go_to_destination')
                if city is None:
                    city = profile.get('travel_goal')
                if city is None:
                    fallback_city = profile.get('home_city') or profile.get('city')
                    if fallback_city:
                        city = fallback_city
            except Exception:
                city = None
            if not city:
                city = ''
            if isinstance(city, str) and not self._is_in_china(city):
                city = ''
            if lang.startswith('en'):
                return f"I'd like to fly to {city if city else 'Sanya'}"
            return f"I'd like to go to {city if city else ''}"

        if _is_budget_question():
            if lang.startswith('en'):
                return "Budget is around 8,000 RMB, willing to adjust slightly."
            return "Budget is around 8,000 RMB, with slight flexibility."

        if lang.startswith('en'):
            return "Could you continue with the next question if anything else is needed?"
        return "Sure, please continue with the next question."

    def _normalize_detected_date_string(self, text: str) -> Optional[str]:
        try:
            res = self._invoke_plugin_hook('normalize_detected_date_string', text)
            if isinstance(res, str) and res:
                return res
        except Exception:
            pass
        return None

    def _ensure_future_dates_with_year(self, text: str, days: int = 30) -> str:
        """Ensure date strings include a year and fall in the future window.

        Converts patterns like "1215" or "12-15" to include the year, ensuring the
        result is not in the past and does not exceed the preferred future window.
        """
        if not text or not isinstance(text, str):
            return text

        try:
            if self._has_explicit_year(text):
                return text
        except Exception:
            pass

        try:
            max_days = int(days) if days is not None else 30
        except Exception:
            max_days = 30
        now = datetime.now()
        prev_target: Optional[datetime] = None
        converted = False

        def _compute_target(month: int, day: int) -> Optional[datetime]:
            nonlocal prev_target
            try:
                candidate = datetime(year=now.year, month=month, day=day)
            except Exception:
                return None
            if candidate < now:
                try:
                    candidate = candidate.replace(year=now.year + 1)
                except Exception:
                    return None
            if (candidate - now).days > max_days:
                candidate = now + timedelta(days=max_days)
            if prev_target and candidate <= prev_target:
                candidate = prev_target + timedelta(days=1)
            prev_target = candidate
            return candidate

        def _replace_cn(match):
            nonlocal converted
            month = int(match.group(1))
            day = int(match.group(2))
            target = _compute_target(month, day)
            if not target:
                return match.group(0)
            converted = True
            return target.strftime('%Y-%m-%d')

        def _replace_md(match):
            nonlocal converted
            month = int(match.group(1))
            day = int(match.group(2))
            target = _compute_target(month, day)
            if not target:
                return match.group(0)
            converted = True
            return target.strftime('%Y-%m-%d')

        out = text
        try:
            pattern_cn = re.compile(r'(\d{1,2})[-/](\d{1,2})')
            out = pattern_cn.sub(_replace_cn, out)
        except Exception:
            pass

        try:
            pattern_md = re.compile(r'(?<!\d)(\d{1,2})[-/](\d{1,2})(?!\d)')
            out = pattern_md.sub(_replace_md, out)
        except Exception:
            pass

        if converted:
            return out

        try:
            fallback = self._generate_near_future_dates(round_trip=('round trip' in text or 'return' in text.lower()))
            if fallback:
                return fallback
        except Exception:
            pass
        return out

    def _build_plugin_context(self, context: Optional[Dict] = None, question: Optional[str] = None, orchestrator_response: Optional[str] = None) -> Dict:
        ctx = {}
        base = context or {}
        ctx['task_id'] = base.get('task_id')
        ctx['conversation_history'] = base.get('conversation_history', getattr(self, 'conversation_history', []))
        ctx['last_user_reply'] = base.get('last_user_reply') or (self.conversation_history[-1]['content'] if self.conversation_history and isinstance(self.conversation_history[-1], dict) and self.conversation_history[-1].get('role') == 'user' else None)
        ctx['orchestrator_response'] = orchestrator_response
        ctx['round'] = base.get('round', 0)
        ctx['user_profile'] = getattr(self, 'profile', None)
        ctx['lang'] = getattr(self, 'lang', 'zh')
        ctx['question'] = question
        return ctx

    def _handle_rejected_candidate(self, original_candidate: str, reason: str, context: Dict, round_trip: bool = False) -> Dict:
        """Log rejection and generate a near-future replacement.

        Returns a dict with metadata so callers (benchmark) can record original vs accepted values.
        """
        out = {
            'timestamp': self.get_timestamp(),
            'original_candidate': original_candidate,
            'rejection_reason': reason,
            'task_id': (context or {}).get('task_id') if isinstance(context, dict) else None,
            'round': (context or {}).get('round') if isinstance(context, dict) else None
        }
        try:
            os.makedirs('output', exist_ok=True)
            path = os.path.join('output', 'clarify_rejections.jsonl')
            with open(path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(out, ensure_ascii=False) + '\n')
        except Exception:
            pass

        # generate replacement
        try:
            replacement = self._generate_near_future_dates(round_trip=round_trip)
        except Exception:
            replacement = self._generate_near_future_dates(round_trip=round_trip)

        return {
            'reply': replacement,
            'original_candidate': original_candidate,
            'rejection_reason': reason,
            'final_reply': replacement
        }

    def normalize_user_response(self, resp, question: Optional[str] = None) -> str:
        try:
            if resp is None:
                return ''
            if isinstance(resp, dict):
                if 'value' in resp and isinstance(resp['value'], (str, int, float, bool)):
                    return str(resp['value'])
                if len(resp) == 1:
                    v = list(resp.values())[0]
                    return str(v) if not isinstance(v, (dict, list)) else json.dumps(v, ensure_ascii=False)
                parts = []
                for k, v in resp.items():
                    parts.append(f"{k}: {v}")
                return '; '.join(parts)
            if isinstance(resp, (list, tuple)):
                if len(resp) == 0:
                    return ''
                first = resp[0]
                if isinstance(first, (str, int, float, bool)):
                    return str(first)
                if isinstance(first, dict):
                    if 'value' in first and isinstance(first['value'], (str, int, float, bool)):
                        return str(first['value'])
                    if len(first) == 1:
                        return str(list(first.values())[0])
                    return '; '.join(f"{k}: {v}" for k, v in first.items())
                return ', '.join(str(x) for x in resp)
            if isinstance(resp, str):
                s = resp.strip()
                if not s:
                    return ''
                if (s.startswith('{') and s.endswith('}')) or (s.startswith('[') and s.endswith(']')):
                    try:
                        parsed = json.loads(s)
                    except Exception:
                        parsed = self._extract_json_from_text(s)
                    if parsed is not None:
                        return self.normalize_user_response(parsed)
                parsed = self._extract_json_from_text(s)
                if parsed is not None:
                    return self.normalize_user_response(parsed)
                single = ' '.join(s.split())
                try:
                    if question and isinstance(question, str):
                        qtype = self._classify_question(question)
                        if qtype == 'destination':
                            if (self._contains_specific_dates(single) or self._has_month_day(single) or self._has_explicit_year(single)):
                                try:
                                    cand = self._invoke_plugin_hook('generate_candidate_list', question, self._build_plugin_context(None, question), max_candidates=5)
                                    if isinstance(cand, (list, tuple)) and len(cand) > 0:
                                        first = cand[0]
                                        if isinstance(first, dict):
                                            for k in ('destination', 'place', 'location', 'value'):
                                                if first.get(k):
                                                    return first.get(k)
                                    return self._generate_quick_answer('destination')
                                except Exception:
                                    return self._generate_quick_answer('destination')
                except Exception:
                    pass
                try:
                    plug_norm = None
                    try:
                        plug_norm = self._invoke_plugin_hook('normalize_slot_value', 'dates', single, getattr(self, 'lang', 'zh'))
                    except Exception:
                        plug_norm = None
                    if plug_norm and isinstance(plug_norm, str) and plug_norm.strip():
                        try:
                            further = self._normalize_detected_date_string(plug_norm)
                            if further:
                                return further
                        except Exception:
                            pass
                        return plug_norm
                    dt_norm = self._normalize_detected_date_string(single)
                    if dt_norm:
                        return dt_norm
                except Exception:
                    pass
                return single
            return str(resp)
        except Exception:
            try:
                return str(resp)
            except Exception:
                return ''
 