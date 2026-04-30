import json
import random
import argparse
from typing import List, Dict, Any, Generator
from dataclasses import dataclass
import time
try:
    import openai
except Exception:
    openai = None
import requests
import urllib3
import os
import sys
import certifi
import tempfile
import uuid
from pathlib import Path
from typing import Optional
import re
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FALLBACK_QUERY_LOG = PROJECT_ROOT / 'output' / 'travel_llm_fallback_queries.txt'

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

_LLM_CLIENT_AVAILABLE = False
LLMClient = None
get_default_client = None
try:
    # Try absolute package import first
    from datasets.llm_client import LLMClient, get_default_client
    _LLM_CLIENT_AVAILABLE = True
except Exception:
    try:
        # Try relative import (when running as package)
        from .llm_client import LLMClient, get_default_client
        _LLM_CLIENT_AVAILABLE = True
    except Exception:
        try:
            # When running the file directly, ensure project root is on sys.path
            from pathlib import Path
            proj_root = Path(__file__).resolve().parents[1]
            if str(proj_root) not in sys.path:
                sys.path.insert(0, str(proj_root))
            from datasets.llm_client import LLMClient, get_default_client
            _LLM_CLIENT_AVAILABLE = True
        except Exception:
            _LLM_CLIENT_AVAILABLE = False


# NOTE: AVAILABLE_TRAVEL_AGENTS and AGENTS_INDEX are established below.
# We prefer to load a repository-level `config/travel_agents.json` if present
# to serve as the canonical authoritative agent list for generation.

# Manual synonym mapping to map common generator agent names to canonical
# agent names discovered under `agents_cards/travel`. This helps ensure
# generated `expected_subagents` refer strictly to available agent names.
SYNONYM_MAP = {
    # flights
    'search_flights': 'flight_search',
    'flight_search': 'flight_search',
    # hotels / accommodation
    'search_hotels': 'hotel_accommodation_recommendation',
    'search_hostels': 'hotel_accommodation_recommendation',
    'search_vacation_rentals': 'hotel_accommodation_recommendation',
    'hotel_accommodation_recommendation': 'hotel_accommodation_recommendation',
    # restaurants / dining
    'search_restaurants': 'restaurant_recommendation',
    'restaurant_recommendation': 'restaurant_recommendation',
    # weather
    'get_weather_forecast': 'weather_forecast_check',
    'weather_forecast_check': 'weather_forecast_check',
    # planning / navigation
    'travel_planning': 'travel_planning',
    'create_itinerary': 'travel_planning',
    'calculate_travel_time': 'travel_planning',
    # news / search
    'news_search': 'news_search',
    # events / movies fallback
    'search_events': 'movie_recommendation',
    'movie_recommendation': 'movie_recommendation',
    # city info -> travel planning as best fit
    'get_city_info': 'travel_planning'
}


def _sanitize_expected_subagents(requested: List[str]) -> List[str]:
    """Return a cleaned list of expected subagents that exist in AVAILABLE_TRAVEL_AGENTS.

    - Keeps order and uniqueness.
    - Attempts simple fuzzy matching if exact names are not found (token overlap).
    - If result is empty, fall back to a single random available agent (if any).
    """
    if not requested:
        return []
    cleaned = []
    avail = set(AVAILABLE_TRAVEL_AGENTS)
    for t in requested:
        if not isinstance(t, str):
            continue
        tt = t.strip().lower()
        # apply direct synonym mapping first
        mapped = SYNONYM_MAP.get(tt)
        if mapped and mapped in avail and mapped not in cleaned:
            cleaned.append(mapped)
            continue

        # canonicalize similar to discovery rules
        ttcanon = re.sub(r'[^a-z0-9]+', '_', re.sub(r'(_agent)?(_en|_zh|_cn)?$', '', tt)).strip('_')
        if ttcanon in avail and ttcanon not in cleaned:
            cleaned.append(ttcanon)
            continue
    # fuzzy: try token overlap (with simple singularization)
    if not cleaned and avail:
        avail_list = list(avail)
        best_matches = []
        for t in requested:
            toks = re.findall(r'[a-z0-9]+', t.lower())
            if not toks:
                continue
            # consider singular forms by stripping trailing 's'
            toks_normalized = set([tok.rstrip('s') for tok in toks])
            scores = []
            for a in avail_list:
                a_toks = set(re.findall(r'[a-z0-9]+', a.lower()))
                # score by intersection size
                score = len(toks_normalized & set([tok.rstrip('s') for tok in a_toks]))
                scores.append((score, a))
            # pick best-scoring available agent (score > 0)
            scores.sort(reverse=True)
            if scores and scores[0][0] > 0:
                cand = scores[0][1]
                if cand not in cleaned:
                    cleaned.append(cand)
                    # stop at first good match for this requested token
                    continue
    # final fallback: pick one (avoid None)
    if not cleaned and avail:
        # Respect strict matching mode: if STRICT_AGENT_MATCH is set to a
        # truthy value in the environment, do NOT silently fall back to a
        # random/first available agent. Instead, return an empty list so
        # callers can detect the mismatch and mark the task for review.
        strict_flag = os.environ.get('STRICT_AGENT_MATCH') or os.environ.get('STRICT_AGENT_MATCHING')
        try:
            strict = bool(str(strict_flag).lower() in ('1', 'true', 'yes', 'on'))
        except Exception:
            strict = False

        if strict:
            # log a short warning for audit (caller may persist full details)
            try:
                print(f"[sanitize] Strict agent match enabled and no canonical match found for requested: {requested}")
            except Exception:
                pass
            return []

        # Non-strict (legacy) behavior: pick a deterministic first candidate
        try:
            cleaned.append(sorted(list(avail))[0])
        except Exception:
            pass

    return cleaned


def _build_agents_index() -> Dict[str, str]:
    """Build a short capability summary for each discovered travel agent.

    Returns a dict mapping canonical agent name -> short summary string.
    The summary is intentionally short (one sentence) and excludes URLs or
    other potential sensitive fields.
    """
    index = {}
    try:
        base = Path(__file__).resolve().parents[1]
        travel_dir = base / 'agents_cards' / 'travel'
        if travel_dir.exists() and travel_dir.is_dir():
            for f in travel_dir.iterdir():
                if not f.is_file():
                    continue
                if f.suffix.lower() not in ('.json',):
                    continue
                try:
                    data = json.loads(f.read_text(encoding='utf-8'))
                except Exception:
                    continue

                stem = f.stem
                # ignore generated index/cache files to avoid self-discovery
                if stem.lower() in ('agents_index', 'agents-index'):
                    continue
                low = stem.lower()
                low = re.sub(r'(_agent)?(_en|_zh|_cn|_en_us|_zh_cn)?$', '', low)
                canon = re.sub(r'[^a-z0-9]+', '_', low).strip('_')
                if not canon:
                    continue

                # Build a compact summary
                pieces = []
                # top-level name/description
                name = (data.get('name') or '').strip()
                desc = (data.get('description') or '').strip()
                if name:
                    pieces.append(name if len(name) < 80 else name[:77] + '...')
                if desc:
                    # avoid including URLs or provider details
                    safe_desc = re.sub(r'http[s]?://\S+', '', desc).strip()
                    pieces.append(safe_desc if len(safe_desc) < 140 else safe_desc[:137] + '...')

                # skills
                skills = data.get('skills') or []
                if isinstance(skills, list) and skills:
                    skill_summaries = []
                    for s in skills[:3]:
                        sname = s.get('name') if isinstance(s, dict) else None
                        sdesc = s.get('description') if isinstance(s, dict) else None
                        if sname:
                            if sdesc:
                                skill_summaries.append(f"{sname}: {sdesc.split('.')[0]}")
                            else:
                                skill_summaries.append(sname)
                    if skill_summaries:
                        pieces.append('; '.join(skill_summaries))

                # final summary
                summary = ' - '.join([p for p in pieces if p])
                if not summary:
                    summary = name or desc or canon

                # keep summary short
                if len(summary) > 220:
                    summary = summary[:217] + '...'

                index[canon] = summary
    except Exception:
        pass
    return index


# Agent capability index used for lightweight prompt injection
AGENTS_INDEX = _build_agents_index()


def _load_travel_agents_from_config() -> Dict[str, str]:
    """Load canonical travel agents from `config/travel_agents.json` if present.

    Returns a dict mapping canonical agent name -> short summary.
    """
    cfg = {}
    try:
        project_root = Path(__file__).resolve().parents[1]
        cfg_path = project_root / 'config' / 'travel_agents.json'
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
            except Exception:
                cfg = {}
    except Exception:
        cfg = {}
    return cfg


# Try to prefer an explicit config file (config/travel_agents.json) as the
# authoritative source of agent names/capabilities. If absent, fall back to
# dynamically building the index from agent cards under `agents_cards/travel`.
_config_agents = _load_travel_agents_from_config()
if _config_agents:
    print("[info] Loaded travel agents from config/travel_agents.json")
    print(_config_agents)
    AGENTS_INDEX = _config_agents
    print("[info] AGENTS_INDEX set from config.")
    print(AGENTS_INDEX, type(AGENTS_INDEX))

# Build AVAILABLE_TRAVEL_AGENTS set from AGENTS_INDEX (whether config-driven
# or discovered from agent cards).
AVAILABLE_TRAVEL_AGENTS = set(AGENTS_INDEX["en"].keys())


# --- Complexity factors canonical list and normalization ---------------------------------
# A controlled vocabulary for task `complexity_factors`. Generators and LLM-parsed
# outputs should be normalized to one of these values. Unknown values are discarded.
COMPLEXITY_FACTORS_ENUM = [
    'single_intent',
    'direct_execution',
    'may_require_date_clarification',
    'sequential_execution',
    'data_dependency',
    'multi_objective_optimization',
    'requires_clarification',
    'open_ended',
    'requires_many_clarifications',
    'multi_agent_coordination',
    'family_member_constraints',
    'dependency_handling',
    'hierarchical_execution'
]

# Common synonyms and localized strings mapped to canonical enum values.
COMPLEXITY_SYNONYMS = {
    # direct mappings
    'single_intent': 'single_intent',
    'single intent': 'single_intent',
    'direct_execution': 'direct_execution',
    'direct execution': 'direct_execution',
    'may_require_date_clarification': 'may_require_date_clarification',
    'may require date clarification': 'may_require_date_clarification',
    'sequential_execution': 'sequential_execution',
    'sequential execution': 'sequential_execution',
    'data_dependency': 'data_dependency',
    'data dependency': 'data_dependency',
    'multi_objective_optimization': 'multi_objective_optimization',
    'multi objective optimization': 'multi_objective_optimization',
    'requires_clarification': 'requires_clarification',
    'requires clarification': 'requires_clarification',
    'open_ended': 'open_ended',
    'open ended': 'open_ended',
    'requires_many_clarifications': 'requires_many_clarifications',
    'requires many clarifications': 'requires_many_clarifications',
    'multi_agent_coordination': 'multi_agent_coordination',
    'multi agent coordination': 'multi_agent_coordination',
    'dependency_handling': 'dependency_handling',
    'hierarchical_execution': 'hierarchical_execution',

    # Chinese/localized terms -> best-effort mapping
    '(, )': 'family_member_constraints',
    '': 'family_member_constraints',
    'budget': 'multi_objective_optimization',
    'clarification': 'requires_many_clarifications',
    '(, , , )': 'multi_agent_coordination',
    '': 'multi_agent_coordination',
    '': 'sequential_execution',
    '': 'sequential_execution',
    '': 'data_dependency',
    'clarification': 'requires_clarification',
    '': 'open_ended',
    'clarification': 'requires_many_clarifications'
}


def _normalize_complexity_factors(factors: List[str]) -> List[str]:
    """Normalize a list of complexity factor strings to the canonical enum values.

    - Maps common synonyms (including a small set of Chinese phrases) to canonical values.
    - Removes duplicates while preserving order.
    - Unknown values are discarded.
    """
    if not factors:
        return []
    out = []
    seen = set()
    for f in factors:
        try:
            if not f:
                continue
            # normalize whitespace and punctuation
            s = str(f).strip()
            key = re.sub(r'[\s\-\,\.;:_]+', ' ', s.lower()).strip()

            # direct synonym lookup
            canon = COMPLEXITY_SYNONYMS.get(key)
            # try simpler token form
            if not canon:
                token = key.replace(' ', '_')
                if token in COMPLEXITY_FACTORS_ENUM:
                    canon = token

            # last resort: if the raw lowercased token matches enum
            if not canon and key in COMPLEXITY_FACTORS_ENUM:
                canon = key

            if canon and canon not in seen:
                out.append(canon)
                seen.add(canon)
        except Exception:
            continue
    return out


def _default_complexity_for_level(level: str) -> List[str]:
    """Return a sensible default list of canonical complexity factors for a given level.

    This helps ensure generated tasks always include at least one complexity factor
    even when upstream heuristics or LLM outputs omit or produce unknown tokens.
    """
    mapping = {
        'T1': ['single_intent', 'direct_execution'],
        'T2': ['sequential_execution', 'data_dependency'],
        'T3': ['multi_objective_optimization', 'requires_clarification'],
        'T4': ['open_ended', 'requires_many_clarifications']
    }
    out = mapping.get((level or '').upper(), [])
    # ensure returned values are canonical (filter via enum)
    return [v for v in out if v in COMPLEXITY_FACTORS_ENUM]

# ---------------------------------------------------------------------------------------


def _compose_agent_prompt_context(max_agents: int = 8, max_chars: int = 1000, lang="en") -> str:
    """Compose a short prompt fragment listing available agents and their capabilities.

    Limits the number of agents and total chars to keep the prompt lightweight.
    """
    AGENTS_INDEX_lang = AGENTS_INDEX[lang]
    if not AGENTS_INDEX_lang:
        return ''
    parts = []
    count = 0
    for k in sorted(AGENTS_INDEX_lang.keys()):
        parts.append(f"{k}: {AGENTS_INDEX_lang[k]}")
        count += 1
        if count >= max_agents:
            break
    context = 'Available agents (agent_name: short capability):\n' + '\n'.join(parts)
    if len(context) > max_chars:
        context = context[:max_chars-3] + '...'
    context += '\n\nPlease ONLY select expected_subagents from the listed agent_name values.'
    return context


def _mask_secret(s: Optional[str]) -> str:
    """Mask secrets for safe logging: show first 4/last4 with ellipsis, or <none>."""
    try:
        if not s:
            return '<none>'
        s = str(s)
        if len(s) <= 8:
            return s[:2] + '...' + s[-1:]
        return s[:4] + '...' + s[-4:]
    except Exception:
        return '<masked>'


def _load_dotenv_if_exists(dotenv_path: Optional[str] = None) -> None:
    """Load .env file into environment if present. Prefer python-dotenv if installed, else simple parser.

    This allows users to put MODEL_ENDPOINT and MODEL_BEARER_TOKEN into a .env at project root.
    """
    try:
        from dotenv import load_dotenv
        # prefer explicit path if given
        if dotenv_path:
            load_dotenv(dotenv_path)
        else:
            # look for .env in project root
            root = Path(__file__).resolve().parents[2]
            p = root / '.env'
            if p.exists():
                load_dotenv(p)
            else:
                load_dotenv()
        return
    except Exception:
        # fallback: simple parser
        try:
            root = Path(__file__).resolve().parents[2]
            p = Path(dotenv_path) if dotenv_path else root / '.env'
            if not p.exists():
                return
            with p.open('r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if '=' not in line:
                        continue
                    k, v = line.split('=', 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k and v and k not in os.environ:
                        os.environ[k] = v
        except Exception:
            return


@dataclass
class GeneratedTask:
    task_id: str
    level: str
    query: str
    expected_subagents: List[str]
    expected_clarifications: List[str]
    user_side_milestones: List[str]
    system_side_milestones: List[str]
    complexity_factors: List[str]
    description: str


class TravelTaskGenerator:
    """travelscenarioTaskgenerate"""
    
    def __init__(self, llm_config: Dict = None):
        self.llm_config = llm_config or {
            'api_key': 'your-openai-key',
            'model': 'gpt_oss_120b',
            'temperature': 0.7
        }
        self.agent_repository = self._initialize_agents()
        self.scenario_templates = self._initialize_scenarios()
        
    def _initialize_agents(self) -> Dict[str, List[str]]:
        """initialize"""
        # Use canonical agent names present in `config/travel_agents.json` (or
        # discovered from agent cards). This avoids runtime mapping and ensures
        # templates reference real agent names.
        return {
            "basic": ["weather_forecast_check", "travel_planning", "travel_planning"],
            "transport": ["flight_search", "route_navigation", "route_navigation", "route_navigation"],
            "accommodation": ["hotel_accommodation_recommendation", "hotel_accommodation_recommendation", "hotel_accommodation_recommendation"],
            "planning": ["travel_planning", "travel_planning", "travel_planning"],
            "advanced": ["restaurant_recommendation", "movie_recommendation", "news_search"]
        }
    
    def _initialize_scenarios(self) -> Dict[str, List[str]]:
        """initializescenario"""
        return {
            "T1": [
                "scenario",
                "simplescenario", 
                "scenario"
            ],
            "T2": [
                "scenario",
                "scenario",
                "scenario"
            ],
            "T3": [
                "constraintplanningscenario",
                "complexscenario",
                "scenario"
            ],
            "T4": [
                "scenario",
                "scenario",
                "recommendationscenario"
            ]
        }
    
class TaskGenerationStrategy:
    """Taskgenerate"""
    
    @staticmethod
    def generate_t1_tasks(count: int, lang: str = 'cn') -> List[GeneratedTask]:
        """generateT1leveltask"""
        tasks = []
        # diversity: time//////dimension
        # language: 'cn'(, coverage), 
        # 'en'(), 'mixed'()
        if lang == 'en':
            base_scenarios = [
                {"template": "What's the weather like in {city}?", "agents": ["get_weather_forecast"], "clarifications": []},
                {"template": "Search flights from {from_city} to {to_city}, earliest departure please", "agents": ["search_flights"], "clarifications": ["travel_date"]},
                {"template": "Find hotels near {location}, hostels and budget hotels acceptable", "agents": ["search_hotels"], "clarifications": ["night_count"]},
                {"template": "Recommend top attractions in {city} and opening hours", "agents": ["search_attractions"], "clarifications": []},
                {"template": "Find the cheapest ticket from {from_city} to {to_city} (economy)", "agents": ["search_flights"], "clarifications": ["travel_date"]},
                {"template": "What local events are happening this afternoon in {city}?", "agents": ["search_events"], "clarifications": []}
            ]
        elif lang == 'mixed':
            base_scenarios = [
                {"template": "{city}weather, please include rain forecast", "agents": ["get_weather_forecast"], "clarifications": []},
                {"template": "Search flights {from_city}{to_city}, prefer earliest departure", "agents": ["search_flights"], "clarifications": ["travel_date"]},
                {"template": "{location}hotel", "agents": ["search_hotels"], "clarifications": ["night_count"]},
                {"template": "recommendation{city} and opening times", "agents": ["search_attractions"], "clarifications": []},
                {"template": "{from_city}{to_city} (economy)", "agents": ["search_flights"], "clarifications": ["travel_date"]},
                {"template": "{city} events/?", "agents": ["search_events"], "clarifications": []}
            ]
        else:
            base_scenarios = [
                {"template": "{city}weather", "agents": ["get_weather_forecast"], "clarifications": []},
                {"template": "{from_city}{to_city}flight, ", "agents": ["search_flights"], "clarifications": ["travel_date"]},
                {"template": "{location}hotel, hotel", "agents": ["search_hotels"], "clarifications": ["night_count"]},
                {"template": "recommendation{city}, time", "agents": ["search_attractions"], "clarifications": []},
                {"template": "{from_city}{to_city}()", "agents": ["search_flights"], "clarifications": ["travel_date"]},
                {"template": "{city}/", "agents": ["search_events"], "clarifications": []}
            ]

        # choose city names appropriate to language mode
        if lang == 'en':
            cities = ["Beijing", "Shanghai", "Guangzhou", "Shenzhen", "Hangzhou", "Chengdu", "Xi'an", "Nanjing", "Chongqing", "Tianjin", "Qingdao", "Xiamen"]
        else:
            cities = ["", "", "", "", "", "", "", "", "", "", "", ""]

        for i in range(count):
            scenario = random.choice(base_scenarios)
            city = random.choice(cities)

            if "from_city" in scenario["template"]:
                from_city = random.choice(cities)
                to_city = random.choice([c for c in cities if c != from_city])
                query = scenario["template"].format(from_city=from_city, to_city=to_city, location=to_city)
            else:
                query = scenario["template"].format(city=city, location=city, from_city=city, to_city=city)

            primary_agent = scenario["agents"][0]
            # add small variations to validation rules and complexity
            if lang == 'en':
                validation = [f"contains {primary_agent.split('_')[-1]} information"]
            else:
                validation = [f"{primary_agent.split('_')[-1]}"]
            if primary_agent == 'search_flights':
                if lang == 'en':
                    validation.append('includes price and schedule')
                else:
                    validation.append('flighttime')

            complexity = ['single_intent', 'direct_execution']
            if 'travel_date' in scenario.get('clarifications', []):
                complexity.append('may_require_date_clarification')

            if lang == 'en':
                desc = f"Simple query: related to {primary_agent}"
            else:
                desc = f"simple: {primary_agent} "
            # Occasionally inject colloquial or dialect markers to increase linguistic diversity signals
            try:
                if lang != 'en':
                    # 12% chance add colloquial particle, 8% chance add a dialect token
                    if random.random() < 0.12:
                        query = query + random.choice(['', '', '', '', '', '', ''])
                    if random.random() < 0.08:
                        # append a short dialect token to trigger dialect detection
                        query = query + ' ' + random.choice(['', '', '', '', '', '', ''])
                else:
                    if random.random() < 0.08:
                        query = query + ' ' + random.choice(["gonna", "wanna", "pls", "thx", "kinda"]) 
            except Exception:
                pass

            task = GeneratedTask(
                task_id=f"T1_{i+1:03d}",
                level="T1",
                query=query,
                expected_subagents=scenario["agents"],
                expected_clarifications=scenario["clarifications"],
                user_side_milestones=validation,
                system_side_milestones=[],  
                complexity_factors=_normalize_complexity_factors(complexity),
                description=desc
            )
            tasks.append(task)

        return tasks
    
    @staticmethod
    def generate_t2_tasks(count: int, lang: str = 'cn') -> List[GeneratedTask]:
        """generateT2leveltask"""
        tasks = []
        if lang == 'en':
            sequential_scenarios = [
                {"template": "Check the weather in {city}, then recommend suitable outdoor activities", "agents": ["get_weather_forecast", "search_attractions"], "clarifications": [], "dependency": "weather_based_recommendation"},
                {"template": "Find the airport in {city}, then search nearby hotels and estimate walking time", "agents": ["get_city_info", "search_hotels", "calculate_travel_time"], "clarifications": [], "dependency": "location_based_search"},
                {"template": "Query transport options from {from_city} to {to_city}, then estimate travel time", "agents": ["search_flights", "calculate_travel_time"], "clarifications": ["travel_date"], "dependency": "transport_planning"},
                {"template": "Check event times, then recommend nearby restaurants and estimate arrival options", "agents": ["search_events", "search_restaurants", "calculate_travel_time"], "clarifications": ["event_date"], "dependency": "event_based_plan"},
                {"template": "Check flight arrival time, then find hotel check-in time and transfer options", "agents": ["search_flights", "search_hotels", "get_city_info"], "clarifications": ["flight_number", "travel_date"], "dependency": "arrival_coordination"}
            ]
        elif lang == 'mixed':
            sequential_scenarios = [
                {"template": "{city}weather, then recommend outdoor activities", "agents": ["get_weather_forecast", "search_attractions"], "clarifications": [], "dependency": "weather_based_recommendation"},
                {"template": "Find airport in {city}, hotelwalking time", "agents": ["get_city_info", "search_hotels", "calculate_travel_time"], "clarifications": [], "dependency": "location_based_search"},
                {"template": "{from_city}{to_city}, then estimate travel time", "agents": ["search_flights", "calculate_travel_time"], "clarifications": ["travel_date"], "dependency": "transport_planning"},
                {"template": "time, then recommend restaurants nearby and estimate arrival", "agents": ["search_events", "search_restaurants", "calculate_travel_time"], "clarifications": ["event_date"], "dependency": "event_based_plan"},
                {"template": "flight arrival time, hotel check-in", "agents": ["search_flights", "search_hotels", "get_city_info"], "clarifications": ["flight_number", "travel_date"], "dependency": "arrival_coordination"}
            ]
        else:
            sequential_scenarios = [
                {"template": "{city}weather, recommendation", "agents": ["get_weather_forecast", "search_attractions"], "clarifications": [], "dependency": "weather_based_recommendation"},
                {"template": "{city}, hoteltime", "agents": ["get_city_info", "search_hotels", "calculate_travel_time"], "clarifications": [], "dependency": "location_based_search"},
                {"template": "{from_city}{to_city}, time", "agents": ["search_flights", "calculate_travel_time"], "clarifications": ["travel_date"], "dependency": "transport_planning"},
                {"template": "time, timerecommendationrestaurant", "agents": ["search_events", "search_restaurants", "calculate_travel_time"], "clarifications": ["event_date"], "dependency": "event_based_plan"},
                {"template": "flighttime, hoteltime", "agents": ["search_flights", "search_hotels", "get_city_info"], "clarifications": ["flight_number", "travel_date"], "dependency": "arrival_coordination"}
            ]

        if lang == 'en':
            cities = ["Beijing", "Shanghai", "Guangzhou", "Hangzhou", "Chengdu", "Xi'an", "Nanjing", "Chongqing"]
        else:
            cities = ["", "", "", "", "", "", "", ""]

        for i in range(count):
            scenario = random.choice(sequential_scenarios)
            city = random.choice(cities)

            if "from_city" in scenario["template"]:
                from_city = random.choice(cities)
                to_city = random.choice([c for c in cities if c != from_city])
                query = scenario["template"].format(from_city=from_city, to_city=to_city)
            else:
                # fill placeholders conservatively
                query = scenario["template"].format(city=city)

            if lang == 'en':
                validation = ["execute steps in sequence", f"handle {scenario['dependency']}", "proper_data_flow"]
            else:
                validation = ["", f" {scenario['dependency']}", "proper_data_flow"]
            # sometimes require explicit confirmation rules
            if 'event' in scenario['dependency'] or 'arrival' in scenario['dependency']:
                if lang == 'en':
                    validation.append('check arrival time and availability')
                else:
                    validation.append('time')

            complexity = ["sequential_execution", "data_dependency"]

            if lang == 'en':
                desc = f"Sequential task: {scenario['dependency']}"
            else:
                desc = f"task: {scenario['dependency']}"
            # Occasionally add colloquial/dialect markers or explicit constraint words
            try:
                if lang != 'en':
                    # 10% chance to include a colloquial particle
                    if random.random() < 0.10:
                        query = query + random.choice(['', '', '', ''])
                    # 6% chance to include an explicit constraint keyword to boost constraint detection
                    if random.random() < 0.06:
                        query = query + ' ' + random.choice(['budget', 'time', '', '', ''])
                else:
                    if random.random() < 0.08:
                        query = query + ' ' + random.choice(["pls", "thx", " asap", "budget"])
            except Exception:
                pass

            task = GeneratedTask(
                task_id=f"T2_{i+1:03d}",
                level="T2",
                query=query,
                expected_subagents=scenario["agents"],
                expected_clarifications=scenario["clarifications"],
                user_side_milestones=validation,
                system_side_milestones=[],
                complexity_factors=_normalize_complexity_factors(complexity),
                description=desc
            )
            tasks.append(task)

        return tasks
    
    @staticmethod 
    def generate_t3_tasks(count: int, lang: str = 'cn') -> List[GeneratedTask]:
        """generateT3leveltask"""
        tasks = []
        if lang == 'en':
            complex_scenarios = [
                {"template": "Plan a {days}-day family trip to {city} with a budget of {budget}, suitable for children aged {children_ages}", "agents": ["search_attractions", "search_hotels", "search_restaurants", "calculate_travel_time", "create_itinerary"], "constraints": ["budget", "family_friendly", "time_allocation"], "clarifications": ["children_ages", "exact_dates", "accommodation_preference"]},
                {"template": "Design a business trip in {city} with requirements {constraint1}, {constraint2}, budget {budget_type}", "agents": ["search_flights", "search_hotels", "search_attractions", "calculate_travel_time", "create_itinerary"], "constraints": ["time_efficiency", "business_focused", "location_proximity"], "clarifications": ["meeting_locations", "hotel_standard", "flight_preference"]},
                {"template": "Arrange a multi-city ({cities}) {days}-day tour in {city}, optimizing transport and accommodation budget", "agents": ["search_flights", "search_trains", "search_hotels", "create_itinerary", "calculate_travel_time"], "constraints": ["multi_city_coverage", "budget", "time_efficiency"], "clarifications": ["travel_dates", "priority_cities", "max_daily_travel"]},
                {"template": "Plan a {days}-day deep-dive trip focused on {interest}, include specialty dining and expert guides", "agents": ["search_attractions", "search_restaurants", "search_events", "create_itinerary"], "constraints": ["special_interest", "local_expert_contact", "authentic_experience"], "clarifications": ["interest", "group_size", "exact_dates"]},
                {"template": "Arrange a {days}-day trip in {region} covering {cities}, avoid {avoidance}, suitable for {interest} enthusiasts", "agents": ["search_flights", "search_hotels", "search_attractions", "get_weather_forecast", "calculate_travel_time", "create_itinerary"], "constraints": ["multi_city_coverage", "special_interest", "risk_avoidance"], "clarifications": ["travel_dates", "physical_condition", "detailed_interests"]}
            ]
        elif lang == 'mixed':
            complex_scenarios = [
                {"template": "planning{days}{city}, budget {budget}, {children_ages}", "agents": ["search_attractions", "search_hotels", "search_restaurants", "calculate_travel_time", "create_itinerary"], "constraints": ["budget", "family_friendly", "time_allocation"], "clarifications": ["children_ages", "exact_dates", "accommodation_preference"]},
                {"template": "{city}, {constraint1}, {constraint2}, budget: {budget_type}", "agents": ["search_flights", "search_hotels", "search_attractions", "calculate_travel_time", "create_itinerary"], "constraints": ["time_efficiency", "business_focused", "location_proximity"], "clarifications": ["meeting_locations", "hotel_standard", "flight_preference"]},
                {"template": "{city}city({cities}){days}, optimizebudget", "agents": ["search_flights", "search_trains", "search_hotels", "create_itinerary", "calculate_travel_time"], "constraints": ["multi_city_coverage", "budget", "time_efficiency"], "clarifications": ["travel_dates", "priority_cities", "max_daily_travel"]},
                {"template": "{interest}({days}), foodlocal guiderecommendation", "agents": ["search_attractions", "search_restaurants", "search_events", "create_itinerary"], "constraints": ["special_interest", "local_expert_contact", "authentic_experience"], "clarifications": ["interest", "group_size", "exact_dates"]},
                {"template": "{region}{days}, {cities}, avoid {avoidance}, {interest}", "agents": ["search_flights", "search_hotels", "search_attractions", "get_weather_forecast", "calculate_travel_time", "create_itinerary"], "constraints": ["multi_city_coverage", "special_interest", "risk_avoidance"], "clarifications": ["travel_dates", "physical_condition", "detailed_interests"]}
            ]
        else:
            complex_scenarios = [
                {"template": "planning{days}{city}, budget{budget}, {children_ages}", "agents": ["search_attractions", "search_hotels", "search_restaurants", "calculate_travel_time", "create_itinerary"], "constraints": ["budget", "family_friendly", "time_allocation"], "clarifications": ["children_ages", "exact_dates", "accommodation_preference"]},
                {"template": "{city}, {constraint1}, {constraint2}, budget{budget_type}", "agents": ["search_flights", "search_hotels", "search_attractions", "calculate_travel_time", "create_itinerary"], "constraints": ["time_efficiency", "business_focused", "location_proximity"], "clarifications": ["meeting_locations", "hotel_standard", "flight_preference"]},
                {"template": "{city}city({cities}){days}, budget", "agents": ["search_flights", "search_trains", "search_hotels", "create_itinerary", "calculate_travel_time"], "constraints": ["multi_city_coverage", "budget", "time_efficiency"], "clarifications": ["travel_dates", "priority_cities", "max_daily_travel"]},
                {"template": "{interest}({days}), recommendation", "agents": ["search_attractions", "search_restaurants", "search_events", "create_itinerary"], "constraints": ["special_interest", "local_expert_contact", "authentic_experience"], "clarifications": ["interest", "group_size", "exact_dates"]},
                {"template": "{region}{days}, {cities}, {avoidance}, {interest}", "agents": ["search_flights", "search_hotels", "search_attractions", "get_weather_forecast", "calculate_travel_time", "create_itinerary"], "constraints": ["multi_city_coverage", "special_interest", "risk_avoidance"], "clarifications": ["travel_dates", "physical_condition", "detailed_interests"]}
            ]
        
        # Regions and sample cities; choose English names for en mode
        if lang == 'en':
            regions = {
                "Yunnan": ["Dali", "Lijiang", "Shangri-La", "Kunming", "Puer"],
                "Sichuan": ["Chengdu", "Jiuzhaigou", "Leshan", "Luzhou", "Zigong"],
                "East China": ["Shanghai", "Hangzhou", "Suzhou", "Nanjing", "Ningbo"],
                "South China": ["Guangzhou", "Shenzhen", "Zhuhai", "Foshan", "Dongguan"]
            }
        else:
            regions = {
                "": ["", "", "","",""],
                "": ["", "", "","",""],
                "": ["", "", "","",""],
                "": ["", "", "","",""]
            }
        
        for i in range(count):
            scenario = random.choice(complex_scenarios)
            region = random.choice(list(regions.keys()))
            # join cities with language-appropriate separator
            cities_list = regions[region][:2]
            cities = ', '.join(cities_list) if lang == 'en' else ', '.join(cities_list)
            days = random.choice([3, 5, 7])
            budget = random.choice([3000, 5000, 8000, 10000])

            if "family" in scenario["template"]:
                if lang == 'en':
                    children_ages = random.choice(["3-6 years", "7-12 years", "13-18 years"])
                else:
                    children_ages = random.choice(["3-6", "7-12", "13-18"])
                query = scenario["template"].format(
                    days=days, city=region, budget=budget, children_ages=children_ages
                )
            elif "" in scenario["template"] or "business" in scenario["template"].lower():
                # business templates: choose constraints localized by language
                if lang == 'en':
                    constraints = [
                        "arrive Monday morning leave Friday night",
                        "hotel close to downtown",
                        "itinerary must be time-efficient"
                    ]
                    constraint1, constraint2 = random.sample(constraints, 2)
                    budget_type = random.choice(["no limit but reasonable", "moderate budget", "high standard"])
                else:
                    constraints = [
                        "",
                        "hotel", 
                        ""
                    ]
                    constraint1, constraint2 = random.sample(constraints, 2)
                    budget_type = random.choice(["", "budget", ""])

                query = scenario["template"].format(
                    city=region, constraint1=constraint1, constraint2=constraint2, budget_type=budget_type
                )
            else:
                if lang == 'en':
                    avoidance = random.choice(["altitude sickness", "rainy season", "peak season", "traffic restrictions"])
                    interest = random.choice(["photography", "local cuisine", "history and culture", "natural scenery", "outdoor adventure"])
                    children_ages_choice = random.choice(["3-6 years", "7-12 years", "13-18 years"])
                    budget_type = random.choice(["no limit but reasonable", "moderate budget", "high standard"])
                    group_size = str(random.choice([2, 4, 6]))
                else:
                    avoidance = random.choice(["", "", "travel", ""])
                    interest = random.choice(["", "", "", "", ""])
                    children_ages_choice = random.choice(["3-6", "7-12", "13-18"])
                    budget_type = 'budget'
                    group_size = '2'

                # Provide a rich set of fallback formatting keys so templates
                # with optional placeholders won't raise KeyError.
                fmt = {
                    'region': region,
                    'city': region,
                    'days': days,
                    'cities': cities,
                    'avoidance': avoidance,
                    'interest': interest,
                    'budget': budget,
                    'children_ages': children_ages_choice,
                    'constraint1': '',
                    'constraint2': '',
                    'budget_type': budget_type,
                    'group_size': group_size,
                    'exact_dates': ''
                }
                try:
                    query = scenario["template"].format(**fmt)
                except Exception:
                    # last resort: interpolate only known keys
                    query = scenario["template"].format(city=region, days=days, cities=cities, avoidance=avoidance, interest=interest, budget=budget)
            
            if lang == 'en':
                desc = f"Complex planning: {scenario['constraints'][0]}"
            else:
                desc = f"complexplanning: {scenario['constraints'][0]}"

            cf = _normalize_complexity_factors(scenario["constraints"] + ["multi_objective_optimization"])
            if not cf:
                cf = _default_complexity_for_level('T3')

            task = GeneratedTask(
                task_id=f"T3_{i+1:03d}",
                level="T3",
                query=query,
                expected_subagents=scenario["agents"],
                expected_clarifications=scenario["clarifications"],
                user_side_milestones=([f"satisfies constraint: {constraint}" for constraint in scenario["constraints"]] + ["reasonable_planning"]) if lang == 'en' else ([f"constraint: {constraint}" for constraint in scenario["constraints"]] + ["reasonable_planning"]),
                system_side_milestones=[],
                complexity_factors=cf,
                description=desc
            )
            tasks.append(task)
        
        return tasks
    

class LLMTaskGenerator:
    """LLM-based intelligent task generator"""
    
    def __init__(self, llm_config: Dict = None):
        # 
        self._load_environment_config()
        
        # configuration:  < configuration
        self.llm_config = self._build_llm_config(llm_config or {})

        custom_log = self.llm_config.get('fallback_log_path')
        if custom_log:
            log_path = Path(custom_log)
            if not log_path.is_absolute():
                log_path = PROJECT_ROOT / log_path
        else:
            log_path = FALLBACK_QUERY_LOG
        self.fallback_log_path = log_path
        
        # initialize
        self.generation_prompts = self._initialize_prompts()

        #  LLMClient 
        self.llm_client = None
        if _LLM_CLIENT_AVAILABLE:
            try:
                if self.llm_config:
                    self.llm_client = LLMClient(self.llm_config)
                else:
                    self.llm_client = get_default_client()
            except Exception as e:
                print(f"LLMClient initialization failed: {e}")
                self.llm_client = None

        # 
        self._log_config()
    
    def _load_environment_config(self):
        """configuration"""
        # .envfile
        dotenv_path = os.environ.get('DOTENV_PATH')
        _load_dotenv_if_exists(dotenv_path)
        
        # , .envfile
        if not dotenv_path:
            _load_dotenv_if_exists()
    
    def _build_llm_config(self, user_config: Dict) -> Dict:
        """LLMconfiguration"""
        # configuration
        env_config = {
            'endpoint': os.environ.get('MODEL_ENDPOINT'),
            'token': os.environ.get('MODEL_BEARER_TOKEN') or os.environ.get('MODEL_TOKEN'),
            'api_key': os.environ.get('OPENAI_API_KEY') or os.environ.get('API_KEY'),
            'model': os.environ.get('MODEL_NAME') or os.environ.get('MODEL'),
            'deployment': os.environ.get('MODEL_DEPLOYMENT'),
            'api_version': os.environ.get('MODEL_API_VERSION'),
            'language': os.environ.get('MODEL_LANG') or os.environ.get('MODEL_LANGUAGE'),
            'allow_insecure': self._parse_bool(os.environ.get('MODEL_ALLOW_INSECURE')),
        }
        
        # configuration
        try:
            if env_temperature := os.environ.get('MODEL_TEMPERATURE'):
                env_config['temperature'] = float(env_temperature)
            if env_max_tokens := os.environ.get('MODEL_MAX_TOKENS'):
                env_config['max_tokens'] = int(env_max_tokens)
        except (TypeError, ValueError):
            pass
        
        # configuration:  < userconfiguration
        merged_config = {}
        
        # configuration(None)
        for key, value in env_config.items():
            if value is not None:
                merged_config[key] = value
        
        # userconfiguration(coverage)
        merged_config.update(user_config)
        
        # 
        defaults = {
            'model': 'gpt-4.1',
            'temperature': 0.7,
            'max_tokens': 4096,
            'language': 'cn',
            'allow_insecure': False,
        }
        
        for key, default_value in defaults.items():
            if key not in merged_config:
                merged_config[key] = default_value
        
        return merged_config
    
    def _parse_bool(self, value: Optional[str]) -> bool:
        """"""
        if not value:
            return False
        return value.lower() in ('1', 'true', 'yes', 'on')
    
    def _log_config(self):
        """configuration()"""
        if not os.environ.get('MODEL_DEBUG') and not self.llm_config.get('debug'):
            return
            
        safe_config = {}
        for key, value in self.llm_config.items():
            if 'key' in key.lower() or 'token' in key.lower() or 'secret' in key.lower():
                safe_config[key] = _mask_secret(value)
            else:
                safe_config[key] = value
        
        print(f"[LLM Config] configuration: {safe_config}")

    def _record_fallback_query(self, query: str, reason: Optional[str] = None) -> None:
        """generate, . """
        if not query:
            return

        try:
            log_path = Path(self.fallback_log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.utcnow().isoformat()
            clean_reason = (reason or 'heuristic_fallback').replace('\n', ' ').strip()
            entry = f"[{timestamp}] {clean_reason}\t{query}\n"
            with log_path.open('a', encoding='utf-8') as handle:
                handle.write(entry)
        except Exception as exc:
            # best-effort logging; avoid breaking generation flow
            print(f"failed: {exc}")
    
    def _initialize_prompts(self) -> Dict[str, str]:
        """initializegenerate"""
        # provide prompts for different language modes. Structure: {lang: {T1:..., T2:..., T3:..., T4:...}}
        lang = self.llm_config.get('language', 'cn')

        prompts_cn = {
            "T1": """generatetravelsimpletask(T1level). : 
1. 
2. 
3. clarification
4. 

JSON: 
{
    "query": "language",
    "expected_subagents": [""],
    "user_side_milestones": "user",
    "system_side_milestones": "system"
}""",

            "T2": """generatetraveltask(T2level). : 
1. 2-3task
2. 
3. simpledateclarification
4. 

JSON: 
{
    "query": "language", 
    "expected_subagents": ["1", "2"],
    "expected_clarifications": ["clarification"],
    "user_side_milestones": ["user1", "user2"],
    "system_side_milestones": ["system1", "system2"],
    "complexity_factors": ["complex"]
}""",

            "T3": """generatetravelcomplexplanningtask(T3level). : 
1. constraint(time, budget, )
2. 
3. clarification
4. 

JSON: 
{
    "query": "language",
    "expected_subagents": [""],
    "expected_clarifications": ["clarification"], 
    "user_side_milestones": ["user1", "user2"],
    "system_side_milestones": ["system1", "system2"],
    "complexity_factors": ["complex"],
    "description": "task"
}""",

            "T4": """generatetraveltask(T4level). : 
1. 
2. clarification
3. 
4. 

JSON: 
{
    "query": "language",
    "required_clarifications": ["clarification"],
    "expected_subagents": [""],
    "user_side_milestones": ["user1", "user2"],
    "system_side_milestones": ["system1", "system2"],
    "complexity_factors": ["complex"],
    "description": "task"
}"""
        }

        prompts_en = {
            "T1": """Generate a simple travel query (T1).
Requirements:
1. Single clear intent
2. Only one subagent needed
3. No clarification required
4. Natural concise query

Return JSON:
{
    "query": "natural language query",
    "expected_subagents": ["agent_name"],
    "user_side_milestones": "user milestone",
    "system_side_milestones": "system milestone"
}""",
            "T2": """Generate a sequential travel task (T2).
Requirements:
1. Contains 2-3 dependent subtasks
2. Needs ordered execution
3. May require simple date clarification
4. Shows data passing

Return JSON:
{
    "query": "natural language query",
    "expected_subagents": ["agent1","agent2"],
    "expected_clarifications": ["clarify"],
    "user_side_milestones": ["r1","r2"],
    "system_side_milestones": ["r1","r2"],
    "complexity_factors": ["factors"]
}""",
            "T3": """Generate a complex planning task (T3). Requirements:
1. Multiple constraints (time, budget, preferences)
2. Needs trade-offs and optimization
3. Requires multi-turn clarifications
4. Involves coordination of multiple subagents

Return JSON:
{
    "query": "natural language query",
    "expected_subagents": ["agent_list"],
    "expected_clarifications": ["clarify_list"],
    "user_side_milestones": ["r1"],
    "system_side_milestones": ["r1"],
    "complexity_factors": ["factors"],
    "description": "description"
}""",
            "T4": """Generate an ambiguous/open-ended travel need (T4).
Requirements:
1. Severely missing information
2. Needs proactive multi-turn clarification
3. Requires requirement elicitation
4. Highly open-ended

Return JSON:
{
    "query": "very vague natural language query",
    "required_clarifications": ["must clarify"],
    "expected_subagents": ["agent_list"],
    "user_side_milestones": ["user_side_milestone1", "user_side_milestone2"],
    "system_side_milestones": ["system_side_milestone1", "system_side_milestone2"],
    "complexity_factors": ["factors"],
    "description": "description"
}"""
    }

        prompts_mixed = {
            "T1": """Generate a short Chinese query for travel but include a few English words (e.g., 'flight', 'hotel'). Requirements: single clear intent, one agent, no clarification. Return JSON with 'query' in Chinese mixing some English words.""",
            "T2": """Generate a sequential travel task using primarily Chinese, but you may mix in English terms like 'itinerary' or 'check-in'. Return a JSON object as specified.""",
            "T3": """Generate a complex planning task in Chinese with occasional English words (budget, flight, hotel). Must include multiple constraints and clarifications. Return JSON.""",
            "T4": """Generate a vague/open-ended Chinese task that requires proactive clarification; mix in a couple of English tokens if natural. Return JSON."""
        }

        return {
            'cn': prompts_cn,
            'en': prompts_en,
            'mixed': prompts_mixed
        }
    
    def generate_with_llm(self, level: str, count: int) -> List[GeneratedTask]:
        """Use LLM to generate task for specified level"""
        tasks = []
        # choose language for prompts
        lang = self.llm_config.get('language', 'cn')
        prompts_for_lang = self.generation_prompts.get(lang, self.generation_prompts.get('cn'))
        prompt = prompts_for_lang.get(level, prompts_for_lang.get('T1'))

        # Lightweight inject agent capabilities into prompt
        try:
            agent_context = _compose_agent_prompt_context(lang=lang)
            if agent_context:
                prompt = agent_context + "\n\n" + prompt
        except Exception:
            pass

        for i in range(count):
            try:
                #  LLMClient 
                if not self.llm_client:
                    raise RuntimeError("LLM client not available")
                
                response = self.llm_client.call_llm(prompt)
                task_data = self.llm_client.parse_llm_response(response)
                
                # normalize and sanitize keys returned by LLM (handle common aliases
                # and single-value shortcuts). This ensures LLM-generated tasks have
                # the same shape as rule-generated tasks.
                # expected_subagents (aliases: expected_subagent, expected_agent)
                eagents = task_data.get("expected_subagents") or task_data.get("expected_subagent") or task_data.get("expected_agent") or []
                if isinstance(eagents, str):
                    eagents = [eagents]
                try:
                    eagents = _sanitize_expected_subagents(eagents)
                except Exception:
                    pass

                # expected_clarifications (aliases: required_clarifications, required_clarification)
                expected_clarifications = task_data.get("expected_clarifications") or task_data.get("required_clarifications") or task_data.get("required_clarification") or []
                if isinstance(expected_clarifications, str):
                    expected_clarifications = [expected_clarifications]

                # user_side_milestones (alias: validation_rule)
                user_side_milestones = task_data.get("user_side_milestones") or []
                if isinstance(user_side_milestones, str):
                    user_side_milestones = [user_side_milestones]

                # complexity_factors (alias: complexity_factor) + normalization
                raw_cf = task_data.get("complexity_factors") or task_data.get("complexity_factor") or []
                if isinstance(raw_cf, str):
                    raw_cf = [raw_cf]
                try:
                    cf = _normalize_complexity_factors(raw_cf)
                except Exception:
                    cf = []
                if not cf:
                    cf = _default_complexity_for_level(level)

                # description
                description = task_data.get("description") or task_data.get("desc") or ""

                # query: prefer explicit 'query' field; if absent, fallback to prompt echo
                qtext = task_data.get("query") or task_data.get("text") or None
                if not qtext:
                    # fall back to using the original prompt-derived example if query missing
                    qtext = None

                task = GeneratedTask(
                    task_id=f"{level}_{i+1:03d}",
                    level=level,
                    query=qtext if qtext is not None else task_data.get("query", ""),
                    expected_subagents=eagents,
                    expected_clarifications=expected_clarifications,
                    user_side_milestones=user_side_milestones,
                    system_side_milestones=task_data.get("system_side_milestones") or [],
                    complexity_factors=cf,
                    description=description
                )
                tasks.append(task)
                
            except Exception as e:
                print(f"LLMgeneratetaskfailed: {e}")
                # generate
                fallback_task = self._generate_fallback_task(level, i)
                tasks.append(fallback_task)
        
        return tasks

    def generate_from_query(self, query: str, level_hint: str = None, use_llm: Optional[bool] = None, lang="en") -> GeneratedTask:
        """ query  task . 
 
        configuration LLM( _call_llm)generate JSON : 
        {
            "level": "T1|T2|T3|T4",
            "expected_subagents": [..],
            "expected_clarifications": [..],
            "validation_rules": [..],
            "complexity_factors": [..],
            "description": "..."
        }
 
         LLM configurationfailed, simple. 
         GeneratedTask (task_id  uuid generate). 
        """
        # Inject agent capability context for better agent selection by LLM
        try:
            agent_context = _compose_agent_prompt_context(lang=lang)
        except Exception:
            agent_context = ""
 
        # build a helpful prompt for LLM if available
 
        prompt_cn = """
        taskinputuserquerygeneratetask JSON, : level (T1|T2|T3|T4), expected_subagents (), expected_clarifications (), user_side_milestones (), system_side_milestones (),  complexity_factors (), description (). 
       
        # taskgenerate: 
        1. level, querylevel, levelgenerateJSON. 
            level: 
            T1: simple, , agent, clarification, . 
            T2: task, 2-3task, , simpledateclarification, . 
            T3: complexplanningtask, constraint(time, budget, ), , clarification, . 
            T4: task, , clarification, , . 
        2. : 
            expected_subagents: taskagent
            expected_clarifications: userclarification, userflightdate, clarificationtravel_date
            user_side_milestones: user, task. (15), user. , taskflight, "userXXXXflight"; taskrestaurantrecommendation, "userrecommendationrestaurant". task. 
            system_side_milestones: systemtask, . (15), system. , flight_agentflight"flight_agentflight", restaurant_agentrestaurant"restaurant_agentrecommendationrestaurant". system. 
            complexity_factors: taskcomplex, : 'single_intent', 'direct_execution', 'may_require_date_clarification', 'sequential_execution', 'data_dependency', \
                'multi_objective_optimization', 'requires_clarification', 'open_ended', \
                'requires_many_clarifications', 'multi_agent_coordination', 'family_member_constraints', 'dependency_handling', 'hierarchical_execution'. taskcomplex
            description: task, : <task: flight, hotel>
       
        # 
        generate user_side_milestones  system_side_milestones , user query , , (), . 
       
        # Agent
        {agent_context}
       
        user: {query}
        """.format(query=query, agent_context=agent_context)
 
        prompt_en = """
        Your task is to generate a task JSON for the given user query, including fields: level (T1|T2|T3|T4), expected_subagents (list), expected_clarifications (list), user_side_milestones (list), system_side_milestones (list), complexity_factors (list), description (string).       
        
        # Task Generation Guidelines:
        1. You need to first determine which level the current query belongs to based on the level descriptions defined below, and then generate the corresponding JSON fields according to the requirements of that level.
            Level Descriptions:
            T1: Simple query, single clear intent, only one agent needed, no clarification required, natural concise query.
            T2: Sequential task, contains 2-3 dependent subtasks, needs ordered execution, may require simple date clarification, shows data passing.
            T3: Complex planning task, contains multiple constraints (time, budget, preferences), needs trade-offs and optimization, requires multi-turn clarifications, involves coordination of multiple agents.
            T4: Ambiguous/open-ended need, severely missing information, needs proactive multi-turn clarification, requires requirement elicitation, highly open-ended.
        2. Definitions of other fields:
            expected_subagents: List of agents that may be needed to complete the task.
            expected_clarifications: List of information that may need further clarification from the user, e.g., if the user wants to search for flights but does not provide departure date, then travel_date needs to be clarified.The output here should be key phrases with underscores(like travel_date)
            user_side_milestones: Key events visible to the user in the response, reflecting the end-to-end completion of the task. Each milestone should be concise (no more than 15 characters/words) and represent a user-perceivable action or result. For example, for a flight query, "User notified of flights from XX to XX"; for restaurant recommendations, "User receives restaurant list". This field is intended to measure task completion from the user's perspective.
            system_side_milestones: Key system actions triggered to complete the task, used to evaluate the process execution quality. Each milestone should be concise (no more than 15 characters/words) and reflect concrete system steps. For example, calling flight_agent can be written as "Called flight_agent to query flights"; calling restaurant_agent as "Called restaurant_agent to recommend restaurants". This field is intended to measure the system's execution and orchestration quality from a process perspective.
            complexity_factors: List of complexity factors that the task belongs to. Optional factors include: 'single_intent', 'direct_execution', 'may_require_date_clarification', 'sequential_execution', 'data_dependency', \
                'multi_objective_optimization', 'requires_clarification', 'open_ended', \
                'requires_many_clarifications', 'multi_agent_coordination', 'family_member_constraints', 'dependency_handling', 'hierarchical_execution'. You can flexibly choose and combine complexity factors based on the actual task.
            description: A brief summary description of the task, e.g.: <Sequential Task: Execute flight and hotel search in order>.
       
        # Notes
        When generating user_side_milestones and system_side_milestones, strictly base them on objective information from the user query. Retain all necessary key details, avoid fabricating information (reasonable inferences are allowed), and ensure no important information is omitted.
       
        # Existing Agent Set and Their Capabilities
        {agent_context}
       
        # User Query: {query}
        """.format(query=query, agent_context=agent_context)

        prompt = prompt_cn if lang == 'cn' else prompt_en
        print(prompt)

        parsed: Dict[str, Any] = {}
        fallback_reason: Optional[str] = None
        heuristic_used = False

        # Try LLM first when configuration exists
        if use_llm is None:
            has_llm = bool(self.llm_client and (self.llm_config.get('endpoint') or self.llm_config.get('api_key')))
        else:
            has_llm = bool(use_llm and self.llm_client)

        llm_attempted = False
        llm_success = False
        last_error: Optional[Exception] = None

        if has_llm:
            llm_attempted = True
            for attempt in range(2):
                try:
                    response = self.llm_client.call_llm(prompt)
                    print(f"LLM: {response}")
                    candidate = self.llm_client.parse_llm_response(response) or {}
                    if candidate:
                        parsed = candidate
                        llm_success = True
                        break
                    fallback_reason = 'empty_llm_response'
                    print("LLM, . ")
                except Exception as exc:
                    last_error = exc
                    fallback_reason = f"exception: {exc}"
                    print(f"LLMfailed({attempt + 1}): {exc}")
                if attempt == 0:
                    time.sleep(3)

            if not llm_success and last_error is not None:
                fallback_reason = f"exception: {last_error}"

        # If parsed lacks useful keys, use heuristic fallback
        if not parsed or (not parsed.get('expected_subagents') and not parsed.get('level')):
            heuristic_used = True
            if has_llm:
                if llm_success and parsed:
                    fallback_reason = fallback_reason or 'missing_expected_fields'
                else:
                    fallback_reason = fallback_reason or 'llm_failure'
            q = query.lower()
            agents = []
            clarifs = []
            u_milestone = []
            s_milestone = []
            complexity = []

            # simple keyword -> agent mapping
            if any(k in q for k in ['flight', '', '']):
                agents.append('search_flights')
            if any(k in q for k in ['hotel', '', '']):
                agents.append('search_hotels')
            if any(k in q for k in ['weather', '', '']):
                agents.append('get_weather_forecast')
            if any(k in q for k in ['', '', '', '']):
                agents.append('search_attractions')
            if any(k in q for k in ['', 'planning', '', 'itinerary']):
                agents.append('create_itinerary')
            if any(k in q for k in ['', '', '']):
                agents.append('search_restaurants')
            if any(k in q for k in ['time', 'date', '', '']):
                clarifs.append('travel_date')
            if any(k in q for k in ['budget', '', '']):
                clarifs.append('budget')
            if any(k in q for k in ['', '', '', '']):
                clarifs.append('party_size')

            # estimate level
            if any(k in q for k in ['planning', '', '', '', 'budget', '']):
                level = 'T3'
            elif any(k in q for k in ['', '', '', '', '']):
                level = 'T2'
            elif any(k in q for k in ['', '', '']):
                # ambiguous arrange -> could be T4 if too vague
                if len(clarifs) >= 2 or '' in q and any(k in q for k in ['city', 'budget', 'date']):
                    level = 'T3'
                else:
                    level = 'T2'
            else:
                level = 'T1'

            # validation rules and complexity
            if agents:
                for t in agents:
                    if lang == 'en':
                        u_milestone.append(f"contains {t.split('_')[-1]} information")
                    else:
                        u_milestone.append(f"{t.split('_')[-1]}")
            else:
                if lang == 'en':
                    u_milestone.append('actionable')
                else:
                    u_milestone.append('')


            # validate system-side milestones
            for t in agents:
                if lang == 'en':
                    s_milestone.append(f"invoke {t} agent")
                else:
                    s_milestone.append(f" {t} ")

            if level == 'T1':
                complexity = ['single_intent', 'direct_execution']
            elif level == 'T2':
                complexity = ['sequential_execution', 'data_dependency']
            elif level == 'T3':
                complexity = ['multi_objective_optimization', 'requires_clarification']
            else:
                complexity = ['open_ended', 'requires_many_clarifications']

            parsed = {
                'level': level,
                'expected_subagents': agents or [],
                'expected_clarifications': clarifs or [],
                'user_side_milestones': u_milestone or [],
                'system_side_milestones': s_milestone or [],  # Placeholder for system-side milestones
                'complexity_factors': complexity,
                'description': parsed.get('description') or (f"Automatically inferred task from query: {query}" if (self.llm_config.get('language','cn') == 'en') else f"task: {query}")
            }

        if heuristic_used and has_llm and llm_attempted:
            self._record_fallback_query(query, fallback_reason)

        # If user supplied level_hint, prefer it
        if level_hint:
            parsed['level'] = level_hint

        # normalize fields
        expected_subagents = parsed.get('expected_subagents') or parsed.get('expected_subagent') or parsed.get('expected_agent') or []
        if isinstance(expected_subagents, str):
            expected_subagents = [expected_subagents]

        # Ensure expected_subagents only references available travel agents
        try:
            expected_subagents = _sanitize_expected_subagents(expected_subagents)
        except Exception:
            pass

        expected_clarifications = parsed.get('expected_clarifications') or parsed.get('required_clarifications') or []
        user_side_milestones = parsed.get('user_side_milestones') or []
        system_side_milestones = parsed.get('system_side_milestones') or []
        complexity_factors = parsed.get('complexity_factors') or []
        # normalize complexity factors to canonical enum values
        try:
            complexity_factors = _normalize_complexity_factors(complexity_factors)
        except Exception:
            complexity_factors = complexity_factors or []
        level = parsed.get('level') or 'T1'
        # ensure sensible defaults per inferred level when normalization yields nothing
        try:
            if not complexity_factors:
                complexity_factors = _default_complexity_for_level(level)
        except Exception:
            complexity_factors = complexity_factors or []
        description = parsed.get('description') or ''

        task = GeneratedTask(
            task_id=f"Q_{uuid.uuid4().hex[:8]}",
            level=level,
            query=query,
            expected_subagents=expected_subagents,
            expected_clarifications=expected_clarifications,
            user_side_milestones=user_side_milestones,
            system_side_milestones=system_side_milestones,
            complexity_factors=complexity_factors,
            description=description
        )

        return task
    
    def _generate_fallback_task(self, level: str, index: int) -> GeneratedTask:
        """generatetask"""
        # Use UUID-based task ids for fallback tasks to avoid colliding with
        # template-generated sequential ids when LLM is enabled and a downgrade
        # occurs. This ensures uniqueness across mixed-generation runs.
        uid = uuid.uuid4().hex[:8]
        lang = self.llm_config.get('language', 'cn')

        if lang == 'en':
            fallback_tasks = {
                "T1": GeneratedTask(
                    task_id=f"{level}_FB_{uid}",
                    level="T1",
                    query="What's the weather in Beijing today?",
                    expected_subagents=["get_weather_forecast"],
                    expected_clarifications=[],
                    user_side_milestones=["contains weather information"],
                    system_side_milestones=["invoke get_weather_forecast agent"],
                    complexity_factors=_normalize_complexity_factors(["single_intent"]),
                    description="Simple weather query"
                ),
                "T2": GeneratedTask(
                    task_id=f"{level}_FB_{uid}",
                    level="T2",
                    query="Check the weather in Hangzhou, then recommend outdoor attractions",
                    expected_subagents=["get_weather_forecast", "search_attractions"],
                    expected_clarifications=[],
                    user_side_milestones=["execute steps in sequence"],
                    system_side_milestones=["invoke get_weather_forecast agent", "invoke search_attractions agent"],
                    complexity_factors=_normalize_complexity_factors(["sequential_execution"]),
                    description="Sequential query task"
                )
            }
        else:
            fallback_tasks = {
                "T1": GeneratedTask(
                    task_id=f"{level}_FB_{uid}",
                    level="T1",
                    query="weather",
                    expected_subagents=["get_weather_forecast"],
                    expected_clarifications=[],
                    user_side_milestones=["weather"],
                    system_side_milestones=[" get_weather_forecast "],   
                    complexity_factors=_normalize_complexity_factors(["single_intent"]),
                    description="simpleweather"
                ),
                "T2": GeneratedTask(
                    task_id=f"{level}_FB_{uid}",
                    level="T2", 
                    query="weather, recommendation",
                    expected_subagents=["get_weather_forecast", "search_attractions"],
                    expected_clarifications=[],
                    user_side_milestones=[""],
                    system_side_milestones=[" get_weather_forecast ", " search_attractions "],
                    complexity_factors=_normalize_complexity_factors(["sequential_execution"]),
                    description="task"
                )
            }

        return fallback_tasks.get(level, fallback_tasks["T1"])
    

class TravelTaskPipeline:
    """travelTaskgenerate"""
    
    def __init__(self, llm_config: Dict = None):
        self.llm_generator = LLMTaskGenerator(llm_config or {})
        self.rule_generator = TaskGenerationStrategy()
        
    def generate_dataset(self, config: Dict) -> Dict[str, List[GeneratedTask]]:
        """generate"""
        dataset = {}
        # determine language from llm_config if available; default to 'cn'
        lang = getattr(self, 'llm_generator', None) and self.llm_generator.llm_config.get('language', 'cn') or 'cn'
        
        for level, count in config.items():
            print(f"generate{level}leveltask, : {count}")
            
            if level == "T1":
                tasks = self.rule_generator.generate_t1_tasks(count, lang=lang)
            elif level == "T2":
                tasks = self.rule_generator.generate_t2_tasks(count, lang=lang)
            elif level == "T3":
                # T3LLMgeneratecomplextask
                tasks = self.llm_generator.generate_with_llm(level, count)
            elif level == "T4":
                # T4LLMgenerate
                tasks = self.llm_generator.generate_with_llm(level, count)
            else:
                continue
                
            dataset[level] = tasks
            print(f"successgenerate{len(tasks)}{level}task")
        
        # Sanitize expected_subagents across the generated dataset so that only
        # subagents defined in agents_cards/travel are referenced. This ensures
        # generated tasks don't include subagents that are not available in the
        # travel scenario agentset.
        for lvl, tasks in dataset.items():
            for t in tasks:
                try:
                    t.expected_subagents = _sanitize_expected_subagents(t.expected_subagents or [])
                except Exception:
                    # best-effort: leave as-is on error
                    pass

        # Enforce travel scenario constraints: only China destinations; dates within next 1 month
        for lvl, tasks in dataset.items():
            for t in tasks:
                try:
                    # determine language mode: check env override or infer from characters
                    lang = os.environ.get('TRAVEL_AGENT_LANG', 'cn')
                    # 1) replace foreign city mentions or ensure at least one China city
                    try:
                        t.query = _replace_foreign_city_with_cn(t.query or '', lang=lang)
                    except Exception:
                        pass

                    # 2) normalize any explicit dates to be within next 30 days
                    try:
                        t.query = _enforce_dates_within_next_month(t.query or '')
                    except Exception:
                        pass
                except Exception:
                    continue

        return dataset
    
    def export_dataset(self, dataset: Dict[str, List[GeneratedTask]], format: str = "yaml", include_metadata: bool = False, metadata: Dict = None) -> str:
        """

        include_metadata:  True,  metadata, 
         user_profiles  evaluation_features, maturityevaluate. 
        metadata: ; , . 
        """
        if include_metadata and metadata is None:
            metadata = {
                'user_profiles': ['family', 'business', 'backpacker', 'luxury', 'cultural', 'adventure'],
                'evaluation_features': {
                    'basic_metrics': True,
                    'multi_dimensional': True,
                    'comparative_analysis': False,
                    'predictive_metrics': False
                }
            }

        if format == "yaml":
            return self._export_yaml(dataset, metadata=metadata if include_metadata else None)
        elif format == "json":
            return self._export_json(dataset, metadata=metadata if include_metadata else None)
        else:
            return self._export_yaml(dataset, metadata=metadata if include_metadata else None)
    
    def _export_yaml(self, dataset: Dict[str, List[GeneratedTask]], metadata: Dict = None) -> str:
        """YAML"""
        yaml_content = ""
        # optionally emit metadata first to help the evaluator detect user profiles and evaluation setup
        if metadata:
            yaml_content += "metadata:\n"
            up = metadata.get('user_profiles', [])
            yaml_content += f"  user_profiles: {json.dumps(up, ensure_ascii=False)}\n"
            ef = metadata.get('evaluation_features', {})
            yaml_content += f"  evaluation_features: {json.dumps(ef, ensure_ascii=False)}\n\n"
        yaml_content += "tasks:\n"

        def _yaml_safe_block(s: str) -> str:
            """Return a YAML-safe block scalar representation for s.

            - If s contains newlines or is longer than 120 chars, use a literal block (|) with indentation.
            - Otherwise, emit a quoted single-line string with internal quotes escaped.
            """
            if s is None:
                return '""'
            # normalize to str
            s = str(s)
            # if contains newline, use block scalar; single-line values should be quoted
            if '\n' in s:
                # ensure trailing newline
                if not s.endswith('\n'):
                    s += '\n'
                # build block with correct indentation for YAML
                block = "|\n"
                for line in s.splitlines(keepends=True):
                    block += f"      {line}"
                return block
            # else safe single-line quoted
            esc = s.replace('"', '\\"')
            return f'"{esc}"'

        for level, tasks in dataset.items():
            for task in tasks:
                yaml_content += f"  {task.task_id}:\n"
                yaml_content += f"    level: \"{task.level}\"\n"
                yaml_content += f"    query: { _yaml_safe_block(task.query) }\n"
                yaml_content += f"    expected_subagents: {json.dumps(task.expected_subagents, ensure_ascii=False)}\n"
                yaml_content += f"    expected_clarifications: {json.dumps(task.expected_clarifications, ensure_ascii=False)}\n"
                yaml_content += f"    user_side_milestones: {json.dumps(task.user_side_milestones, ensure_ascii=False)}\n"
                yaml_content += f"    system_side_milestones: {json.dumps(task.system_side_milestones, ensure_ascii=False)}\n"
                yaml_content += f"    complexity_factors: {json.dumps(task.complexity_factors, ensure_ascii=False)}\n"
                yaml_content += f"    description: { _yaml_safe_block(task.description) }\n"

        return yaml_content
    
    def _export_json(self, dataset: Dict[str, List[GeneratedTask]], metadata: Dict = None) -> str:
        """JSON,  metadata"""
        json_data = {}
        if metadata:
            json_data['metadata'] = metadata
        json_data["tasks"] = {}
        for level, tasks in dataset.items():
            for task in tasks:
                json_data["tasks"][task.task_id] = {
                    "level": task.level,
                    "query": task.query,
                    "expected_subagents": task.expected_subagents,
                    "expected_clarifications": task.expected_clarifications,
                    "user_side_milestones": task.user_side_milestones,
                    "system_side_milestones": task.system_side_milestones,
                    "complexity_factors": task.complexity_factors,
                    "description": task.description
                }

        return json.dumps(json_data, ensure_ascii=False, indent=2)
    
    def analyze_dataset(self, dataset: Dict[str, List[GeneratedTask]]) -> Dict:
        """"""
        analysis = {
            "total_tasks": 0,
            "by_level": {},
            "agents_coverage": {},
            "complexity_distribution": {}
        }
        
        all_agents = set()
        all_complexities = set()
        
        for level, tasks in dataset.items():
            analysis["by_level"][level] = len(tasks)
            analysis["total_tasks"] += len(tasks)
            
            for task in tasks:
                # coverage
                for agent in task.expected_subagents:
                    all_agents.add(agent)
                    analysis["agents_coverage"][agent] = analysis["agents_coverage"].get(agent, 0) + 1
                
                # complex
                for factor in task.complexity_factors:
                    all_complexities.add(factor)
                    analysis["complexity_distribution"][factor] = analysis["complexity_distribution"].get(factor, 0) + 1
        
        return analysis
    

def _parse_ratio_string(ratio_str: str) -> Dict[str, float]:
    """Parse a ratio string like 'T1:0.6,T2:0.3,T3:0.1' into a dict."""
    out = {}
    if not ratio_str:
        return out
    for part in ratio_str.split(','):
        if ':' not in part:
            continue
        k, v = part.split(':', 1)
        try:
            out[k.strip().upper()] = float(v)
        except Exception:
            continue
    return out


# ------------------ Travel constraints helpers ------------------
CHINA_CITIES_EN = ["Beijing", "Shanghai", "Guangzhou", "Shenzhen", "Hangzhou", "Chengdu", "Xi'an", "Nanjing", "Chongqing", "Tianjin", "Qingdao", "Xiamen"]
CHINA_CITIES_CN = ["", "", "", "", "", "", "", "", "", "", "", ""]

# Common foreign city names to detect and replace when enforcing China-only constraint
_COMMON_FOREIGN_CITIES = [
    'Paris', 'London', 'New York', 'Los Angeles', 'Tokyo', 'Osaka', 'Seoul', 'Sydney', 'Singapore',
    'Bangkok', 'Berlin', 'Rome', 'Moscow', 'Toronto', 'Vancouver', 'Dubai', 'Barcelona', 'Amsterdam'
]


def _replace_foreign_city_with_cn(query: str, lang: str = 'cn') -> str:
    """If query contains a known foreign city, replace it with a random Chinese city.

    This enforces the policy: only China destinations are supported. If no known foreign
    city is detected but the query does not mention any Chinese city, we append a default
    Chinese destination to the query to make the task China-scoped.
    """
    if not query:
        return query

    q = str(query)
    replaced = False
    # detect explicit foreign city names (case-insensitive)
    for city in _COMMON_FOREIGN_CITIES:
        if re.search(r"\b" + re.escape(city) + r"\b", q, flags=re.IGNORECASE):
            new_city = random.choice(CHINA_CITIES_EN if lang == 'en' else CHINA_CITIES_CN)
            q = re.sub(r"\b" + re.escape(city) + r"\b", new_city, q, flags=re.IGNORECASE)
            replaced = True

    # If no foreign city replaced, ensure at least one China city is present
    all_cn = CHINA_CITIES_EN if lang == 'en' else CHINA_CITIES_CN
    if not any(re.search(re.escape(c), q, flags=re.IGNORECASE) for c in all_cn):
        # append a short clause restricting destination to a China city
        append_city = random.choice(all_cn)
        if lang == 'en':
            q = q.rstrip('?. .!') + f" in {append_city}?"
        else:
            # add a short clarifying phrase in Chinese
            if not q.endswith('. ') and not q.endswith('?') and not q.endswith('!'):
                q = q + f', : {append_city}'
            else:
                q = q + f' : {append_city}'

    return q


def _enforce_dates_within_next_month(query: str) -> str:
    """Normalize/adjust explicit dates in query to ensure they fall within next 30 days.

    - Detect YYYY-MM-DD and Chinese patterns like 'XY' and adjust to an in-range date.
    - If no explicit date but contains 'next month' style, rewrite to a concrete date within 30 days.
    """
    if not query:
        return query
    q = str(query)
    today = datetime.utcnow().date()
    max_date = today + timedelta(days=30)

    # YYYY-MM-DD pattern
    def _fix_iso(match):
        try:
            s = match.group(0)
            dt = datetime.strptime(s, '%Y-%m-%d').date()
            if dt < today or dt > max_date:
                # pick a replacement within next 30 days
                new_dt = today + timedelta(days=random.randint(1, 30))
                return new_dt.isoformat()
            return s
        except Exception:
            return match.group(0)

    q = re.sub(r'\d{4}-\d{2}-\d{2}', _fix_iso, q)

    # Chinese date pattern like 1224 or 35
    def _fix_cn_date(match):
        try:
            mon = int(match.group(1))
            day = int(match.group(2))
            # assume current year; adjust year if necessary
            year = today.year
            try_dt = datetime(year, mon, day).date()
            # if date in past relative to today, bump to next year
            if try_dt < today:
                try_dt = datetime(year + 1, mon, day).date()
            if try_dt < today or try_dt > max_date:
                new_dt = today + timedelta(days=random.randint(1, 30))
                return f"{new_dt.month}{new_dt.day}"
            return f"{try_dt.month}{try_dt.day}"
        except Exception:
            return match.group(0)

    q = re.sub(r'(\d{1,2})(\d{1,2})[]?', _fix_cn_date, q)

    # phrases like 'next month', '' -> replace with a concrete date within next 30 days
    if re.search(r'next month|next month\b', q, flags=re.IGNORECASE) or '' in q:
        new_dt = today + timedelta(days=random.randint(1, 30))
        if re.search(r'next month|next month\b', q, flags=re.IGNORECASE):
            q = re.sub(r'next month\b', new_dt.isoformat(), q, flags=re.IGNORECASE)
        q = q.replace('', f"{new_dt.month}{new_dt.day}")

    return q

# ------------------ end helpers ------------------


def main_cli(argv=None):
    parser = argparse.ArgumentParser(description="Travel task generator CLI")
    parser.add_argument('--mode', choices=['zero', 'from-file'], default='zero', help='Generation mode: zero (from scratch) or from-file (read queries)')
    parser.add_argument('--queries-file', '-q', type=str, help='Path to a newline-delimited queries file (used with --mode from-file)')
    parser.add_argument('--use-llm', action='store_true', help='Allow using configured LLM for generation (T3/T4). Default: disabled')
    parser.add_argument('--t1', type=int, default=20, help='Number of T1 tasks to generate (zero mode)')
    parser.add_argument('--t2', type=int, default=10, help='Number of T2 tasks to generate (zero mode)')
    parser.add_argument('--t3', type=int, default=5, help='Number of T3 tasks to generate (zero mode)')
    parser.add_argument('--t4', type=int, default=5, help='Number of T4 tasks to generate (zero mode)')
    parser.add_argument('--total', type=int, help='Total number of tasks to generate (used with --ratio)')
    parser.add_argument('--ratio', type=str, help='Comma-separated level ratios, e.g. T1:0.6,T2:0.3,T3:0.1')
    parser.add_argument('--model', type=str, help='LLM model name to use (overrides env MODEL_NAME). Examples: gpt_oss_120b, llama3.2-90B')
    parser.add_argument('--lang', '--language', dest='lang', choices=['cn', 'en', 'mixed'], default='en', help='Language mode for generated tasks: cn (Chinese, default), en (English), mixed (Chinese with English tokens)')
    parser.add_argument('--out', '-o', type=str, default='generated_travel_tasks.yaml', help='Output file path')
    parser.add_argument('--format', choices=['yaml', 'json'], default='yaml', help='Output format')
    parser.add_argument('--level-hint', type=str, help='Optional level hint when generating from queries')
    parser.add_argument('--include-metadata', action='store_true', help='Include evaluation metadata in exported file to help maturity evaluator')

    args = parser.parse_args(argv)
    os.environ["TRAVEL_AGENT_LANG"] = args.lang

    # pass selected model (if any) into the pipeline's llm_config
    llm_config = {}
    if args.model:
        llm_config['model'] = args.model
    # include language selection in llm_config so LLM prompts follow requested language
    llm_config['language'] = args.lang
    pipeline = TravelTaskPipeline(llm_config)
    rule_gen = TaskGenerationStrategy()
    llm_gen = pipeline.llm_generator

    dataset: Dict[str, List[GeneratedTask]] = {}

    if args.mode == 'zero':
        # compute counts from ratio/total if provided
        counts = {'T1': args.t1, 'T2': args.t2, 'T3': args.t3, 'T4': args.t4}
        if args.ratio and args.total:
            ratios = _parse_ratio_string(args.ratio)
            # normalize and compute integer counts
            total = args.total
            remaining = total
            computed = {}
            # assign floor first
            for level, r in ratios.items():
                c = int(total * r)
                computed[level] = c
                remaining -= c
            # distribute remaining to levels in descending ratio order
            for level, _ in sorted(ratios.items(), key=lambda x: x[1], reverse=True):
                if remaining <= 0:
                    break
                computed[level] = computed.get(level, 0) + 1
                remaining -= 1
            # fill counts with computed (fall back to defaults if not specified)
            for k in ['T1', 'T2', 'T3', 'T4']:
                counts[k] = computed.get(k, counts[k])

        print(f"generate: {counts}")

        # generate per level
        if counts.get('T1', 0) > 0:
            dataset['T1'] = rule_gen.generate_t1_tasks(counts['T1'], lang=args.lang)
        if counts.get('T2', 0) > 0:
            dataset['T2'] = rule_gen.generate_t2_tasks(counts['T2'], lang=args.lang)
        if counts.get('T3', 0) > 0:
            if args.use_llm:
                dataset['T3'] = llm_gen.generate_with_llm('T3', counts['T3'])
            else:
                dataset['T3'] = rule_gen.generate_t3_tasks(counts['T3'], lang=args.lang)
        if counts.get('T4', 0) > 0:
            if args.use_llm:
                dataset['T4'] = llm_gen.generate_with_llm('T4', counts['T4'])
            else:
                # fallback to rule-based generation when LLM disabled.
                # Use the existing T3 rule generator but relabel tasks to T4
                # so they are correctly identified as T4 in exported datasets.
                tasks = rule_gen.generate_t3_tasks(counts['T4'], lang=args.lang)
                for t in tasks:
                    try:
                        # normalize level and task_id prefix
                        t.level = 'T4'
                        if isinstance(t.task_id, str) and t.task_id.startswith('T3_'):
                            t.task_id = t.task_id.replace('T3_', 'T4_', 1)
                        elif not isinstance(t.task_id, str) or not t.task_id.startswith('T4_'):
                            t.task_id = f"T4_{t.task_id}"
                    except Exception:
                        # best-effort: if anything goes wrong, ensure level is T4
                        t.level = 'T4'
                dataset['T4'] = tasks

    elif args.mode == 'from-file':
        if not args.queries_file:
            raise SystemExit(' --queries-file  --mode from-file')
        qpath = Path(args.queries_file)
        if not qpath.exists():
            raise SystemExit(f'queries file not found: {qpath}')

        dataset = {}
        with qpath.open('r', encoding='utf-8') as f:
            queries = [line.strip() for line in f if line.strip()]

        for q in queries:
            task = llm_gen.generate_from_query(q, level_hint=args.level_hint, use_llm=args.use_llm, lang=args.lang)
            lvl = task.level or 'T1'
            dataset.setdefault(lvl, []).append(task)

    # export
    exporter = pipeline
    # prepare optional metadata for the evaluator when requested
    out_metadata = None
    if args.include_metadata:
        out_metadata = {
            'user_profiles': ['family', 'business', 'backpacker', 'luxury', 'cultural', 'adventure'],
            'evaluation_features': {
                'basic_metrics': True,
                'multi_dimensional': True,
                'comparative_analysis': False,
                'predictive_metrics': False
            }
        }

    outp = Path(args.out)
    export_format = args.format

    content: str = ''

    try:
        if export_format == 'yaml':
            content = exporter.export_dataset(dataset, format='yaml', include_metadata=args.include_metadata, metadata=out_metadata)
        else:
            content = exporter.export_dataset(dataset, format='json', include_metadata=args.include_metadata, metadata=out_metadata)
    except Exception as exc:
        print(f"generate: {exc}")
        if dataset:
            try:
                fallback_content = exporter.export_dataset(dataset, format='yaml', include_metadata=args.include_metadata, metadata=out_metadata)
                outp.parent.mkdir(parents=True, exist_ok=True)
                outp.write_text(fallback_content, encoding='utf-8')
                print(f"generate {outp} (yaml)")
            except Exception as inner_exc:
                print(f"failed: {inner_exc}")
        raise

    outp.parent.mkdir(parents=True, exist_ok=True)

    try:
        outp.write_text(content, encoding='utf-8')
        print(f"generate {outp} ({export_format})")
    except Exception as exc:
        print(f"file {outp} failed: {exc}")
        if dataset:
            try:
                partial_path = outp.with_name(outp.name + '.partial')
                partial_path.parent.mkdir(parents=True, exist_ok=True)
                partial_path.write_text(content, encoding='utf-8')
                print(f"generate {partial_path} ({export_format})")
            except Exception as inner_exc:
                print(f"failed: {inner_exc}")
        raise


if __name__ == '__main__':
    main_cli()