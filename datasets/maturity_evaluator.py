from dataclasses import dataclass
from typing import Dict, List, Set, Any
from enum import Enum
import argparse
# optional heavy deps: make yaml optional so module can import without PyYAML installed
try:
    import yaml
    _YAML_AVAILABLE = True
except Exception:
    yaml = None
    _YAML_AVAILABLE = False
import os
import json
import re
import sys
from pathlib import Path

# Use unified LLM client - fix import issues
_LLM_CLIENT_AVAILABLE = False
_LLM_CLIENT_IMPORT_ERROR = None

try:
    # First try relative import (when used as module)
    from .llm_client import LLMClient, get_default_client
    _LLM_CLIENT_AVAILABLE = True
except ImportError:
    try:
        # If relative import fails, try absolute import (when running script directly)
        # Add parent directory to Python path
        current_dir = Path(__file__).parent
        parent_dir = current_dir.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))
        
        from datasets.llm_client import LLMClient, get_default_client
        _LLM_CLIENT_AVAILABLE = True
    except ImportError as e:
        _LLM_CLIENT_AVAILABLE = False
        _LLM_CLIENT_IMPORT_ERROR = str(e)
        # Print detailed error for debugging
        import traceback
        _LLM_CLIENT_IMPORT_ERROR = traceback.format_exc()
except Exception as e:
    _LLM_CLIENT_AVAILABLE = False
    _LLM_CLIENT_IMPORT_ERROR = str(e)
    import traceback
    _LLM_CLIENT_IMPORT_ERROR = traceback.format_exc()

# If LLM client is unavailable, provide clearer error message
if not _LLM_CLIENT_AVAILABLE:
    print("Warning: LLM client unavailable, LLM-related features will be disabled")
    if _LLM_CLIENT_IMPORT_ERROR:
        print(f"Import error details: {_LLM_CLIENT_IMPORT_ERROR}")

class MaturityLevel(Enum):
    """maturitylevel"""
    BRONZE = 1    # 
    SILVER = 2    #   
    GOLD = 3      # 
    PLATINUM = 4  # 

@dataclass
class MaturityDimension:
    """maturitydimension"""
    name: str
    description: str
    bronze_criteria: List[str]
    silver_criteria: List[str] 
    gold_criteria: List[str]
    platinum_criteria: List[str]
    weight: float = 1.0

@dataclass
class DatasetMaturityReport:
    """maturityreport"""
    overall_level: MaturityLevel
    dimension_scores: Dict[str, float]
    dimension_levels: Dict[str, MaturityLevel]
    recommendations: List[str]
    total_score: float
    max_score: float


class TravelDatasetMaturityModel:
    """travelscenariomaturity"""
    
    def __init__(self):
        self.dimensions = self._initialize_dimensions()
    
    def _initialize_dimensions(self) -> List[MaturityDimension]:
        """initialize4maturitydimension: Task, Content Diversity, Interaction Depth, Capability Coverage"""
        return [
            MaturityDimension(
                name="task",
                description="taskcomplexconstraint",
                bronze_criteria=["T1simpletask", "task"],
                silver_criteria=["coverageT1-T2task", "task"],
                gold_criteria=["coverageT1-T3task", "complexconstrainttask", ""],
                platinum_criteria=["coverageT1-T4", "", ""],
                weight=0.35
            ),

            MaturityDimension(
                name="content_diversity",
                description="scenario/language/userdiversity",
                bronze_criteria=["scenariolanguage", "user"],
                silver_criteria=["coveragescenario", "", "user"],
                gold_criteria=["language", "scenariocoverage", "complexuser"],
                platinum_criteria=["scenario/language", "coverage", "user"],
                weight=0.25
            ),

            MaturityDimension(
                name="interaction_depth",
                description="dialogueclarification",
                bronze_criteria=["dialogue", "simpleclarification"],
                silver_criteria=["dialogue(2-3)", ""],
                gold_criteria=["dialogue(4-6)", "", ""],
                platinum_criteria=["dialogue(7+)", "", "dialogue"],
                weight=0.20
            ),

            MaturityDimension(
                name="capability_coverage",
                description="/coverage",
                bronze_criteria=["coverage(<50%)", "simple"],
                silver_criteria=["coverage(50-80%)", ""],
                gold_criteria=["coverage(80-95%)", "complex", "error"],
                platinum_criteria=["coverage(95%+)", "", ""],
                weight=0.20
            ),
        ]
    

class MaturityAssessmentEngine:
    """maturityevaluate"""
    
    def __init__(self, model: TravelDatasetMaturityModel):
        self.model = model
    
    def assess_dataset(self, dataset: Dict, metadata: Dict = None) -> DatasetMaturityReport:
        """evaluatematurity"""
        dimension_scores = {}
        dimension_levels = {}
        
        # evaluatedimension
        for dimension in self.model.dimensions:
            score = self._assess_dimension(dimension, dataset, metadata)
            level = self._score_to_level(score)
            
            dimension_scores[dimension.name] = score
            dimension_levels[dimension.name] = level
        
        # maturity
        total_score = self._calculate_total_score(dimension_scores)
        overall_level = self._score_to_level(total_score)
        
        # generaterecommendation
        recommendations = self._generate_recommendations(dimension_scores, dimension_levels)
        
        return DatasetMaturityReport(
            overall_level=overall_level,
            dimension_scores=dimension_scores,
            dimension_levels=dimension_levels,
            recommendations=recommendations,
            total_score=total_score,
            max_score=100.0
        )
    
    def _assess_dimension(self, dimension: MaturityDimension, dataset: Dict, metadata: Dict) -> float:
        """evaluatedimension"""
        if dimension.name == "task_complexity":
            return self._assess_task_complexity(dataset)
        elif dimension.name == "scenario_diversity":
            return self._assess_scenario_diversity(dataset)
        elif dimension.name == "linguistic_diversity":
            return self._assess_linguistic_diversity(dataset, metadata)
        elif dimension.name == "user_diversity":
            return self._assess_user_diversity(dataset, metadata)
        elif dimension.name == "subagent_coverage":
            return self._assess_subagent_coverage(dataset)
        elif dimension.name == "constraint_variety":
            return self._assess_constraint_variety(dataset)
        elif dimension.name == "conversation_depth":
            return self._assess_conversation_depth(dataset, metadata)
        # back-compat: new condensed dimensions
        elif dimension.name == "task":
            return self._assess_task(dataset)
        elif dimension.name == "content_diversity":
            return self._assess_content_diversity(dataset, metadata)
        elif dimension.name == "interaction_depth":
            return self._assess_conversation_depth(dataset, metadata)
        elif dimension.name == "capability_coverage":
            return self._assess_subagent_coverage(dataset)
        else:
            return 0.0
    
    def _assess_task_complexity(self, dataset: Dict) -> float:
        """evaluatetaskcomplex"""
        tasks = dataset.get('tasks', {})
        level_distribution = {}
        
        for task_id, task in tasks.items():
            level = task.get('level', 'T1')
            level_distribution[level] = level_distribution.get(level, 0) + 1
        
        total_tasks = len(tasks)
        if total_tasks == 0:
            return 0.0
        
        # complex
        score = 0.0
        if level_distribution.get('T1', 0) > 0:
            score += 10  # 
        
        # T2task
        t2_ratio = level_distribution.get('T2', 0) / total_tasks
        score += min(t2_ratio * 20, 20)
        
        # T3task  
        t3_ratio = level_distribution.get('T3', 0) / total_tasks
        score += min(t3_ratio * 30, 30)
        
        # T4task
        t4_ratio = level_distribution.get('T4', 0) / total_tasks
        score += min(t4_ratio * 40, 40)
        
        return min(score, 100.0)
    
    def _assess_scenario_diversity(self, dataset: Dict) -> float:
        """evaluatescenariodiversity"""
        tasks = dataset.get('tasks', {})
        scenarios = set()
        
        scenario_keywords = {
            'city': ['city', '', '', '', 'city', 'downtown', 'urban'],
            'nature': ['', '', '', '', '', 'nature', 'outdoor', 'hiking', 'scenery'],
            'culture': ['', '', '', '', '', 'culture', 'heritage', 'museum', 'historic'],
            'business': ['', '', '', '', 'business', 'conference', 'corporate', 'meeting'],
            'family': ['', '', '', '', '', 'family', 'kids', 'children', 'parent'],
            'adventure': ['', '', '', '', 'adventure', 'extreme', 'challenge', 'thrill'],
            'luxury': ['', '', '', '', 'luxury', 'premium', 'five star', 'high-end'],
            'budget': ['', '', 'budget', '', 'budget', 'cheap', 'affordable', 'low-cost'],
            'events': ['', '', '', '', 'event', 'festival', 'show', 'exhibition'],
            'food': ['', 'restaurant', '', '', 'food', 'cuisine', 'restaurant', 'dining'],
            'weather': ['weather', '', '', '', 'weather', 'climate', 'temperature', 'rain'],
            'transport': ['', '', '', '', '', '', '', 'transport', 'transit', 'bus', 'subway', 'vehicle', 'flight','train']
        }
        
        for task_id, task in tasks.items():
            query = (task.get('query', '') or '').lower()
            for scenario_type, keywords in scenario_keywords.items():
                if any(keyword in query for keyword in keywords):
                    scenarios.add(scenario_type)
        
        # scenariodiversity
        unique_scenarios = len(scenarios)
        return min(unique_scenarios * 12.5, 100.0)  # 8scenario
    
    def _assess_linguistic_diversity(self, dataset: Dict, metadata: Dict) -> float:
        """evaluatelanguagediversity

        metadata may contain 'lang' or 'language' with values: 'CN', 'EN', 'mixed'.
        We adjust weighting of features depending on the expected dataset language to avoid
        mismatches between dataset reality and scoring assumptions.
        """
        tasks = dataset.get('tasks', {})
        language_features = {
            'english': 0,
            'dialect': 0,
            'colloquial': 0,
            'complex_syntax': 0
        }

        # collect per-task detailed detections so debug can print them
        per_task_details: Dict[str, Dict[str, Any]] = {}
        cn_dialect_indicators = ['', '', '', '', '', '', '', '', '', '']
        cn_colloquial = ['', '', '', '', '', '', '', '']
        en_colloquial = ["'m", "n't", "'re", "gonna", "wanna", "gotta", "lol", "pls", "thx", "ya", "cuz", "kinda", "sorta"]

        for task_id, task in tasks.items():
            query = task.get('query', '')
            # Normalize
            q = query.strip()

            # detect script presence: cjk vs latin
            has_cjk = bool(re.search(r'[\u4e00-\u9fff]', q))
            has_latin = bool(re.search(r'[A-Za-z]', q))

            # per-task flags
            is_english = False
            is_dialect = False
            is_colloquial = False
            is_complex = False

            # EN detection: require a minimum word token count to avoid false positives
            if has_latin and len(re.findall(r"[A-Za-z]{2,}", q)) >= 1:
                language_features['english'] += 1
                is_english = True

            # Chinese dialect indicators
            if has_cjk and any(indicator in q for indicator in cn_dialect_indicators):
                language_features['dialect'] += 1
                is_dialect = True

            # colloquial markers
            if has_cjk and any(tok in q for tok in cn_colloquial):
                language_features['colloquial'] += 1
                is_colloquial = True
            elif has_latin and any(re.search(rf"\b{re.escape(tok)}\b", q, re.IGNORECASE) for tok in en_colloquial):
                language_features['colloquial'] += 1
                is_colloquial = True

            # Complex syntax detection
            if has_cjk:
                if len(q) > 60 and re.search(r'[, ; : —]', q):
                    language_features['complex_syntax'] += 1
                    is_complex = True
            if has_latin:
                clause_separators = len(re.findall(r'[;,]', q))
                conj_count = len(re.findall(r'\b(and|but|which|that|because|however|therefore)\b', q, re.IGNORECASE))
                if len(q) > 80 and (clause_separators + conj_count) >= 1:
                    language_features['complex_syntax'] += 1
                    is_complex = True

            per_task_details[task_id] = {
                'query': q,
                'english': is_english,
                'dialect': is_dialect,
                'colloquial': is_colloquial,
                'complex_syntax': is_complex
            }

        total_tasks = len(tasks)
        if total_tasks == 0:
            return 0.0

        # determine requested language expectation from metadata
        lang_hint = None
        if metadata:
            lang_hint = (metadata.get('lang') or metadata.get('language') or '').upper()

        # Define weighting schemes per language expectation
        # Each mapping gives maximum contribution for each feature (summing to ~100 when combined with base)
        schemes = {
            'CN': {
                'base': 25,
                'english_max': 25,
                'dialect_max': 16.67,
                'colloquial_max': 16.67,
                'complex_syntax_max': 16.67
            },
            'EN': {
                'base': 10,
                'english_max': 60,
                'dialect_max': 0,
                'colloquial_max': 15,
                'complex_syntax_max': 15
            },
            'MIXED': {
                'base': 15,
                'english_max': 40,
                'dialect_max': 12.5,
                'colloquial_max': 16.25,
                'complex_syntax_max': 16.25
            }
        }

        scheme = schemes.get(lang_hint, schemes['CN'])

        score = float(scheme['base'])

        # add contributions proportional to detected ratios
        ratios = {f: (language_features.get(f, 0) / total_tasks) for f in language_features}

        score += min(ratios['english'] * scheme['english_max'], scheme['english_max'])
        score += min(ratios['dialect'] * scheme['dialect_max'], scheme['dialect_max'])
        score += min(ratios['colloquial'] * scheme['colloquial_max'], scheme['colloquial_max'])
        score += min(ratios['complex_syntax'] * scheme['complex_syntax_max'], scheme['complex_syntax_max'])
        # If debug requested, print per-task detection details and ratios
        debug = False
        if metadata and isinstance(metadata, dict):
            debug = bool(metadata.get('debug'))

        if debug:
            try:
                import pprint
                details = {
                    'language_features_counts': language_features,
                    'ratios': ratios,
                    'scheme': scheme
                }
                print('\n=== Linguistic detection debug (per-dataset summary) ===')
                pprint.pprint(details)

                # Print per-task detections in a compact table-like form
                print('\n--- Per-task linguistic detections ---')
                for tid, det in per_task_details.items():
                    flags = []
                    if det.get('english'):
                        flags.append('EN')
                    if det.get('dialect'):
                        flags.append('DIA')
                    if det.get('colloquial'):
                        flags.append('COL')
                    if det.get('complex_syntax'):
                        flags.append('CMP')
                    flags_str = ','.join(flags) if flags else 'none'
                    # short query preview (max 80 chars)
                    preview = det.get('query','')
                    if len(preview) > 80:
                        preview = preview[:77] + '...'
                    print(f"{tid}: [{flags_str}] {preview}")

                print('=== End linguistic debug ===\n')
            except Exception:
                pass

        return min(score, 100.0)

    def _assess_task(self, dataset: Dict) -> float:
        """evaluatetaskcomplexconstraint( task_complexity  constraint_variety)"""
        tc = self._assess_task_complexity(dataset)
        cv = self._assess_constraint_variety(dataset)
        # 
        return min((tc + cv) / 2.0, 100.0)

    def _assess_content_diversity(self, dataset: Dict, metadata: Dict) -> float:
        """scenario, languageuserdiversitydiversity"""
        sd = self._assess_scenario_diversity(dataset)
        ld = self._assess_linguistic_diversity(dataset, metadata)
        ud = self._assess_user_diversity(dataset, metadata)
        return min((sd + ld + ud) / 3.0, 100.0)
    
    def _assess_user_diversity(self, dataset: Dict, metadata: Dict) -> float:
        """evaluateuserdiversity"""
        # metadatauser, task
        user_profiles = metadata.get('user_profiles', ['default'])
        
        profile_types = {
            'business': ['', '', '', ''],
            'family': ['', '', '', ''],
            'backpacker': ['', '', '', ' hostel'],
            'luxury': ['', '', '', ''],
            'cultural': ['', '', '', ''],
            'adventure': ['', '', '', '']
        }
        
        # taskuser
        tasks = dataset.get('tasks', {})
        detected_profiles = set()
        
        for task_id, task in tasks.items():
            query = task.get('query', '')
            for profile_type, keywords in profile_types.items():
                if any(keyword in query for keyword in keywords):
                    detected_profiles.add(profile_type)
        
        # user
        all_profiles = set(user_profiles) | detected_profiles
        unique_profiles = len(all_profiles)
        
        return min(unique_profiles * 16.67, 100.0)  # 6user
    
    def _assess_subagent_coverage(self, dataset: Dict) -> float:
        """evaluatecoverage(task `expected_subagents`)"""
        all_subagents = {
            'get_weather_forecast', 'search_flights', 'search_hotels', 
            'search_attractions', 'calculate_travel_time', 'create_itinerary',
            'search_restaurants', 'get_city_info', 'check_calendar_conflict',
            'search_trains', 'search_buses', 'search_events', 'calculate_budget'
        }

        tasks = dataset.get('tasks', {})
        used_subagents = set()

        for task_id, task in tasks.items():
            subs = task.get('expected_subagents', [])
            if isinstance(subs, (list, set)):
                used_subagents.update(subs)
            # also inspect system_side_milestones for explicit agent invocations
            s_milestones = task.get('system_side_milestones', [])
            if s_milestones:
                for rule in s_milestones:
                    try:
                        r = str(rule)
                    except Exception:
                        continue
                    # try to detect known agent names mentioned in milestone strings
                    for a in all_subagents:
                        if a in r:
                            used_subagents.add(a)

        # if no declarations present, return 0 (or could be treated as N/A in future)
        if not used_subagents:
            return 0.0

        coverage_ratio = len(used_subagents) / len(all_subagents)
        return min(coverage_ratio * 100, 100.0)
    
    def _assess_constraint_variety(self, dataset: Dict) -> float:
        """evaluateconstraintdiversity"""
        constraint_types = {
            'time': ['time', 'date', '', '', ''],
            'budget': ['budget', '', '', '', ''],
            'preference': ['', '', '', '', ''],
            'location': ['', '', '', 'location', ''],
            'quality': ['', '', '', '', ''],
            'accessibility': ['', '', '', ''],
            'safety': ['', '', '', '']
        }
        
        tasks = dataset.get('tasks', {})
        detected_constraints = set()
        
        for task_id, task in tasks.items():
            query = task.get('query', '')
            user_side_milestones = task.get('user_side_milestones', [])
            system_side_milestones = task.get('system_side_milestones', [])
            
            # constraint
            for constraint_type, keywords in constraint_types.items():
                if any(keyword in query for keyword in keywords):
                    detected_constraints.add(constraint_type)
            
            # constraint
            # consider both user-side and system-side milestones when detecting constraints
            for rule in (user_side_milestones or []) + (system_side_milestones or []):
                try:
                    rule_lower = str(rule).lower()
                except Exception:
                    rule_lower = ''
                for constraint_type in constraint_types.keys():
                    if constraint_type in rule_lower:
                        detected_constraints.add(constraint_type)
        
        unique_constraints = len(detected_constraints)
        return min(unique_constraints * 14.29, 100.0)  # 7constraint
    
    def _assess_conversation_depth(self, dataset: Dict, metadata: Dict) -> float:
        """evaluatedialogue"""
        # metadatadialogue, taskcomplex
        avg_clarifications = 0
        tasks = dataset.get('tasks', {})
        
        for task_id, task in tasks.items():
            clarifications = task.get('expected_clarifications', [])
            avg_clarifications += len(clarifications)
        
        if tasks:
            avg_clarifications /= len(tasks)
        
        # clarification
        if avg_clarifications == 0:
            return 25.0  # dialogue
        elif avg_clarifications <= 2:
            return 50.0  # simple
        elif avg_clarifications <= 4:
            return 75.0  # 
        else:
            return 100.0  # dialogue
    
    def _calculate_total_score(self, dimension_scores: Dict[str, float]) -> float:
        """"""
        total_score = 0.0
        total_weight = 0.0
        
        for dimension in self.model.dimensions:
            score = dimension_scores.get(dimension.name, 0.0)
            total_score += score * dimension.weight
            total_weight += dimension.weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _score_to_level(self, score: float) -> MaturityLevel:
        """maturitylevel"""
        if score >= 90:
            return MaturityLevel.PLATINUM
        elif score >= 75:
            return MaturityLevel.GOLD
        elif score >= 60:
            return MaturityLevel.SILVER
        else:
            return MaturityLevel.BRONZE
    
    def _generate_recommendations(self, scores: Dict[str, float], levels: Dict[str, MaturityLevel]) -> List[str]:
        """generaterecommendation"""
        recommendations = []
        
        for dimension_name, score in scores.items():
            level = levels[dimension_name]
            
            if level == MaturityLevel.BRONZE:
                if dimension_name in ("task", "task_complexity"):
                    recommendations.append("T3-T4complextask, ")
                elif dimension_name in ("content_diversity", "linguistic_diversity"):
                    recommendations.append("languagetask")
            
            elif level == MaturityLevel.SILVER:
                if dimension_name in ("content_diversity", "user_diversity"):
                    recommendations.append("user, coverageuser")
                elif dimension_name in ("task", "constraint_variety"):
                    recommendations.append("complexconstraintscenario")
            
            elif level == MaturityLevel.GOLD:
                if dimension_name in ("interaction_depth", "conversation_depth"):
                    recommendations.append("dialogue, ")
                elif dimension_name in ("capability_coverage", "subagent_coverage"):
                    recommendations.append("coverage")
        
        # recommendation
        overall_level = self._score_to_level(self._calculate_total_score(scores))
        if overall_level == MaturityLevel.BRONZE:
            recommendations.append("recommendationtaskcomplexscenariodiversity")
        elif overall_level == MaturityLevel.SILVER:
            recommendations.append("recommendationlanguagediversityusercoverage")
        
        return recommendations[:5]  # 5recommendation
    

def demonstrate_maturity_assessment():
    """maturityevaluate"""
    
    # maturity
    maturity_model = TravelDatasetMaturityModel()
    assessment_engine = MaturityAssessmentEngine(maturity_model)
    
    # 
    sample_dataset = {
        'tasks': {
            'T1_001': {
                'level': 'T1',
                'query': 'weather',
                'expected_subagents': ['get_weather_forecast'],
                'expected_clarifications': [],
                'user_side_milestones': ['must_contain_weather_info']
            },
            'T2_001': {
                'level': 'T2', 
                'query': 'weatherrecommendation',
                'expected_subagents': ['get_weather_forecast', 'search_attractions'],
                'expected_clarifications': [],
                'user_side_milestones': ['must_execute_in_sequence']
            },
            'T3_001': {
                'level': 'T3',
                'query': 'planning5, budget5000, ',
                'expected_subagents': ['search_attractions', 'search_hotels', 'create_itinerary'],
                'expected_clarifications': ['children_ages', 'exact_dates'],
                'user_side_milestones': ['must_satisfy_budget', 'must_be_family_friendly']
            }
        }
    }
    
    # 
    metadata = {
        'user_profiles': ['family', 'business'],
        'evaluation_features': {
            'basic_metrics': True,
            'multi_dimensional': True,
            'comparative_analysis': False,
            'predictive_metrics': False
        }
    }
    
    # evaluate
    report = assessment_engine.assess_dataset(sample_dataset, metadata)
    
    # outputreport
    print("=== maturityevaluatereport ===")
    print(f"maturity: {report.overall_level.name}")
    print(f": {report.total_score:.1f}/100")
    
    print("\ndimension:")
    for dim_name, score in report.dimension_scores.items():
        level = report.dimension_levels[dim_name]
        print(f"  {dim_name}: {score:.1f} ({level.name})")
    
    print("\nrecommendation:")
    for i, recommendation in enumerate(report.recommendations, 1):
        print(f"  {i}. {recommendation}")


def assess_dataset_from_file(file_path: str, metadata: Dict = None, verbose: bool = True) -> DatasetMaturityReport:
    """Load a dataset (YAML or JSON) from file and run maturity assessment.

        Expected YAML shape (like generated_travel_tasks.yaml):
    tasks:
      TASK_ID:
        level: "T1"
        query: "..."
                expected_subagents: [...]
        expected_clarifications: [...]
        user_side_milestones: [...]
        system_side_milestones: [...]

    Returns DatasetMaturityReport and prints summary when verbose=True.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        if file_path.lower().endswith(('.yaml', '.yml')):
            data = yaml.safe_load(f)
        else:
            # assume json
            data = json.load(f)

    # The evaluator expects {'tasks': {id: {...}}}
    dataset = {'tasks': {}}
    # If the dataset file contains a top-level 'metadata' section, merge it with provided metadata.
    file_metadata = data.get('metadata') or {}
    # Merge file metadata as defaults, allow explicit metadata param to override
    merged_metadata = {}
    if isinstance(file_metadata, dict):
        merged_metadata.update(file_metadata)
    if isinstance(metadata, dict):
        # metadata passed by caller (CLI flags or external file) should take precedence
        merged_metadata.update(metadata)
    # use merged metadata for assessment
    metadata = merged_metadata

    src_tasks = data.get('tasks') or {}
    # normalize tasks: ensure keys and fields exist
    for tid, t in src_tasks.items():
        dataset['tasks'][tid] = {
            'level': t.get('level', 'T1'),
            'query': t.get('query', ''),
            'expected_subagents': t.get('expected_subagents', t.get('tools', [])),
            'expected_clarifications': t.get('expected_clarifications', t.get('clarifications', [])),
            'user_side_milestones': t.get('user_side_milestones', []),
            'system_side_milestones': t.get('system_side_milestones', []),
            # preserve optional fields if present (generator now emits these)
            'complexity_factors': t.get('complexity_factors', []),
            'description': t.get('description', '')
        }

    # run assessment
    maturity_model = TravelDatasetMaturityModel()
    engine = MaturityAssessmentEngine(maturity_model)
    report = engine.assess_dataset(dataset, metadata=metadata or {})

    if verbose:
        _print_report(report, title="=== maturityevaluatereport (file) ===", file_path=file_path)

    return report


def _print_report(report: DatasetMaturityReport, title: str = None, file_path: str = None) -> None:
    """Pretty-print a `DatasetMaturityReport` to stdout."""
    if title:
        print(title)
    if file_path:
        print(f"inputfile: {file_path}")
    print(f"maturity: {report.overall_level.name}")
    print(f": {report.total_score:.1f}/100")

    print("\ndimension:")
    for dim_name, score in report.dimension_scores.items():
        level = report.dimension_levels[dim_name]
        print(f"  {dim_name}: {score:.1f} ({level.name})")

    if report.recommendations:
        print("\nrecommendation:")
        for idx, recommendation in enumerate(report.recommendations, 1):
            print(f"  {idx}. {recommendation}")


def _summarize_dataset_for_prompt(dataset: Dict) -> Dict[str, Any]:
    """Create a compact summary of the dataset to include in prompts to an LLM."""
    tasks = dataset.get('tasks', {})
    total_tasks = len(tasks)
    levels = {}
    subagents = set()
    sample_queries = []
    for i, (tid, t) in enumerate(tasks.items()):
        lvl = t.get('level', 'T1')
        levels[lvl] = levels.get(lvl, 0) + 1
        subs = t.get('expected_subagents', [])
        if isinstance(subs, (list, set)):
            subagents.update(subs)
        if i < 5:
            q = t.get('query','')
            sample_queries.append(q if len(q) < 200 else q[:197] + '...')

    return {
        'total_tasks': total_tasks,
        'level_distribution': levels,
        'declared_subagents': sorted(list(subagents))[:50],
        'sample_queries': sample_queries
    }


def _call_llm_for_scoring(dataset: Dict, metadata: Dict, dimension_names: List[str], llm_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Call an LLM to request per-dimension numeric scores (0-100) and short comments.

    Returns a dict with keys: 'dimension_scores' (mapping name->float) and optional 'comments'.
    This function uses the centralized LLMClient for HTTP/SDK calls and parsing.
    """
    llm_client_cls = globals().get('LLMClient') if _LLM_CLIENT_AVAILABLE else None
    default_client_factory = globals().get('get_default_client') if _LLM_CLIENT_AVAILABLE else None

    if not _LLM_CLIENT_AVAILABLE:
        # 
        try:
            # 
            current_dir = Path(__file__).parent
            llm_client_path = current_dir / 'llm_client.py'
            if llm_client_path.exists():
                import importlib.util
                spec = importlib.util.spec_from_file_location("llm_client", llm_client_path)
                llm_client_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(llm_client_module)
                llm_client_cls = llm_client_module.LLMClient
                default_client_factory = llm_client_module.get_default_client
            else:
                #  datasets 
                parent_dir = current_dir.parent
                if str(parent_dir) not in sys.path:
                    sys.path.insert(0, str(parent_dir))
                from datasets.llm_client import LLMClient as _ImportedLLMClient, get_default_client as _imported_default_client
                llm_client_cls = _ImportedLLMClient
                default_client_factory = _imported_default_client
        except Exception as e:
            error_msg = f'LLM client not available. Initial import error: {_LLM_CLIENT_IMPORT_ERROR}. Dynamic import error: {e}'
            raise RuntimeError(error_msg)

    if llm_client_cls is None or default_client_factory is None:
        raise RuntimeError('LLM client not available. Unable to resolve client factory.')

    # Build succinct dataset summary
    summary = _summarize_dataset_for_prompt(dataset)

    # Build the instruction prompt asking for JSON output
    dims = ', '.join(dimension_names)
    prompt = (
        f"You are an objective evaluator. Given a travel dataset summary, score the following maturity "
        f"dimensions on a 0-100 scale (integers or floats): {dims}. "
        "Return a single valid JSON object with keys: 'dimension_scores' (map of name->number) and 'comments' "
        "(map of name->short string explaining the score). Do not include extra fields. "
        "If you cannot determine a dimension, return 0 for that dimension. Keep the JSON compact.\n\n"
    )

    payload_content = {
        'dataset_summary': summary,
        'metadata': metadata or {}
    }

    # Append the payload to the user message (as JSON) so the model can parse details if needed.
    user_message = prompt + "\n\nDataset summary (JSON):\n" + json.dumps(payload_content, ensure_ascii=False)

    # Use centralized LLMClient
    try:
        # Build LLM configuration
        config = {}
        if isinstance(llm_config, dict):
            config.update(llm_config)
        # metadata may contain embedded llm_config
        if isinstance(metadata, dict) and metadata.get('llm_config'):
            try:
                config.update(metadata.get('llm_config') or {})
            except Exception:
                pass

        # build client: prefer explicit config, else default
        client = llm_client_cls(config) if config else default_client_factory()

        resp_text = client.call_llm(user_message)

        # parse JSON using client's method
        parsed = client.parse_llm_response(resp_text)

        dim_scores = parsed.get('dimension_scores') if isinstance(parsed, dict) else None
        comments = parsed.get('comments') if isinstance(parsed, dict) else {}
        if not isinstance(dim_scores, dict):
            raise RuntimeError('LLM response does not contain dimension_scores mapping')

        normalized = {}
        for dn in dimension_names:
            val = dim_scores.get(dn)
            try:
                vnum = float(val)
            except Exception:
                vnum = 0.0
            if vnum < 0:
                vnum = 0.0
            if vnum > 100:
                vnum = 100.0
            normalized[dn] = vnum

        return {'dimension_scores': normalized, 'comments': comments}
    except Exception as e:
        # Bubble up the error so caller can decide to fallback to heuristics
        raise RuntimeError(f'LLM scoring failed: {e}')

def generate_maturity_roadmap():
    """Generate maturity improvement roadmap"""
    roadmap = {
        MaturityLevel.BRONZE: [
            "task(T1-T2)",
            "coveragetravelscenario", 
            "evaluate",
            "coverage"
        ],
        MaturityLevel.SILVER: [
            "complextask(T3)",
            "language",
            "user", 
            "dimensionevaluate",
            "scenariodiversity"
        ],
        MaturityLevel.GOLD: [
            "task(T4)",
            "coverage",
            "complexconstraint",
            "dialogueevaluate",
            ""
        ],
        MaturityLevel.PLATINUM: [
            "scenario",
            "language",
            "evaluate",
            "",
            ""
        ]
    }
    
    print("=== maturity ===")
    for level, steps in roadmap.items():
        print(f"\n{level.name} level:")
        for step in steps:
            print(f"  • {step}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataset maturity evaluator')
    parser.add_argument('--file', '-f', help='Path to dataset file (YAML or JSON) to evaluate')
    parser.add_argument('--metadata', '-m', help='Path to metadata file (YAML or JSON) to provide evaluation features/user profiles')
    parser.add_argument('--lang', '-l', help='Expected dataset language: CN, EN, or mixed (affects linguistic scoring)', choices=['CN','EN','mixed'], default=None)
    parser.add_argument('--out-file', '-o', help='Path to write JSON report output')
    parser.add_argument('--no-demo', action='store_true', help='Skip built-in demo and only run file assessment if provided')
    parser.add_argument('--debug', action='store_true', help='Print per-task linguistic detection details for debugging')
    parser.add_argument('--use-llm', action='store_true', help='Use an LLM to score dimensions instead of internal heuristics')
    # LLM configuration is read from metadata, config/llm.yaml, or env vars (LLM_CONFIG_PATH, OPENAI_API_KEY, MODEL_ENDPOINT)
    args = parser.parse_args()

    if args.file:
        try:
            # start with empty metadata, then load file metadata (if any), then override with CLI flags
            metadata = {}
            if args.metadata:
                if not os.path.exists(args.metadata):
                    print(f"Metadata file not found: {args.metadata}")
                else:
                    with open(args.metadata, 'r', encoding='utf-8') as mf:
                        if args.metadata.lower().endswith(('.yaml', '.yml')):
                            file_md = yaml.safe_load(mf) or {}
                        else:
                            file_md = json.load(mf)
                    # merge file metadata
                    metadata.update(file_md)

            # CLI overrides
            if args.lang:
                metadata['lang'] = args.lang
            if args.debug:
                metadata['debug'] = True

            # If LLM scoring requested, attempt to call LLM and use its numeric scores
            if args.use_llm:
                try:
                    maturity_model = TravelDatasetMaturityModel()
                    engine = MaturityAssessmentEngine(maturity_model)
                    # prepare normalized dataset (same shape as engine expects)
                    # reuse logic from assess_dataset_from_file to load/normalize tasks
                    # load the file again to get the normalized `dataset` structure
                    with open(args.file, 'r', encoding='utf-8') as f:
                        if args.file.lower().endswith(('.yaml', '.yml')):
                            raw = yaml.safe_load(f) or {}
                        else:
                            raw = json.load(f) or {}

                    src_tasks = raw.get('tasks') or {}
                    dataset = {'tasks': {}}
                    for tid, t in src_tasks.items():
                        dataset['tasks'][tid] = {
                            'level': t.get('level', 'T1'),
                            'query': t.get('query', ''),
                            'expected_subagents': t.get('expected_subagents', t.get('tools', [])),
                            'expected_clarifications': t.get('expected_clarifications', t.get('clarifications', [])),
                            'user_side_milestones': t.get('user_side_milestones', []),
                            'system_side_milestones': t.get('system_side_milestones', []),
                            'complexity_factors': t.get('complexity_factors', []),
                            'description': t.get('description', '')
                        }

                    dimension_names = [d.name for d in maturity_model.dimensions]
                    try:
                        # build llm_config preference: allow metadata to embed 'llm_config'
                        llm_config = None
                        if isinstance(metadata, dict) and metadata.get('llm_config'):
                            llm_config = metadata.get('llm_config')
                        # call scoring with config resolution inside the function
                        llm_result = _call_llm_for_scoring(dataset, metadata or {}, dimension_names, llm_config=llm_config)
                        dim_scores = llm_result.get('dimension_scores', {})
                        # build report from llm scores
                        dimension_levels = {k: engine._score_to_level(v) for k, v in dim_scores.items()}
                        total_score = engine._calculate_total_score(dim_scores)
                        overall_level = engine._score_to_level(total_score)
                        recommendations = engine._generate_recommendations(dim_scores, dimension_levels)
                        report = DatasetMaturityReport(
                            overall_level=overall_level,
                            dimension_scores=dim_scores,
                            dimension_levels=dimension_levels,
                            recommendations=recommendations,
                            total_score=total_score,
                            max_score=100.0
                        )
                        # also print any comments from LLM
                        comments = llm_result.get('comments') or {}
                        if comments and args.debug:
                            print('\nLLM comments:')
                            for k, v in comments.items():
                                print(f"  {k}: {v}")
                        _print_report(report, title="=== maturityevaluatereport (LLM) ===", file_path=args.file)
                    except Exception as e:
                        print('LLM scoring failed, falling back to internal scoring. Error:', e)
                        report = assess_dataset_from_file(args.file, metadata=metadata, verbose=True)
                except Exception as e:
                    print('Error preparing dataset for LLM scoring:', e)
                    report = assess_dataset_from_file(args.file, metadata=metadata, verbose=True)
            else:
                report = assess_dataset_from_file(args.file, metadata=metadata, verbose=True)

            if args.out_file:
                # serialize report to JSON and write
                out = {
                    'overall_level': report.overall_level.name,
                    'total_score': report.total_score,
                    'max_score': report.max_score,
                    'dimension_scores': report.dimension_scores,
                    'dimension_levels': {k: v.name for k, v in report.dimension_levels.items()},
                    'recommendations': report.recommendations,
                }
                try:
                    with open(args.out_file, 'w', encoding='utf-8') as of:
                        json.dump(out, of, ensure_ascii=False, indent=2)
                    print(f"Report written to: {args.out_file}")
                except Exception as e:
                    print(f"Failed to write report to {args.out_file}: {e}")
        except Exception as e:
            print('Error assessing dataset file:', e)