from pathlib import Path
import yaml
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluators.evaluator import RuleBasedEvaluator

cfg_path = Path('datasets/travel/tasks.yaml')
with cfg_path.open('r', encoding='utf-8') as f:
    tasks_cfg = yaml.safe_load(f)

be = RuleBasedEvaluator(tasks_config={'tasks': tasks_cfg.get('tasks')})

candidates = [
    "I want to go somewhere nice for a trip soon.",
    "Plan a 4-day warm-weather trip anywhere with full daily plans.",
    "Help me plan a trip to Shanghai next month.",
    "I need a travel plan for somewhere warm.",
    "Book a flight from Beijing to Shanghai on Dec 20",
]

for q in candidates:
    m = be._find_matching_task(q)
    print(q)
    print('->', m['task_id'] if m else None)
    print('---')
