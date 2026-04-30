import json
from pathlib import Path
p=Path(r'c:\workspace\benchmark-mutliagent\reports\benchmark_report_20251203_212046.json')
print('exists',p.exists())
with p.open('r',encoding='utf-8') as f:
    d=json.load(f)
print(type(d), list(d.keys()))
print('has summary?', 'summary' in d, 'has details?', 'details' in d, 'has tasks?', 'tasks' in d)
print('len d', len(d))
