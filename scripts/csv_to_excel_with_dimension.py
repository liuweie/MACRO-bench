import csv
from openpyxl import Workbook

csv_path = r"c:\workspace\cogbenchmark\docs\evaluator_llm_enhanced_metrics.csv"
out_xlsx = r"c:\workspace\cogbenchmark\docs\evaluator_llm_enhanced_metrics.xlsx"

# Mapping MetricKey to Dimension
dimension_map = {
    'milestone_completion_rate': 'Business',
    'system_milestone_completion_rate': 'Business',
    'task_success_rate': 'Business',
    'task_completion_rate': 'Business',
    'orchestration_efficiency': 'Orchestration',
    'agent_routing_accuracy': 'Orchestration',
    'clarification_efficiency': 'Orchestration',
    'time_efficiency': 'Efficiency',
    'average_rounds': 'Efficiency',
    'orchestration_latency_seconds': 'Efficiency',
    'response_quality': 'Performance',
    'travel_context_understanding': 'Performance'
}

rows = []
with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for r in reader:
        key = r.get('MetricKey')
        dim = dimension_map.get(key, '')
        r['Dimension'] = dim
        rows.append(r)

# write to xlsx
wb = Workbook()
ws = wb.active
ws.title = 'LLM Enhanced Metrics'

# headers
if rows:
    headers = list(rows[0].keys())
    ws.append(headers)
    for row in rows:
        ws.append([row.get(h, '') for h in headers])

wb.save(out_xlsx)
print(f"Wrote {len(rows)} rows to {out_xlsx}")
