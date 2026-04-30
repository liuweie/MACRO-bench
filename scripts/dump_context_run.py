import json
import sys
from pathlib import Path

# ensure cogbenchmark package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plugins.html_reporter_plugin import HTMLReporter

REPORT_P = Path(r'c:\workspace\benchmark-mutliagent\reports\benchmark_report_20251203_212046.json')
OUT_HTML = Path('output') / 'sample_from_benchmark_mutliagent.html'

if not REPORT_P.exists():
    print('REPORT_JSON_NOT_FOUND:', REPORT_P)
    raise SystemExit(1)

with REPORT_P.open('r', encoding='utf-8') as f:
    data = json.load(f)

reporter = HTMLReporter()
reporter.generate_report(data, str(OUT_HTML))
print('DONE')
