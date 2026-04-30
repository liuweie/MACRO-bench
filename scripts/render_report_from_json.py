import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from plugins.html_reporter_plugin import HTMLReporter

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--json', required=True, help='Path to report JSON')
parser.add_argument('--out', required=False, help='Desired output path (html or json)', default=None)
args = parser.parse_args()

json_path = Path(args.json)
if not json_path.exists():
    print(f'ERROR: json file not found: {json_path}')
    raise SystemExit(2)

with json_path.open('r', encoding='utf-8') as f:
    report = json.load(f)

out_path = Path(args.out) if args.out else json_path

reporter = HTMLReporter(config={})
print(f'Generating report from {json_path} -> {out_path}')
try:
    reporter.generate_report(report, str(out_path))
    print('generate_report returned successfully')
except Exception as e:
    print('generate_report raised exception:', repr(e))

# list files around output
html_path = out_path.with_suffix('.html') if out_path.suffix.lower() != '.html' else out_path
json_copy = html_path.with_suffix('.json')
dbg = html_path.with_suffix('.debug.json')
print('\nResult files:')
for p in (html_path, json_copy, dbg):
    try:
        print(p, '->', 'exists' if p.exists() else 'MISSING', 'size=' + (str(p.stat().st_size) if p.exists() else '0'))
    except Exception as e:
        print(p, '-> error checking', e)

print('\nListing output/ dir (most recent):')
for p in sorted(Path('output').glob('*'), key=lambda x: x.stat().st_mtime, reverse=True)[:20]:
    try:
        print(p.name, p.stat().st_mtime, p.stat().st_size)
    except Exception:
        print(p.name, 'stat failed')
