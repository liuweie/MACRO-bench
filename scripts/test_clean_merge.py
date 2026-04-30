from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from plugins.latc_orchestrator_plugin import _clean_merged_text

sample = '''### Agent Router
#### Receives a query from user:
- 12 月 15–18 日这段时间我是打算待在大阪。
#### Break down the task into the following steps:
Recived query: 12 月 15–18 日这段时间我是打算待在大阪。
从高德地图获取None相关内容
从高德地图获取None相关内容
从高德地图获取None相关内容
从高德地图获取None相关内容
Hotel
Hotel
inquiries
inquiries
have
have
been
been
made
made
according
according
to
to
your
your
requirements
requirements
,
,
but
but
no
no
eligible
eligible
hotels
hotels
in
in
Osaka
Osaka
,
,
Japan
Japan
were
were
found
found
in
in
the
the
search
search
results
results
.
.
The
The
tool
tool
repeatedly
repeatedly
returned
returned
hotel
hotel
information
information
in
in
Hong
Hong
Kong
Kong
Special
Special
Administrative
Administrative
Region
Region
,
,
which
which
does
does
not
not
match
match
your
your
destination
destination
.
.
Please
Please
check
check
if
if
the
the
destination
destination
is
is
correct
correct
or
or
try
try
using
using
more
more
specific
specific
search
search
keywords
keywords
(
(
e
e
.g
.g
.,
.,
""
Os
Os
aka
aka
city
city
center
center
hotel
hotel
""
")
")
and
and
try
try
again
again
.
.
answer
answer
finished
finished
### Agent Router
- **hotel accommodation recommendation agent** execution finished
- Requests user input to continue the task.'''

print('--- Original (truncated) ---')
print(sample[:500])
print('\n--- Cleaned ---')
print(_clean_merged_text(sample)[:1000])
