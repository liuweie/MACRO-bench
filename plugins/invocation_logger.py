import json
import time
from pathlib import Path
from threading import Lock
from typing import Optional, Dict, Any

_LOG_DIR = Path('output') / 'collected_jsons'
_LOCK = Lock()


def _ensure_dir():
    _LOG_DIR.mkdir(parents=True, exist_ok=True)


def _sanitize_task_id(task_id: str) -> str:
    if not task_id:
        return ''
    return str(task_id).replace(' ', '_').replace('/', '_').replace('\\', '_')


def _run_timestamp() -> str:
    return time.strftime('%Y%m%d_%H%M%S')


def get_log_path(task_id: Optional[str] = None) -> Path:
    """Return a Path to append logs to.

    If task_id is provided, returns a per-task file `plugin_invocations_<taskid>.jsonl`.
    Otherwise returns a run-level file stamped with current timestamp.
    """
    _ensure_dir()
    if task_id:
        safe = _sanitize_task_id(task_id)
        return _LOG_DIR / f'plugin_invocations_task_{safe}.jsonl'
    ts = _run_timestamp()
    return _LOG_DIR / f'plugin_invocations_{ts}.jsonl'


def log_invocation(entry: Dict[str, Any], task_id: Optional[str] = None) -> None:
    """Append a structured JSON line to the appropriate invocation log.

    The function will add a timestamp (`ts`) in ISO format if not present.
    It is safe to call from multiple threads/processes on the same host
    (uses a simple threading.Lock for in-process safety).
    """
    try:
        if not isinstance(entry, dict):
            entry = {'message': str(entry)}
        if 'ts' not in entry:
            entry['ts'] = time.strftime('%Y-%m-%dT%H:%M:%S')

        path = get_log_path(task_id)
        line = json.dumps(entry, ensure_ascii=False)
        with _LOCK:
            with open(path, 'a', encoding='utf-8') as fh:
                fh.write(line + '\n')
    except Exception:
        # best-effort: don't raise to avoid breaking plugin flows
        try:
            # fallback to writing a plain text diagnostics file
            _ensure_dir()
            ts = int(time.time())
            fname = _LOG_DIR / f'plugin_invocations_fallback_{ts}.log'
            with open(fname, 'a', encoding='utf-8') as fh:
                fh.write(str(entry) + '\n')
        except Exception:
            pass
