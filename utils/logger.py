import logging
import json
from datetime import datetime
from typing import Dict, Any

class BenchmarkLogger:
    def __init__(self, log_file: str = "benchmark_logs.jsonl", console_full: bool = False, console_max_len: int = 20000):
        """Create a benchmark logger.

        :param log_file: path to jsonl log file where full payloads/responses are stored
        :param console_full: if True, print full payload/response to console for DEBUG logs
        :param console_max_len: maximum characters to print to console for payload/response
        """
        self.log_file = log_file
        self.console_full = bool(console_full)
        self.console_max_len = int(console_max_len) if console_max_len is not None else 20000
        self.setup_logging()
    
    def setup_logging(self):
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        # Use a generic logger name for clarity across benchmarks
        self.logger = logging.getLogger('Benchmark')
    
    def log_task_start(self, task_id: str, query: str):
        self._write_log({
            'timestamp': datetime.now().isoformat(),
            'level': 'INFO',
            'event': 'task_start',
            'task_id': task_id,
            'query': query
        })
    
    def log_orchestrator_call(self, task_id: str, payload: Dict, response: Dict):
        # Serialize payload and response safely and truncate to avoid excessively large logs
        try:
            payload_s = json.dumps(payload, ensure_ascii=False, default=str)
        except Exception:
            payload_s = str(payload)
        try:
            response_s = json.dumps(response, ensure_ascii=False, default=str)
        except Exception:
            response_s = str(response)

        # truncate long strings for file storage but keep original strings for console control
        max_len = 20000
        payload_for_file = payload_s
        response_for_file = response_s
        if len(payload_for_file) > max_len:
            payload_for_file = payload_for_file[:max_len] + '...'
        if len(response_for_file) > max_len:
            response_for_file = response_for_file[:max_len] + '...'

        self._write_log({
            'timestamp': datetime.now().isoformat(),
            'level': 'DEBUG',
            'event': 'orchestrator_call',
            'task_id': task_id,
            'payload': payload_for_file,
            'response': response_for_file,
            'payload_raw': payload_s,
            'response_raw': response_s,
            'payload_size': len(str(payload)),
            'response_status': 'success' if response else 'error'
        })

    def log_clarification_diagnostic(self, task_id: str, round_number: int, question: str, user_response: str, rule_match: Dict = None, used_strategy: str = None):
        """Log diagnostics for a clarification handling event.

        Includes the assistant question, simulated user response, rule match info (if any), and which strategy was used.
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'level': 'DEBUG',
            'event': 'clarification_diagnostic',
            'task_id': task_id,
            'round': round_number,
            'question': question,
            'user_response': (user_response[:2000] + '...') if user_response and len(user_response) > 2000 else user_response,
            'used_strategy': used_strategy,
            'rule_match': rule_match
        }
        self._write_log(entry)
    
    def log_evaluation_result(self, task_id: str, scores: Dict[str, float]):
        self._write_log({
            'timestamp': datetime.now().isoformat(),
            'level': 'INFO',
            'event': 'evaluation_complete',
            'task_id': task_id,
            'scores': scores
        })
    
    def log_error(self, task_id: str, error: str):
        self._write_log({
            'timestamp': datetime.now().isoformat(),
            'level': 'ERROR',
            'event': 'error',
            'task_id': task_id,
            'error': error
        })
    
    def _write_log(self, log_entry: Dict[str, Any]):
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        
        # 同时输出到控制台
        lvl = log_entry.get('level', 'INFO')
        try:
            if lvl == 'ERROR':
                self.logger.error(f"{log_entry['event']} - {log_entry.get('error', '')}")
            elif lvl == 'INFO':
                self.logger.info(f"{log_entry['event']} - task: {log_entry.get('task_id', '')}")
            elif lvl == 'DEBUG':
                # Optionally print full payload/response to console when console_full is enabled.
                ev = log_entry.get('event', '')
                tid = log_entry.get('task_id', '')
                ps = log_entry.get('payload_raw', log_entry.get('payload'))
                rs = log_entry.get('response_raw', log_entry.get('response'))
                if self.console_full:
                    try:
                        # limit console output length to avoid flooding the terminal
                        ps_print = (ps[:self.console_max_len] + '...') if ps and len(ps) > self.console_max_len else ps
                        rs_print = (rs[:self.console_max_len] + '...') if rs and len(rs) > self.console_max_len else rs
                        self.logger.debug(f"{ev} - task: {tid} - payload_size: {log_entry.get('payload_size', 0)} - response_status: {log_entry.get('response_status')}")
                        self.logger.debug(f"PAYLOAD:\n{ps_print}")
                        self.logger.debug(f"RESPONSE:\n{rs_print}")
                    except Exception:
                        self.logger.debug(f"{ev} - task: {tid} - (failed to print full payload/response)")
                else:
                    # print a concise debug summary to console, full payload/response are saved in the jsonl file
                    self.logger.debug(f"{ev} - task: {tid} - payload_size: {log_entry.get('payload_size', 0)} - response_status: {log_entry.get('response_status')}")
            else:
                self.logger.info(f"{log_entry['event']} - task: {log_entry.get('task_id', '')}")
        except Exception:
            # fallback: always log as info
            try:
                self.logger.info(f"{log_entry.get('event', '')} - task: {log_entry.get('task_id', '')}")
            except Exception:
                pass

    def info(self, message: str):
        # Accept printf-style formatting or plain message
        try:
            self.logger.info(message)
        except Exception:
            # fallback: convert to string
            self.logger.info(str(message))
    
    def error(self, message: str):
        try:
            self.logger.error(message)
        except Exception:
            self.logger.error(str(message))

    def warning(self, message: str):
        try:
            self.logger.warning(message)
        except Exception:
            self.logger.warning(str(message))

    def debug(self, message: str):
        try:
            self.logger.debug(message)
        except Exception:
            self.logger.debug(str(message))