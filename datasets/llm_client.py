"""
Reusable LLM client utilities.

This module centralizes HTTP/SDK calls and response parsing so other
modules (e.g., travel_generator, maturity_evaluator) can reuse the same
logic and error handling.

API:
 - call_llm(prompt, llm_config)
 - call_openai_sdk(prompt, llm_config)
 - parse_llm_response(response)
 - parse_fallback(response)
"""
from typing import Dict, Any, Optional, Union
import os
import json
from pathlib import Path
import time
import requests
import yaml
try:
    from loguru import logger as log
    _LOG_AVAILABLE = True
except Exception:
    _LOG_AVAILABLE = False
    class _SimpleLogger:
        def __init__(self):
            # Honor environment variable to silence LLM HTTP/debug logs during benchmark runs
            self._silent = os.environ.get('COGBENCH_SILENT_LLM', '1').strip() not in ('0', 'false', 'False')
        def debug(self, *args, **kwargs):
            if not self._silent:
                print('[DEBUG]', *args)
        def info(self, *args, **kwargs):
            if not self._silent:
                print('[INFO]', *args)
        def warning(self, *args, **kwargs):
            if not self._silent:
                print('[WARN]', *args)
        def error(self, *args, **kwargs):
            if not self._silent:
                print('[ERROR]', *args)
    log = _SimpleLogger()

# Reduce noisy urllib3/request debug logs by default (can be overridden by env/host config)
try:
    import logging as _logging
    _logging.getLogger('urllib3').setLevel(_logging.WARNING)
    _logging.getLogger('requests').setLevel(_logging.WARNING)
except Exception:
    pass


class LLMClient:
    """Class encapsulating LLM configuration and call helpers.

    Usage patterns:
      - LLMClient.from_env()  # builds from env and config/llm.yaml
      - LLMClient(config_dict)
      - client.call_llm(prompt)

    The class centralizes config resolution and provides `call_llm`,
    `call_openai_sdk`, `parse_llm_response`, and `parse_fallback` methods.
    """

    _ENV_INITIALIZED = False

    def __init__(self, config: Optional[Dict[str, Any]] = None, config_path: Optional[str] = None):
        self.config = config.copy() if isinstance(config, dict) else {}
        # normalize keys
        self._resolve_from_env_and_file(config_path)

    @classmethod
    def _load_dotenv_once(cls, explicit_path: Optional[str] = None) -> None:
        if cls._ENV_INITIALIZED:
            return

        candidates = []
        if explicit_path:
            candidates.append(Path(explicit_path))
        env_defined = os.environ.get('DOTENV_PATH')
        if env_defined:
            env_path = Path(env_defined)
            if env_path not in candidates:
                candidates.append(env_path)
        candidates.append(Path('.env'))

        for candidate in candidates:
            if not candidate:
                continue
            try:
                if not candidate.exists():
                    continue
            except Exception:
                continue
            try:
                from dotenv import load_dotenv
                load_dotenv(str(candidate), override=False)
            except Exception:
                try:
                    for line in candidate.read_text(encoding='utf-8').splitlines():
                        line = line.strip()
                        if not line or line.startswith('#') or '=' not in line:
                            continue
                        key, value = line.split('=', 1)
                        key = key.strip()
                        if not key or key in os.environ:
                            continue
                        os.environ[key] = value.strip().strip('"').strip("'")
                except Exception:
                    pass

        cls._ENV_INITIALIZED = True

    @staticmethod
    def _collect_env_overrides() -> Dict[str, Any]:
        aliases = {
            'endpoint': ['LLM_ENDPOINT', 'MODEL_ENDPOINT', 'LLM_API_URL'],
            'api_key': ['LLM_API_KEY', 'MODEL_BEARER_TOKEN', 'OPENAI_API_KEY', 'API_KEY'],
            'token': ['LLM_TOKEN', 'MODEL_TOKEN'],
            'model': ['LLM_MODEL', 'MODEL_NAME'],
            'deployment': ['LLM_DEPLOYMENT', 'MODEL_DEPLOYMENT'],
            'api_version': ['LLM_API_VERSION', 'MODEL_API_VERSION'],
            'temperature': ['LLM_TEMPERATURE', 'MODEL_TEMPERATURE'],
            'max_tokens': ['LLM_MAX_TOKENS', 'MODEL_MAX_TOKENS'],
            'max_completion_tokens': ['LLM_MAX_COMPLETION_TOKENS', 'MODEL_MAX_COMPLETION_TOKENS'],
            'language': ['LLM_LANGUAGE', 'LLM_LANG', 'MODEL_LANGUAGE', 'MODEL_LANG'],
            'allow_insecure': ['LLM_ALLOW_INSECURE', 'MODEL_ALLOW_INSECURE'],
            'debug': ['LLM_DEBUG', 'MODEL_DEBUG'],
            'ca_bundle': ['LLM_CA_BUNDLE', 'MODEL_CA_BUNDLE'],
            'timeout': ['LLM_TIMEOUT', 'LLM_HTTP_TIMEOUT', 'MODEL_HTTP_TIMEOUT'],
            'http_retries': ['LLM_HTTP_RETRIES', 'MODEL_HTTP_RETRIES'],
            'retry_backoff': ['LLM_HTTP_RETRY_BACKOFF', 'MODEL_HTTP_RETRY_BACKOFF'],
            'connect_timeout': ['LLM_CONNECT_TIMEOUT', 'MODEL_CONNECT_TIMEOUT'],
            'read_timeout': ['LLM_READ_TIMEOUT', 'MODEL_READ_TIMEOUT'],
        }

        cfg: Dict[str, Any] = {}
        for target, env_keys in aliases.items():
            for env_key in env_keys:
                value = os.environ.get(env_key)
                if value is not None and value != '':
                    cfg[target] = value
                    break

        def _to_bool(value: Any) -> bool:
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() in {'1', 'true', 'yes', 'y', 'on'}

        def _to_optional_float(value: Any) -> Optional[float]:
            try:
                return float(value)
            except Exception:
                return None

        def _to_optional_int(value: Any) -> Optional[int]:
            try:
                return int(value)
            except Exception:
                return None

        if 'temperature' in cfg:
            temp = _to_optional_float(cfg['temperature'])
            if temp is not None:
                cfg['temperature'] = temp
            else:
                cfg.pop('temperature', None)

        if 'max_tokens' in cfg:
            max_tok = _to_optional_int(cfg['max_tokens'])
            if max_tok is not None:
                cfg['max_tokens'] = max_tok
            else:
                cfg.pop('max_tokens', None)

        if 'allow_insecure' in cfg:
            cfg['allow_insecure'] = _to_bool(cfg['allow_insecure'])

        if 'debug' in cfg:
            cfg['debug'] = _to_bool(cfg['debug'])

        if 'language' in cfg and isinstance(cfg['language'], str):
            cfg['language'] = cfg['language'].strip().lower()

        if 'timeout' in cfg:
            timeout_val = _to_optional_float(cfg['timeout'])
            if timeout_val is not None and timeout_val > 0:
                cfg['timeout'] = timeout_val
            else:
                cfg.pop('timeout', None)

        if 'http_retries' in cfg:
            retries_val = _to_optional_int(cfg['http_retries'])
            if retries_val is not None and retries_val >= 0:
                cfg['http_retries'] = retries_val
            else:
                cfg.pop('http_retries', None)

        if 'retry_backoff' in cfg:
            backoff_val = _to_optional_float(cfg['retry_backoff'])
            if backoff_val is not None and backoff_val >= 0:
                cfg['retry_backoff'] = backoff_val
            else:
                cfg.pop('retry_backoff', None)

        if 'connect_timeout' in cfg:
            connect_val = _to_optional_float(cfg['connect_timeout'])
            if connect_val is not None and connect_val > 0:
                cfg['connect_timeout'] = connect_val
            else:
                cfg.pop('connect_timeout', None)

        if 'read_timeout' in cfg:
            read_val = _to_optional_float(cfg['read_timeout'])
            if read_val is not None and read_val > 0:
                cfg['read_timeout'] = read_val
            else:
                cfg.pop('read_timeout', None)

        if 'api_key' in cfg and 'token' not in cfg:
            cfg['token'] = cfg['api_key']
        if 'token' in cfg and 'api_key' not in cfg:
            cfg['api_key'] = cfg['token']

        return cfg

    @staticmethod
    def _load_config_file(config_path: Optional[str]) -> Dict[str, Any]:
        path_hint = config_path or os.environ.get('LLM_CONFIG_PATH')
        if not path_hint:
            path_hint = os.path.join('config', 'llm.yaml')

        path = Path(path_hint)
        if not path.exists():
            return {}

        try:
            if path.suffix.lower() in ('.yaml', '.yml'):
                with path.open('r', encoding='utf-8') as handle:
                    data = yaml.safe_load(handle) or {}
            else:
                with path.open('r', encoding='utf-8') as handle:
                    data = json.load(handle)
            return data if isinstance(data, dict) else {}
        except Exception as exc:
            log.warning(f'Failed to load LLM config from {path}: {exc}')
            return {}

    @staticmethod
    def _normalize_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(cfg or {})

        def _normalize_float(key: str):
            if key in normalized and not isinstance(normalized[key], (int, float)):
                try:
                    normalized[key] = float(normalized[key])
                except Exception:
                    normalized.pop(key, None)

        def _normalize_int(key: str):
            if key in normalized and not isinstance(normalized[key], int):
                try:
                    normalized[key] = int(normalized[key])
                except Exception:
                    normalized.pop(key, None)

        def _normalize_bool(key: str):
            if key in normalized and not isinstance(normalized[key], bool):
                val = str(normalized[key]).strip().lower()
                normalized[key] = val in {'1', 'true', 'yes', 'y', 'on'}

        if 'language' in normalized and isinstance(normalized['language'], str):
            normalized['language'] = normalized['language'].strip().lower()

        _normalize_float('temperature')
        _normalize_int('max_tokens')
        _normalize_int('max_completion_tokens')
        _normalize_bool('allow_insecure')
        _normalize_bool('debug')
        _normalize_float('timeout')
        _normalize_int('http_retries')
        _normalize_float('retry_backoff')
        _normalize_float('connect_timeout')
        _normalize_float('read_timeout')

        if 'api_key' in normalized and 'token' not in normalized:
            normalized['token'] = normalized['api_key']
        if 'token' in normalized and 'api_key' not in normalized:
            normalized['api_key'] = normalized['token']

        return normalized

    @staticmethod
    def _apply_env_aliases() -> None:
        alias_groups = [
            ['LLM_ENDPOINT', 'MODEL_ENDPOINT', 'LLM_API_URL'],
            ['LLM_API_KEY', 'OPENAI_API_KEY', 'API_KEY', 'MODEL_BEARER_TOKEN'],
            ['LLM_TOKEN', 'MODEL_TOKEN'],
            ['LLM_MODEL', 'MODEL_NAME', 'MODEL'],
            ['LLM_DEPLOYMENT', 'MODEL_DEPLOYMENT'],
            ['LLM_API_VERSION', 'MODEL_API_VERSION'],
            ['LLM_TEMPERATURE', 'MODEL_TEMPERATURE'],
            ['LLM_MAX_TOKENS', 'MODEL_MAX_TOKENS'],
            ['LLM_MAX_COMPLETION_TOKENS', 'MODEL_MAX_COMPLETION_TOKENS'],
            ['LLM_LANGUAGE', 'LLM_LANG', 'MODEL_LANGUAGE', 'MODEL_LANG'],
            ['LLM_ALLOW_INSECURE', 'MODEL_ALLOW_INSECURE'],
            ['LLM_DEBUG', 'MODEL_DEBUG'],
            ['LLM_CA_BUNDLE', 'MODEL_CA_BUNDLE'],
        ]

        for group in alias_groups:
            value = None
            for key in group:
                candidate = os.environ.get(key)
                if candidate not in (None, ''):
                    value = candidate
                    break
            if value is None:
                continue
            for key in group:
                if os.environ.get(key) in (None, ''):
                    os.environ[key] = value

    @classmethod
    def from_env(cls, config_path: Optional[str] = None):
        dotenv_hint = config_path if config_path and str(config_path).lower().endswith('.env') else None
        cls._load_dotenv_once(dotenv_hint)
        cls._apply_env_aliases()

        file_cfg = cls._load_config_file(config_path)
        env_cfg = cls._collect_env_overrides()

        merged: Dict[str, Any] = {}
        merged.update(file_cfg)
        merged.update(env_cfg)

        return cls(cls._normalize_config(merged), config_path=config_path)

    def _resolve_from_env_and_file(self, config_path: Optional[str] = None):
        dotenv_hint = config_path if config_path and str(config_path).lower().endswith('.env') else None
        self._load_dotenv_once(dotenv_hint)
        self._apply_env_aliases()

        file_cfg = self._load_config_file(config_path)
        env_cfg = self._collect_env_overrides()

        merged: Dict[str, Any] = {}
        merged.update(file_cfg)
        merged.update(env_cfg)
        merged.update(self.config)

        self.config = self._normalize_config(merged)

    def call_llm(self, prompt, override_config: Optional[Dict[str, Any]] = None) -> str:
        """Call the LLM.

        `prompt` may be either a string (legacy) or a list of messages like
        [{'role': 'system','content':...}, {'role':'assistant',...}, {'role':'user',...}].
        `override_config` can pass temporary options like `temperature`/`max_tokens`.
        """
        # merge instance config with any override for this call
        cfg: Dict[str, Any] = dict(self.config or {})
        if override_config:
            try:
                cfg.update(override_config)
            except Exception:
                pass

        endpoint = cfg.get('endpoint')
        token = cfg.get('token') or cfg.get('api_key')
        model_name = cfg.get('model', 'gpt-4.1')

        log.debug(f'call_llm start: endpoint={bool(endpoint)}, model={model_name}, token_set={bool(token)}')
        if not endpoint or not token:
            log.info('No endpoint or token configured; falling back to SDK call')
            # attempt SDK path
            return self.call_openai_sdk(prompt, override_config=cfg)

        is_azure = 'openai.azure.com' in endpoint

        if is_azure:
            # For Azure, URL path must reference the deployment name. Prefer
            # an explicit 'deployment' config (e.g., MODEL_DEPLOYMENT). If not
            # present, fall back to the configured 'model'.
            deployment = self.config.get('deployment') or self.config.get('model') or model_name
            api_version = self.config.get('api_version', '2025-04-01-preview')
            url = f"{endpoint.rstrip('/')}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
            headers = { 'Content-Type': 'application/json', 'api-key': token }
            # log chosen deployment vs model for diagnostics
            try:
                log.debug(f'Azure request: deployment={deployment}, model={model_name}, api_version={api_version}')
            except Exception:
                pass
        else:
            if not endpoint.endswith('/chat/completions'):
                if endpoint.endswith('/v1'):
                    url = f"{endpoint.rstrip('/')}/chat/completions"
                else:
                    url = f"{endpoint.rstrip('/')}/v1/chat/completions"
            else:
                url = endpoint
            # This provider requires Bearer token authentication
            headers = { 'Content-Type': 'application/json', 'Authorization': f"Bearer {token}" }


        # Build messages payload: support both legacy string prompt and list-of-messages
        if isinstance(prompt, list):
            # assume prompt is already a list of message dicts
            messages = prompt
        else:
            messages = [ {'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': prompt} ]


        payload = {
            'model': model_name,
            'messages': messages,
            'temperature': cfg.get('temperature', 0.7),
            'stream': False
        }

        max_completion_tokens = cfg.get('max_completion_tokens')
        max_tokens = cfg.get('max_tokens')
        if max_completion_tokens is not None:
            payload['max_completion_tokens'] = max_completion_tokens
        elif max_tokens is not None:
            payload['max_tokens'] = max_tokens
        else:
            payload['max_tokens'] = 4096  # preserve previous default upper bound

        verify_ssl = not bool(self.config.get('allow_insecure', False))

        timeout_base = cfg.get('timeout') or self.config.get('timeout') or 60.0
        connect_timeout = cfg.get('connect_timeout') or self.config.get('connect_timeout')
        read_timeout = cfg.get('read_timeout') or self.config.get('read_timeout')
        if connect_timeout or read_timeout:
            timeout_setting: Union[float, tuple] = (
                float(connect_timeout or timeout_base),
                float(read_timeout or timeout_base)
            )
        else:
            timeout_setting = float(timeout_base)

        max_attempts = max(1, int(cfg.get('http_retries') or self.config.get('http_retries') or 1))
        backoff_seconds = float(cfg.get('retry_backoff') or self.config.get('retry_backoff') or 2.0)

        def _post_payload(current_payload: Dict[str, Any]):
            masked_headers = {}
            for hk, hv in headers.items():
                mask = ('key' in hk.lower()) or ('token' in hk.lower()) or (hk.lower() == 'authorization')
                masked_headers[hk] = '<masked>' if mask else hv
            log.debug(f'HTTP LLM POST url={url} headers={masked_headers} payload_keys={list(current_payload.keys())} verify_ssl={verify_ssl}')
            log.debug(f'HTTP LLM POST payload sample: {json.dumps(current_payload, ensure_ascii=False)[:500]}')
            response = requests.post(url, json=current_payload, headers=headers, timeout=timeout_setting, verify=verify_ssl)
            log.debug(f'HTTP response status={response.status_code}')
            response.raise_for_status()
            return response

        parsed = None
        error: Optional[Exception] = None

        attempt = 0
        while attempt < max_attempts:
            attempt += 1
            try:
                resp = _post_payload(payload)
                parsed = resp.json()
                log.debug(f'HTTP response parsed type={type(parsed)}')
                error = None
                break
            except Exception as exc:
                error = exc
                if attempt < max_attempts:
                    try:
                        from requests.exceptions import ReadTimeout, ConnectTimeout
                        if isinstance(exc, (ReadTimeout, ConnectTimeout)):
                            log.warning(f'HTTP LLM attempt {attempt} timed out; retrying after {backoff_seconds} seconds')
                            time.sleep(backoff_seconds)
                            continue
                    except Exception:
                        pass
                break

        if error is not None:
            from requests.exceptions import HTTPError  # safe: requests dependency is available when HTTP path is used

            if isinstance(error, HTTPError):
                response_obj = getattr(error, 'response', None)
                status_code = getattr(response_obj, 'status_code', None)
                try:
                    body_text = response_obj.text if response_obj is not None else ''
                except Exception:
                    body_text = ''

                if status_code == 400 and body_text and "'max_tokens'" in body_text and 'max_completion_tokens' in body_text:
                    fallback_payload = dict(payload)
                    fallback_payload.pop('max_tokens', None)
                    fallback_payload['max_completion_tokens'] = (
                        cfg.get('max_completion_tokens')
                        or cfg.get('max_tokens')
                        or 4096
                    )
                    log.info('HTTP provider rejected max_tokens; retrying with max_completion_tokens')
                    log.debug(f'Retrying HTTP LLM POST with payload keys={list(fallback_payload.keys())}')
                    try:
                        resp = _post_payload(fallback_payload)
                        parsed = resp.json()
                        log.debug(f'HTTP response parsed type={type(parsed)}')
                        error = None
                    except Exception as retry_exc:
                        error = retry_exc

        if error is not None:
            # Handle common requests timeouts/connection errors with clearer message
            try:
                from requests.exceptions import ReadTimeout, ConnectTimeout, RequestException
                if isinstance(error, ReadTimeout):
                    log.error(f'HTTP LLM call timed out contacting {url}: {error}')
                    raise RuntimeError(f'LLM HTTP request timed out: {error}')
                if isinstance(error, ConnectTimeout):
                    log.error(f'HTTP LLM connection timed out contacting {url}: {error}')
                    raise RuntimeError(f'LLM HTTP connection timed out: {error}')
                if isinstance(error, RequestException):
                    # If Azure returns 401, provide actionable hint about deployment
                    status = getattr(error, 'response', None)
                    if status is not None:
                        try:
                            sc = int(status.status_code)
                        except Exception:
                            sc = None
                        try:
                            body = status.text[:2000]
                        except Exception:
                            body = '<unreadable-response-body>'
                    else:
                        sc = None
                        body = None
                    if sc == 401:
                        log.error(f'HTTP LLM request failed with 401 for URL {url}: {error}')
                        log.error('401 from Azure often means the deployment name is incorrect or the key lacks permission. Confirm MODEL_DEPLOYMENT matches a valid deployment and MODEL_BEARER_TOKEN / OPENAI_API_KEY is correct.')
                        if body:
                            log.error(f'Response body: {body}')
                    else:
                        log.error(f'HTTP LLM request failed contacting {url}: {error} (status={sc})')
                        if body:
                            log.error(f'Response body: {body}')
                    raise RuntimeError(f'LLM HTTP request failed: {error}')
            except Exception:
                # fallback for when requests.exceptions cannot be imported
                log.error(f'HTTP LLM request raised exception contacting {url}: {error}')
                raise RuntimeError(f'LLM HTTP request failed: {error}')

        if isinstance(parsed, dict):
            choices = parsed.get('choices') or parsed.get('outputs') or []
            if isinstance(choices, list) and choices:
                first = choices[0]
                message = first.get('message') or first.get('delta') or {}
                if isinstance(message, dict):
                    content = message.get('content') or (message.get('content', {}).get('parts') if isinstance(message.get('content', {}), dict) else None)
                    if isinstance(content, list) and content:
                        return content[0]
                    if isinstance(content, str):
                        return content
                text = first.get('text')
                if text:
                    return text

        return json.dumps(parsed, ensure_ascii=False)


    def call_openai_sdk(self, prompt, override_config: Optional[Dict[str, Any]] = None) -> str:
        api_key = (override_config or self.config).get('api_key')
        if not api_key:
            raise RuntimeError('No API key configured for OpenAI SDK')
        cfg = dict(self.config or {})
        if override_config:
            cfg.update(override_config)
        model = cfg.get('model', 'gpt-4.1')
        temperature = cfg.get('temperature', 0.7)
        max_tokens = cfg.get('max_tokens', 500)

        log.debug(f'SDK call_openai_sdk start: model={model}, temperature={temperature}, max_tokens={max_tokens}')
        try:
            try:
                from openai import OpenAI
                log.debug('Using modern openai.OpenAI client')
                client = OpenAI(api_key=api_key)
                # support both string prompt and messages list
                if isinstance(prompt, list):
                    messages = prompt
                else:
                    messages = [{'role': 'user', 'content': prompt}]
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                content = response.choices[0].message.content
                log.debug('SDK response length=%s', len(str(content)))
                return content
            except ImportError:
                try:
                    import openai
                except Exception:
                    log.error('openai SDK is not installed')
                    raise RuntimeError('openai package is not installed for SDK-based calls')
                log.debug('Using legacy openai package')
                openai.api_key = api_key
                if isinstance(prompt, list):
                    messages = prompt
                else:
                    messages = [{'role': 'user', 'content': prompt}]
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                first = response.choices[0]
                if hasattr(first, 'message'):
                    content = first.message.content
                else:
                    content = first.get('text') or str(first)
                log.debug('Legacy SDK response length=%s', len(str(content)))
                return content
        except Exception as e:
            log.error(f'OpenAI SDK call failed: {e}')
            raise RuntimeError(f'OpenAI SDK call failed: {e}')


    def parse_llm_response(self, response: str) -> Dict[str, Any]:
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end != -1 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
        except Exception:
            pass
        return self.parse_fallback(response)


    def parse_fallback(self, response: str) -> Dict[str, Any]:
        lines = response.split('\n')
        task_data: Dict[str, Any] = {}
        for line in lines:
            low = line.lower()
            if 'query' in low and ':' in line:
                task_data['query'] = line.split(':', 1)[1].strip().strip('"')
            elif 'agents' in low and ':' in line:
                agents_str = line.split(':', 1)[1].strip().strip('[]"')
                task_data['expected_subagents'] = [t.strip() for t in agents_str.split(',') if t.strip()]
        return task_data


# Compatibility helpers: keep module-level functions that delegate to a default client
_DEFAULT_CLIENT: Optional[LLMClient] = None

def get_default_client() -> LLMClient:
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is None:
        _DEFAULT_CLIENT = LLMClient.from_env()
    return _DEFAULT_CLIENT


def call_llm(prompt: str, llm_config: Dict[str, Any]) -> str:
    client = LLMClient(llm_config) if llm_config else get_default_client()
    return client.call_llm(prompt)


def call_openai_sdk(prompt: str, llm_config: Dict[str, Any]) -> str:
    client = LLMClient(llm_config) if llm_config else get_default_client()
    return client.call_openai_sdk(prompt)


def parse_llm_response(response: str) -> Dict[str, Any]:
    return get_default_client().parse_llm_response(response)


def parse_fallback(response: str) -> Dict[str, Any]:
    return get_default_client().parse_fallback(response)
