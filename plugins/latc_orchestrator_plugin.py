import json
import time
from typing import Generator, Dict, Any, List
import requests
from .specs import hookimpl
from .invocation_logger import log_invocation
import re
import string


def _clean_merged_text(text: str) -> str:
    """Normalize merged stream text:
    - collapse newlines and excessive whitespace
    - remove adjacent duplicate tokens (case- and punctuation-insensitive)
    - normalize spacing around punctuation
    """
    if not text or not isinstance(text, str):
        return text
    # collapse newlines to spaces and normalize whitespace
    s = text.replace('\n', ' ')
    s = re.sub(r'\s+', ' ', s).strip()

    # split into tokens (words or standalone punctuation)
    try:
        tokens = re.findall(r"\w+|[^\w\s]", s, flags=re.UNICODE)
    except Exception:
        tokens = s.split()

    dedup = []
    prev_norm = None
    for tok in tokens:
        # normalize token for comparison: drop non-word chars and lower
        norm = re.sub(r'[^\w]', '', tok, flags=re.UNICODE).lower()
        if norm and prev_norm == norm:
            # adjacent duplicate word -> skip
            continue
        if not norm and dedup and dedup[-1] == tok:
            # adjacent duplicate punctuation -> skip
            continue
        dedup.append(tok)
        if norm:
            prev_norm = norm

    result = ' '.join(dedup)
    # remove space before common punctuation
    result = re.sub(r'\s+([,.;:!?])', r'\1', result)
    # collapse repeated punctuation
    result = re.sub(r'([,.;:!?])\1+', r'\1', result)

    # Sentence/paragraph-level deduplication: split into sentences (by
    # sentence-ending punctuation or newline) and collapse adjacent
    # identical segments. This removes cases like
    # "...search results. ...search results." or repeated blocks.
    try:
        # split on Chinese/English sentence enders or on multiple newlines
        segs = re.split(r'(?<=[。！？.!?])\s+|\n+', result)
        cleaned_segs = []
        prev = None
        for s in segs:
            s_strip = s.strip()
            if not s_strip:
                continue
            if prev is not None and s_strip == prev:
                # skip repeated adjacent segment
                continue
            cleaned_segs.append(s_strip)
            prev = s_strip
        result = ' '.join(cleaned_segs)
    except Exception:
        pass

    return result


class LatcOrchestratorClient:
    def __init__(self, url: str):
        self.url = url
        self.session = requests.Session()

    def post_stream(self, payload: Dict[str, Any], timeout: int = 60):
        return self.session.post(
            self.url, 
            json=payload, 
            headers={"Content-Type": "application/json"}, 
            stream=True, 
            timeout=timeout
        )


@hookimpl
def get_supported_orchestrators():
    return ["LATC"]


@hookimpl
def create_orchestrator_client(orchestrator_type: str, config: Dict[str, Any]):
    if not orchestrator_type or orchestrator_type.lower() not in ("latc", "latc_orchestrator"):
        return None
    
    url = config.get("url") or config.get("orchestrator_url")
    if not url:
        raise ValueError("LATC orchestrator plugin requires 'url' in config")
    return LatcOrchestratorClient(url)


def _extract_text_from_part(part: Any, out: List[str]):
    """从多种格式的数据中提取文本"""
    if not part:
        return
    
    if isinstance(part, str):
        # avoid appending the same string twice in a row
        if not out or out[-1] != part:
            out.append(part)
        return
    
    data = part.get("data") or part
    if isinstance(data, dict):
        for key in ("text", "content", "body", "message"):
            val = data.get(key)
            if isinstance(val, str) and val.strip():
                sval = val.strip()
                if not out or out[-1] != sval:
                    out.append(sval)
        
        parts = data.get("parts") or data.get("children") or data.get("segments")
        if isinstance(parts, list):
            for p in parts:
                _extract_text_from_part(p, out)
    
    for key in ("text", "content", "body", "message"):
        val = part.get(key)
        if isinstance(val, str) and val.strip():
            sval = val.strip()
            if not out or out[-1] != sval:
                out.append(sval)


@hookimpl
def call_orchestrator_stream(orchestrator_client: LatcOrchestratorClient, 
                           payload: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
    """调用LATC orchestrator并解析流式响应"""
    try:
        if hasattr(orchestrator_client, 'post_stream'):
            resp = orchestrator_client.post_stream(payload)
        elif hasattr(orchestrator_client, 'call'):
            resp = orchestrator_client.call(payload)
        elif hasattr(orchestrator_client, 'post'):
            resp = orchestrator_client.post(payload)
        else:
            url = getattr(orchestrator_client, 'url', None)
            if not url:
                raise AttributeError("orchestrator_client缺少必要的调用方法")
            resp = requests.post(url, json=payload, 
                               headers={"Content-Type": "application/json"}, 
                               stream=True, timeout=60)
    except Exception as e:
        yield {"error": "request_failed", "message": str(e)}
        return

    # 处理HTTP错误
    if hasattr(resp, 'status_code') and resp.status_code != 200:
        try:
            yield {"error": f"http_{resp.status_code}", "body": resp.text}
        except:
            yield {"error": f"http_{resp.status_code}", "body": None}
        return

    # 处理流式响应
    buffer = ""
    
    if hasattr(resp, 'iter_lines'):
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw:
                continue
            
            line = raw.strip()
            if line.startswith("data:"):
                data = line[5:].strip()
                yield from _parse_json_data(data, buffer)
                buffer = ""
            else:
                try:
                    yield json.loads(line)
                except:
                    buffer += line
    elif isinstance(resp, dict):
        yield resp
    elif hasattr(resp, '__iter__') and not isinstance(resp, (str, bytes)):
        for raw in resp:
            if isinstance(raw, dict):
                yield raw
            elif isinstance(raw, (bytes, bytearray)):
                try:
                    line = raw.decode('utf-8').strip()
                    yield from _parse_json_data(line, buffer)
                except:
                    yield {"text": str(raw)}
    else:
        try:
            yield json.loads(resp.text)
        except:
            yield {"text": str(resp)}


def _parse_json_data(data: str, buffer: str = ""):
    """解析JSON数据，支持缓冲"""
    try:
        yield json.loads(data)
    except:
        buffer += data
        try:
            yield json.loads(buffer)
        except:
            pass


@hookimpl
def process_stream_response(orchestrator_client: LatcOrchestratorClient, 
                          stream_generator: Generator[Dict[str, Any], None, None]) -> Dict[str, Any]:
    """处理流式响应并规范化输出"""
    status = None
    artifacts = []
    text_parts = []
    raw_chunks = []

    for chunk in stream_generator:
        raw_chunks.append(chunk)
        if not isinstance(chunk, dict):
            continue

        # 更新状态
        if chunk.get("type") in ("status-update", "status") or chunk.get("status"):
            status = chunk.get("status") or chunk.get("type")

        # 提取文本和工件
        art = chunk.get("artifact") or chunk.get("artifacts") or chunk.get("data")
        if art:
            if isinstance(art, list):
                artifacts.extend(art)
                for a in art:
                    _extract_text_from_part(a, text_parts)
            else:
                artifacts.append(art)
                _extract_text_from_part(art, text_parts)

        if chunk.get("text"):
            text_parts.append(chunk["text"])

    merged = "\n".join(filter(None, text_parts))
    # Clean merged text: collapse newlines/whitespace and deduplicate adjacent tokens
    try:
        merged = _clean_merged_text(merged)
    except Exception:
        pass

    # 提取元数据
    meta = {}
    for chunk in raw_chunks:
        if not isinstance(chunk, dict):
            continue
        
        for key in ("userId", "sessionId", "transactionId", "stepId", "kind", "status", "subAgent"):
            if key in chunk and key not in meta:
                meta[key] = chunk[key]
        
        if isinstance(chunk.get('metadata'), dict):
            meta.update(chunk['metadata'])

    # 规范化ID字段
    if meta.get('transactionId'):
        meta['transaction_id'] = meta['transactionId']
    if meta.get('stepId'):
        meta['step_id'] = meta['stepId']

    # 构建工件结构
    collected_artifacts = {}
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        
        aid = artifact.get('artifactId') or artifact.get('artifact_id') or f"artifact_{len(collected_artifacts)}"
        entry = collected_artifacts.setdefault(aid, {
            'artifactId': aid,
            'name': artifact.get('name'),
            'text_parts': []
        })
        
        tparts = artifact.get('text_parts') or artifact.get('textParts') or artifact.get('parts') or []
        if isinstance(tparts, list):
            for part in tparts:
                if isinstance(part, str):
                    entry['text_parts'].append(part)
                elif isinstance(part, dict):
                    for key in ('text', 'content', 'body', 'message'):
                        if key in part and isinstance(part[key], str):
                            entry['text_parts'].append(part[key])

    result = {
        "status": status or ("completed" if merged else None),
        "final_output": merged if merged else None,
        "internal_logs": [],
        "transaction_id": meta.get('transaction_id'),
        "step_id": meta.get('step_id'),
        "collected_json": {
            'meta': meta,
            'artifacts': collected_artifacts,
            'final_status': status or ("completed" if merged else None),
            'merged_text': merged
        },
        "raw_chunks": raw_chunks,
    }

    # 检测澄清请求
    clarification = _detect_clarification(raw_chunks, merged)
    if clarification:
        clar_type = 'input-required'
        result.update({
            'clarification_question': clarification,
            'clarification_requested': True,
            'clarification_type': clar_type
        })
        
        if 'meta' in result['collected_json']:
            result['collected_json']['meta'].update({
                'clarification': True,
                'clarification_type': clar_type
            })
        
        _log_clarification(meta, clarification, clar_type)

    return result


def _detect_clarification(chunks: List[Dict], merged_text: str) -> str:
    """检测澄清请求"""
    # 仅在状态严格等于 'input-required' 时触发澄清（其他状态如 'user_query' 或 'input_required' 将被忽略）
    clar_keys = ('input-required',)
    
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue

        status = chunk.get('status') or chunk.get('type') or chunk.get('kind')
        # 精确匹配：只有当 status 的小写形式完全等于 'input-required' 才视作需要用户输入的澄清
        if status and str(status).lower() == 'input-required':
            # 优先返回 chunk 中明确的 final_output/text 或 data.parts 的拼接文本
            question = chunk.get('final_output') or chunk.get('text')
            if not question and isinstance(chunk.get('data'), dict):
                parts = chunk.get('data').get('parts') or []
                texts = []
                for part in parts:
                    if isinstance(part, str):
                        texts.append(part)
                    elif isinstance(part, dict):
                        text = part.get('text') or part.get('content') or part.get('message')
                        if text:
                            texts.append(str(text))
                question = '\n'.join(texts) if texts else None
            return question or merged_text
    
    return None


def _log_clarification(meta: Dict, clarification: str, clar_type: str):
    """记录澄清日志"""
    task_id = meta.get('transaction_id') or meta.get('transactionId')
    try:
        log_invocation({
            'hook': 'process_stream_response',
            'plugin': 'latc_orchestrator_plugin',
            'clarification_sample': str(clarification)[:200],
            'clarification_type': clar_type,
            'message': 'LATC plugin detected clarification request',
            'transaction_id': meta.get('transaction_id'),
            'step_id': meta.get('step_id'),
        }, task_id=task_id)
    except:
        pass


def _normalize_query(raw: Any) -> str:
    """规范化查询文本"""
    if raw is None:
        return ''
    
    if isinstance(raw, dict):
        if 'value' in raw:
            return _normalize_query(raw.get('value'))
        try:
            return json.dumps(raw, ensure_ascii=False)
        except:
            return str(raw)
    
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = raw.decode('utf-8')
        except:
            raw = str(raw)
    
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return ''
        
        # 尝试解析JSON
        if (text.startswith('{') and text.endswith('}')) or (text.startswith('[') and text.endswith(']')):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict) and 'value' in parsed:
                    return str(parsed['value'])
                elif isinstance(parsed, str):
                    return parsed
                elif isinstance(parsed, dict):
                    parts = [f"{k}: {v}" for k, v in parsed.items()]
                    return '; '.join(parts)
                elif isinstance(parsed, list):
                    return ', '.join(str(x) for x in parsed)
                else:
                    return json.dumps(parsed, ensure_ascii=False)
            except:
                return text
        
        return text
    
    return str(raw)


@hookimpl
def create_orchestrator_payload(orchestrator_client: LatcOrchestratorClient, 
                              conversation_state: dict, current_query: str, 
                              is_initial: bool) -> Dict[str, Any]:
    """创建LATC兼容的请求载荷"""
    # 获取历史记录
    history = []
    raw_hist = []
    
    if hasattr(conversation_state, 'history'):
        raw_hist = getattr(conversation_state, 'history') or []
    elif isinstance(conversation_state, dict):
        raw_hist = conversation_state.get('history') or []
    
    for entry in raw_hist:
        if not isinstance(entry, dict):
            history.append({'role': 'user', 'content': _normalize_query(entry)})
            continue
        
        normalized = dict(entry)
        normalized['content'] = _normalize_query(entry.get('content'))
        normalized.setdefault('role', 'user')
        history.append(normalized)

    # 构建基础载荷
    user_id = (getattr(conversation_state, 'user_id', None) or 
               conversation_state.get('user_id') if isinstance(conversation_state, dict) else None or 
               'benchmark_user')
    
    session_id = (getattr(conversation_state, 'session_id', None) or 
                  conversation_state.get('session_id') if isinstance(conversation_state, dict) else None or 
                  'benchmark_session')

    lang_value = getattr(conversation_state, 'lang', None)
    if lang_value is None and isinstance(conversation_state, dict):
        lang_value = conversation_state.get('lang')

    payload = {
        'userId': user_id,
        'sessionId': session_id,
        'messageId': f"msg_{int(time.time())}",
        'history': history,
        'query': _normalize_query(current_query),
        'is_initial': bool(is_initial),
        'lang': lang_value or 'en'
    }

    # 添加历史记录ID
    for entry in payload['history']:
        if isinstance(entry, dict):
            if user_id and 'userId' not in entry:
                entry['userId'] = user_id
            if session_id and 'sessionId' not in entry:
                entry['sessionId'] = session_id

    # 添加事务ID（如果是后续请求）
    if not is_initial:
        txn = None
        stp = None

        if hasattr(conversation_state, 'root_transaction_id'):
            txn = getattr(conversation_state, 'root_transaction_id') or getattr(conversation_state, 'transaction_id', None)
        elif isinstance(conversation_state, dict):
            txn = conversation_state.get('root_transaction_id') or conversation_state.get('transaction_id')

        if hasattr(conversation_state, 'step_id'):
            stp = getattr(conversation_state, 'step_id')
        elif isinstance(conversation_state, dict):
            stp = conversation_state.get('step_id')

        if txn:
            payload['transactionId'] = txn
        if stp:
            payload['stepId'] = stp

    return payload


def register_plugin(pm):
    """注册插件（可选）"""
    try:
        import sys
        module = sys.modules.get(__name__)
        if module:
            pm.register(module)
            log_invocation({
                'hook': 'register_plugin', 
                'plugin': 'latc_orchestrator_plugin',
                'message': '插件注册成功'
            }, task_id=None)
    except:
        pass