import os
import sys
from typing import Optional

# Simple cross-platform console color helper. Uses ANSI escapes when available.
# Enable with environment variable `COGBENCH_COLOR=1` (default: enabled if stdout is a tty).

def _supports_color() -> bool:
    env = os.environ.get('COGBENCH_COLOR')
    if env is not None:
        return env.strip() not in ('0', 'false', 'False')
    try:
        return sys.stdout.isatty()
    except Exception:
        return False

USE_COLOR = _supports_color()

RESET = '\u001b[0m' if USE_COLOR else ''
BOLD = '\u001b[1m' if USE_COLOR else ''

# Foreground colors
FG_RED = '\u001b[31m' if USE_COLOR else ''
FG_GREEN = '\u001b[32m' if USE_COLOR else ''
FG_YELLOW = '\u001b[33m' if USE_COLOR else ''
FG_BLUE = '\u001b[34m' if USE_COLOR else ''
FG_MAGENTA = '\u001b[35m' if USE_COLOR else ''
FG_CYAN = '\u001b[36m' if USE_COLOR else ''
FG_WHITE = '\u001b[37m' if USE_COLOR else ''


def color_text(text: str, color: str = '', bold: bool = False) -> str:
    if not USE_COLOR or not color:
        return text
    prefix = ''
    if bold:
        prefix += BOLD
    prefix += color
    return f"{prefix}{text}{RESET}"


# Control printing of verbose LLM request/response interaction logs.
# Set environment variable `COGBENCH_PRINT_LLM=0` to disable these prints.
VERBOSE_LLM_INTERACTION = os.environ.get('COGBENCH_PRINT_LLM', '1').strip() not in ('0', 'false', 'False')


def print_conv_entry(idx: int, timestamp: str, role: str, content: str, round_no: Optional[int] = None, status: Optional[str] = None, sub_agent: Optional[str] = None, origin: Optional[str] = None):
    """打印会话条目：索引、时间、角色、轮次、内容（重点信息着色）"""
    role_label = role.upper()
    if role_label == 'USER':
        role_col = FG_CYAN
    elif role_label == 'ASSISTANT' or role_label == 'AGENT':
        role_col = FG_GREEN
    else:
        role_col = FG_MAGENTA

    round_part = f" {color_text('[round:'+str(round_no)+']', FG_YELLOW, bold=True)}" if round_no is not None else ''
    header = f"[CONV][#{idx}][{timestamp}]" + round_part + ' '
    header_colored = color_text(header, FG_WHITE, bold=True)
    role_colored = color_text(f"{role_label}:", role_col, bold=True)

    # 状态和子智能体高亮
    status_part = ''
    if status:
        st = str(status).lower()
        if st == 'completed':
            status_col = FG_GREEN
        elif st == 'input-required':
            status_col = FG_YELLOW
        elif st == 'next-step':
            status_col = FG_CYAN
        elif st == 'error':
            status_col = FG_RED
        elif st == 'pending':
            status_col = FG_MAGENTA
        else:
            status_col = FG_MAGENTA
        status_part = f" {color_text('['+str(status)+']', status_col, bold=True)}"

    sub_agent_part = f" {color_text('('+str(sub_agent)+')', FG_MAGENTA, bold=False)}" if sub_agent else ''
    origin_part = ''
    if origin:
        origin_part = f" {color_text('[origin:'+str(origin)+']', FG_MAGENTA, bold=False)}"

    # Shorten long content for header but still print full content in next line
    preview = content if len(content) <= 200 else content[:197] + '...'
    print(f"{header_colored}{status_part}{sub_agent_part}{origin_part} {role_colored} {preview}")
    # Only print full multi-line content when verbose LLM interaction logging is enabled
    if len(preview) != len(content) and VERBOSE_LLM_INTERACTION:
        print(color_text('  (full content):', FG_BLUE, bold=False) + ' ' + content)


def print_llm_interaction(timestamp: str, messages, response, metadata=None):
    if not VERBOSE_LLM_INTERACTION:
        return
    ts = timestamp
    print(color_text(f"[LLM][{ts}] ↗ Request:", FG_YELLOW, bold=True))
    for m in messages:
        role = m.get('role', 'user')
        content = m.get('content', '')
        role_col = FG_CYAN if role == 'user' else FG_GREEN
        print(f"  {color_text(role.upper()+':', role_col, bold=False)} {content}")
    print(color_text(f"[LLM][{ts}] ↘ Response:", FG_YELLOW, bold=True))
    print(color_text(str(response), FG_WHITE, bold=False))
    if metadata:
        print(color_text(f"[LLM][{ts}] meta: {metadata}", FG_MAGENTA, bold=False))


def print_info(msg: str):
    print(color_text(msg, FG_BLUE, bold=True))


def print_warning(msg: str):
    print(color_text(msg, FG_YELLOW, bold=True))


def print_error(msg: str):
    print(color_text(msg, FG_RED, bold=True))