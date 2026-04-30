from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional
import json
import os
import traceback
from datasets.llm_client import LLMClient
from plugins.manager import global_plugin_manager
from ..utils import console as console_utils


class BaseUserSimulator(ABC):
    def __init__(self, user_profile: str = "profile_001", llm_config: Dict = None, 
                 lang: str = 'zh', plugin_call_timeout: float = 0.5, 
                 profile_path: Optional[str] = None, domain: str = 'travel'):
        
        self.conversation_history = []
        self.lang = str(lang).lower() if lang else 'zh'
        self.profile_path = profile_path
        self.domain = domain
        
        # Get domain plugin via plugin manager
        self.plugin_manager = global_plugin_manager
        self.domain_plugin = self._get_domain_plugin()
        
        # LLM client initialization
        if llm_config:
            try:
                self.llm_config = llm_config
                self._llm_client = LLMClient(self.llm_config)
            except Exception:
                self._llm_client = LLMClient.from_env()
                self.llm_config = getattr(self._llm_client, 'config', {}) or {}
        else:
            try:
                self._llm_client = LLMClient.from_env()
                self.llm_config = getattr(self._llm_client, 'config', {}) or {}
            except Exception:
                self._llm_client = None
                self.llm_config = {}
        
        try:
            self.plugin_call_timeout = float(plugin_call_timeout or 0.5)
        except Exception:
            self.plugin_call_timeout = 0.5
        
        # Load user profile via plugin
        self.profile = self._load_profile(user_profile)

    def _get_domain_plugin(self):
        """Get domain plugin via plugin manager"""
        return self.plugin_manager.get_domain_plugin(self.domain)

    def _load_profile(self, profile_name: str) -> Dict:
        """Load user profile via plugin"""
        # Try plugin first
        plugin_profile = self.plugin_manager.get_domain_profile(self.domain, profile_name)
        if plugin_profile:
            return plugin_profile
        
        # Fallback to local config
        try:
            import yaml
            path = getattr(self, 'profile_path', None) or 'config/user_profiles.yaml'
            with open(path, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f) or {}
                users = cfg.get('user_profiles') or cfg.get('profiles') or {}
                
                if profile_name in users:
                    return users[profile_name]
                if 'profile_001' in users:
                    return users['profile_001']
        except Exception:
            pass
        
        # Default base profile
        return {
            "name": "Standard User",
            "traits": [],
            "communication_style": "natural",
            "typical_responses": {}
        }

    @abstractmethod
    def generate_clarification_response(self, question: str, context: Dict) -> str:
        """Generate clarification response (must be implemented by subclass)"""
        raise NotImplementedError()

    def get_timestamp(self) -> str:
        """Get current timestamp"""
        return datetime.now().isoformat()

    def record_interaction(self, role: str, content: str, context: Dict = None):
        """Record interaction history and print to console (with simple suppression)"""
        entry = {
            'role': role,
            'content': content,
            'timestamp': self.get_timestamp(),
            'context': context or {}
        }
        # Append to history first so index/timestamp reflect final state
        self.conversation_history.append(entry)

        # Determine whether to print this entry to console
        try:
            round_no = None
            if isinstance(context, dict) and 'round' in context:
                round_no = context.get('round')

            idx = len(self.conversation_history)
            ts = entry.get('timestamp')

            status = None
            sub_agent = None
            # try to extract JSON-like status information if available
            try:
                if isinstance(content, str) and hasattr(self, '_extract_json_from_text'):
                    parsed = self._extract_json_from_text(content)
                    if isinstance(parsed, dict):
                        status = status or parsed.get('status') or parsed.get('agentStatus')
                        sub_agent = sub_agent or parsed.get('subAgentName') or parsed.get('sub_agent_name') or parsed.get('agent')
            except Exception:
                status = status

            should_print = True
            try:
                if isinstance(role, str) and role.lower() in ('assistant', 'agent'):
                    cstr = str(content or '').strip()
                    # suppress very short fragments (likely per-token pieces)
                    if len(cstr) <= 2 and (' ' not in cstr):
                        should_print = False
            except Exception:
                should_print = True

            if should_print:
                console_utils.print_conv_entry(idx, ts, role, content, round_no, status=status, sub_agent=sub_agent)
        except Exception:
            # best-effort fallback printing
            try:
                print(f"[CONV] {role}: {content}")
            except Exception:
                pass
    def clear_history(self):
        """Clear conversation history"""
        try:
            self.conversation_history = []
        except Exception:
            self.conversation_history = []
