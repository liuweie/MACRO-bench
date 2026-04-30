from typing import Dict

from .specs import hookimpl
from cogbenchmark.user_simulator.travel_simulator import TravelUserSimulator


class TravelDomainPlugin:
    """Travel domain plugin exposing simulation and profiling hooks."""

    SUPPORTED_DOMAINS = {"travel", "tourism"}

    @hookimpl
    def get_supported_domains(self):
        return list(self.SUPPORTED_DOMAINS)

    @hookimpl
    def create_user_simulator(self, domain: str, config: dict):
        if domain in self.SUPPORTED_DOMAINS:
            return TravelUserSimulator(**config)
        return None

    @hookimpl
    def get_domain_profile(self, domain: str, profile_name: str):
        if domain not in self.SUPPORTED_DOMAINS:
            return None

        profiles: Dict[str, Dict[str, object]] = {
            "profile_001": {
                "name": "商务旅行者",
                "traits": ["注重效率", "预算中等", "喜欢便捷交通", "需要商务设施"],
                "communication_style": "简洁专业",
            },
            "profile_002": {
                "name": "家庭游客",
                "traits": ["注重亲子设施", "预算有限", "喜欢景点游玩", "需要家庭房"],
                "communication_style": "友好细致",
            },
        }
        return profiles.get(profile_name, profiles["profile_001"])

    @hookimpl
    def build_conversation_context(self, domain: str, question: str, context: dict, **kwargs):
        if domain not in self.SUPPORTED_DOMAINS:
            return None

        system_prompt = """你正在模拟一个旅行用户的角色回答助手的澄清问题。"""
        messages = [{"role": "system", "content": system_prompt}]

        conversation_history = context.get("conversation_history", []) if isinstance(context, dict) else []
        for entry in conversation_history[-6:]:
            if entry.get("role") == "assistant":
                messages.append({"role": "assistant", "content": entry.get("content")})
            elif entry.get("role") == "user":
                messages.append({"role": "user", "content": entry.get("content")})

        messages.append({"role": "assistant", "content": question})
        messages.append({"role": "user", "content": "请根据你的角色直接回答这个问题。"})
        return messages

    @hookimpl
    def classify_question(self, domain: str, question: str, lang: str):
        if domain not in self.SUPPORTED_DOMAINS:
            return None

        q_lower = (question or "").lower()
        if any(keyword in q_lower for keyword in ["时间", "日期", "什么时候", "几天"]):
            return "travel_dates"
        if any(keyword in q_lower for keyword in ["预算", "价格", "花费", "多少钱"]):
            return "budget_range"
        if any(keyword in q_lower for keyword in ["哪里", "目的地", "城市", "地方"]):
            return "destination"
        if any(keyword in q_lower for keyword in ["喜欢", "兴趣", "爱好", "想玩"]):
            return "interests"
        return "general"

    @hookimpl
    def generate_candidate_list(self, domain: str, question: str, context: dict, max_candidates: int):
        if domain not in self.SUPPORTED_DOMAINS:
            return None

        import random
        from datetime import datetime, timedelta

        question_type = self.classify_question(domain, question, "zh")
        candidates = []

        if question_type == "travel_dates":
            now = datetime.now()
            for _ in range(min(max_candidates, 3)):
                start = now + timedelta(days=random.randint(7, 30))
                end = start + timedelta(days=random.randint(2, 7))
                candidates.append(f"{start.year}-{start.month:02d}-{start.day:02d} to {end.year}-{end.month:02d}-{end.day:02d}")
        elif question_type == "destination":
            destinations = ["北京", "上海", "杭州", "成都", "西安", "广州", "深圳", "三亚"]
            candidates = random.sample(destinations, min(max_candidates, len(destinations)))
        elif question_type == "budget_range":
            budgets = ["5000元左右", "8000-10000元", "15000元以上", "2000-3000元"]
            candidates = random.sample(budgets, min(max_candidates, len(budgets)))

        return candidates

    @hookimpl
    def contains_specific_dates(self, domain: str, text: str):
        if domain not in self.SUPPORTED_DOMAINS:
            return None

        import re

        date_patterns = [
            r"\d{4}-\d{2}-\d{2}",
            r"\d{1,2}月\d{1,2}日",
            r"\d{1,2}/\d{1,2}/\d{4}",
            r"下周[一二三四五六日]",
            r"下个月\d{1,2}号",
        ]
        text_lower = (text or "").lower()
        return any(re.search(pattern, text_lower) for pattern in date_patterns) or any(
            keyword in text_lower for keyword in ["明天", "后天", "下周", "下个月"]
        )


def register_plugin(pm):
    pm.register(TravelDomainPlugin())
