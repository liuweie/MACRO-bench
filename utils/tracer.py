import json
import uuid
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum


class AgentStatus(Enum):
    """智能体状态枚举"""
    COMPLETED = "completed"
    INPUT_REQUIRED = "input-required"
    NEXT_STEP = "next-step"
    ERROR = "error"
    PENDING = "pending"


@dataclass
class Step:
    """步骤信息"""
    string_value: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {"string_value": self.string_value}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Step':
        return cls(string_value=data.get("string_value", ""))


@dataclass
class Conversation:
    """对话记录"""
    sub_agent_name: str
    response: str
    status: str
    query: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "subAgentName": self.sub_agent_name,
            "response": self.response,
            "status": self.status,
            "query": self.query
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Conversation':
        return cls(
            sub_agent_name=data.get("subAgentName", ""),
            response=data.get("response", ""),
            status=data.get("status", ""),
            query=data.get("query", "")
        )


@dataclass
class InternalRerouting:
    """内部重定向记录"""
    last_sub_agent_name: str
    new_sub_agent_name: str
    last_status: str
    new_status: str
    expired: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "lastSubAgentName": self.last_sub_agent_name,
            "newSubAgentName": self.new_sub_agent_name,
            "lastStatus": self.last_status,
            "newStatus": self.new_status,
            "expired": self.expired
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InternalRerouting':
        return cls(
            last_sub_agent_name=data.get("lastSubAgentName", ""),
            new_sub_agent_name=data.get("newSubAgentName", ""),
            last_status=data.get("lastStatus", ""),
            new_status=data.get("newStatus", ""),
            expired=data.get("expired", 0)
        )


@dataclass
class Span:
    """单个回合的跨度信息"""
    input_query: str
    input_history: Optional[str] = None
    steps: Union[str, List[Dict[str, Any]]] = ""
    routing_time: int = 0
    conversations: List[Conversation] = field(default_factory=list)
    internal_rerouting: List[InternalRerouting] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __post_init__(self):
        # 确保 steps 是字符串格式
        if isinstance(self.steps, list):
            self.steps = json.dumps({"values": self.steps})
    
    @property
    def steps_list(self) -> List[Step]:
        """将 steps JSON 字符串解析为 Step 对象列表"""
        if isinstance(self.steps, str) and self.steps:
            try:
                data = json.loads(self.steps)
                return [Step.from_dict(item) for item in data.get("values", [])]
            except json.JSONDecodeError:
                return []
        return []
    
    @property
    def total_conversations(self) -> int:
        """总对话数量"""
        return len(self.conversations)
    
    @property
    def completed_conversations(self) -> int:
        """完成的对话数量"""
        return sum(1 for conv in self.conversations if conv.status == AgentStatus.COMPLETED.value)
    
    @property
    def avg_response_length(self) -> float:
        """平均响应长度"""
        if not self.conversations:
            return 0.0
        total_length = sum(len(conv.response) for conv in self.conversations)
        return total_length / len(self.conversations)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "inputQuery": self.input_query,
            "inputHistory": self.input_history,
            "steps": self.steps,
            "routingTime": self.routing_time,
            "conversations": [conv.to_dict() for conv in self.conversations],
            "internalRerouting": [reroute.to_dict() for reroute in self.internal_rerouting]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Span':
        return cls(
            input_query=data.get("inputQuery", ""),
            input_history=data.get("inputHistory"),
            steps=data.get("steps", ""),
            routing_time=data.get("routingTime", 0),
            conversations=[Conversation.from_dict(conv) for conv in data.get("conversations", [])],
            internal_rerouting=[InternalRerouting.from_dict(reroute) 
                               for reroute in data.get("internalRerouting", [])]
        )


@dataclass
class RoundResult:
    """单个回合的结果"""
    round: int
    span: Span
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: Optional[str] = None
    
    @property
    def duration(self) -> Optional[float]:
        """回合持续时间（秒）"""
        if self.start_time and self.end_time:
            try:
                start = datetime.fromisoformat(self.start_time)
                end = datetime.fromisoformat(self.end_time)
                return (end - start).total_seconds()
            except (ValueError, TypeError):
                return None
        return None
    
    def complete_round(self):
        """标记回合完成"""
        self.end_time = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "round": self.round,
            "span": self.span.to_dict(),
            "startTime": self.start_time,
            "endTime": self.end_time
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RoundResult':
        span_data = data.get("span", {})
        return cls(
            round=data.get("round", 0),
            span=Span.from_dict(span_data),
            start_time=data.get("startTime", datetime.now().isoformat()),
            end_time=data.get("endTime")
        )


@dataclass
class BenchmarkOpenTelemetryResult:
    """OpenTelemetry 基准测试结果"""
    transaction_id: str
    rounds: List[RoundResult]
    test_name: Optional[str] = None
    environment: str = "production"
    version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # 如果 transaction_id 未提供，生成一个
        if not self.transaction_id:
            self.transaction_id = str(uuid.uuid4())
    
    @property
    def total_rounds(self) -> int:
        """总回合数"""
        return len(self.rounds)
    
    @property
    def total_conversations(self) -> int:
        """总对话数量"""
        return sum(round_result.span.total_conversations for round_result in self.rounds)
    
    @property
    def total_routing_time(self) -> int:
        """总路由时间（纳秒）"""
        return sum(round_result.span.routing_time for round_result in self.rounds)
    
    @property
    def avg_round_duration(self) -> Optional[float]:
        """平均回合持续时间（秒）"""
        durations = [r.duration for r in self.rounds if r.duration is not None]
        return sum(durations) / len(durations) if durations else None
    
    @property
    def success_rate(self) -> float:
        """整体成功率"""
        if not self.total_conversations:
            return 0.0
        total_completed = sum(
            round_result.span.completed_conversations 
            for round_result in self.rounds
        )
        return total_completed / self.total_conversations
    
    def add_round(self, round_result: RoundResult):
        """添加一个回合结果"""
        self.rounds.append(round_result)
    
    def get_round(self, round_number: int) -> Optional[RoundResult]:
        """获取指定回合的结果"""
        for round_result in self.rounds:
            if round_result.round == round_number:
                return round_result
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式（用于JSON序列化）"""
        return {
            "transactionId": self.transaction_id,
            "testName": self.test_name,
            "environment": self.environment,
            "version": self.version,
            "createdAt": self.created_at,
            "metadata": self.metadata,
            "rounds": [round_result.to_dict() for round_result in self.rounds],
            "summary": {
                "totalRounds": self.total_rounds,
                "totalConversations": self.total_conversations,
                "totalRoutingTime": self.total_routing_time,
                "successRate": self.success_rate,
                "avgRoundDuration": self.avg_round_duration
            }
        }
    
    def to_json(self, indent: int = 2, ensure_ascii: bool = False) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=ensure_ascii)
    
    def save_to_file(self, filepath: str):
        """保存结果到文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkOpenTelemetryResult':
        """从字典创建实例"""
        rounds_data = data.get("rounds", [])
        return cls(
            transaction_id=data.get("transactionId", str(uuid.uuid4())),
            rounds=[RoundResult.from_dict(round_data) for round_data in rounds_data],
            test_name=data.get("testName"),
            environment=data.get("environment", "production"),
            version=data.get("version", "1.0.0"),
            created_at=data.get("createdAt", datetime.now().isoformat()),
            metadata=data.get("metadata", {})
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'BenchmarkOpenTelemetryResult':
        """从JSON字符串创建实例"""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'BenchmarkOpenTelemetryResult':
        """从文件加载实例"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return cls.from_json(f.read())


# 使用示例
if __name__ == "__main__":
    # 1. 从示例JSON创建实例
    example_json = """{
        "transactionId": "dc7c7872-6636-4c07-9fab-6af40452a08d",
        "rounds": [
            {
                "round": 0,
                "span": {
                    "inputQuery": "北京三日游",
                    "inputHistory": null,
                    "steps": "{\\"values\\": [{\\"string_value\\": \\"查询北京旅游景点详细信息以及游玩路线...\\"}]}",
                    "routingTime": 6686678956,
                    "conversations": [
                        {
                            "subAgentName": "tourism agent",
                            "response": "好的，以下是一个为期三天的北京旅游景点详细信息...",
                            "status": "completed",
                            "query": "查询北京旅游景点详细信息以及游玩路线..."
                        }
                    ],
                    "internalRerouting": [
                        {
                            "lastSubAgentName": "route navigation agent",
                            "newSubAgentName": "movie recommendation agent",
                            "lastStatus": "completed",
                            "newStatus": "next-step",
                            "expired": 8100642562
                        }
                    ]
                }
            }
        ]
    }"""
    
    # 从JSON加载
    result = BenchmarkOpenTelemetryResult.from_json(example_json)
    
    # 2. 查看统计信息
    print(f"Transaction ID: {result.transaction_id}")
    print(f"Total Rounds: {result.total_rounds}")
    print(f"Total Conversations: {result.total_conversations}")
    print(f"Success Rate: {result.success_rate:.2%}")
    print(f"Total Routing Time: {result.total_routing_time} ns")
    
    # 3. 添加新的回合
    new_span = Span(
        input_query="上海两日游",
        steps=[{"string_value": "查询上海景点"}],
        routing_time=1234567890
    )
    
    new_conversation = Conversation(
        sub_agent_name="tourism agent",
        response="上海有外滩、东方明珠等景点...",
        status="completed",
        query="查询上海景点"
    )
    new_span.conversations.append(new_conversation)
    
    new_round = RoundResult(
        round=1,
        span=new_span
    )
    new_round.complete_round()
    
    result.add_round(new_round)
    
    # 4. 转换为JSON
    output_json = result.to_json(indent=2)
    print("\n转换为JSON后的结构:")
    print(output_json[:500] + "...")  # 只打印前500字符
    
    # 5. 保存到文件
    result.save_to_file("benchmark_result.json")
    
    # 6. 从文件加载
    loaded_result = BenchmarkOpenTelemetryResult.load_from_file("benchmark_result.json")
    print(f"\n从文件加载的结果ID: {loaded_result.transaction_id}")