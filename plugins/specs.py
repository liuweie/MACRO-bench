import pluggy

# 创建hook规范管理器
hookspec = pluggy.HookspecMarker("cogbenchmark")
hookimpl = pluggy.HookimplMarker("cogbenchmark")

class DomainHookSpec:
    """领域相关钩子规范"""
    
    @hookspec
    def get_supported_domains(self):
        """返回插件支持的领域列表"""
        pass
    
    @hookspec
    def create_user_simulator(self, domain: str, config: dict):
        """创建领域特定的用户模拟器"""
        pass
    
    @hookspec
    def get_domain_profile(self, domain: str, profile_name: str):
        """获取领域特定的用户画像"""
        pass
    
    @hookspec
    def build_conversation_context(self, domain: str, question: str, context: dict, **kwargs):
        """构建领域特定的对话上下文"""
        pass
    
    @hookspec
    def classify_question(self, domain: str, question: str, lang: str):
        """领域特定的问题分类"""
        pass
    
    @hookspec
    def generate_candidate_list(self, domain: str, question: str, context: dict, max_candidates: int):
        """生成领域特定的候选回复"""
        pass
    
    @hookspec
    def contains_specific_dates(self, domain: str, text: str):
        """检查是否包含特定日期（领域特定）"""
        pass
    
    @hookspec
    def normalize_slot_value(self, domain: str, slot_type: str, value: str, lang: str):
        """规范化领域特定的槽位值"""
        pass


class OrchestratorHookSpec:
    """Orchestrator相关钩子规范"""
    
    @hookspec
    def get_supported_orchestrators(self):
        """返回插件支持的orchestrator类型"""
        pass
    
    @hookspec
    def create_orchestrator_client(self, orchestrator_type: str, config: dict):
        """创建orchestrator客户端"""
        pass
    
    @hookspec
    def call_orchestrator_stream(self, orchestrator_client, payload: dict):
        """调用orchestrator并返回流式响应"""
        pass
    
    @hookspec
    def process_stream_response(self, orchestrator_client, stream_generator):
        """处理orchestrator的流式响应"""
        pass

    @hookspec
    def create_orchestrator_payload(self, orchestrator_client, conversation_state: dict, current_query: str, is_initial: bool):
        """构建发送到orchestrator的请求载荷（由orchestrator插件实现）"""
        pass


class EvaluatorHookSpec:
    """评测器相关钩子规范"""
    
    @hookspec
    def get_supported_evaluators(self):
        """返回插件支持的评测器类型"""
        pass
    
    @hookspec
    def create_evaluator(self, evaluator_type: str, config: dict):
        """创建评测器"""
        pass
    
    @hookspec
    def evaluate_task(self, evaluator, task_config: dict, orchestrator_response: dict, world_state: dict, debug: bool):
        """执行任务评测"""
        pass
    
    @hookspec
    def get_metric_weights(self, evaluator):
        """获取评测指标的权重配置"""
        pass


class ReporterHookSpec:
    """报告生成器相关钩子规范"""
    
    @hookspec
    def get_supported_reporters(self):
        """返回插件支持的报告生成器类型"""
        pass
    
    @hookspec
    def create_reporter(self, reporter_type: str, config: dict):
        """创建报告生成器"""
        pass
    
    @hookspec
    def generate_report(self, reporter, results: dict, output_path: str, **kwargs):
        """生成评测报告"""
        pass
    
    @hookspec
    def get_report_templates(self, reporter):
        """获取报告模板"""
        pass