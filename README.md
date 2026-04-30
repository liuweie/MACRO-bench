# CogBenchmark 插件化评测框架

基于 pluggy 的多领域、多组件插件化评测框架，支持灵活扩展和配置。

## 特性

- 🎯 **多领域支持**: 支持旅游、电商、金融等多个领域
- 🔌 **插件化架构**: 基于 pluggy 的灵活插件系统
- 🧩 **组件化设计**: Orchestrator、Evaluator、Reporter 均可插件化
- ⚙️ **配置驱动**: YAML 配置文件管理所有组件
- 📊 **智能评估**: 集成 LLM Judge 和规则评估
- 📈 **丰富报告**: 支持 HTML、JSON 等多种报告格式

## 快速开始

### 安装

```bash
pip install -r requirements.txt
pip install -e .

### run command for generate data from seed query
python ./datasets/travel_generator.py --mode from-file --queries-file ./scripts/demo_query.txt --out ./datasets/demo_data/demo_query.json --format json

### run command for trace collection
python run_benchmark.py --counts 1 --debug --max-clarifications 24 --max-subagent-clarification-rounds 12 --lang en --domain travel

### run command for evaluation
python evaluators/evaluator.py -t ./datasets/travel/tasks.yaml -a ./config/travel_agents.json -i ./trace_output/af_trace/converted_travel_tasks_v1_20251224_1130_exp1_3reset.jsonl -v

### run multi-times for stability test
bash run_stability.sh NUM_RUNS