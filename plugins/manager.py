import pluggy
from typing import Dict, List, Any, Optional, Tuple
from .specs import (
    DomainHookSpec, OrchestratorHookSpec, 
    EvaluatorHookSpec, ReporterHookSpec
)
import importlib
import pkgutil
import traceback
import time
from pathlib import Path
from .invocation_logger import log_invocation


class ReporterPipeline:
    """Simple pipeline allowing multiple reporter plugins to run sequentially."""

    def __init__(self, reporter_entries: List[Tuple[str, Any]], domain: Optional[str] = None, chain_name: Optional[str] = None):
        if not reporter_entries:
            raise ValueError('ReporterPipeline requires at least one reporter entry')
        self._reporter_entries = reporter_entries
        self._primary_reporter = reporter_entries[0][1]
        self._plugin_ids = [entry[0] for entry in reporter_entries]
        self.domain = domain
        self.chain_name = chain_name

    def generate_report(self, results: dict, output_path: str, **kwargs):
        aggregated = []
        primary_report = None

        for plugin_id, reporter in self._reporter_entries:
            try:
                report_obj = reporter.generate_report(results, output_path, **kwargs)
                aggregated.append({'plugin': plugin_id, 'action': 'ok'})
                if primary_report is None and isinstance(report_obj, dict):
                    primary_report = report_obj
            except Exception as exc:
                aggregated.append({'plugin': plugin_id, 'action': 'error', 'error': str(exc)})
                continue

        if primary_report is None:
            primary_report = {}

        pipeline_meta = primary_report.setdefault('_report_pipeline', [])
        pipeline_meta.extend(aggregated)
        return primary_report

    def __getattr__(self, item):
        return getattr(self._primary_reporter, item)

    @property
    def reporters(self) -> List[Any]:
        return [entry[1] for entry in self._reporter_entries]

    @property
    def plugin_ids(self) -> List[str]:
        return list(self._plugin_ids)


class CompositeOrchestratorClient:
    """Wrapper that keeps track of orchestrator clients returned by multiple plugins."""

    def __init__(self, client_entries: List[Tuple[str, Any]], chain: Optional[List[str]] = None, domain: Optional[str] = None, chain_name: Optional[str] = None):
        if not client_entries:
            raise ValueError('CompositeOrchestratorClient requires at least one client entry')
        self._client_entries = client_entries
        self._primary_id, self._primary_client = client_entries[0]
        self.chain = list(chain) if chain else []
        self.domain = domain
        self.chain_name = chain_name

    def get_client(self, plugin_id: str):
        for pid, client in self._client_entries:
            if pid == plugin_id:
                return client
        return self._primary_client

    def iter_entries(self) -> List[Tuple[str, Any]]:
        return list(self._client_entries)

    @property
    def primary(self):
        return self._primary_client

    @property
    def plugin_ids(self) -> List[str]:
        return [pid for pid, _ in self._client_entries]

    def __getattr__(self, item):
        return getattr(self._primary_client, item)

class PluginManager:
    """统一的插件管理器"""
    
    def __init__(self):
        self.pm = pluggy.PluginManager("cogbenchmark")
        
        # 注册所有钩子规范
        self.pm.add_hookspecs(DomainHookSpec)
        self.pm.add_hookspecs(OrchestratorHookSpec)
        self.pm.add_hookspecs(EvaluatorHookSpec)
        self.pm.add_hookspecs(ReporterHookSpec)
        
        # 插件缓存（先初始化以避免在发现期间引用未初始化的属性）
        self._domain_plugins = {}
        self._orchestrator_plugins = {}
        self._evaluator_plugins = {}
        self._reporter_plugins = {}
        self._plugin_priority = {}
        self._orchestrator_clients = {}
        self._last_payload_chain = None
        self._last_stream_chain = None
        self._last_call_chain = None
        self._config_data = {}
        self._hook_chains = {}
        self._domain_settings = {}
        self._orchestrator_chain_info = {}
        self._reporter_chain_info: Dict[int, Dict[str, Any]] = {}
        self._evaluator_chain_info: Dict[int, Dict[str, Any]] = {}
        self._evaluator_instance_map: Dict[int, Dict[str, Any]] = {}
        self._pipeline_history: Dict[str, List[Dict[str, Any]]] = {}
        # 记录已加载的模块，避免重复导入/注册
        self._loaded_modules = set()

        # 注意: 不在 __init__ 中立即自动发现插件，以避免在模块导入期间触发
        # 插件模块的顶级导入（这些模块可能会导入本 manager 导致循环导入）。
        # 请在模块底部创建全局实例后调用 `global_plugin_manager._discover_plugins()`。
        
    def _discover_plugins(self):
        """自动发现和加载插件"""
        # 1. 从配置文件加载
        self._load_config_plugins()

        # 2. 从插件包目录动态导入
        try:
            base_packages = []
            try:
                base_packages.append(importlib.import_module('cogbenchmark.plugins'))
            except ModuleNotFoundError:
                import sys
                parent = Path(__file__).resolve().parents[1]
                if str(parent) not in sys.path:
                    sys.path.insert(0, str(parent))
                try:
                    base_packages.append(importlib.import_module('cogbenchmark.plugins'))
                except ModuleNotFoundError:
                    pass

            try:
                base_packages.append(importlib.import_module('plugins'))
            except ModuleNotFoundError:
                pass

            for plugins_pkg in base_packages:
                if not hasattr(plugins_pkg, '__path__'):
                    continue
                base_name = getattr(plugins_pkg, '__name__', 'plugins')
                for finder, name, ispkg in pkgutil.iter_modules(plugins_pkg.__path__):
                    if not name.endswith('_plugin'):
                        continue
                        plugin_module_path = f'{base_name}.{name}'
                        if plugin_module_path in self._loaded_modules:
                            continue
                        try:
                            module = importlib.import_module(plugin_module_path)
                            try:
                                if hasattr(module, 'register_plugin'):
                                    module.register_plugin(self.pm)
                                else:
                                    try:
                                        self.pm.register(module)
                                    except Exception:
                                        for attr_name in dir(module):
                                            attr = getattr(module, attr_name)
                                            try:
                                                if hasattr(attr, '__class__') and hasattr(attr, '__module__'):
                                                    self.pm.register(attr)
                                                    break
                                            except Exception:
                                                continue

                                try:
                                    outdir = Path('output/collected_jsons')
                                    outdir.mkdir(parents=True, exist_ok=True)
                                    ts = int(time.time())
                                    fname = outdir / f"plugin_discovered_{name}_{ts}.log"
                                    with open(fname, 'w', encoding='utf-8') as fh:
                                        fh.write(f"discovered: {plugin_module_path}\n")
                                except Exception:
                                    pass
                            except Exception:
                                try:
                                    outdir = Path('output/collected_jsons')
                                    outdir.mkdir(parents=True, exist_ok=True)
                                    ts = int(time.time())
                                    fname = outdir / f"plugin_register_error_{name}_{ts}.log"
                                    with open(fname, 'w', encoding='utf-8') as fh:
                                        fh.write(f"Failed to register plugin module: {plugin_module_path}\n")
                                        fh.write(traceback.format_exc())
                                except Exception:
                                    pass
                            self._loaded_modules.add(plugin_module_path)
                        except Exception as e:
                            try:
                                outdir = Path('output/collected_jsons')
                                outdir.mkdir(parents=True, exist_ok=True)
                                ts = int(time.time())
                                fname = outdir / f"plugin_discover_error_{name}_{ts}.log"
                                with open(fname, 'w', encoding='utf-8') as fh:
                                    fh.write(f"Failed to discover plugin: {plugin_module_path}\n")
                                    fh.write(str(e) + "\n\n")
                                    fh.write(traceback.format_exc())
                            except Exception:
                                pass
        except Exception as e:
            try:
                outdir = Path('output/collected_jsons')
                outdir.mkdir(parents=True, exist_ok=True)
                ts = int(time.time())
                fname = outdir / f"plugin_discover_error_main_{ts}.log"
                with open(fname, 'w', encoding='utf-8') as fh:
                    fh.write(f"Failed to discover plugins: {e}\n")
                    fh.write(traceback.format_exc())
            except Exception:
                pass
    
    def _load_config_plugins(self):
        """从配置文件加载插件"""
        try:
            import yaml
            with open('config/plugins.yaml', 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}

            self._config_data = config
            self._hook_chains = config.get('hook_chains') or {}
            self._domain_settings = config.get('domains') or {}

            # capture priority map early for deterministic execution order
            priority_map = config.get('plugin_priority') or {}
            if isinstance(priority_map, dict):
                self._plugin_priority = priority_map

            # Primary list of modules to load
            modules = config.get('plugin_modules') or []
            # Backwards-compatible: if user provided a `plugins` mapping, derive module paths
            if not modules and isinstance(config.get('plugins'), dict):
                for name in config.get('plugins').keys():
                    modules.append(f"plugins.{name}")

            for plugin_path in modules:
                candidates: List[str] = []
                if plugin_path:
                    candidates.append(plugin_path)
                    if not plugin_path.startswith('cogbenchmark.'):
                        candidates.append(f'cogbenchmark.{plugin_path}')

                imported = False
                for plugin_path_to_import in candidates:
                    if plugin_path_to_import in self._loaded_modules:
                        imported = True
                        break
                    try:
                        module = importlib.import_module(plugin_path_to_import)
                        try:
                            if hasattr(module, 'register_plugin'):
                                module.register_plugin(self.pm)
                            else:
                                try:
                                    self.pm.register(module)
                                except Exception:
                                    for attr_name in dir(module):
                                        attr = getattr(module, attr_name)
                                        try:
                                            if hasattr(attr, '__class__') and hasattr(attr, '__module__'):
                                                self.pm.register(attr)
                                                break
                                        except Exception:
                                            continue
                        except Exception:
                            try:
                                outdir = Path('output/collected_jsons')
                                outdir.mkdir(parents=True, exist_ok=True)
                                ts = int(time.time())
                                fname = outdir / f"plugin_register_error_{plugin_path_to_import.replace('.', '_')}_{ts}.log"
                                with open(fname, 'w', encoding='utf-8') as fh:
                                    fh.write(f"Failed to register plugin module: {plugin_path_to_import}\n")
                                    fh.write(traceback.format_exc())
                            except Exception:
                                pass
                        self._loaded_modules.add(plugin_path_to_import)
                        imported = True
                        break
                    except Exception as e:
                        try:
                            outdir = Path('output/collected_jsons')
                            outdir.mkdir(parents=True, exist_ok=True)
                            ts = int(time.time())
                            safe_name = plugin_path.replace('/', '_').replace('.', '_')
                            fname = outdir / f"plugin_load_error_{safe_name}_{ts}.log"
                            with open(fname, 'w', encoding='utf-8') as fh:
                                fh.write(f"Failed to load plugin: {plugin_path_to_import}\n")
                                fh.write(str(e) + "\n\n")
                                fh.write(traceback.format_exc())
                        except Exception:
                            pass
                if imported:
                    continue
        except Exception:
            pass

    def _get_plugin_priority(self, plugin_obj) -> int:
        """Resolve configured priority for a plugin (lower value loads earlier)."""
        if not self._plugin_priority:
            return 1000

        candidates = []

        name = getattr(plugin_obj, '__name__', None)
        if name:
            candidates.append(name)

        module_name = getattr(plugin_obj, '__module__', None)
        if module_name:
            candidates.append(module_name)

        plugin_class = getattr(plugin_obj, '__class__', None)
        if plugin_class:
            class_name = getattr(plugin_class, '__name__', None)
            class_module = getattr(plugin_class, '__module__', None)
            if class_name:
                candidates.append(class_name)
            if class_module:
                candidates.append(class_module)
            if class_name and class_module:
                candidates.append(f"{class_module}.{class_name}")

        normalized = []
        for cand in candidates:
            normalized.append(cand)
            if isinstance(cand, str) and '.' in cand:
                normalized.append(cand.split('.')[-1])

        for key in normalized:
            if key in self._plugin_priority:
                try:
                    return int(self._plugin_priority[key])
                except Exception:
                    return 1000

        return 1000

    def _sorted_hook_impls(self, hook) -> list:
        """Return hook implementations sorted by configured priority then pluggy order."""
        impls = list(hook.get_hookimpls())
        if not impls:
            return impls

        def sort_key(impl):
            plugin_obj = getattr(impl, 'plugin', None)
            priority = self._get_plugin_priority(plugin_obj) if plugin_obj else 1000
            order_hint = getattr(impl, 'order', 0)
            return (priority, order_hint)

        impls.sort(key=sort_key)
        return impls

    def _plugin_identifiers(self, plugin_obj) -> set:
        identifiers = set()
        if plugin_obj is None:
            return identifiers

        name = getattr(plugin_obj, '__name__', None)
        if name:
            identifiers.add(name)
            if '.' in name:
                identifiers.add(name.split('.')[-1])

        module_name = getattr(plugin_obj, '__module__', None)
        if module_name:
            identifiers.add(module_name)
            if '.' in module_name:
                identifiers.add(module_name.split('.')[-1])

        plugin_class = getattr(plugin_obj, '__class__', None)
        if plugin_class:
            class_name = getattr(plugin_class, '__name__', None)
            class_module = getattr(plugin_class, '__module__', None)
            if class_name:
                identifiers.add(class_name)
            if class_module:
                identifiers.add(class_module)
                if '.' in class_module:
                    identifiers.add(class_module.split('.')[-1])
            if class_name and class_module:
                identifiers.add(f"{class_module}.{class_name}")

        return identifiers

    def _select_chain(self, hook_key: Optional[str], domain: Optional[str], chain_name: Optional[str]) -> List[str]:
        if not hook_key:
            return []

        chain_config = self._hook_chains.get(hook_key, {}) if isinstance(self._hook_chains, dict) else {}

        if chain_name and isinstance(chain_config, dict):
            chain = chain_config.get(chain_name)
            if chain:
                return list(chain)

        domain_key = domain or ''
        if domain_key:
            domain_settings = self._domain_settings.get(domain_key, {}) if isinstance(self._domain_settings, dict) else {}
            if domain_settings:
                domain_chain = domain_settings.get(f"{hook_key}_chain")
                if domain_chain:
                    return list(domain_chain)

        if isinstance(chain_config, dict):
            if domain_key:
                domain_chain = chain_config.get(domain_key)
                if domain_chain:
                    return list(domain_chain)
            default_chain = chain_config.get('default_chain')
            if default_chain:
                return list(default_chain)
        elif isinstance(chain_config, list):
            return list(chain_config)

        return []

    def _iter_hook_impls(self, hook, hook_key: Optional[str] = None, domain: Optional[str] = None, chain_name: Optional[str] = None):
        impls = self._sorted_hook_impls(hook)
        if not impls:
            return []

        chain = self._select_chain(hook_key, domain, chain_name)
        if not chain:
            return impls

        matched = []
        used_impls = set()
        impl_identifier_map = []
        for impl in impls:
            identifiers = self._plugin_identifiers(getattr(impl, 'plugin', None))
            impl_identifier_map.append((impl, identifiers))

        for entry in chain:
            entry_str = str(entry)
            for impl, identifiers in impl_identifier_map:
                if impl in used_impls:
                    continue
                if entry_str in identifiers:
                    matched.append(impl)
                    used_impls.add(impl)
                    break

        for impl, _ in impl_identifier_map:
            if impl not in used_impls:
                matched.append(impl)

        return matched

    def get_domain_settings(self, domain: Optional[str]) -> Dict[str, Any]:
        """Return a copy of configured settings for the given domain."""
        if not domain:
            return {}
        if isinstance(self._domain_settings, dict):
            settings = self._domain_settings.get(domain)
            if isinstance(settings, dict):
                return dict(settings)
        return {}

    def _extract_task_domain(self, task_config: Optional[dict]) -> Optional[str]:
        if not isinstance(task_config, dict):
            return None
        domain = task_config.get('domain') or task_config.get('task_domain')
        if domain:
            return domain
        config_section = task_config.get('config') if isinstance(task_config.get('config'), dict) else {}
        domain = config_section.get('domain') or config_section.get('task_domain')
        return domain
    
    def register_plugin(self, plugin):
        """手动注册插件"""
        try:
            # Deduplicate by plugin class and module to avoid double-registration
            existing = list(self.pm.get_plugins())
            for p in existing:
                try:
                    if p is plugin:
                        return
                    if p.__class__.__name__ == plugin.__class__.__name__:
                        # same plugin implementation class already registered
                        return
                    if getattr(p, '__module__', None) == getattr(plugin, '__module__', None):
                        return
                except Exception:
                    continue
            self.pm.register(plugin)
            try:
                outdir = Path('output/collected_jsons')
                outdir.mkdir(parents=True, exist_ok=True)
                ts = int(time.time())
                fname = outdir / f"plugin_registered_{plugin.__class__.__name__}_{ts}.log"
                with open(fname, 'w', encoding='utf-8') as fh:
                    fh.write(f"registered plugin: {plugin.__class__.__name__}\n")
            except Exception:
                pass
        except Exception:
            try:
                # best-effort fallback: attempt direct registration
                self.pm.register(plugin)
            except Exception:
                pass
    
    # ========== Domain 相关方法 ==========
    
    def get_domain_plugin(self, domain: str):
        """获取特定领域的插件"""
        if domain in self._domain_plugins:
            return self._domain_plugins[domain]
        
        # 查询所有插件支持的领域
        plugins = self.pm.hook.get_supported_domains()
        for plugin_result in plugins:
            if plugin_result and domain in plugin_result:
                self._domain_plugins[domain] = plugin_result
                return plugin_result
        
        return None
    
    def create_user_simulator(self, domain: str, config: dict):
        """创建用户模拟器"""
        hook = self.pm.hook.create_user_simulator
        for impl in self._iter_hook_impls(hook, hook_key='domain', domain=domain, chain_name=config.get('chain') if isinstance(config, dict) else None):
            plugin_id = getattr(impl.plugin, '__name__', None) or impl.plugin.__class__.__name__
            try:
                sim = impl.function(domain=domain, config=config)
                # log invocation (structured)
                try:
                    log_invocation({'hook': 'create_user_simulator', 'plugin': plugin_id, 'message': 'create_user_simulator invoked'}, task_id=None)
                except Exception:
                    pass
                if sim:
                    self._last_plugin_invocation = {'hook': 'create_user_simulator', 'plugin': plugin_id, 'ts': time.time()}
                    try:
                        log_invocation({'hook': 'create_user_simulator', 'plugin': plugin_id, 'message': 'create_user_simulator handled'}, task_id=None)
                    except Exception:
                        pass
                    return sim
            except Exception as e:
                try:
                    outdir = Path('output/collected_jsons')
                    outdir.mkdir(parents=True, exist_ok=True)
                    ts = int(time.time())
                    fname = outdir / f"plugin_error_{plugin_id}_{ts}.log"
                    with open(fname, 'w', encoding='utf-8') as fh:
                        fh.write(str(e))
                except Exception:
                    pass
                continue
        return None
    
    def get_domain_profile(self, domain: str, profile_name: str):
        """获取领域特定的用户画像"""
        result = self.pm.hook.get_domain_profile(domain=domain, profile_name=profile_name)
        for profile in result:
            if profile:
                return profile
        return None
    
    def build_conversation_context(self, domain: str, question: str, context: dict, **kwargs):
        """构建对话上下文"""
        result = self.pm.hook.build_conversation_context(
            domain=domain, question=question, context=context, **kwargs
        )
        for ctx in result:
            if ctx:
                return ctx
        return None
    
    # ========== Orchestrator 相关方法 ==========
    
    def create_orchestrator_client(self, orchestrator_type: str, config: dict):
        """创建orchestrator客户端"""
        hook = self.pm.hook.create_orchestrator_client
        domain = None
        chain_name = None
        if isinstance(config, dict):
            domain = config.get('domain') or config.get('task_domain')
            chain_name = config.get('chain') or config.get('pipeline') or config.get('hook_chain')
        impls = self._iter_hook_impls(hook, hook_key='orchestrator', domain=domain, chain_name=chain_name)
        client_entries: List[Tuple[str, Any]] = []
        for impl in impls:
            plugin_id = getattr(impl.plugin, '__name__', None) or impl.plugin.__class__.__name__
            try:
                client = impl.function(orchestrator_type=orchestrator_type, config=config)
                try:
                    log_invocation({'hook': 'create_orchestrator_client', 'plugin': plugin_id, 'orchestrator_type': orchestrator_type, 'message': 'create_orchestrator_client handled'}, task_id=None)
                except Exception:
                    pass
                if client:
                    client_entries.append((plugin_id, client))
                    self._last_plugin_invocation = {'hook': 'create_orchestrator_client', 'plugin': plugin_id, 'ts': time.time()}
            except Exception as e:
                try:
                    outdir = Path('output/collected_jsons')
                    outdir.mkdir(parents=True, exist_ok=True)
                    ts = int(time.time())
                    fname = outdir / f"plugin_error_{plugin_id}_{ts}.log"
                    with open(fname, 'w', encoding='utf-8') as fh:
                        fh.write(str(e))
                except Exception:
                    pass
                continue
        if not client_entries:
            return None
        self._orchestrator_clients[orchestrator_type] = client_entries
        chain_used = self._select_chain('orchestrator', domain, chain_name)
        if len(client_entries) == 1:
            client = client_entries[0][1]
            self._orchestrator_chain_info[id(client)] = {'domain': domain, 'chain_name': chain_name, 'chain': chain_used}
            return client
        try:
            composite = CompositeOrchestratorClient(client_entries, chain=chain_used, domain=domain, chain_name=chain_name)
            self._orchestrator_chain_info[id(composite)] = {'domain': domain, 'chain_name': chain_name, 'chain': chain_used}
            return composite
        except Exception:
            fallback_client = client_entries[0][1]
            self._orchestrator_chain_info[id(fallback_client)] = {'domain': domain, 'chain_name': chain_name, 'chain': chain_used}
            return fallback_client
    
    def call_orchestrator_stream(self, orchestrator_client, payload: dict):
        """调用orchestrator"""
        hook = self.pm.hook.call_orchestrator_stream
        composite = orchestrator_client if isinstance(orchestrator_client, CompositeOrchestratorClient) else None
        chain_info = self._orchestrator_chain_info.get(id(orchestrator_client), {})
        domain = chain_info.get('domain')
        chain_name = chain_info.get('chain_name')
        if composite:
            domain = composite.domain if composite.domain else domain
            chain_name = composite.chain_name if composite.chain_name else chain_name
        impls = self._iter_hook_impls(hook, hook_key='orchestrator', domain=domain, chain_name=chain_name)
        stream_result = None
        chain_meta: List[Dict[str, Any]] = []

        for impl in impls:
            plugin_id = getattr(impl.plugin, '__name__', None) or impl.plugin.__class__.__name__
            try:
                client_for_plugin = composite.get_client(plugin_id) if composite else orchestrator_client
                resp = impl.function(orchestrator_client=client_for_plugin, payload=payload)
                try:
                    log_invocation({'hook': 'call_orchestrator_stream', 'plugin': plugin_id, 'payload_summary': str(payload)[:200], 'message': 'call_orchestrator_stream invoked'}, task_id=None)
                except Exception:
                    pass
                if resp:
                    action = 'replace' if stream_result is None else 'update'
                    resp_payload = resp

                    if isinstance(resp, tuple) and len(resp) == 2 and isinstance(resp[0], str):
                        directive, content = resp
                        if directive.lower() in ('replace', 'update', 'merge'):
                            action = directive.lower()
                            resp_payload = content

                    if isinstance(resp_payload, dict) and 'stream' in resp_payload:
                        stream_payload = resp_payload.get('stream')
                    else:
                        stream_payload = resp_payload

                    if stream_result is None or action == 'replace':
                        stream_result = stream_payload
                    else:
                        stream_result = stream_payload if stream_payload is not None else stream_result

                    chain_meta.append({'plugin': plugin_id, 'action': action})
                    self._last_plugin_invocation = {'hook': 'call_orchestrator_stream', 'plugin': plugin_id, 'ts': time.time(), 'payload': payload}
            except Exception as e:
                try:
                    outdir = Path('output/collected_jsons')
                    outdir.mkdir(parents=True, exist_ok=True)
                    ts = int(time.time())
                    fname = outdir / f"plugin_error_{plugin_id}_{ts}.log"
                    with open(fname, 'w', encoding='utf-8') as fh:
                        fh.write(str(e))
                except Exception:
                    pass
                continue
        if chain_meta:
            self._last_call_chain = chain_meta
            self._record_pipeline('orchestrator_call', chain_meta, domain=domain, chain_name=chain_name)
        return stream_result

    def create_orchestrator_payload(self, orchestrator_client, conversation_state: dict, current_query: str, is_initial: bool):
        """让 orchestrator 插件负责构建发送载荷，插件可以根据自身需要定制历史、路由字段等。"""
        hook = self.pm.hook.create_orchestrator_payload
        composite = orchestrator_client if isinstance(orchestrator_client, CompositeOrchestratorClient) else None
        chain_info = self._orchestrator_chain_info.get(id(orchestrator_client), {})
        domain = chain_info.get('domain')
        chain_name = chain_info.get('chain_name')
        if composite:
            domain = composite.domain if composite.domain else domain
            chain_name = composite.chain_name if composite.chain_name else chain_name
        impls = self._iter_hook_impls(hook, hook_key='orchestrator', domain=domain, chain_name=chain_name)
        payload_result = None
        chain_meta: List[Dict[str, Any]] = []
        for impl in impls:
            plugin_id = getattr(impl.plugin, '__name__', None) or impl.plugin.__class__.__name__
            try:
                client_for_plugin = composite.get_client(plugin_id) if composite else orchestrator_client
                payload = impl.function(
                    orchestrator_client=client_for_plugin,
                    conversation_state=conversation_state,
                    current_query=current_query,
                    is_initial=is_initial
                )
                try:
                    log_invocation({'hook': 'create_orchestrator_payload', 'plugin': plugin_id, 'payload_summary': str(current_query)[:200], 'message': 'create_orchestrator_payload invoked'}, task_id=None)
                except Exception:
                    pass
                if payload is None:
                    continue

                action = 'replace' if payload_result is None else 'update'
                payload_data = payload

                if isinstance(payload, tuple) and len(payload) == 2 and isinstance(payload[0], str):
                    directive, content = payload
                    if directive.lower() in ('replace', 'update', 'merge'):
                        action = directive.lower()
                        payload_data = content

                if not isinstance(payload_data, dict):
                    continue

                if payload_result is None or action == 'replace':
                    payload_result = dict(payload_data)
                else:
                    payload_result.update(payload_data)

                chain_meta.append({'plugin': plugin_id, 'action': action})
                self._last_plugin_invocation = {
                    'hook': 'create_orchestrator_payload',
                    'plugin': plugin_id,
                    'ts': time.time(),
                    'payload_sample': str(current_query)[:200]
                }
            except Exception as e:
                try:
                    outdir = Path('output/collected_jsons')
                    outdir.mkdir(parents=True, exist_ok=True)
                    ts = int(time.time())
                    fname = outdir / f"plugin_error_{plugin_id}_{ts}.log"
                    with open(fname, 'w', encoding='utf-8') as fh:
                        fh.write(str(e))
                except Exception:
                    pass
                continue
        if payload_result is not None:
            self._last_payload_chain = chain_meta
            self._record_pipeline('orchestrator_payload', chain_meta, domain=domain, chain_name=chain_name)
        return payload_result

    def process_stream_response(self, orchestrator_client, stream_generator):
        """让插件处理流式响应并记录哪个插件处理了请求"""
        hook = self.pm.hook.process_stream_response
        composite = orchestrator_client if isinstance(orchestrator_client, CompositeOrchestratorClient) else None
        chain_info = self._orchestrator_chain_info.get(id(orchestrator_client), {})
        domain = chain_info.get('domain')
        chain_name = chain_info.get('chain_name')
        if composite:
            domain = composite.domain if composite.domain else domain
            chain_name = composite.chain_name if composite.chain_name else chain_name
        impls = self._iter_hook_impls(hook, hook_key='orchestrator', domain=domain, chain_name=chain_name)
        stream_data = []
        if stream_generator is not None:
            try:
                if isinstance(stream_generator, list):
                    stream_data = list(stream_generator)
                else:
                    stream_data = list(stream_generator)
            except Exception:
                stream_data = []

        result = None
        chain_meta: List[Dict[str, Any]] = []

        for impl in impls:
            plugin_id = getattr(impl.plugin, '__name__', None) or impl.plugin.__class__.__name__
            try:
                client_for_plugin = composite.get_client(plugin_id) if composite else orchestrator_client
                local_iter = iter(stream_data)
                resp = impl.function(orchestrator_client=client_for_plugin, stream_generator=local_iter)
                try:
                    log_invocation({'hook': 'process_stream_response', 'plugin': plugin_id, 'message': 'process_stream_response handled'}, task_id=None)
                except Exception:
                    pass
                if resp is None:
                    continue

                action = 'replace' if result is None else 'update'
                resp_payload = resp

                if isinstance(resp, tuple) and len(resp) == 2 and isinstance(resp[0], str):
                    directive, content = resp
                    if directive.lower() in ('replace', 'update', 'merge'):
                        action = directive.lower()
                        resp_payload = content

                if isinstance(resp_payload, dict):
                    if result is None or action == 'replace':
                        result = dict(resp_payload)
                    else:
                        result.update(resp_payload)
                    chain_meta.append({'plugin': plugin_id, 'action': action})

                    if isinstance(result, dict) and result.get('clarification_requested'):
                        try:
                            clar_q = result.get('clarification_question')
                            clar_type = result.get('clarification_type')
                            task_id = result.get('transaction_id') or (result.get('collected_json', {}) or {}).get('meta', {}).get('transactionId')
                            log_invocation({'hook': 'clarification_detected', 'plugin': plugin_id, 'clarification_sample': str(clar_q)[:200], 'clarification_type': clar_type}, task_id=task_id)
                            from cogbenchmark.utils.logger import BenchmarkLogger
                            bl = BenchmarkLogger(log_file='benchmark_logs.jsonl', console_full=False)
                            bl.log_clarification_diagnostic(task_id=task_id, round_number=0, question=str(clar_q), user_response=None, rule_match=None, used_strategy=clar_type)
                        except Exception:
                            pass
                else:
                    result = resp_payload
                    chain_meta.append({'plugin': plugin_id, 'action': action, 'non_dict': True})

                self._last_plugin_invocation = {'hook': 'process_stream_response', 'plugin': plugin_id, 'ts': time.time()}
            except Exception as e:
                try:
                    outdir = Path('output/collected_jsons')
                    outdir.mkdir(parents=True, exist_ok=True)
                    ts = int(time.time())
                    fname = outdir / f"plugin_error_{plugin_id}_{ts}.log"
                    with open(fname, 'w', encoding='utf-8') as fh:
                        fh.write(str(e))
                except Exception:
                    pass
                continue

        if chain_meta:
            self._last_stream_chain = chain_meta
            self._record_pipeline('orchestrator_stream', chain_meta, domain=domain, chain_name=chain_name)

        if isinstance(result, dict):
            result.setdefault('_stream_pipeline', chain_meta)
        return result
    
    # ========== Evaluator 相关方法 ==========
    
    def create_evaluator(self, evaluator_type: str, config: dict):
        """创建评测器"""
        hook = self.pm.hook.create_evaluator
        evaluator_entries: List[Tuple[str, Any]] = []
        domain = None
        chain_name = None
        if isinstance(config, dict):
            domain = config.get('domain') or config.get('task_domain')
            chain_name = config.get('chain') or config.get('pipeline') or config.get('hook_chain')
        chain_used = self._select_chain('evaluator', domain, chain_name)
        for impl in self._iter_hook_impls(hook, hook_key='evaluator', domain=domain, chain_name=chain_name):
            plugin_id = getattr(impl.plugin, '__name__', None) or impl.plugin.__class__.__name__
            try:
                evaluator = impl.function(evaluator_type=evaluator_type, config=config)
                try:
                    log_invocation({'hook': 'create_evaluator', 'plugin': plugin_id, 'evaluator_type': evaluator_type, 'message': 'create_evaluator handled'}, task_id=None)
                except Exception:
                    pass
                if evaluator:
                    evaluator_entries.append((plugin_id, evaluator))
                    self._last_plugin_invocation = {'hook': 'create_evaluator', 'plugin': plugin_id, 'ts': time.time()}
            except Exception as e:
                try:
                    outdir = Path('output/collected_jsons')
                    outdir.mkdir(parents=True, exist_ok=True)
                    ts = int(time.time())
                    fname = outdir / f"plugin_error_{plugin_id}_{ts}.log"
                    with open(fname, 'w', encoding='utf-8') as fh:
                        fh.write(str(e))
                except Exception:
                    pass
                continue
        if not evaluator_entries:
            return None

        try:
            selection_meta = [{'plugin': pid, 'action': 'selected'} for pid, _ in evaluator_entries]
            if selection_meta:
                self._record_pipeline('evaluators', selection_meta, domain=domain, chain_name=chain_name)
        except Exception:
            pass

        primary = None
        for _, evaluator_obj in evaluator_entries:
            if hasattr(evaluator_obj, 'evaluate'):
                primary = evaluator_obj
                break
        if primary is None:
            primary = evaluator_entries[0][1]

        try:
            self._evaluator_chain_info[id(primary)] = {
                'domain': domain,
                'chain_name': chain_name,
                'chain': chain_used,
                'plugins': [pid for pid, _ in evaluator_entries],
            }
            self._evaluator_instance_map[id(primary)] = {pid: obj for pid, obj in evaluator_entries}
        except Exception:
            pass

        return primary
    
    def evaluate_task(self, evaluator, task_config: dict, orchestrator_response: dict, 
                     world_state: dict, debug: bool):
        """执行任务评测"""
        hook = self.pm.hook.evaluate_task
        domain = self._extract_task_domain(task_config)
        chain_name = None
        if isinstance(task_config, dict):
            chain_name = task_config.get('evaluator_chain')
        impls = self._iter_hook_impls(hook, hook_key='evaluator', domain=domain, chain_name=chain_name)
        aggregated = None
        chain_meta: List[Dict[str, Any]] = []
        instance_map = {}
        chain_info = {}
        try:
            instance_map = self._evaluator_instance_map.get(id(evaluator), {})
            chain_info = self._evaluator_chain_info.get(id(evaluator), {})
        except Exception:
            instance_map = {}
            chain_info = {}
        for impl in impls:
            plugin_id = getattr(impl.plugin, '__name__', None) or impl.plugin.__class__.__name__
            try:
                plugin_evaluator = instance_map.get(plugin_id, evaluator)
                score = impl.function(evaluator=plugin_evaluator, task_config=task_config, orchestrator_response=orchestrator_response, world_state=world_state, debug=debug)
                try:
                    t_id = task_config.get('id') or task_config.get('task_id') or None
                    log_invocation({'hook': 'evaluate_task', 'plugin': plugin_id, 'task': t_id, 'message': 'evaluate_task handled'}, task_id=t_id)
                except Exception:
                    pass
                if score:
                    action = 'replace' if aggregated is None else 'update'
                    score_payload = score

                    if isinstance(score, tuple) and len(score) == 2 and isinstance(score[0], str):
                        directive, content = score
                        if directive.lower() in ('replace', 'update', 'merge'):
                            action = directive.lower()
                            score_payload = content

                    if isinstance(score_payload, dict):
                        if aggregated is None or action == 'replace':
                            aggregated = dict(score_payload)
                        else:
                            for key, value in score_payload.items():
                                if key in aggregated and isinstance(aggregated[key], dict) and isinstance(value, dict):
                                    aggregated[key].update(value)
                                else:
                                    aggregated[key] = value
                        chain_meta.append({'plugin': plugin_id, 'action': action, 'keys': list(score_payload.keys())})
                    else:
                        aggregated = score_payload
                        chain_meta.append({'plugin': plugin_id, 'action': action, 'non_dict': True})

                    self._last_plugin_invocation = {'hook': 'evaluate_task', 'plugin': plugin_id, 'ts': time.time(), 'task': task_config}
            except Exception as e:
                try:
                    outdir = Path('output/collected_jsons')
                    outdir.mkdir(parents=True, exist_ok=True)
                    ts = int(time.time())
                    fname = outdir / f"plugin_error_{plugin_id}_{ts}.log"
                    with open(fname, 'w', encoding='utf-8') as fh:
                        fh.write(str(e))
                except Exception:
                    pass
                continue
        try:
            recorded = {entry.get('plugin') for entry in chain_meta}
            for plugin_id in chain_info.get('plugins', []):
                if plugin_id not in recorded:
                    chain_meta.append({'plugin': plugin_id, 'action': 'skipped'})
        except Exception:
            pass
        if aggregated is not None and isinstance(aggregated, dict):
            aggregated.setdefault('_evaluator_pipeline', chain_meta)
            self._record_pipeline('evaluators', chain_meta, domain=domain, chain_name=chain_name)
        return aggregated
    
    # ========== Reporter 相关方法 ==========
    
    def create_reporter(self, reporter_type: str, config: dict):
        """创建报告生成器"""
        hook = self.pm.hook.create_reporter
        reporter_entries: List[Tuple[str, Any]] = []
        domain = None
        chain_name = None
        if isinstance(config, dict):
            domain = config.get('domain') or config.get('task_domain')
            chain_name = config.get('chain') or config.get('pipeline') or config.get('hook_chain')
        chain_used = self._select_chain('reporter', domain, chain_name)
        for impl in self._iter_hook_impls(hook, hook_key='reporter', domain=domain, chain_name=chain_name):
            plugin_id = getattr(impl.plugin, '__name__', None) or impl.plugin.__class__.__name__
            try:
                reporter = impl.function(reporter_type=reporter_type, config=config)
                try:
                    log_invocation({'hook': 'create_reporter', 'plugin': plugin_id, 'reporter_type': reporter_type, 'message': 'create_reporter handled'}, task_id=None)
                except Exception:
                    pass
                if reporter:
                    reporter_entries.append((plugin_id, reporter))
                    self._last_plugin_invocation = {'hook': 'create_reporter', 'plugin': plugin_id, 'ts': time.time()}
            except Exception as e:
                try:
                    outdir = Path('output/collected_jsons')
                    outdir.mkdir(parents=True, exist_ok=True)
                    ts = int(time.time())
                    fname = outdir / f"plugin_error_{plugin_id}_{ts}.log"
                    with open(fname, 'w', encoding='utf-8') as fh:
                        fh.write(str(e))
                except Exception:
                    pass
                continue
        if not reporter_entries:
            return None
        try:
            selection_meta = [{'plugin': pid, 'action': 'selected'} for pid, _ in reporter_entries]
            if selection_meta:
                self._record_pipeline('reporters', selection_meta, domain=domain, chain_name=chain_name)
        except Exception:
            pass
        plugins_selected = [pid for pid, _ in reporter_entries]
        if len(reporter_entries) == 1:
            reporter_obj = reporter_entries[0][1]
            self._reporter_chain_info[id(reporter_obj)] = {
                'domain': domain,
                'chain_name': chain_name,
                'chain': chain_used,
                'plugins': plugins_selected,
            }
            return reporter_obj
        try:
            pipeline = ReporterPipeline(reporter_entries, domain=domain, chain_name=chain_name)
            self._reporter_chain_info[id(pipeline)] = {
                'domain': domain,
                'chain_name': chain_name,
                'chain': chain_used,
                'plugins': plugins_selected,
            }
            return pipeline
        except Exception:
            reporter_obj = reporter_entries[0][1]
            self._reporter_chain_info[id(reporter_obj)] = {
                'domain': domain,
                'chain_name': chain_name,
                'chain': chain_used,
                'plugins': plugins_selected[:1],
            }
            return reporter_entries[0][1]
    
    def generate_report(self, reporter, results: dict, output_path: str, **kwargs):
        """生成报告"""
        chain_info = self._reporter_chain_info.get(id(reporter), {})
        domain = kwargs.get('domain') or chain_info.get('domain')
        chain_name = kwargs.get('chain_name') or chain_info.get('chain_name')

        report_result = None
        chain_meta: List[Dict[str, Any]] = []

        try:
            if isinstance(reporter, ReporterPipeline):
                report_result = reporter.generate_report(results, output_path, **kwargs)
                chain_meta = [{'plugin': pid, 'action': 'run'} for pid in reporter.plugin_ids]
            elif reporter is not None:
                report_result = reporter.generate_report(results, output_path, **kwargs)
                plugin_name = getattr(reporter, '__class__', type('Reporter', (), {})).__name__
                chain_meta = [{'plugin': plugin_name, 'action': 'run'}]
            else:
                report_result = None
        except TypeError:
            # fallback for reporters that do not accept kwargs
            if isinstance(reporter, ReporterPipeline):
                report_result = reporter.generate_report(results, output_path)
                chain_meta = [{'plugin': pid, 'action': 'run'} for pid in reporter.plugin_ids]
            elif reporter is not None:
                report_result = reporter.generate_report(results, output_path)
                plugin_name = getattr(reporter, '__class__', type('Reporter', (), {})).__name__
                chain_meta = [{'plugin': plugin_name, 'action': 'run'}]

        if chain_meta:
            self._record_pipeline('reporters', chain_meta, domain=domain, chain_name=chain_name)

        if report_result is None:
            try:
                hook_outputs = self.pm.hook.generate_report(
                    reporter=reporter,
                    results=results,
                    output_path=output_path,
                    **kwargs,
                )
                if hook_outputs:
                    report_result = hook_outputs[-1]
            except Exception:
                report_result = None

        return report_result
    
    # ========== 工具方法 ==========
    
    def list_plugins(self):
        """列出所有已注册的插件"""
        return self.pm.get_plugins()
    
    def list_supported_domains(self):
        """列出所有支持的领域"""
        domains = set()
        plugins = self.pm.hook.get_supported_domains()
        for plugin_result in plugins:
            if plugin_result:
                domains.update(plugin_result)
        return list(domains)
    
    def list_supported_orchestrators(self):
        """列出所有支持的orchestrator类型"""
        orchestrators = set()
        plugins = self.pm.hook.get_supported_orchestrators()
        for plugin_result in plugins:
            if plugin_result:
                orchestrators.update(plugin_result)
        return list(orchestrators)
    
    def list_supported_evaluators(self):
        """列出所有支持的评测器类型"""
        evaluators = set()
        plugins = self.pm.hook.get_supported_evaluators()
        for plugin_result in plugins:
            if plugin_result:
                evaluators.update(plugin_result)
        return list(evaluators)
    
    def list_supported_reporters(self):
        """列出所有支持的报告生成器类型"""
        reporters = set()
        plugins = self.pm.hook.get_supported_reporters()
        for plugin_result in plugins:
            if plugin_result:
                reporters.update(plugin_result)
        return list(reporters)

    def _record_pipeline(self, key: str, chain_meta: List[Dict[str, Any]], domain: Optional[str] = None, chain_name: Optional[str] = None):
        if not chain_meta:
            return
        entry = {
            'timestamp': time.time(),
            'domain': domain,
            'chain_name': chain_name,
            'chain': [dict(item) for item in chain_meta],
        }
        self._pipeline_history.setdefault(key, []).append(entry)

    def describe_pipeline(self, pipeline_type: str, last_only: bool = True) -> Dict[str, Any]:
        records = self._pipeline_history.get(pipeline_type, [])
        result: Dict[str, Any] = {
            'pipeline_type': pipeline_type,
            'records_available': len(records)
        }

        last_record = records[-1] if records else None

        if last_record is None:
            fallback_chain = None
            if pipeline_type == 'orchestrator_payload':
                fallback_chain = self._last_payload_chain
            elif pipeline_type == 'orchestrator_stream':
                fallback_chain = self._last_stream_chain
            elif pipeline_type == 'orchestrator_call':
                fallback_chain = self._last_call_chain
            elif pipeline_type == 'evaluators':
                fallback_chain = None

            if fallback_chain:
                last_record = {
                    'timestamp': time.time(),
                    'domain': None,
                    'chain_name': None,
                    'chain': list(fallback_chain)
                }

        result['last_record'] = last_record

        if not last_only:
            result['history'] = list(records)

        return result


# 全局插件管理器实例
global_plugin_manager = PluginManager()

# 自动触发插件发现（延迟调用以避免在模块导入期间发生循环导入）
try:
    global_plugin_manager._discover_plugins()
except Exception as e:
    # 如果发现插件时有错误，打印但不要中断模块导入
    try:
        print(f"Plugin discovery failed at import time: {e}")
    except Exception:
        pass
