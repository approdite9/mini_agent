"""mini_agent —— 手搓的 Agent Runtime（不依赖任何外部 Agent 框架）。

分包结构（仿 HelloAgents）：
- core/         : 基础原语（llm 抽象 + adapters + factory、events、session）
- runtime/      : 执行语义层 + 控制层内核（状态机、Controller、EventLog、回放）
- control/      : 控制平面策略（backpressure、budget、tool policy）
- tools/        : 工具注册（registry）+ 内置工具（builtin）+ 隔离运行时（runtime）
- context/      : 上下文工程（compaction、缓存断点、压力度量）
- memory/       : 分层记忆（持久化 store）
- observability/: 系统级可观测（tracer、metrics、inspector）
- service / cli / server / web : 门面与 Execution Inspector 前端

顶层再导出常用公开名，保持 `from mini_agent import X` 稳定。
"""

# ---- core ----
from .core.llm import (
    LLMClient, LLMError, LLMResponse, StreamCallback, ToolCallRequest,
)
# 顶层只暴露与厂商无关的入口（工厂 + 离线脚本后端）；具体厂商适配器类需从
# mini_agent.core.llm_adapters 导入，保持内核公开面不点名任何厂商。
from .core.llm_adapters import ScriptedLLM
from .core.llm_factory import available_providers, create_llm
from .core.events import AgentEvent, EventCallback
from .core.session import Session, SessionManager

# ---- runtime（控制器 + 执行语义；Agent 为 Controller 的兼容别名）----
from .runtime.controller import Agent, AgentResult, Controller
from .runtime.state import RunState, ExecutionState
from .runtime.replay import EventLog, Replayer, ReplayLLM
from .runtime.execution_context import ExecutionContext

# ---- control（控制平面）----
from .control.budget import Budget
from .control.backpressure import BackpressureController
from .control.policy import ToolPolicy, ToolRule

# ---- tools ----
from .tools.registry import Tool, ToolContext, ToolError, ToolRegistry
from .tools.builtin import build_default_registry

# ---- context / memory ----
from .context.manager import ContextManager
from .memory.store import MemoryStore

# ---- observability ----
from .observability.tracer import TraceEvent, Tracer
from .observability.metrics import aggregate_metrics, compute_run_metrics

# ---- service 门面 ----
from .service import AgentService, SessionBusyError

__all__ = [
    "LLMClient", "LLMError", "LLMResponse", "StreamCallback", "ToolCallRequest",
    "ScriptedLLM", "available_providers", "create_llm",
    "AgentEvent", "EventCallback", "Session", "SessionManager",
    "Agent", "Controller", "AgentResult",
    "RunState", "ExecutionState", "EventLog", "Replayer", "ReplayLLM", "ExecutionContext",
    "Budget", "BackpressureController", "ToolPolicy", "ToolRule",
    "compute_run_metrics", "aggregate_metrics",
    "Tool", "ToolContext", "ToolError", "ToolRegistry", "build_default_registry",
    "ContextManager", "MemoryStore",
    "TraceEvent", "Tracer",
    "AgentService", "SessionBusyError",
]
