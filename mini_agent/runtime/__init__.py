"""mini_agent.runtime —— 执行语义层 + 控制层内核。

- state      : RunState / ExecutionState（显式状态机 + 迁移历史 + 快照/重建）
- controller : Controller（驱动状态机的 ReAct 内核；Agent 为兼容别名）
- replay     : EventLog（append-only 事件日志）/ Replayer（从日志重建执行过程）
- stream_adapter : TurnStream（Provider delta -> 语义事件）
"""

from .state import ExecutionState, IllegalTransition, RunState, Transition
from .replay import EventLog, Replayer, ReplayLLM
from .stream_adapter import TurnStream
from .execution_context import ExecutionContext, MemoryView
from .controller import Agent, AgentResult, Controller

__all__ = [
    "RunState", "ExecutionState", "Transition", "IllegalTransition",
    "EventLog", "Replayer", "ReplayLLM", "TurnStream",
    "ExecutionContext", "MemoryView",
    "Controller", "Agent", "AgentResult",
]
