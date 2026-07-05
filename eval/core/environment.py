"""Environment —— 一个任务的隔离执行环境（配置化，可注入故障）。

对齐 AgentBench 的"环境"概念，但落到本 Runtime：环境决定这次评测里
- 有哪些工具（可注入 flaky/故障工具）
- 预算与工具策略（用于 budget_adherence / 隔离验证）
- 初始记忆/会话状态
Runner 用给定的真实 LLM 在此环境上跑完整 Controller。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from mini_agent import AgentService, MemoryStore
from mini_agent.control.budget import Budget
from mini_agent.control.policy import ToolPolicy
from mini_agent.tools.builtin import build_default_registry
from mini_agent.tools.registry import Tool, ToolError, ToolRegistry


class Environment:
    def __init__(
        self,
        extra_tools: Optional[List[Tool]] = None,
        budget: Optional[Budget] = None,
        tool_policy: Optional[ToolPolicy] = None,
        seed_memory: Optional[Dict[str, str]] = None,
        max_tool_turns: int = 12,
    ):
        self.extra_tools = extra_tools or []
        self.budget = budget
        self.tool_policy = tool_policy
        self.seed_memory = seed_memory or {}
        self.max_tool_turns = max_tool_turns

    def _registry(self) -> ToolRegistry:
        reg = build_default_registry()
        for tool in self.extra_tools:
            reg.register(tool)
        return reg

    def build(self, base_dir: Path, llm) -> Tuple[AgentService, str]:
        service = AgentService(
            llm=llm, base_dir=base_dir, registry=self._registry(),
            max_tool_turns=self.max_tool_turns,
            budget=self.budget, tool_policy=self.tool_policy)
        for key, value in self.seed_memory.items():
            service.global_memory.set(key, value, source="eval_seed")
        session = service.create_session(title="eval")
        return service, session.id


def flaky_tool(name: str, description: str, succeed_on_attempt: int,
               success_value: str, params: Optional[Dict[str, Any]] = None) -> Tool:
    """构造一个前 (succeed_on_attempt-1) 次失败、之后成功的工具，用于 error_recovery。
    失败信息明确提示"请重试"，考察 Agent 看到 tool_error 后能否自愈。"""
    state = {"attempts": 0}

    def func(args: Dict[str, Any], ctx) -> str:
        state["attempts"] += 1
        if state["attempts"] < succeed_on_attempt:
            raise ToolError(f"服务暂时不可用（第 {state['attempts']} 次），请重试。")
        return success_value

    return Tool(name=name, description=description,
                parameters=params or {"type": "object", "properties": {}}, func=func)
