"""BackpressureController —— 资源压力 → 控制信号（控制平面的核心决策者）。

把原本内联在 Agent 循环里的"是否压缩"决策，连同预算/成本/时间/延迟压力，统一
收敛到一处。它读取压力指标，输出一个 ControlSignal；由 Controller 执行动作。
LLM 完全不参与这些决策。

压力 → 动作：
- 预算耗尽（token/cost/tool_call/time）           → early_stop
- 上下文压力（context_tokens 超阈值）             → compact
- 成本压力高（接近成本上限）且配置了更便宜的模型 → downgrade
- 否则                                            → continue

上下文压力沿用 ContextManager 的阈值判断（单一事实来源），因此不配置额外预算时
压缩触发点与重构前完全一致。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from ..context.manager import ContextManager
    from ..core.session import Session
    from ..runtime.execution_context import ExecutionContext

CONTINUE = "continue"
COMPACT = "compact"
DOWNGRADE = "downgrade"
EARLY_STOP = "early_stop"


@dataclass
class ControlSignal:
    action: str = CONTINUE
    reason: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)


class BackpressureController:
    def __init__(self, context: "ContextManager",
                 downgrade_cost_ratio: float = 0.8,
                 latency_p95_ms: Optional[float] = None):
        self.context = context
        self.downgrade_cost_ratio = downgrade_cost_ratio
        self.latency_p95_ms = latency_p95_ms

    def evaluate(self, session: "Session", exec_ctx: "ExecutionContext",
                 recent_latency_ms: Optional[float] = None) -> ControlSignal:
        # 1) 预算耗尽 → 早停（系统托底，不问 LLM）
        exhausted = exec_ctx.budget.exhausted()
        if exhausted:
            return ControlSignal(EARLY_STOP, f"budget_exhausted:{exhausted}",
                                 {"budget": exec_ctx.budget.snapshot()})

        # 2) 延迟压力 → 早停（配置了 p95 阈值且近期超标）
        if (self.latency_p95_ms is not None and recent_latency_ms is not None
                and recent_latency_ms > self.latency_p95_ms):
            return ControlSignal(EARLY_STOP, f"latency_pressure({recent_latency_ms:.0f}ms)",
                                 {"threshold_ms": self.latency_p95_ms})

        # 3) 上下文压力 → 压缩（沿用 ContextManager 阈值，单一事实来源）
        if self.context.should_compact(session):
            return ControlSignal(COMPACT, "context_pressure",
                                 {"context_tokens": session.context_tokens()})

        # 4) 成本压力高且有更便宜的后备模型 → 降级
        ratio = exec_ctx.budget.cost_pressure()
        if (ratio is not None and ratio >= self.downgrade_cost_ratio
                and exec_ctx.fallback_llm is not None):
            return ControlSignal(DOWNGRADE, f"cost_pressure({ratio:.2f})",
                                 {"ratio": ratio})

        return ControlSignal(CONTINUE, "ok")
