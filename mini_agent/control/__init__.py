"""mini_agent.control —— 控制平面策略（系统拥有，LLM 不参与）。

- budget       : Budget（token/cost/time/tool-call/turns 上限 + 记账，per-run 隔离）
- backpressure : BackpressureController（压力 → compact/downgrade/early_stop 信号）
- policy       : ToolPolicy / ToolRule（permission/timeout/retry/rate-limit/circuit）
"""

from .budget import Budget
from .backpressure import (
    BackpressureController, ControlSignal,
    CONTINUE, COMPACT, DOWNGRADE, EARLY_STOP,
)
from .policy import ToolPolicy, ToolRule, ALLOW, DENY, CONFIRM

__all__ = [
    "Budget",
    "BackpressureController", "ControlSignal",
    "CONTINUE", "COMPACT", "DOWNGRADE", "EARLY_STOP",
    "ToolPolicy", "ToolRule", "ALLOW", "DENY", "CONFIRM",
]
