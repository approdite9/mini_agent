"""ToolPolicy —— 工具执行的控制策略（由 tools/runtime.py 的隔离层强制执行）。

每个工具（或默认）可配置：
- permission     : allow / deny / confirm（confirm 在无人值守时按 deny 处理并说明）
- timeout_s      : 单次执行超时（None=不限）
- max_retries    : 失败后的重试次数（0=不重试）
- rate_per_minute: 每分钟调用次数上限（None=不限）
- circuit_threshold: 连续失败达到该值即打开熔断，后续快速失败（0=不启用）

默认策略 = 全 allow、无超时、无重试、无限流、无熔断，因此不配置 policy 时
工具执行行为与重构前完全一致。策略只是声明，决策与计数由 ToolRuntime 落地，
计数状态存放在每个 run 独占的 ExecutionContext.policy_state 里，不跨 run 泄漏。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

ALLOW = "allow"
DENY = "deny"
CONFIRM = "confirm"


@dataclass(frozen=True)
class ToolRule:
    permission: str = ALLOW
    timeout_s: Optional[float] = None
    max_retries: int = 0
    rate_per_minute: Optional[int] = None
    circuit_threshold: int = 0


class ToolPolicy:
    def __init__(self, default: Optional[ToolRule] = None,
                 rules: Optional[Dict[str, ToolRule]] = None):
        self.default = default or ToolRule()
        self.rules = dict(rules or {})

    def rule_for(self, name: str) -> ToolRule:
        return self.rules.get(name, self.default)

    def set(self, name: str, rule: ToolRule) -> None:
        self.rules[name] = rule
