"""Budget —— 每个 run 独占的资源上限（Execution Context Isolation 的一部分）。

系统（而非 LLM）为一次 run 设定 token / 成本 / 墙钟时间 / 工具调用次数 / 轮次 上限，
并在执行中记账。超限由控制平面（BackpressureController）转成 early_stop 信号，
Controller 据此进入 STOPPED_BUDGET 终态 —— retry/终止判定不经过 LLM。

所有上限默认 None（不限），仅 max_turns 有默认值，因此不配置 Budget 时行为与
重构前完全一致。Budget 是纯数据 + 记账，不做决策（决策在 backpressure.py）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

_MTOK = 1_000_000


@dataclass
class Budget:
    max_tokens: Optional[int] = None          # 累计（input+output+cache）token 上限
    max_cost_usd: Optional[float] = None       # 累计估算成本上限
    max_wall_seconds: Optional[float] = None   # 墙钟时间上限
    max_tool_calls: Optional[int] = None       # 工具调用次数上限
    max_turns: int = 16                        # 轮次上限（原 max_tool_turns 迁入）

    # ---- 运行时记账 ----
    spent_tokens: int = 0
    spent_cost_usd: float = 0.0
    tool_calls: int = 0
    started_at: float = field(default_factory=time.time)

    def clone(self) -> "Budget":
        """每个 run 用模板克隆一份独立预算，计数互不影响（隔离）。"""
        return Budget(
            max_tokens=self.max_tokens, max_cost_usd=self.max_cost_usd,
            max_wall_seconds=self.max_wall_seconds, max_tool_calls=self.max_tool_calls,
            max_turns=self.max_turns)

    # ---- 记账 ----
    def charge(self, usage: Dict[str, int], pricing: Optional[Dict[str, Any]] = None) -> None:
        self.spent_tokens += (
            int(usage.get("input_tokens", 0) or 0)
            + int(usage.get("output_tokens", 0) or 0)
            + int(usage.get("cache_read_input_tokens", 0) or 0)
            + int(usage.get("cache_creation_input_tokens", 0) or 0))
        if pricing and pricing.get("input") is not None and pricing.get("output") is not None:
            read_mul = pricing.get("cache_read_multiplier", 1.0)
            write_mul = pricing.get("cache_write_multiplier", 1.0)
            self.spent_cost_usd += (
                usage.get("input_tokens", 0) * pricing["input"]
                + usage.get("output_tokens", 0) * pricing["output"]
                + usage.get("cache_read_input_tokens", 0) * pricing["input"] * read_mul
                + usage.get("cache_creation_input_tokens", 0) * pricing["input"] * write_mul
            ) / _MTOK

    def charge_tool_call(self) -> None:
        self.tool_calls += 1

    # ---- 查询 ----
    def elapsed(self) -> float:
        return time.time() - self.started_at

    def exhausted(self) -> Optional[str]:
        """返回首个被突破的上限名（作为 early_stop 原因），未超限返回 None。"""
        if self.max_tokens is not None and self.spent_tokens >= self.max_tokens:
            return f"token_budget({self.spent_tokens}/{self.max_tokens})"
        if self.max_cost_usd is not None and self.spent_cost_usd >= self.max_cost_usd:
            return f"cost_budget(${self.spent_cost_usd:.4f}/${self.max_cost_usd})"
        if self.max_tool_calls is not None and self.tool_calls >= self.max_tool_calls:
            return f"tool_call_budget({self.tool_calls}/{self.max_tool_calls})"
        if self.max_wall_seconds is not None and self.elapsed() >= self.max_wall_seconds:
            return f"time_budget({self.elapsed():.1f}s/{self.max_wall_seconds}s)"
        return None

    def cost_pressure(self) -> Optional[float]:
        """成本已用比例（0~1+），无成本上限时返回 None —— 供降级决策参考。"""
        if self.max_cost_usd:
            return self.spent_cost_usd / self.max_cost_usd
        return None

    def snapshot(self) -> Dict[str, Any]:
        return {
            "spent_tokens": self.spent_tokens,
            "spent_cost_usd": round(self.spent_cost_usd, 6),
            "tool_calls": self.tool_calls,
            "elapsed_s": round(self.elapsed(), 3),
            "limits": {
                "max_tokens": self.max_tokens, "max_cost_usd": self.max_cost_usd,
                "max_tool_calls": self.max_tool_calls,
                "max_wall_seconds": self.max_wall_seconds, "max_turns": self.max_turns,
            },
        }
