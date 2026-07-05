"""运行审计（Run Inspector）—— Agent 的"飞行记录仪"。

框架的事件流本来就记录了一切；本模块把它变成产品能力：
每次 run 自动沉淀一条结构化审计记录（轮次、逐工具耗时与成败、真实 token
用量、缓存命中率、估算成本、压缩次数），随会话持久化。

实现方式是事件流架构的又一次收益：RunCollector 只是 on_event 的一个
中间消费者（透传给前端的同时旁路记账），Agent 循环零改动。
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from ..runtime.controller import AgentResult
from ..core.events import AgentEvent, EventCallback

_MTOK = 1_000_000


def estimate_cost(usage: Dict[str, int],
                  pricing: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """按 Provider 声明的定价元数据估算成本。无定价时返回 None（只看 token）。"""
    if not pricing:
        return None
    p_in, p_out = pricing.get("input"), pricing.get("output")
    if p_in is None or p_out is None:
        return None
    read_mul = pricing.get("cache_read_multiplier", 1.0)
    write_mul = pricing.get("cache_write_multiplier", 1.0)
    amount = (
        usage.get("input_tokens", 0) * p_in
        + usage.get("output_tokens", 0) * p_out
        + usage.get("cache_read_input_tokens", 0) * p_in * read_mul
        + usage.get("cache_creation_input_tokens", 0) * p_in * write_mul
    ) / _MTOK
    return {"amount": round(amount, 6), "currency": pricing.get("currency", "USD")}


def cache_hit_rate(usage: Dict[str, int]) -> Optional[float]:
    """缓存命中率 = 缓存读 / 全部输入侧 token。输入为 0 时返回 None。"""
    read = usage.get("cache_read_input_tokens", 0)
    total = (usage.get("input_tokens", 0) + read
             + usage.get("cache_creation_input_tokens", 0))
    return round(read / total, 4) if total > 0 else None


class RunCollector:
    """事件流中间件：向下游透传事件，同时旁路收集，run 结束后产出审计记录。"""

    def __init__(self, downstream: Optional[EventCallback] = None):
        self._downstream = downstream
        self._events: List[AgentEvent] = []

    def __call__(self, event: AgentEvent) -> None:
        self._events.append(event)
        if self._downstream is not None:
            self._downstream(event)

    # ------------------------------------------------------------------
    def build(self, result: AgentResult, user_input: str,
              model: str = "", pricing: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        started_at = self._events[0].timestamp if self._events else time.time()
        ended_at = self._events[-1].timestamp if self._events else started_at

        # 逐工具执行单元：tool_start -> tool_result / tool_error（自带 latency）
        tools: List[Dict[str, Any]] = []
        compactions = 0
        memory_updates = 0
        for event in self._events:
            if event.type in ("tool_result", "tool_error"):
                tools.append({
                    "tool": event.data.get("tool", ""),
                    "ok": event.type == "tool_result",
                    "duration_ms": int(event.data.get("duration_ms", 0)),
                })
            elif event.type == "compaction":
                compactions += 1
            elif event.type == "memory_update":
                memory_updates += 1

        return {
            "id": f"r{int(started_at * 1000)}",
            "started_at": time.strftime("%H:%M:%S", time.localtime(started_at)),
            "input": user_input[:300],
            "answer": (result.answer or "")[:300],
            "error": result.error,
            "model": model,
            "turns": result.turns,
            "duration_ms": int((ended_at - started_at) * 1000),
            "tools": tools,
            "tools_ok": sum(1 for t in tools if t["ok"]),
            "tools_failed": sum(1 for t in tools if not t["ok"]),
            "usage": dict(result.usage),
            "cache_hit_rate": cache_hit_rate(result.usage),
            "cost": estimate_cost(result.usage, pricing),
            "compactions": compactions,
            "memory_updates": memory_updates,
        }


def format_run(record: Dict[str, Any]) -> str:
    """审计记录 -> 单行摘要（CLI 与日志共用）。"""
    usage = record.get("usage", {})
    parts = [
        record.get("model") or "?",
        f"{record.get('turns', 0)}轮",
        f"工具{record.get('tools_ok', 0)}✓" + (
            f"{record['tools_failed']}✗" if record.get("tools_failed") else ""),
        f"{record.get('duration_ms', 0) / 1000:.1f}s",
        f"tok {usage.get('input_tokens', 0)}+{usage.get('cache_read_input_tokens', 0)}c"
        f"/{usage.get('output_tokens', 0)}",
    ]
    rate = record.get("cache_hit_rate")
    if rate is not None:
        parts.append(f"cache {rate * 100:.0f}%")
    cost = record.get("cost")
    if cost:
        parts.append(f"≈${cost['amount']:.4f}")
    if record.get("compactions"):
        parts.append(f"压缩×{record['compactions']}")
    return " · ".join(parts)
