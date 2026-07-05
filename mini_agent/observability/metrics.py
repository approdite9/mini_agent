"""系统级结构化指标 —— 从持久化事件日志计算，进程重启后仍可复算。

不同于 inspector 的"面向人的审计摘要"，本模块给出机器可聚合的低层指标：
- counters   : 工具调用/成功/失败/重试/超时/拒绝/熔断、压缩、降级、状态迁移、轮次
- histograms : 工具执行延迟（min/avg/p50/p95/max）
- gauges     : 终态、最终上下文规模、预算花费

事件日志是唯一事实来源（runtime/replay.py::EventLog），因此指标可离线复算、可聚合。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _percentile(values: List[float], pct: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((pct / 100.0) * (len(s) - 1)))))
    return s[k]


def _histogram(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "avg": None, "p50": None, "p95": None, "max": None}
    return {
        "count": len(values),
        "min": min(values),
        "avg": round(sum(values) / len(values), 2),
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "max": max(values),
    }


def compute_run_metrics(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """从一次 run 的事件日志计算结构化指标。"""
    counters = {
        "turns": 0, "transitions": 0, "tool_calls": 0, "tool_ok": 0, "tool_error": 0,
        "tool_retry": 0, "tool_timeout": 0, "tool_denied": 0, "circuit_open": 0,
        "compactions": 0, "downgrades": 0, "early_stops": 0,
    }
    tool_latencies: List[float] = []
    final_state: Optional[str] = None
    context_tokens = 0
    budget_snapshot: Dict[str, Any] = {}
    run_id = ""
    session_id = ""

    for e in events:
        run_id = run_id or e.get("run_id", "")
        session_id = session_id or e.get("session_id", "")
        etype = e.get("type")
        data = e.get("data", {})
        if etype == "control_transition":
            counters["transitions"] += 1
            final_state = e.get("to_state", final_state)
        elif etype == "turn_start":
            counters["turns"] = max(counters["turns"], int(data.get("turn", 0)))
        elif etype == "tool_start":
            counters["tool_calls"] += 1
        elif etype == "tool_result":
            counters["tool_ok"] += 1
            tool_latencies.append(float(data.get("duration_ms", 0)))
        elif etype == "tool_error":
            counters["tool_error"] += 1
            tool_latencies.append(float(data.get("duration_ms", 0)))
        elif etype == "tool_retry":
            counters["tool_retry"] += 1
        elif etype == "tool_timeout":
            counters["tool_timeout"] += 1
        elif etype == "tool_denied":
            counters["tool_denied"] += 1
        elif etype == "circuit_open":
            counters["circuit_open"] += 1
        elif etype == "compaction":
            counters["compactions"] += 1
        elif etype == "backpressure_signal":
            if data.get("action") == "downgrade":
                counters["downgrades"] += 1
            elif data.get("action") == "early_stop":
                counters["early_stops"] += 1
        elif etype == "usage":
            context_tokens = (int(data.get("input_tokens", 0) or 0)
                              + int(data.get("cache_read_input_tokens", 0) or 0)
                              + int(data.get("cache_creation_input_tokens", 0) or 0))
        elif etype == "budget_charged":
            budget_snapshot = data

    return {
        "run_id": run_id,
        "session_id": session_id,
        "final_state": final_state,
        "counters": counters,
        "tool_latency_ms": _histogram(tool_latencies),
        "gauges": {
            "context_tokens": context_tokens,
            "budget_spent_tokens": budget_snapshot.get("spent_tokens", 0),
            "budget_spent_cost_usd": budget_snapshot.get("spent_cost_usd", 0.0),
        },
    }


def aggregate_metrics(run_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """跨多个 run 聚合：counters 求和、延迟合并、成功率。"""
    total = {k: 0 for k in (
        "turns", "transitions", "tool_calls", "tool_ok", "tool_error", "tool_retry",
        "tool_timeout", "tool_denied", "circuit_open", "compactions", "downgrades",
        "early_stops")}
    for m in run_metrics:
        for k, v in m.get("counters", {}).items():
            total[k] = total.get(k, 0) + v
    tool_total = total["tool_ok"] + total["tool_error"]
    states: Dict[str, int] = {}
    for m in run_metrics:
        st = m.get("final_state")
        if st:
            states[st] = states.get(st, 0) + 1
    return {
        "runs": len(run_metrics),
        "counters": total,
        "tool_success_rate": round(total["tool_ok"] / tool_total, 4) if tool_total else None,
        "final_states": states,
    }
