"""执行事件流 —— 从"日志"升级为"执行语义"。

事件不再只是"发生了什么"的记录，而承载 **execution semantics + control transitions**：
每个事件可携带 `run_id`、`from_state`/`to_state`（状态迁移）、`reason`（控制平面为什么
这么做）。UI 与 replay 都只消费事件，禁止解析原始 LLM 文本。

事件词汇表分三类：

UI 生命周期事件（渲染用，保持向后兼容）：
- run_start / run_end            一次 run 的开始 / 结束（run_end 携带最终答案）
- turn_start                     一轮 LLM 调用开始
- thinking_start/delta/end       思考层边界与流式片段（end 携带完整内容与耗时）
- assistant_delta / assistant_final  最终回答流式片段 / 完整内容
- tool_start / tool_result / tool_error  工具执行单元（含 duration_ms）
- plan_update / memory_update    结构化计划变更 / 记忆写入
- compaction / usage / run_stats / error

执行语义事件（控制平面 + 状态机，本次新增）：
- control_transition             状态迁移（from_state/to_state/reason）
- backpressure_signal            背压信号（compact/downgrade/partial/early_stop + 原因）
- tool_denied / tool_retry / tool_timeout / circuit_open   工具隔离层的控制动作
- budget_charged / budget_exhausted   预算记账 / 预算耗尽
- state_snapshot                 run 级状态快照

一次 run 的全部事件按序落盘为 append-only 日志（见 runtime/replay.py::EventLog），
即可完整重建执行过程 —— 这是 record & replay 的数据基础。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


@dataclass
class ExecutionEvent:
    """执行事件。相比旧 AgentEvent 增加 run_id 与状态迁移语义字段（均可选，向后兼容）。"""

    type: str
    session_id: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    run_id: Optional[str] = None
    from_state: Optional[str] = None
    to_state: Optional[str] = None
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "type": self.type,
            "session_id": self.session_id,
            "data": self.data,
            "timestamp": self.timestamp,
        }
        # 语义字段仅在存在时写出，保持旧消费者（UI）payload 干净
        if self.run_id is not None:
            d["run_id"] = self.run_id
        if self.from_state is not None or self.to_state is not None:
            d["from_state"] = self.from_state
            d["to_state"] = self.to_state
        if self.reason is not None:
            d["reason"] = self.reason
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExecutionEvent":
        return cls(
            type=d["type"],
            session_id=d.get("session_id", ""),
            data=d.get("data", {}),
            timestamp=d.get("timestamp", 0.0),
            run_id=d.get("run_id"),
            from_state=d.get("from_state"),
            to_state=d.get("to_state"),
            reason=d.get("reason"),
        )


# 向后兼容别名：既有 CLI/server/测试仍以 AgentEvent 构造/消费事件。
AgentEvent = ExecutionEvent

# 前端/收集器传入的事件回调签名
EventCallback = Callable[[ExecutionEvent], None]

# ---- 执行语义事件类型常量（供控制器/控制平面使用，避免裸字符串拼写错误）----
EVT_CONTROL_TRANSITION = "control_transition"
EVT_BACKPRESSURE_SIGNAL = "backpressure_signal"
EVT_TOOL_DENIED = "tool_denied"
EVT_TOOL_RETRY = "tool_retry"
EVT_TOOL_TIMEOUT = "tool_timeout"
EVT_CIRCUIT_OPEN = "circuit_open"
EVT_BUDGET_CHARGED = "budget_charged"
EVT_BUDGET_EXHAUSTED = "budget_exhausted"
EVT_STATE_SNAPSHOT = "state_snapshot"
