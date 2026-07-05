"""执行状态机 —— 把一次 run 的执行语义显式化。

现状（重构前）：Agent.run() 是一个隐式的 while 循环，"现在处于哪个阶段""为什么
从这个阶段走到下一个阶段"只存在于代码控制流里，无法观测、无法重建、无法复现。

本模块把它变成一等对象：
- RunState        : 一次 run 可能处于的显式状态（含 4 个终态）
- ExecutionState  : 持有当前状态 + 迁移历史；transition(to, reason) 校验合法迁移
                    并记录"为什么"（reason 即控制平面的决策依据）
- snapshot()      : 任意时刻可完全序列化的 run 级状态快照（支撑重建/复现）

状态机本身不做任何决策，只负责"合法性 + 记录"。决策由控制平面（control/）给出，
由 Controller 调用 transition() 落地。这是 "Event ≠ 日志，而是 execution semantics +
state machine + control transitions" 的地基。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class RunState(str, Enum):
    CREATED = "created"              # 刚创建，尚未开始
    RUNNING = "running"             # 顶层活跃：准备发起下一步
    MODEL_CALL = "model_call"       # 正在调用/流式接收模型（LLM = Actor）
    TOOL_DISPATCH = "tool_dispatch" # 已拿到工具调用请求，控制平面正在决定门控/顺序
    TOOL_EXEC = "tool_exec"         # 正在执行某个工具（受隔离层保护）
    COMPACTING = "compacting"       # 背压触发的上下文压缩
    PAUSED = "paused"               # 单步调试暂停（step-by-step）
    # ---- 终态 ----
    COMPLETED = "completed"             # 正常完成（end_turn）
    FAILED = "failed"                   # LLM/系统级错误终止
    STOPPED_BUDGET = "stopped_budget"   # 控制平面因预算/早停而停止
    STOPPED_MAX_TURNS = "stopped_max_turns"  # 达到最大轮次

    @property
    def is_terminal(self) -> bool:
        return self in _TERMINAL


_TERMINAL = {
    RunState.COMPLETED, RunState.FAILED,
    RunState.STOPPED_BUDGET, RunState.STOPPED_MAX_TURNS,
}

# 合法迁移表：from -> 允许到达的 to 集合。任何不在表中的迁移都是编程错误。
_LEGAL: Dict[RunState, set] = {
    RunState.CREATED: {RunState.RUNNING},
    RunState.RUNNING: {
        RunState.MODEL_CALL, RunState.COMPACTING, RunState.PAUSED,
        RunState.STOPPED_BUDGET, RunState.STOPPED_MAX_TURNS, RunState.FAILED,
    },
    RunState.COMPACTING: {RunState.RUNNING, RunState.FAILED},
    RunState.PAUSED: {RunState.RUNNING, RunState.MODEL_CALL},
    RunState.MODEL_CALL: {
        RunState.TOOL_DISPATCH, RunState.COMPLETED, RunState.FAILED,
    },
    RunState.TOOL_DISPATCH: {RunState.TOOL_EXEC, RunState.FAILED},
    RunState.TOOL_EXEC: {RunState.TOOL_EXEC, RunState.TOOL_DISPATCH,
                         RunState.RUNNING, RunState.FAILED},
}


class IllegalTransition(Exception):
    """尝试了状态机不允许的迁移 —— 属于控制器编程错误，应当在测试期暴露。"""


@dataclass
class Transition:
    from_state: RunState
    to_state: RunState
    reason: str
    timestamp: float
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from": self.from_state.value,
            "to": self.to_state.value,
            "reason": self.reason,
            "timestamp": self.timestamp,
            "detail": self.detail,
        }


@dataclass
class ExecutionState:
    """一次 run 的执行状态：当前状态 + 全部迁移历史。可完全重建。"""

    run_id: str
    session_id: str
    current: RunState = RunState.CREATED
    turn: int = 0
    history: List[Transition] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)

    def transition(self, to: RunState, reason: str, **detail: Any) -> Transition:
        if to not in _LEGAL.get(self.current, set()):
            raise IllegalTransition(
                f"非法迁移 {self.current.value} -> {to.value}（reason={reason}）")
        t = Transition(self.current, to, reason, time.time(), dict(detail))
        self.history.append(t)
        self.current = to
        return t

    @property
    def is_terminal(self) -> bool:
        return self.current.is_terminal

    def snapshot(self) -> Dict[str, Any]:
        """run 级状态快照：可序列化、可用于重建与观测。"""
        return {
            "run_id": self.run_id,
            "session_id": self.session_id,
            "current": self.current.value,
            "turn": self.turn,
            "started_at": self.started_at,
            "terminal": self.is_terminal,
            "transitions": [t.to_dict() for t in self.history],
        }

    @classmethod
    def reconstruct(cls, transitions: List[Dict[str, Any]],
                    run_id: str = "", session_id: str = "") -> "ExecutionState":
        """从持久化的迁移记录重建状态机（用于 replay / 从日志重建状态）。"""
        state = cls(run_id=run_id, session_id=session_id)
        for d in transitions:
            t = Transition(
                from_state=RunState(d["from"]),
                to_state=RunState(d["to"]),
                reason=d.get("reason", ""),
                timestamp=d.get("timestamp", 0.0),
                detail=d.get("detail", {}),
            )
            state.history.append(t)
            state.current = t.to_state
        return state
