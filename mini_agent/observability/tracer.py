"""执行日志与工具调用 trace，保证 Agent 全过程可观测、可追踪。

每个事件带时间戳、会话 ID、事件类型与负载。事件既保存在内存里
（供 CLI /trace 命令与测试断言使用），也可选择实时打印到终端。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TraceEvent:
    timestamp: float
    session_id: str
    event_type: str  # user_input / llm_call / llm_output / thinking / tool_call / tool_result / tool_error / final_answer / format_error / llm_error / max_turns
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "type": self.event_type,
            "data": self.data,
        }

    def format(self) -> str:
        ts = time.strftime("%H:%M:%S", time.localtime(self.timestamp))
        payload = " ".join(
            f"{k}={self._short(v)}" for k, v in self.data.items()
        )
        return f"[{ts}] [{self.session_id}] {self.event_type:<13} {payload}"

    @staticmethod
    def _short(value: Any, limit: int = 120) -> str:
        text = str(value).replace("\n", "\\n")
        return text if len(text) <= limit else text[: limit - 3] + "..."


class Tracer:
    def __init__(self, echo: bool = False) -> None:
        self.echo = echo  # True 时事件实时打印到终端
        self.events: List[TraceEvent] = []

    def log(self, session_id: str, event_type: str, **data: Any) -> TraceEvent:
        event = TraceEvent(
            timestamp=time.time(),
            session_id=session_id,
            event_type=event_type,
            data=data,
        )
        self.events.append(event)
        if self.echo:
            print("  · " + event.format())
        return event

    def for_session(self, session_id: str) -> List[TraceEvent]:
        return [e for e in self.events if e.session_id == session_id]

    def of_type(self, event_type: str, session_id: Optional[str] = None) -> List[TraceEvent]:
        return [
            e for e in self.events
            if e.event_type == event_type
            and (session_id is None or e.session_id == session_id)
        ]

    def dump(self, session_id: Optional[str] = None) -> str:
        events = self.for_session(session_id) if session_id else self.events
        return "\n".join(e.format() for e in events) or "(无 trace 记录)"
