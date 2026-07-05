"""执行日志（EventLog）与重建（Replayer）—— record & replay 的数据层。

一次 run 的全部执行事件（状态迁移、模型响应、工具执行、背压信号、预算记账……）
按序 append 到 append-only JSONL：`<runs_dir>/<run_id>.jsonl`。有了它：
- 任意 run 可从日志**完整重建**执行过程（状态时间线、每步为什么发生）
- Phase 3 的 ReplayLLM 直接回放已录制的模型响应，实现确定性调试（不再真调 API）

EventLog 在 runs_dir 为 None 时退化为纯内存（测试用），仍保留完整事件序列。
写盘用"边写边 flush 的追加"，进程崩溃也能保留已写入的前缀。
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.events import EVT_CONTROL_TRANSITION, ExecutionEvent
from ..core.llm import LLMClient, LLMError, LLMResponse, ToolCallRequest
from .state import ExecutionState


class EventLog:
    """单个 run 的 append-only 事件日志。"""

    def __init__(self, run_id: str, runs_dir: Optional[Path] = None):
        self.run_id = run_id
        self._events: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._path: Optional[Path] = None
        self._fh = None
        if runs_dir is not None:
            runs_dir = Path(runs_dir)
            runs_dir.mkdir(parents=True, exist_ok=True)
            self._path = runs_dir / f"{run_id}.jsonl"
            self._fh = self._path.open("a", encoding="utf-8")

    @property
    def path(self) -> Optional[Path]:
        return self._path

    def append(self, event: ExecutionEvent) -> None:
        record = event.to_dict()
        record.setdefault("run_id", self.run_id)
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            self._events.append(record)
            if self._fh is not None:
                self._fh.write(line + "\n")
                self._fh.flush()

    def events(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(e) for e in self._events]

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None

    def __enter__(self) -> "EventLog":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class Replayer:
    """从持久化 EventLog 重建一次 run 的执行过程。"""

    def __init__(self, events: List[Dict[str, Any]], run_id: str = "", session_id: str = ""):
        self.events = events
        self.run_id = run_id
        self.session_id = session_id

    @classmethod
    def from_file(cls, path: Path) -> "Replayer":
        path = Path(path)
        events: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        run_id = events[0].get("run_id", "") if events else path.stem
        session_id = events[0].get("session_id", "") if events else ""
        return cls(events, run_id=run_id, session_id=session_id)

    @classmethod
    def from_run(cls, run_id: str, runs_dir: Path) -> "Replayer":
        return cls.from_file(Path(runs_dir) / f"{run_id}.jsonl")

    def transitions(self) -> List[Dict[str, Any]]:
        """从事件流中提取状态迁移记录。"""
        out: List[Dict[str, Any]] = []
        for e in self.events:
            if e.get("type") == EVT_CONTROL_TRANSITION:
                out.append({
                    "from": e.get("from_state"),
                    "to": e.get("to_state"),
                    "reason": e.get("reason", ""),
                    "timestamp": e.get("timestamp", 0.0),
                    "detail": e.get("data", {}),
                })
        return out

    def model_responses(self) -> List[Dict[str, Any]]:
        """提取录制的模型响应（供 ReplayLLM 确定性回放）。"""
        return [e["data"] for e in self.events if e.get("type") == "model_response"]

    def iter_states(self):
        """单步执行：逐事件 yield (event, 到该事件为止重建的 ExecutionState)。
        供确定性调试与前端 replay scrub —— 每步都能看到当时的状态。"""
        state = ExecutionState(run_id=self.run_id, session_id=self.session_id)
        for e in self.events:
            if e.get("type") == EVT_CONTROL_TRANSITION:
                from .state import RunState, Transition
                t = Transition(
                    from_state=RunState(e["from_state"]), to_state=RunState(e["to_state"]),
                    reason=e.get("reason", ""), timestamp=e.get("timestamp", 0.0),
                    detail=e.get("data", {}))
                state.history.append(t)
                state.current = t.to_state
            yield ExecutionEvent.from_dict(e), state

    def reconstruct_state(self) -> ExecutionState:
        """重建终态状态机 —— 状态时间线与原 run 逐帧一致。"""
        return ExecutionState.reconstruct(
            self.transitions(), run_id=self.run_id, session_id=self.session_id)

    def steps(self) -> List[ExecutionEvent]:
        """把日志逐条还原为 ExecutionEvent，供单步执行 / 前端 replay scrub。"""
        return [ExecutionEvent.from_dict(e) for e in self.events]


class ReplayLLM(LLMClient):
    """确定性回放后端：按序回放已录制的 model_response，不再真调 API。

    把 ReplayLLM 装进 Controller 重跑同一份初始会话，即可**逐帧复现**原 run 的
    执行轨迹（模型的不确定性被录制响应消除，工具/控制是确定的、会真实重放）。
    若真实执行偏离了录制轨迹（响应被提前用尽），抛 LLMError 明确暴露。
    """

    def __init__(self, responses: List[Dict[str, Any]], label: str = "replay"):
        self._responses = list(responses)
        self._i = 0
        self._label = label
        self.calls: List[Dict[str, Any]] = []
        self.pricing = None

    def describe(self) -> str:
        return self._label

    @classmethod
    def from_events(cls, events: List[Dict[str, Any]], label: str = "replay") -> "ReplayLLM":
        return cls([e["data"] for e in events if e.get("type") == "model_response"], label)

    @classmethod
    def from_run(cls, run_id: str, runs_dir: Path, label: str = "replay") -> "ReplayLLM":
        return cls.from_events(Replayer.from_run(run_id, runs_dir).events, label)

    def complete(self, *, system, messages, tools=None, on_delta=None) -> LLMResponse:
        self.calls.append({"messages": len(messages), "tools": len(tools or [])})
        if self._i >= len(self._responses):
            raise LLMError("回放响应已用尽：真实执行偏离了录制轨迹（非确定性重放）")
        rec = self._responses[self._i]
        self._i += 1

        thinking = rec.get("thinking", "")
        text = rec.get("text", "")
        if on_delta:  # 复现流式，使流适配器产出与原 run 一致的事件
            if thinking:
                on_delta("thinking_delta", {"text": thinking})
            if text:
                on_delta("text_delta", {"text": text})

        tool_calls = [
            ToolCallRequest(id=c["id"], name=c["name"], arguments=c["arguments"])
            for c in rec.get("tool_calls", [])
        ]
        return LLMResponse(
            stop_reason=rec.get("stop_reason", "end_turn"),
            text=text, thinking=thinking, tool_calls=tool_calls,
            raw_content=rec.get("raw_content", []), usage=rec.get("usage", {}))
