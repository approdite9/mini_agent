"""Controller —— 执行语义层 + 控制层的内核（取代旧 Agent.run 的隐式循环）。

与旧实现的本质区别：一次 run 的执行被显式建模为一个**状态机**（runtime/state.py），
控制器在每个阶段之间做**显式迁移**并记录"为什么"（reason 即控制平面的决策依据），
所有事件（含状态迁移）按序落入 **append-only 事件日志**（runtime/replay.py），
使任意 run 可从日志完整重建、复现。

LLM 只作为 Actor：仅输出 reasoning / tool-selection / final-answer。调度、循环、
错误处理、终止判定全部由本控制器承担 —— LLM 不决定 execution flow。

（Phase 1 先把控制流显式化并可观测/可复现；压缩/预算/工具隔离等策略的"决策"
在 Phase 2 进一步外提到 control/ 与 tools/runtime.py。当前压缩决策仍内联调用
ContextManager，但已被包在显式的 COMPACTING 状态迁移中。）
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..context.manager import ContextManager
from ..control.backpressure import (
    COMPACT, DOWNGRADE, EARLY_STOP, BackpressureController,
)
from ..control.budget import Budget
from ..control.policy import ToolPolicy
from ..core.events import (
    EVT_CONTROL_TRANSITION, EVT_STATE_SNAPSHOT, EventCallback, ExecutionEvent,
)
from ..core.llm import LLMClient, LLMError, LLMResponse
from ..core.session import Session
from ..memory.store import MemoryStore
from ..observability.tracer import Tracer
from ..tools.registry import ToolContext, ToolRegistry
from ..tools.runtime import ToolRuntime
from .execution_context import ExecutionContext
from .replay import EventLog
from .state import ExecutionState, RunState
from .stream_adapter import TurnStream


@dataclass
class AgentResult:
    answer: str
    turns: int
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)  # {name, arguments, result, ok}
    usage: Dict[str, int] = field(default_factory=dict)             # 本次 run 的累计 usage
    error: Optional[str] = None
    run_id: str = ""                                                # 事件日志定位用
    final_state: str = ""                                           # 终态（RunState.value）


class _RunEmitter:
    """把一次 run 的所有事件写入 EventLog，并（可选）转发给前端 on_event。

    - emit(type, **data)         : 落日志 + 转发（UI 事件）
    - emit_transition(transition): 落日志 + 转发（control_transition 语义事件）
    - log_only(type, **data)     : 仅落日志（如 state_snapshot / model_response，不进 SSE）
    """

    def __init__(self, session_id: str, run_id: str, state: ExecutionState,
                 log: EventLog, on_event: Optional[EventCallback]):
        self.session_id = session_id
        self.run_id = run_id
        self.state = state
        self.log = log
        self._on_event = on_event

    def emit(self, event_type: str, **data: Any) -> None:
        ev = ExecutionEvent(type=event_type, session_id=self.session_id,
                            data=data, run_id=self.run_id)
        self.log.append(ev)
        if self._on_event is not None:
            self._on_event(ev)

    def log_only(self, event_type: str, **data: Any) -> None:
        self.log.append(ExecutionEvent(type=event_type, session_id=self.session_id,
                                       data=data, run_id=self.run_id))

    def transition(self, to: RunState, reason: str, **detail: Any) -> None:
        t = self.state.transition(to, reason, **detail)
        ev = ExecutionEvent(
            type=EVT_CONTROL_TRANSITION, session_id=self.session_id,
            data=dict(detail), run_id=self.run_id,
            from_state=t.from_state.value, to_state=t.to_state.value, reason=t.reason)
        self.log.append(ev)
        if self._on_event is not None:
            self._on_event(ev)


class Controller:
    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
        global_memory: Optional[MemoryStore] = None,
        tracer: Optional[Tracer] = None,
        context_manager: Optional[ContextManager] = None,
        max_tool_turns: int = 16,
        runs_dir: Optional[Path] = None,
        backpressure: Optional[BackpressureController] = None,
        tool_policy: Optional[ToolPolicy] = None,
        budget: Optional[Budget] = None,
        fallback_llm: Optional[LLMClient] = None,
    ) -> None:
        self.llm = llm
        self.registry = registry
        # 注意不能写 `global_memory or ...`：空 MemoryStore 的 __len__ 为 0 是 falsy
        self.global_memory = global_memory if global_memory is not None else MemoryStore()
        self.tracer = tracer or Tracer()
        self.context = context_manager or ContextManager()
        self.max_tool_turns = max_tool_turns
        self.runs_dir = Path(runs_dir) if runs_dir else None
        # ---- 控制平面（默认值均保持重构前行为）----
        self.backpressure = backpressure or BackpressureController(self.context)
        self.tool_runtime = ToolRuntime(self.registry, tool_policy)
        # 预算模板：每个 run 克隆一份独立计数
        self.budget_template = budget or Budget(max_turns=max_tool_turns)
        self.fallback_llm = fallback_llm

    # ------------------------------------------------------------------
    def run(self, session: Session, user_input: str,
            on_event: Optional[EventCallback] = None,
            attachments: Optional[List[Dict[str, Any]]] = None) -> AgentResult:
        session.request_count += 1
        run_id = f"{session.id}-r{session.request_count}"
        state = ExecutionState(run_id=run_id, session_id=session.id)
        log = EventLog(run_id, self.runs_dir)
        em = _RunEmitter(session.id, run_id, state, log, on_event)
        exec_ctx = ExecutionContext(
            run_id=run_id, session=session, budget=self.budget_template.clone(),
            global_memory=self.global_memory, fallback_llm=self.fallback_llm)

        try:
            return self._drive(session, user_input, attachments, state, em, run_id, exec_ctx)
        finally:
            # 全局记忆写回缓冲提交（读一致/写隔离，成功或失败均落已缓冲的写入）
            exec_ctx.commit_memory()
            log.close()

    # ------------------------------------------------------------------
    def _drive(self, session, user_input, attachments, state, em, run_id, exec_ctx) -> AgentResult:
        session.history.append(self._user_message(user_input, attachments))
        session.save()
        self.tracer.log(session.id, "user_input", content=user_input,
                        attachments=len(attachments or []))
        em.emit("run_start", input=user_input, attachments=len(attachments or []))
        em.transition(RunState.RUNNING, "user_input")

        tool_calls_acc: List[Dict[str, Any]] = []
        run_usage: Dict[str, int] = {}
        budget = exec_ctx.budget
        recent_latency_ms: Optional[float] = None

        for turn in range(1, budget.max_turns + 1):
            state.turn = turn

            # ---- 控制平面：背压评估（拥有 压缩/降级/早停 决策，取代内联 should_compact）----
            signal = self.backpressure.evaluate(session, exec_ctx, recent_latency_ms)
            turn_llm = self.llm
            if signal.action == EARLY_STOP:
                em.emit("backpressure_signal", action=EARLY_STOP, reason=signal.reason)
                return self._finalize_stopped(
                    session, state, em, run_id, run_usage, tool_calls_acc,
                    reason=signal.reason)
            if signal.action == COMPACT:
                em.emit("backpressure_signal", action=COMPACT, reason=signal.reason)
                em.transition(RunState.COMPACTING, signal.reason,
                              context_tokens=session.context_tokens())
                compacted = self.context.compact(session, self.llm, on_event=em.emit)
                if compacted:
                    self.tracer.log(session.id, "compaction",
                                    context_tokens=session.context_tokens())
                    session.last_usage = {}  # 压缩后旧的规模数据失效
                em.transition(RunState.RUNNING,
                              "compacted" if compacted else "compact_noop")
            elif signal.action == DOWNGRADE and exec_ctx.fallback_llm is not None:
                em.emit("backpressure_signal", action=DOWNGRADE, reason=signal.reason)
                turn_llm = exec_ctx.fallback_llm  # 仅本轮降级到更便宜的模型

            # ---- 组装请求并流式调用（LLM = Actor）----
            memory_items = (exec_ctx.memory_view.items() if session.use_global_memory
                            else session.local_memory)
            scope = "全局共享" if session.use_global_memory else "会话私有"
            system, messages = self.context.build_request(
                session, MemoryStore.render_index(memory_items), scope)

            em.emit("turn_start", turn=turn)
            em.transition(RunState.MODEL_CALL, "need_action", turn=turn)
            self.tracer.log(session.id, "llm_call", turn=turn, messages=len(messages))
            stream = TurnStream(em.emit)
            call_started = time.time()
            try:
                response = turn_llm.complete(
                    system=system, messages=messages,
                    tools=self.registry.tool_specs(exec_ctx.tool_namespace),
                    on_delta=stream,
                )
            except LLMError as exc:
                stream.close()
                em.transition(RunState.FAILED, "llm_error", error=str(exc))
                self.tracer.log(session.id, "llm_error", error=str(exc))
                em.emit("error", message=str(exc))
                answer = f"抱歉，本次请求失败（LLM 错误）：{exc}"
                session.history.append({"role": "assistant", "content": answer})
                session.save()
                em.emit("run_end", answer=answer, error=str(exc), usage=run_usage)
                em.log_only(EVT_STATE_SNAPSHOT, **state.snapshot())
                return AgentResult(answer=answer, turns=turn, tool_calls=tool_calls_acc,
                                   usage=run_usage, error=f"llm_error: {exc}",
                                   run_id=run_id, final_state=state.current.value)
            stream.close()
            recent_latency_ms = (time.time() - call_started) * 1000

            # 完整录制模型响应，供 Phase 3 的 ReplayLLM 确定性回放（仅落日志，不进 SSE）
            em.log_only("model_response", stop_reason=response.stop_reason,
                        text=response.text, thinking=response.thinking,
                        raw_content=response.raw_content, usage=response.usage,
                        tool_calls=[{"id": c.id, "name": c.name, "arguments": c.arguments}
                                    for c in response.tool_calls])

            self._record_usage(session, response, run_usage)
            budget.charge(response.usage, turn_llm.pricing)  # 系统记账，非 LLM
            em.log_only("budget_charged", **budget.snapshot())
            em.emit("usage", turn=turn, **response.usage)
            if response.thinking:
                self.tracer.log(session.id, "thinking", turn=turn, content=response.thinking)
            if response.text:
                self.tracer.log(session.id, "assistant_text", turn=turn, content=response.text)
                em.emit("assistant_final", turn=turn, content=response.text)

            session.history.append({"role": "assistant", "content": response.raw_content})

            # ---- 工具调用分支：本轮所有结果合并进一条 user 消息 ----
            if response.wants_tools:
                em.transition(RunState.TOOL_DISPATCH, "tool_use",
                              tools=[c.name for c in response.tool_calls])
                result_blocks: List[Dict[str, Any]] = []
                for call in response.tool_calls:
                    em.transition(RunState.TOOL_EXEC, "exec_tool", tool=call.name)
                    budget.charge_tool_call()
                    observation, ok = self._execute_tool(
                        session, em.emit, exec_ctx, call.name, call.arguments, call.id)
                    tool_calls_acc.append({"name": call.name, "arguments": call.arguments,
                                           "result": observation, "ok": ok})
                    result_blocks.append({
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": observation,
                        "is_error": not ok,
                    })
                session.history.append({"role": "user", "content": result_blocks})
                session.save()
                em.transition(RunState.RUNNING, "tools_done")
                continue

            # ---- 最终回答 ----
            answer = response.text
            session.save()
            em.transition(RunState.COMPLETED, "end_turn")
            self.tracer.log(session.id, "final_answer", turn=turn, answer=answer)
            em.emit("run_end", answer=answer, turns=turn, usage=run_usage)
            em.log_only(EVT_STATE_SNAPSHOT, **state.snapshot())
            return AgentResult(answer=answer, turns=turn, tool_calls=tool_calls_acc,
                               usage=run_usage, run_id=run_id,
                               final_state=state.current.value)

        # ---- 超出最大轮次：优雅收尾 ----
        em.transition(RunState.STOPPED_MAX_TURNS, "max_turns", limit=budget.max_turns)
        self.tracer.log(session.id, "max_turns", limit=budget.max_turns)
        answer = (f"本次请求已达到最大工具调用轮次({budget.max_turns})仍未完成，已停止。"
                  f"可以拆分任务后继续，我会基于已有进展接着做。")
        session.history.append({"role": "assistant", "content": answer})
        session.save()
        em.emit("run_end", answer=answer, error="max_turns_exceeded", usage=run_usage)
        em.log_only(EVT_STATE_SNAPSHOT, **state.snapshot())
        return AgentResult(answer=answer, turns=budget.max_turns,
                           tool_calls=tool_calls_acc, usage=run_usage,
                           error="max_turns_exceeded", run_id=run_id,
                           final_state=state.current.value)

    # ------------------------------------------------------------------
    def _finalize_stopped(self, session, state, em, run_id, run_usage,
                          tool_calls_acc, reason: str) -> AgentResult:
        """控制平面早停（预算/延迟）：进入 STOPPED_BUDGET 终态并优雅收尾。"""
        em.transition(RunState.STOPPED_BUDGET, reason)
        self.tracer.log(session.id, "stopped_budget", reason=reason)
        answer = (f"本次请求因资源上限（{reason}）被系统提前停止，已保留现有进展。"
                  f"可放宽预算或拆分任务后继续。")
        session.history.append({"role": "assistant", "content": answer})
        session.save()
        em.emit("run_end", answer=answer, error="stopped_budget", usage=run_usage)
        em.log_only(EVT_STATE_SNAPSHOT, **state.snapshot())
        return AgentResult(answer=answer, turns=state.turn, tool_calls=tool_calls_acc,
                           usage=run_usage, error=f"stopped_budget: {reason}",
                           run_id=run_id, final_state=state.current.value)

    # ------------------------------------------------------------------
    @staticmethod
    def _user_message(text: str,
                      attachments: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        if not attachments:
            return {"role": "user", "content": text}
        blocks: List[Dict[str, Any]] = list(attachments)
        if text:
            blocks.append({"type": "text", "text": text})
        return {"role": "user", "content": blocks}

    @staticmethod
    def _record_usage(session: Session, response: LLMResponse,
                      run_usage: Dict[str, int]) -> None:
        session.record_usage(response.usage)
        for key, value in response.usage.items():
            run_usage[key] = run_usage.get(key, 0) + int(value or 0)

    def _execute_tool(self, session: Session, emit, exec_ctx, name: str,
                      args: Dict[str, Any], call_id: str) -> tuple:
        # 全局记忆写走隔离视图（缓冲，run 结束提交）；本地记忆仍即时落库
        ctx = ToolContext(session=session, global_memory=exec_ctx.memory_view, emit=emit)
        self.tracer.log(session.id, "tool_start", tool=name, arguments=args)
        emit("tool_start", id=call_id, tool=name, arguments=args)
        started = time.time()
        # 经隔离层执行：timeout/retry/permission/rate-limit/circuit-breaker
        outcome = self.tool_runtime.run(name, args, ctx, exec_ctx, emit, call_id)
        duration_ms = int((time.time() - started) * 1000)
        if outcome.ok:
            self.tracer.log(session.id, "tool_result", tool=name, result=outcome.observation,
                            duration_ms=duration_ms, attempts=outcome.attempts)
            emit("tool_result", id=call_id, tool=name, result=outcome.observation,
                 duration_ms=duration_ms, attempts=outcome.attempts)
        else:
            self.tracer.log(session.id, "tool_error", tool=name, error=outcome.observation,
                            duration_ms=duration_ms, attempts=outcome.attempts)
            emit("tool_error", id=call_id, tool=name, error=outcome.observation,
                 duration_ms=duration_ms, attempts=outcome.attempts)
        return outcome.observation, outcome.ok


# 向后兼容别名：既有代码/测试以 Agent 命名控制器。
Agent = Controller
