"""执行语义内核测试：状态机、事件日志、从日志重建、Controller 的控制迁移与可复现。"""

import json
import tempfile
import unittest
from pathlib import Path

from mini_agent import AgentService, ScriptedLLM
from mini_agent.core.events import (
    EVT_CONTROL_TRANSITION, EVT_STATE_SNAPSHOT, ExecutionEvent,
)
from mini_agent.core.llm import LLMClient, LLMResponse
from mini_agent.runtime import (
    EventLog, ExecutionState, IllegalTransition, Replayer, RunState,
)


class NonStreamingLLM(LLMClient):
    """返回最终文本但从不调用 on_delta 的 Provider（模拟 qwen-max 不逐字流式）。"""

    def __init__(self, text: str):
        self._text = text
        self.pricing = None

    def describe(self) -> str:
        return "nonstream"

    def complete(self, *, system, messages, tools=None, on_delta=None) -> LLMResponse:
        return LLMResponse(stop_reason="end_turn", text=self._text,
                           raw_content=[{"type": "text", "text": self._text}],
                           usage={"input_tokens": 10, "output_tokens": 5})


class TestStateMachine(unittest.TestCase):
    def test_legal_happy_path(self):
        s = ExecutionState(run_id="r1", session_id="s1")
        self.assertEqual(s.current, RunState.CREATED)
        s.transition(RunState.RUNNING, "user_input")
        s.transition(RunState.MODEL_CALL, "need_action")
        s.transition(RunState.TOOL_DISPATCH, "tool_use")
        s.transition(RunState.TOOL_EXEC, "exec_tool")
        s.transition(RunState.TOOL_EXEC, "exec_tool")   # 并行多工具，合法自环
        s.transition(RunState.RUNNING, "tools_done")
        s.transition(RunState.MODEL_CALL, "need_action")
        s.transition(RunState.COMPLETED, "end_turn")
        self.assertTrue(s.is_terminal)
        self.assertEqual(len(s.history), 8)

    def test_illegal_transition_raises(self):
        s = ExecutionState(run_id="r", session_id="s")
        # CREATED 只能到 RUNNING
        with self.assertRaises(IllegalTransition):
            s.transition(RunState.MODEL_CALL, "skip")
        s.transition(RunState.RUNNING, "ok")
        s.transition(RunState.MODEL_CALL, "ok")
        # MODEL_CALL 不能直接回 RUNNING（必须经 TOOL 或终态）
        with self.assertRaises(IllegalTransition):
            s.transition(RunState.RUNNING, "bad")

    def test_terminal_states(self):
        for terminal in (RunState.COMPLETED, RunState.FAILED,
                         RunState.STOPPED_BUDGET, RunState.STOPPED_MAX_TURNS):
            self.assertTrue(terminal.is_terminal)
        self.assertFalse(RunState.RUNNING.is_terminal)

    def test_snapshot_and_reconstruct_roundtrip(self):
        s = ExecutionState(run_id="r1", session_id="s1")
        s.transition(RunState.RUNNING, "user_input")
        s.transition(RunState.MODEL_CALL, "need_action")
        s.transition(RunState.COMPLETED, "end_turn")
        snap = s.snapshot()
        # 快照可 JSON 序列化
        json.dumps(snap)
        rebuilt = ExecutionState.reconstruct(snap["transitions"], "r1", "s1")
        self.assertEqual(rebuilt.current, RunState.COMPLETED)
        self.assertEqual([t.to_dict() for t in rebuilt.history],
                         [t.to_dict() for t in s.history])


class TestEventLog(unittest.TestCase):
    def test_in_memory_mode(self):
        log = EventLog("r1", runs_dir=None)
        log.append(ExecutionEvent(type="run_start", session_id="s1", run_id="r1"))
        log.append(ExecutionEvent(type="run_end", session_id="s1", run_id="r1"))
        events = log.events()
        self.assertEqual([e["type"] for e in events], ["run_start", "run_end"])
        self.assertIsNone(log.path)

    def test_jsonl_persistence_and_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            with EventLog("r1", runs_dir=Path(tmp)) as log:
                log.append(ExecutionEvent(type="a", session_id="s", run_id="r1", data={"x": 1}))
                log.append(ExecutionEvent(
                    type=EVT_CONTROL_TRANSITION, session_id="s", run_id="r1",
                    from_state="created", to_state="running", reason="user_input"))
            path = Path(tmp) / "r1.jsonl"
            self.assertTrue(path.is_file())
            lines = path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)
            reloaded = Replayer.from_file(path)
            self.assertEqual([e["type"] for e in reloaded.events], ["a", EVT_CONTROL_TRANSITION])
            # 语义字段随盘往返
            self.assertEqual(reloaded.events[1]["reason"], "user_input")


class TestControllerSemantics(unittest.TestCase):
    """Controller 把执行显式化为状态迁移，并全程落可复现的事件日志。"""

    def _service(self, tmp, script):
        return AgentService(llm=ScriptedLLM(script), base_dir=Path(tmp))

    def test_run_emits_control_transitions_in_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(tmp, [
                ScriptedLLM.call("calculator", {"expression": "2+2"}),
                ScriptedLLM.say("等于 4"),
            ])
            sid = service.create_session().id
            events = []
            result = service.send(sid, "算 2+2", on_event=events.append)

            transitions = [(e.from_state, e.to_state, e.reason)
                           for e in events if e.type == EVT_CONTROL_TRANSITION]
            # 关键迁移语义齐全且顺序合理
            tos = [t[1] for t in transitions]
            self.assertEqual(tos[0], "running")            # 首个迁移进入 RUNNING
            self.assertIn("model_call", tos)
            self.assertIn("tool_dispatch", tos)
            self.assertIn("tool_exec", tos)
            self.assertEqual(tos[-1], "completed")         # 终态 COMPLETED
            # 每个迁移都带 reason（“为什么发生”）
            self.assertTrue(all(r for _, _, r in transitions))
            self.assertEqual(result.final_state, "completed")
            # run_end 仍是 on_event 流的最后一个事件（UI 契约不破）
            self.assertEqual(events[-1].type, "run_end")

    def test_run_is_reconstructable_from_persisted_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(tmp, [
                ScriptedLLM.call("weather", {"city": "北京"}),
                ScriptedLLM.say("北京晴"),
            ])
            sid = service.create_session().id
            result = service.send(sid, "北京天气")

            # 事件日志已落盘
            runs_dir = Path(tmp) / "runs"
            log_path = runs_dir / f"{result.run_id}.jsonl"
            self.assertTrue(log_path.is_file())

            # 从日志重建的状态时间线与在线 run 的终态一致
            replayer = Replayer.from_run(result.run_id, runs_dir)
            rebuilt = replayer.reconstruct_state()
            self.assertEqual(rebuilt.current.value, result.final_state)
            self.assertEqual(rebuilt.current, RunState.COMPLETED)

            # 日志完整记录了模型响应（供确定性回放）与状态快照
            types = [e["type"] for e in replayer.events]
            self.assertIn("model_response", types)
            self.assertIn(EVT_STATE_SNAPSHOT, types)
            # 工具执行单元在日志中成对
            starts = {e["data"]["id"] for e in replayer.events if e["type"] == "tool_start"}
            results = {e["data"]["id"] for e in replayer.events
                       if e["type"] in ("tool_result", "tool_error")}
            self.assertEqual(starts, results)

    def test_max_turns_reaches_stopped_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = AgentService(
                llm=ScriptedLLM([ScriptedLLM.call("search", {"query": "x"})] * 5),
                base_dir=Path(tmp), max_tool_turns=3)
            sid = service.create_session().id
            result = service.send(sid, "一直查")
            self.assertEqual(result.error, "max_turns_exceeded")
            self.assertEqual(result.final_state, RunState.STOPPED_MAX_TURNS.value)

    def test_llm_error_reaches_failed_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(tmp, [])  # 立即耗尽 -> LLMError
            sid = service.create_session().id
            result = service.send(sid, "hi")
            self.assertTrue(result.error.startswith("llm_error"))
            self.assertEqual(result.final_state, RunState.FAILED.value)

    def test_non_streaming_provider_still_carries_answer(self):
        """回归：即使 Provider 不逐字流式（无 assistant_delta），最终答案仍必须
        通过 assistant_final 与 run_end 携带，供任意前端渲染。"""
        with tempfile.TemporaryDirectory() as tmp:
            service = AgentService(
                llm=NonStreamingLLM("我是通义千问。"), base_dir=Path(tmp))
            sid = service.create_session().id
            events = []
            result = service.send(sid, "你是什么模型", on_event=events.append)

            types = [e.type for e in events]
            self.assertNotIn("assistant_delta", types)  # 确实没有逐字流式
            # 但答案通过权威事件到达前端
            finals = [e for e in events if e.type == "assistant_final"]
            self.assertEqual(finals[0].data["content"], "我是通义千问。")
            run_end = [e for e in events if e.type == "run_end"][0]
            self.assertEqual(run_end.data["answer"], "我是通义千问。")
            self.assertEqual(result.answer, "我是通义千问。")


if __name__ == "__main__":
    unittest.main()
