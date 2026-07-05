"""Phase 3 测试：录制-回放（确定性复现）、从日志重建、系统级指标、持久化可观测。"""

import tempfile
import unittest
from pathlib import Path

from mini_agent import (
    AgentService, Controller, MemoryStore, ReplayLLM, ScriptedLLM, Session,
    build_default_registry, compute_run_metrics,
)
from mini_agent.observability.metrics import aggregate_metrics
from mini_agent.runtime.replay import Replayer
from mini_agent.runtime.state import RunState


def _service(tmp, script, **kw):
    return AgentService(llm=ScriptedLLM(script), base_dir=Path(tmp), **kw)


class TestRecordAndReplay(unittest.TestCase):
    def test_replay_reproduces_trajectory_frame_by_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            svc = _service(tmp, [
                ScriptedLLM.call("calculator", {"expression": "6*7"}, thinking="算一下"),
                ScriptedLLM.say("等于 42"),
            ])
            sid = svc.create_session().id
            original = svc.send(sid, "算 6*7")
            orig_tr = [(t["from"], t["to"], t["reason"])
                       for t in svc.run_transitions(original.run_id)]

            # ReplayLLM 回放录制响应，重跑全新会话
            rllm = ReplayLLM.from_run(original.run_id, svc.runs_dir)
            replay_dir = Path(tmp) / "replay"
            ctrl = Controller(llm=rllm, registry=build_default_registry(),
                              global_memory=MemoryStore(), runs_dir=replay_dir)
            replayed = ctrl.run(Session(id="rep"), "算 6*7")
            replay_tr = [(t["from"], t["to"], t["reason"])
                         for t in Replayer.from_run(replayed.run_id, replay_dir).transitions()]

            self.assertEqual(orig_tr, replay_tr)                 # 逐帧一致
            self.assertEqual(replayed.final_state, original.final_state)
            self.assertEqual(replayed.answer, original.answer)

    def test_replay_makes_no_real_api_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            svc = _service(tmp, [ScriptedLLM.call("search", {"query": "x"}),
                                 ScriptedLLM.say("done")])
            sid = svc.create_session().id
            r = svc.send(sid, "查")
            rllm = ReplayLLM.from_run(r.run_id, svc.runs_dir)
            ctrl = Controller(llm=rllm, registry=build_default_registry(),
                              global_memory=MemoryStore())
            ctrl.run(Session(id="rep"), "查")
            # ReplayLLM 只回放录制的 2 个响应，不产生任何真实请求
            self.assertEqual(len(rllm.calls), 2)

    def test_replay_exhaustion_raises_on_divergence(self):
        # 只喂 1 个录制响应，但轨迹需要 2 次 -> 明确暴露"偏离录制"
        rllm = ReplayLLM([{"stop_reason": "tool_use", "text": "", "thinking": "",
                           "raw_content": [], "usage": {},
                           "tool_calls": [{"id": "t1", "name": "search",
                                           "arguments": {"query": "x"}}]}])
        ctrl = Controller(llm=rllm, registry=build_default_registry(),
                          global_memory=MemoryStore())
        result = ctrl.run(Session(id="div"), "查")
        # 第 2 轮回放耗尽 -> LLMError -> FAILED 终态（不静默）
        self.assertEqual(result.final_state, RunState.FAILED.value)


class TestLogReconstruction(unittest.TestCase):
    def test_reconstruct_state_matches_online_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            svc = _service(tmp, [ScriptedLLM.say("你好")])
            sid = svc.create_session().id
            r = svc.send(sid, "hi")
            rebuilt = Replayer.from_run(r.run_id, svc.runs_dir).reconstruct_state()
            self.assertEqual(rebuilt.current.value, r.final_state)

    def test_iter_states_yields_incremental_states(self):
        with tempfile.TemporaryDirectory() as tmp:
            svc = _service(tmp, [ScriptedLLM.call("calculator", {"expression": "1+1"}),
                                 ScriptedLLM.say("2")])
            sid = svc.create_session().id
            r = svc.send(sid, "算")
            states = [state.current.value
                      for _event, state in Replayer.from_run(r.run_id, svc.runs_dir).iter_states()]
            # 起于 created，止于 completed，单调推进
            self.assertEqual(states[0], "created")
            self.assertEqual(states[-1], "completed")
            self.assertIn("tool_exec", states)


class TestMetrics(unittest.TestCase):
    def test_run_metrics_from_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            svc = _service(tmp, [
                ScriptedLLM.call("weather", {"city": "北平"}),   # 失败（未知城市）
                ScriptedLLM.call("weather", {"city": "北京"}),   # 成功
                ScriptedLLM.say("北京晴"),
            ])
            sid = svc.create_session().id
            r = svc.send(sid, "北京天气")
            m = compute_run_metrics(svc.run_events(r.run_id))
            self.assertEqual(m["final_state"], "completed")
            self.assertEqual(m["counters"]["tool_calls"], 2)
            self.assertEqual(m["counters"]["tool_ok"], 1)
            self.assertEqual(m["counters"]["tool_error"], 1)
            self.assertEqual(m["counters"]["turns"], 3)
            self.assertGreater(m["counters"]["transitions"], 0)
            self.assertEqual(m["tool_latency_ms"]["count"], 2)

    def test_aggregate_over_multiple_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            svc = _service(tmp, [
                ScriptedLLM.say("a"),
                ScriptedLLM.call("calculator", {"expression": "1+1"}),
                ScriptedLLM.say("b"),
            ])
            sid = svc.create_session().id
            svc.send(sid, "第一次")
            svc.send(sid, "第二次")
            agg = svc.metrics(sid)["aggregate"]
            self.assertEqual(agg["runs"], 2)
            self.assertEqual(agg["counters"]["tool_ok"], 1)
            self.assertEqual(agg["final_states"], {"completed": 2})
            self.assertEqual(agg["tool_success_rate"], 1.0)


class TestPersistedObservability(unittest.TestCase):
    def test_runs_survive_process_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            svc = _service(tmp, [ScriptedLLM.say("ok")])
            sid = svc.create_session().id
            r = svc.send(sid, "hi")

            # 模拟进程重启：新 service 实例，同 base_dir
            svc2 = AgentService(llm=ScriptedLLM([]), base_dir=Path(tmp))
            self.assertIn(r.run_id, svc2.list_runs(sid))
            events = svc2.run_events(r.run_id)   # 事件日志重启后仍可读
            self.assertTrue(any(e["type"] == "run_start" for e in events))
            # 重启后仍可复算指标
            self.assertEqual(svc2.metrics(sid)["aggregate"]["runs"], 1)


if __name__ == "__main__":
    unittest.main()
