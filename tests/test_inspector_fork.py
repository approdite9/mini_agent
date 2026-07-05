"""旗舰功能测试：运行审计（Run Inspector）与 会话分叉 + 跨模型接力。"""

import tempfile
import unittest
from pathlib import Path

from mini_agent import AgentService, ScriptedLLM, SessionManager
from mini_agent.observability.inspector import RunCollector, cache_hit_rate, estimate_cost, format_run


class TestCostAndCacheMetrics(unittest.TestCase):
    PRICING = {"input": 5.0, "output": 25.0,
               "cache_read_multiplier": 0.1, "cache_write_multiplier": 1.25,
               "currency": "USD"}

    def test_estimate_cost(self):
        usage = {"input_tokens": 1_000_000, "output_tokens": 100_000,
                 "cache_read_input_tokens": 2_000_000,
                 "cache_creation_input_tokens": 400_000}
        cost = estimate_cost(usage, self.PRICING)
        # 5 + 2.5 + 2M*5*0.1/1M=1.0 + 0.4M*5*1.25/1M=2.5
        self.assertAlmostEqual(cost["amount"], 5.0 + 2.5 + 1.0 + 2.5, places=6)
        self.assertEqual(cost["currency"], "USD")

    def test_no_pricing_returns_none(self):
        self.assertIsNone(estimate_cost({"input_tokens": 100}, None))

    def test_cache_hit_rate(self):
        self.assertAlmostEqual(
            cache_hit_rate({"input_tokens": 200, "cache_read_input_tokens": 800}), 0.8)
        self.assertIsNone(cache_hit_rate({}))


class TestRunCollector(unittest.TestCase):
    def _run(self, tmp, pricing=None):
        llm = ScriptedLLM([
            ScriptedLLM.call("calculator", {"expression": "2+2"}, thinking="算一下",
                             usage={"input_tokens": 100, "output_tokens": 10,
                                    "cache_read_input_tokens": 300,
                                    "cache_creation_input_tokens": 0}),
            ScriptedLLM.say("等于 4", usage={"input_tokens": 120, "output_tokens": 5,
                                             "cache_read_input_tokens": 0,
                                             "cache_creation_input_tokens": 0}),
        ])
        if pricing:
            llm.pricing = pricing
        service = AgentService(llm=llm, base_dir=Path(tmp))
        session = service.create_session()
        service.send(session.id, "2+2 等于几")
        return service, session

    def test_run_record_persisted_with_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            service, session = self._run(tmp)
            records = service.runs(session.id)
            self.assertEqual(len(records), 1)
            r = records[0]
            self.assertEqual(r["turns"], 2)
            self.assertEqual(r["model"], "scripted")
            self.assertEqual(r["tools_ok"], 1)
            self.assertEqual(r["tools_failed"], 0)
            self.assertEqual(r["tools"][0]["tool"], "calculator")
            self.assertGreaterEqual(r["tools"][0]["duration_ms"], 0)
            self.assertEqual(r["usage"]["input_tokens"], 220)
            # 缓存命中率 = 300 / (220 + 300)
            self.assertAlmostEqual(r["cache_hit_rate"], 300 / 520, places=3)
            self.assertIsNone(r["cost"])  # 无定价 -> 只看 token
            self.assertIn("2+2", r["input"])
            self.assertIn("等于 4", r["answer"])
            # 随会话文件持久化，重启可见
            reloaded = SessionManager(Path(tmp)).get(session.id)
            self.assertEqual(len(reloaded.runs), 1)

    def test_cost_computed_when_pricing_declared(self):
        pricing = {"input": 5.0, "output": 25.0, "cache_read_multiplier": 0.1,
                   "cache_write_multiplier": 1.25, "currency": "USD"}
        with tempfile.TemporaryDirectory() as tmp:
            service, session = self._run(tmp, pricing=pricing)
            cost = service.runs(session.id)[0]["cost"]
            expected = (220 * 5.0 + 15 * 25.0 + 300 * 5.0 * 0.1) / 1_000_000
            self.assertAlmostEqual(cost["amount"], round(expected, 6), places=6)

    def test_collector_forwards_events_downstream(self):
        events = []
        collector = RunCollector(events.append)
        with tempfile.TemporaryDirectory() as tmp:
            llm = ScriptedLLM([ScriptedLLM.say("ok")])
            service = AgentService(llm=llm, base_dir=Path(tmp))
            session = service.create_session()
            service.send(session.id, "hi", on_event=collector)
        self.assertIn("run_end", [e.type for e in events])  # 透传不丢事件

    def test_format_run_readable(self):
        with tempfile.TemporaryDirectory() as tmp:
            service, session = self._run(tmp)
            line = format_run(service.runs(session.id)[0])
            self.assertIn("2轮", line)
            self.assertIn("工具1✓", line)
            self.assertIn("cache", line)


class TestForkAndRelay(unittest.TestCase):
    def _service(self, tmp):
        """默认 llm + 可按 provider 名分发的 llm_factory（跨模型接力测试桩）。"""
        scripts = {
            "alt": ScriptedLLM([ScriptedLLM.say("我是接力模型，你之前说了你叫小明")],
                               label="alt/mock-model"),
        }
        return AgentService(
            llm=ScriptedLLM([ScriptedLLM.say("你好，小明！"),
                             ScriptedLLM.say("默认模型的回答")]),
            base_dir=Path(tmp),
            llm_factory=lambda provider=None, **kw: scripts[provider],
        )

    def test_fork_copies_history_and_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(tmp)
            s1 = service.create_session(title="主线")
            service.send(s1.id, "我叫小明")
            s1.plan = [{"title": "步骤", "status": "done"}]
            s1.save()

            fork = service.fork_session(s1.id)
            self.assertEqual(len(fork.history), len(s1.history))
            self.assertEqual(fork.plan, s1.plan)
            self.assertEqual(fork.parent, f"{s1.id}@{len(s1.history)}")
            # 深拷贝：改分叉不影响主线
            fork.history.append({"role": "user", "content": "只在分叉里"})
            self.assertNotIn("只在分叉里", str(s1.history))

    def test_fork_at_message_boundary_validated(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(tmp)
            s1 = service.create_session()
            service.send(s1.id, "我叫小明")  # 历史: [user, assistant]
            fork = service.fork_session(s1.id, at_message=2)
            self.assertEqual(len(fork.history), 2)
            with self.assertRaisesRegex(ValueError, "非法分叉点"):
                service.fork_session(s1.id, at_message=1)  # 切在 user 后 -> 不完整

    def test_cross_model_relay(self):
        """跨模型接力：换 provider 后，新模型收到同一份 canonical 历史继续对话。"""
        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(tmp)
            s1 = service.create_session()
            service.send(s1.id, "我叫小明")               # 默认模型回答

            label = service.set_provider(s1.id, "alt")     # 中途换模型
            self.assertEqual(label, "alt/mock-model")
            self.assertEqual(service.get_session(s1.id).provider, "alt")

            result = service.send(s1.id, "我叫什么？")
            self.assertIn("接力模型", result.answer)
            # 新模型确实收到了旧模型时期的历史（Provider 中立格式直接可用）
            alt_llm = service._llm_for("alt")
            seen = str(alt_llm.calls[0]["messages"])
            self.assertIn("我叫小明", seen)
            self.assertIn("你好，小明", seen)
            # 审计记录标注了各自的模型
            models = [r["model"] for r in service.runs(s1.id)]
            self.assertEqual(models, ["scripted", "alt/mock-model"])

    def test_set_provider_back_to_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(tmp)
            s1 = service.create_session(provider="alt")
            self.assertEqual(s1.provider, "alt")
            service.set_provider(s1.id, "default")
            self.assertIsNone(service.get_session(s1.id).provider)

    def test_fork_with_provider_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = self._service(tmp)
            s1 = service.create_session()
            service.send(s1.id, "我叫小明")
            fork = service.fork_session(s1.id, provider="alt")
            self.assertEqual(fork.provider, "alt")
            result = service.send(fork.id, "我叫什么？")
            self.assertIn("接力模型", result.answer)


if __name__ == "__main__":
    unittest.main()
