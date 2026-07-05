"""真实 LLM API 集成测试（按需运行，默认自动跳过，不产生意外费用）。

运行方式：
    export MINI_AGENT_LIVE=1
    export ANTHROPIC_API_KEY=sk-...        # 跑 Claude 用例
    export DASHSCOPE_API_KEY=sk-...        # 跑 Qwen 用例（两者都设则跑接力用例）
    python -m unittest tests.test_live_llm -v

验证的是离线测试覆盖不到的部分：真实流式增量、真实原生工具调用决策、
真实 usage 字段、跨厂商接力在真实 API 上成立。用例刻意小而便宜。
"""

import os
import tempfile
import unittest
from pathlib import Path

from mini_agent import Agent, AgentService, MemoryStore, Session, build_default_registry
from mini_agent import create_llm
from mini_agent.core.llm_adapters import AnthropicLLM, QwenLLM

LIVE = os.environ.get("MINI_AGENT_LIVE") == "1"
HAS_ANTHROPIC = bool(os.environ.get("ANTHROPIC_API_KEY"))
HAS_QWEN = bool(os.environ.get("DASHSCOPE_API_KEY"))

try:
    import anthropic  # noqa: F401
    HAS_ANTHROPIC_SDK = True
except ImportError:
    HAS_ANTHROPIC_SDK = False
try:
    import openai  # noqa: F401
    HAS_OPENAI_SDK = True
except ImportError:
    HAS_OPENAI_SDK = False

anthropic_ready = LIVE and HAS_ANTHROPIC and HAS_ANTHROPIC_SDK
qwen_ready = LIVE and HAS_QWEN and HAS_OPENAI_SDK


def _tool_loop_assertions(case, llm):
    """通用断言：真实模型自主决策调用 calculator 并给出正确答案。"""
    events = []
    agent = Agent(llm=llm, registry=build_default_registry(),
                  global_memory=MemoryStore(), max_tool_turns=5)
    session = Session(id="live1")
    result = agent.run(
        session,
        "请调用 calculator 工具计算 137*4，然后用一句话告诉我结果。",
        on_event=events.append)

    case.assertIsNone(result.error, f"run 失败: {result.error} / {result.answer}")
    calc_calls = [c for c in result.tool_calls if c["name"] == "calculator"]
    case.assertGreaterEqual(len(calc_calls), 1, "模型未调用 calculator")
    case.assertTrue(any(c["result"] == "548" for c in calc_calls))
    case.assertIn("548", result.answer)
    # 事件生命周期在真实流上同样成立
    types = [e.type for e in events]
    case.assertIn("tool_start", types)
    case.assertIn("tool_result", types)
    case.assertIn("assistant_delta", types)
    case.assertEqual(types.count("thinking_start"), types.count("thinking_end"))
    # 真实 usage
    case.assertGreater(result.usage.get("input_tokens", 0)
                       + result.usage.get("cache_read_input_tokens", 0)
                       + result.usage.get("cache_creation_input_tokens", 0), 0)
    case.assertGreater(result.usage.get("output_tokens", 0), 0)


@unittest.skipUnless(anthropic_ready,
                     "需 MINI_AGENT_LIVE=1 + ANTHROPIC_API_KEY + anthropic SDK")
class TestLiveAnthropic(unittest.TestCase):
    def test_stream_and_response_shape(self):
        llm = AnthropicLLM(max_tokens=2048)
        deltas = []
        response = llm.complete(
            system=[{"type": "text", "text": "只输出答案本身，不要解释。"}],
            messages=[{"role": "user", "content": "1+1=?（只回答数字）"}],
            on_delta=lambda t, d: deltas.append(t),
        )
        self.assertIn("2", response.text)
        self.assertEqual(response.stop_reason, "end_turn")
        self.assertIn("text_delta", deltas)  # 真实流式增量
        self.assertGreater(response.usage["output_tokens"], 0)

    def test_native_tool_loop(self):
        _tool_loop_assertions(self, AnthropicLLM(max_tokens=4096))


@unittest.skipUnless(qwen_ready,
                     "需 MINI_AGENT_LIVE=1 + DASHSCOPE_API_KEY + openai SDK")
class TestLiveQwen(unittest.TestCase):
    def test_stream_and_response_shape(self):
        llm = QwenLLM(max_tokens=512)
        deltas = []
        response = llm.complete(
            system=[{"type": "text", "text": "只输出答案本身，不要解释。"}],
            messages=[{"role": "user", "content": "1+1=?（只回答数字）"}],
            on_delta=lambda t, d: deltas.append(t),
        )
        self.assertIn("2", response.text)
        self.assertIn("text_delta", deltas)
        self.assertGreater(response.usage["output_tokens"], 0)

    def test_native_tool_loop(self):
        _tool_loop_assertions(self, QwenLLM(max_tokens=2048))


@unittest.skipUnless(anthropic_ready and qwen_ready,
                     "跨厂商接力需同时具备两个 Provider 的 key 与 SDK")
class TestLiveCrossProviderRelay(unittest.TestCase):
    def test_relay_preserves_context_across_vendors(self):
        """真实跨厂商接力：厂商 A 记住的事实，换到厂商 B 能接着用。"""
        with tempfile.TemporaryDirectory() as tmp:
            service = AgentService(
                llm=create_llm(provider="anthropic"),
                base_dir=Path(tmp),
                llm_factory=create_llm,
            )
            session = service.create_session()
            service.send(session.id,
                         "我的项目代号是「蓝鲸七号」。请只回复：收到。")

            service.set_provider(session.id, "qwen")
            result = service.send(session.id, "我的项目代号是什么？只回答代号本身。")

            self.assertIn("蓝鲸七号", result.answer)
            models = [r["model"] for r in service.runs(session.id)]
            self.assertTrue(models[0].startswith("anthropic/"))
            self.assertTrue(models[1].startswith("qwen/"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
