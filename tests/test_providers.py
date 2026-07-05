"""Provider 层测试：工厂选择、Qwen 双向转换（离线，不依赖 SDK/网络）、解耦守卫。"""

import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from mini_agent import LLMError, create_llm
from mini_agent.core.llm_adapters import AnthropicLLM, QwenLLM
from mini_agent.core.llm_adapters import (
    StreamAccumulator, to_provider_messages, to_provider_tools,
)


class TestFactory(unittest.TestCase):
    def test_env_selects_qwen(self):
        with mock.patch.dict(os.environ, {"LLM_PROVIDER": "qwen",
                                          "DASHSCOPE_API_KEY": "sk-test"}, clear=False):
            llm = create_llm()
        self.assertIsInstance(llm, QwenLLM)

    def test_env_selects_anthropic_by_default(self):
        env = {k: v for k, v in os.environ.items() if k != "LLM_PROVIDER"}
        env["ANTHROPIC_API_KEY"] = "sk-test"
        with mock.patch.dict(os.environ, env, clear=True):
            llm = create_llm()
        self.assertIsInstance(llm, AnthropicLLM)

    def test_explicit_provider_overrides_env(self):
        with mock.patch.dict(os.environ, {"LLM_PROVIDER": "anthropic",
                                          "DASHSCOPE_API_KEY": "sk-test"}, clear=False):
            self.assertIsInstance(create_llm(provider="qwen"), QwenLLM)

    def test_model_override_via_env(self):
        with mock.patch.dict(os.environ, {"LLM_PROVIDER": "qwen",
                                          "DASHSCOPE_API_KEY": "sk-test",
                                          "LLM_MODEL": "qwen-max"}, clear=False):
            self.assertEqual(create_llm().model, "qwen-max")

    def test_missing_api_key_rejected(self):
        env = {k: v for k, v in os.environ.items()
               if k not in ("DASHSCOPE_API_KEY",)}
        env["LLM_PROVIDER"] = "qwen"
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(LLMError, "DASHSCOPE_API_KEY"):
                create_llm()

    def test_unknown_provider_rejected(self):
        with self.assertRaisesRegex(LLMError, "不支持的 LLM_PROVIDER"):
            create_llm(provider="gpt5-turbo-pro")


class TestQwenRequestConversion(unittest.TestCase):
    """框架 canonical 消息/ToolSpec -> chat.completions 格式。"""

    def test_system_blocks_merged_and_cache_control_ignored(self):
        system = [
            {"type": "text", "text": "核心", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "记忆索引"},
        ]
        out = to_provider_messages(system, [])
        self.assertEqual(out[0]["role"], "system")
        self.assertIn("核心", out[0]["content"])
        self.assertIn("记忆索引", out[0]["content"])
        self.assertNotIn("cache_control", out[0])

    def test_full_conversation_roundtrip_shape(self):
        messages = [
            {"role": "user", "content": "查天气"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "内部推理", "signature": "sig"},  # 应被跳过
                {"type": "text", "text": "我来查"},
                {"type": "tool_use", "id": "call_1", "name": "weather",
                 "input": {"city": "北京"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "call_1",
                 "content": "晴 26°C", "is_error": False},
            ]},
        ]
        out = to_provider_messages([], messages)

        self.assertEqual(out[0], {"role": "user", "content": "查天气"})
        assistant = out[1]
        self.assertEqual(assistant["role"], "assistant")
        self.assertEqual(assistant["content"], "我来查")
        self.assertNotIn("thinking", str(assistant))  # 思考不进请求
        tc = assistant["tool_calls"][0]
        self.assertEqual(tc["id"], "call_1")
        self.assertEqual(tc["function"]["name"], "weather")
        self.assertIn("北京", tc["function"]["arguments"])  # JSON 字符串
        tool_msg = out[2]
        self.assertEqual(tool_msg, {"role": "tool", "tool_call_id": "call_1",
                                    "content": "晴 26°C"})

    def test_error_result_marked(self):
        messages = [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "c1",
             "content": "城市不存在", "is_error": True}]}]
        out = to_provider_messages([], messages)
        self.assertTrue(out[0]["content"].startswith("[error]"))

    def test_image_block_converted_to_image_url(self):
        """多模态接口：canonical image block -> image_url 多模态分片。"""
        messages = [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64",
                                         "media_type": "image/jpeg", "data": "QUJD"}},
            {"type": "text", "text": "图里是什么？"},
        ]}]
        out = to_provider_messages([], messages)
        parts = out[0]["content"]
        self.assertIsInstance(parts, list)
        self.assertEqual(parts[0]["type"], "image_url")
        self.assertEqual(parts[0]["image_url"]["url"], "data:image/jpeg;base64,QUJD")
        self.assertEqual(parts[1], {"type": "text", "text": "图里是什么？"})

    def test_tool_specs_to_function_format(self):
        specs = [{"name": "calc", "description": "算数",
                  "input_schema": {"type": "object",
                                   "properties": {"expression": {"type": "string"}},
                                   "required": ["expression"]}}]
        out = to_provider_tools(specs)
        self.assertEqual(out[0]["type"], "function")
        fn = out[0]["function"]
        self.assertEqual(fn["name"], "calc")
        self.assertEqual(fn["parameters"]["required"], ["expression"])


def _chunk(content=None, reasoning=None, tool_calls=None, finish=None, usage=None):
    delta = SimpleNamespace(content=content, reasoning_content=reasoning,
                            tool_calls=tool_calls)
    choice = SimpleNamespace(delta=delta, finish_reason=finish)
    return SimpleNamespace(choices=[choice], usage=usage)


def _tc(index, id=None, name=None, arguments=None):
    return SimpleNamespace(index=index, id=id,
                           function=SimpleNamespace(name=name, arguments=arguments))


class TestQwenStreamAccumulation(unittest.TestCase):
    """流式 chunk 聚合 -> 统一 LLMResponse。"""

    def test_text_and_reasoning_stream(self):
        deltas = []
        acc = StreamAccumulator(lambda t, d: deltas.append((t, d["text"])))
        acc.feed(_chunk(reasoning="想一"))
        acc.feed(_chunk(reasoning="想二"))
        acc.feed(_chunk(content="你好"))
        acc.feed(_chunk(content="！", finish="stop"))
        response = acc.finalize()

        self.assertEqual(response.text, "你好！")
        self.assertEqual(response.thinking, "想一想二")
        self.assertEqual(response.stop_reason, "end_turn")
        self.assertEqual(deltas[0], ("thinking_delta", "想一"))
        self.assertEqual(deltas[-1], ("text_delta", "！"))
        # raw_content 为框架 canonical blocks；思考不写入历史、不伪造签名块
        self.assertEqual(response.raw_content, [{"type": "text", "text": "你好！"}])

    def test_tool_call_fragments_reassembled(self):
        acc = StreamAccumulator()
        acc.feed(_chunk(tool_calls=[_tc(0, id="call_abc", name="weather")]))
        acc.feed(_chunk(tool_calls=[_tc(0, arguments='{"cit')]))
        acc.feed(_chunk(tool_calls=[_tc(0, arguments='y": "北京"}')], finish="tool_calls"))
        response = acc.finalize()

        self.assertEqual(response.stop_reason, "tool_use")
        call = response.tool_calls[0]
        self.assertEqual((call.id, call.name, call.arguments),
                         ("call_abc", "weather", {"city": "北京"}))
        block = response.raw_content[0]
        self.assertEqual(block["type"], "tool_use")
        self.assertEqual(block["input"], {"city": "北京"})

    def test_parallel_tool_calls_by_index(self):
        acc = StreamAccumulator()
        acc.feed(_chunk(tool_calls=[_tc(0, id="c0", name="weather", arguments='{"city": "北京"}'),
                                    _tc(1, id="c1", name="weather", arguments='{"city": "上海"}')],
                        finish="tool_calls"))
        response = acc.finalize()
        self.assertEqual([c.arguments["city"] for c in response.tool_calls],
                         ["北京", "上海"])

    def test_invalid_arguments_degrade_to_empty(self):
        """参数流损坏时降级为 {}，交给工具层 schema 校验反馈模型自纠。"""
        acc = StreamAccumulator()
        acc.feed(_chunk(tool_calls=[_tc(0, id="c0", name="calc", arguments='{"broken')],
                        finish="tool_calls"))
        response = acc.finalize()
        self.assertEqual(response.tool_calls[0].arguments, {})

    def test_usage_mapping(self):
        usage = SimpleNamespace(
            prompt_tokens=120, completion_tokens=30,
            prompt_tokens_details=SimpleNamespace(cached_tokens=80))
        acc = StreamAccumulator()
        acc.feed(_chunk(content="ok", finish="stop"))
        acc.feed(SimpleNamespace(choices=[], usage=usage))  # usage 常在末尾空 choices 块
        response = acc.finalize()
        self.assertEqual(response.usage, {
            "input_tokens": 120, "output_tokens": 30,
            "cache_read_input_tokens": 80, "cache_creation_input_tokens": 0,
        })

    def test_length_finish_maps_to_max_tokens(self):
        acc = StreamAccumulator()
        acc.feed(_chunk(content="截断", finish="length"))
        self.assertEqual(acc.finalize().stop_reason, "max_tokens")


class TestDecouplingGuard(unittest.TestCase):
    """守卫测试：框架内核禁止出现任何 Provider 名称 —— 把解耦原则变成可执行约束。"""

    FORBIDDEN = ["anthropic", "claude", "qwen", "dashscope", "openai", "gpt", "gemini"]
    # 唯一允许出现 Provider 名称的地方：Provider 适配层与工厂
    PROVIDER_LAYER = {"llm_adapters.py", "llm_factory.py"}

    def test_core_modules_are_provider_free(self):
        """自动遍历整个 mini_agent 包树：除 Provider 适配层外，任何文件出现
        Provider 名称即失败。重构后目录变化无需手改清单。"""
        pkg = Path(__file__).resolve().parent.parent / "mini_agent"
        violations = []
        for path in sorted(pkg.rglob("*.py")):
            if "__pycache__" in path.parts or path.name in self.PROVIDER_LAYER:
                continue
            rel = path.relative_to(pkg)
            for i, line in enumerate(path.read_text(encoding="utf-8").lower().splitlines(), 1):
                for word in self.FORBIDDEN:
                    if word in line:
                        violations.append(f"{rel}:{i} 含 '{word}'")
        self.assertEqual(violations, [],
                         "框架内核泄漏了 Provider 名称:\n" + "\n".join(violations))


if __name__ == "__main__":
    unittest.main()
