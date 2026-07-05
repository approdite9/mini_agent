"""Agent 内核测试：原生工具循环、并行调用、错误恢复、轮次上限、事件流、追问。"""

import unittest

from mini_agent import (
    Agent, ContextManager, MemoryStore, ScriptedLLM, Session, Tracer,
    build_default_registry,
)


def make_agent(script, **kwargs):
    tracer = Tracer()
    memory = MemoryStore()
    agent = Agent(
        llm=ScriptedLLM(script),
        registry=build_default_registry(),
        global_memory=memory,
        tracer=tracer,
        **kwargs,
    )
    return agent, tracer, memory


def collect_events(agent, session, text):
    events = []
    result = agent.run(session, text, on_event=events.append)
    return result, events


class TestAgentLoop(unittest.TestCase):
    def test_direct_answer_no_tool(self):
        agent, tracer, _ = make_agent(
            [ScriptedLLM.say("你好！", thinking="不需要工具")])
        session = Session(id="a1")
        result, events = collect_events(agent, session, "你好")

        self.assertEqual(result.answer, "你好！")
        self.assertEqual(result.turns, 1)
        self.assertEqual(result.tool_calls, [])
        self.assertIsNone(result.error)
        # 事件流完整性
        types = [e.type for e in events]
        self.assertEqual(types[0], "run_start")
        self.assertEqual(types[-1], "run_end")
        self.assertIn("thinking_delta", types)
        self.assertIn("assistant_delta", types)
        self.assertIn("usage", types)
        # 思考层带生命周期边界，且在回答之前完整闭合
        self.assertLess(types.index("thinking_start"), types.index("thinking_delta"))
        self.assertLess(types.index("thinking_delta"), types.index("thinking_end"))
        self.assertLess(types.index("thinking_end"), types.index("assistant_delta"))
        thinking_end = [e for e in events if e.type == "thinking_end"][0]
        self.assertEqual(thinking_end.data["content"], "不需要工具")
        self.assertGreaterEqual(thinking_end.data["duration_ms"], 0)
        # 回答有完整内容事件（UI 不需要自己拼 delta）
        final = [e for e in events if e.type == "assistant_final"][0]
        self.assertEqual(final.data["content"], "你好！")
        # 思考过程进 trace
        self.assertEqual(tracer.of_type("thinking")[0].data["content"], "不需要工具")

    def test_tool_call_then_answer(self):
        agent, tracer, _ = make_agent([
            ScriptedLLM.call("calculator", {"expression": "6*7"}, thinking="要算数"),
            ScriptedLLM.say("结果是 42"),
        ])
        session = Session(id="a2")
        result, events = collect_events(agent, session, "6*7 等于多少")

        self.assertEqual(result.answer, "结果是 42")
        self.assertEqual(result.turns, 2)
        call = result.tool_calls[0]
        self.assertEqual((call["name"], call["result"], call["ok"]), ("calculator", "42", True))

        # 历史：assistant 带原生 tool_use block；下一条 user 带 tool_result block
        assistant_msg = session.history[1]
        self.assertEqual(assistant_msg["role"], "assistant")
        tool_use = [b for b in assistant_msg["content"] if b["type"] == "tool_use"][0]
        result_msg = session.history[2]
        self.assertEqual(result_msg["role"], "user")
        self.assertEqual(result_msg["content"][0]["type"], "tool_result")
        self.assertEqual(result_msg["content"][0]["tool_use_id"], tool_use["id"])
        self.assertFalse(result_msg["content"][0]["is_error"])
        # 事件流里有完整的工具执行单元：start -> result（带延迟）
        types = [e.type for e in events]
        self.assertIn("tool_start", types)
        self.assertIn("tool_result", types)
        result_event = [e for e in events if e.type == "tool_result"][0]
        self.assertGreaterEqual(result_event.data["duration_ms"], 0)

    def test_parallel_tool_calls_merged_in_one_message(self):
        """一轮多个 tool_use：所有 tool_result 必须合并进同一条 user 消息（API 要求）。"""
        multi = {"text": "", "thinking": "", "tool_calls": [
            {"name": "weather", "arguments": {"city": "北京"}},
            {"name": "weather", "arguments": {"city": "上海"}},
        ]}
        agent, _, _ = make_agent([multi, ScriptedLLM.say("两地天气已查")])
        session = Session(id="a3")
        result, _ = collect_events(agent, session, "北京和上海的天气")

        self.assertEqual(len(result.tool_calls), 2)
        result_msg = session.history[2]
        blocks = result_msg["content"]
        self.assertEqual(len(blocks), 2)
        self.assertTrue(all(b["type"] == "tool_result" for b in blocks))
        # id 与 tool_use 一一对应
        use_ids = [b["id"] for b in session.history[1]["content"] if b["type"] == "tool_use"]
        self.assertEqual([b["tool_use_id"] for b in blocks], use_ids)

    def test_tool_error_fed_back_and_recovered(self):
        agent, tracer, _ = make_agent([
            ScriptedLLM.call("weather", {"city": "北平"}),
            ScriptedLLM.call("weather", {"city": "北京"}),
            ScriptedLLM.say("北京今天晴。"),
        ])
        session = Session(id="a4")
        result, _ = collect_events(agent, session, "北平天气如何")

        self.assertEqual(result.answer, "北京今天晴。")
        self.assertFalse(result.tool_calls[0]["ok"])
        self.assertTrue(result.tool_calls[1]["ok"])
        # 错误经 is_error=true 的 tool_result 反馈给模型
        error_block = session.history[2]["content"][0]
        self.assertTrue(error_block["is_error"])
        self.assertEqual(len(tracer.of_type("tool_error")), 1)

    def test_unknown_tool_handled(self):
        agent, _, _ = make_agent([
            ScriptedLLM.call("teleport", {}),
            ScriptedLLM.say("抱歉，我没有那个能力。"),
        ])
        session = Session(id="a5")
        result, _ = collect_events(agent, session, "传送我回家")
        self.assertIsNone(result.error)
        self.assertIn("未知工具", result.tool_calls[0]["result"])

    def test_max_turns_exceeded(self):
        script = [ScriptedLLM.call("search", {"query": "ReAct"}) for _ in range(5)]
        agent, tracer, _ = make_agent(script, max_tool_turns=3)
        session = Session(id="a6")
        result, events = collect_events(agent, session, "一直查")

        self.assertEqual(result.error, "max_turns_exceeded")
        self.assertEqual(result.turns, 3)
        self.assertEqual(len(result.tool_calls), 3)
        self.assertIn("最大工具调用轮次", result.answer)
        self.assertEqual(events[-1].type, "run_end")
        self.assertEqual(events[-1].data["error"], "max_turns_exceeded")

    def test_llm_error_graceful(self):
        agent, tracer, _ = make_agent([])
        session = Session(id="a7")
        result, events = collect_events(agent, session, "hi")
        self.assertTrue(result.error.startswith("llm_error"))
        self.assertIn("失败", result.answer)
        self.assertIn("error", [e.type for e in events])
        self.assertEqual(session.history[-1]["role"], "assistant")  # 会话仍可继续

    def test_followup_sees_previous_context(self):
        agent, _, _ = make_agent([
            ScriptedLLM.say("你好，小明！"),
            ScriptedLLM.say("你叫小明。"),
        ])
        session = Session(id="a8")
        agent.run(session, "我叫小明")
        agent.run(session, "我叫什么？")
        contents = []
        for m in agent.llm.calls[1]["messages"]:
            c = m["content"]
            contents.append(c if isinstance(c, str) else str(c))
        joined = " ".join(contents)
        self.assertIn("我叫小明", joined)
        self.assertIn("你好，小明", joined)

    def test_run_usage_accumulated(self):
        agent, _, _ = make_agent([
            ScriptedLLM.call("calculator", {"expression": "1+1"},
                             usage={"input_tokens": 100, "output_tokens": 10,
                                    "cache_read_input_tokens": 0, "cache_creation_input_tokens": 50}),
            ScriptedLLM.say("2", usage={"input_tokens": 120, "output_tokens": 5,
                                        "cache_read_input_tokens": 150,
                                        "cache_creation_input_tokens": 0}),
        ])
        session = Session(id="a9")
        result, _ = collect_events(agent, session, "1+1")
        self.assertEqual(result.usage["input_tokens"], 220)
        self.assertEqual(result.usage["cache_read_input_tokens"], 150)
        # 会话累计 + 最近一轮上下文规模
        self.assertEqual(session.total_usage["input_tokens"], 220)
        self.assertEqual(session.context_tokens(), 120 + 150)

    def test_compaction_triggered_in_run(self):
        """上下文超阈值时，run 内自动压缩后再调用 LLM。"""
        cm = ContextManager(max_context_tokens=500, keep_recent_messages=2)
        agent, tracer, _ = make_agent(
            [ScriptedLLM.say("压缩摘要在此"),   # 第 1 次调用被 compact 的 summarize 消耗
             ScriptedLLM.say("继续回答")],
            context_manager=cm)
        session = Session(id="a10")
        for i in range(8):
            session.history.append({"role": "user", "content": f"旧问题{i}"})
            session.history.append({"role": "assistant", "content": [{"type": "text", "text": f"旧回答{i}"}]})
        session.last_usage = {"input_tokens": 99999}  # 模拟真实 usage 超限

        result, events = collect_events(agent, session, "新问题")
        self.assertEqual(result.answer, "继续回答")
        self.assertIn("compaction", [e.type for e in events])
        self.assertIn("[前情摘要", session.history[0]["content"])

    def test_plan_tool_emits_plan_update(self):
        agent, _, _ = make_agent([
            ScriptedLLM.call("plan", {"action": "set", "steps": ["调研", "实现"]}),
            ScriptedLLM.say("计划已定，开始执行。"),
        ])
        session = Session(id="a11")
        result, events = collect_events(agent, session, "帮我完成一个复杂任务")
        plan_events = [e for e in events if e.type == "plan_update"]
        self.assertEqual(len(plan_events), 1)
        self.assertEqual(len(session.plan), 2)
        self.assertEqual(session.plan[0]["title"], "调研")

    def test_memory_update_event_emitted(self):
        agent, _, _ = make_agent([
            ScriptedLLM.call("memory", {"action": "save", "key": "城市", "value": "北京"}),
            ScriptedLLM.say("记住了"),
        ])
        session = Session(id="a13")
        _, events = collect_events(agent, session, "记住我在北京")
        mem_events = [e for e in events if e.type == "memory_update"]
        self.assertEqual(len(mem_events), 1)
        self.assertEqual(mem_events[0].data["key"], "城市")
        self.assertIn("全局", mem_events[0].data["scope"])

    def test_multimodal_attachments_interface(self):
        """多模态接口：附件以框架 canonical image block 进入历史。"""
        agent, _, _ = make_agent([ScriptedLLM.say("我看到了一张图")])
        session = Session(id="a14")
        image = {"type": "image", "source": {"type": "base64",
                                             "media_type": "image/png", "data": "aGk="}}
        agent.run(session, "这是什么？", attachments=[image])

        user_msg = session.history[0]
        self.assertIsInstance(user_msg["content"], list)
        self.assertEqual(user_msg["content"][0]["type"], "image")
        self.assertEqual(user_msg["content"][1], {"type": "text", "text": "这是什么？"})
        # Provider 收到的就是 canonical 块（是否支持由 Provider 转换层决定）
        sent = agent.llm.calls[0]["messages"][0]["content"]
        self.assertEqual(sent[0]["type"], "image")

    def test_memory_index_in_system_prompt(self):
        agent, _, memory = make_agent([ScriptedLLM.say("好的")])
        memory.set("常驻城市", "北京")
        session = Session(id="a12")
        agent.run(session, "hi")
        system = agent.llm.calls[0]["system"]
        self.assertIn("常驻城市: 北京", system[1]["text"])
        self.assertIn("全局共享", system[1]["text"])
        # 工具以原生 schema 传入
        tool_names = [t["name"] for t in agent.llm.calls[0]["tools"]]
        self.assertIn("calculator", tool_names)


if __name__ == "__main__":
    unittest.main()
