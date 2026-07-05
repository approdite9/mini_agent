"""上下文管理测试：prompt cache 断点放置、usage 驱动的历史压缩。"""

import unittest

from mini_agent import ContextManager, ScriptedLLM, Session
from mini_agent.context.manager import _is_plain_user_message


def make_session(history=None, context_tokens=0):
    session = Session(id="c1")
    session.history = history or []
    if context_tokens:
        session.last_usage = {"input_tokens": context_tokens}
    return session


class TestBuildRequest(unittest.TestCase):
    def setUp(self):
        self.cm = ContextManager()

    def test_cache_breakpoints_placement(self):
        session = make_session([
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": [{"type": "text", "text": "你好！"}]},
            {"role": "user", "content": "再见"},
        ])
        system, messages = self.cm.build_request(session, "- k: v", "全局共享")

        # system 块 1 = 冻结核心 prompt，带缓存断点；块 2 = 记忆索引，不带
        self.assertEqual(system[0]["cache_control"], {"type": "ephemeral"})
        self.assertNotIn("cache_control", system[1])
        self.assertIn("k: v", system[1]["text"])
        self.assertIn("全局共享", system[1]["text"])

        # 最后一条消息的最后一个 block 带断点（多轮增量缓存）
        last_block = messages[-1]["content"][-1]
        self.assertEqual(last_block["cache_control"], {"type": "ephemeral"})

    def test_history_not_mutated(self):
        session = make_session([{"role": "user", "content": "你好"}])
        self.cm.build_request(session, "(空)", "全局共享")
        # 原始历史保持纯净：不含 cache_control，content 仍是 str
        self.assertEqual(session.history, [{"role": "user", "content": "你好"}])

    def test_core_prompt_is_frozen(self):
        """核心 system prompt 必须字节级稳定，两次构建完全一致（缓存前提）。"""
        s1 = make_session([{"role": "user", "content": "a"}])
        s2 = make_session([{"role": "user", "content": "b"}])
        sys1, _ = self.cm.build_request(s1, "- x: 1", "全局共享")
        sys2, _ = self.cm.build_request(s2, "- y: 2", "会话私有")
        self.assertEqual(sys1[0], sys2[0])  # 记忆不同、会话不同，核心块不变


class TestCompaction(unittest.TestCase):
    def _long_history(self, n_rounds=10):
        """构造含工具调用配对的长历史。"""
        history = []
        for i in range(n_rounds):
            history.append({"role": "user", "content": f"问题{i}"})
            history.append({"role": "assistant", "content": [
                {"type": "tool_use", "id": f"t{i}", "name": "search", "input": {"query": f"q{i}"}}]})
            history.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": f"t{i}", "content": f"结果{i}"}]})
            history.append({"role": "assistant", "content": [
                {"type": "text", "text": f"回答{i}"}]})
        return history

    def test_should_compact_by_real_usage(self):
        cm = ContextManager(max_context_tokens=1000)
        self.assertFalse(cm.should_compact(make_session(context_tokens=999)))
        self.assertTrue(cm.should_compact(make_session(context_tokens=1001)))

    def test_compact_replaces_old_history_with_summary(self):
        cm = ContextManager(max_context_tokens=100, keep_recent_messages=6)
        session = make_session(self._long_history(), context_tokens=99999)
        llm = ScriptedLLM([ScriptedLLM.say("这是压缩摘要")])
        events = []
        original_len = len(session.history)

        compacted = cm.compact(session, llm, on_event=lambda t, **d: events.append((t, d)))

        self.assertTrue(compacted)
        self.assertLess(len(session.history), original_len)
        self.assertIn("[前情摘要", session.history[0]["content"])
        self.assertIn("这是压缩摘要", session.history[0]["content"])
        self.assertEqual(events[0][0], "compaction")

    def test_compact_boundary_never_splits_tool_pair(self):
        """切割点必须是纯用户消息 —— tool_result 不能与其 tool_use 分离。"""
        cm = ContextManager(keep_recent_messages=6)
        session = make_session(self._long_history())
        llm = ScriptedLLM([ScriptedLLM.say("摘要")])
        cm.compact(session, llm)

        # 摘要后的第一条真实消息必须是纯 user 文本，不能是孤儿 tool_result
        first_real = session.history[1]
        self.assertTrue(_is_plain_user_message(first_real),
                        f"压缩边界拆散了工具配对: {first_real}")

    def test_compact_skips_short_history(self):
        cm = ContextManager(keep_recent_messages=12)
        session = make_session([{"role": "user", "content": "唯一一条"}])
        llm = ScriptedLLM([])
        self.assertFalse(cm.compact(session, llm))
        self.assertEqual(len(session.history), 1)

    def test_compact_survives_llm_failure(self):
        """压缩失败不致命：历史原样保留，本轮照常进行。"""
        cm = ContextManager(keep_recent_messages=4)
        session = make_session(self._long_history())
        original = [dict(m) for m in session.history]
        llm = ScriptedLLM([])  # 立即耗尽 -> LLMError
        self.assertFalse(cm.compact(session, llm))
        self.assertEqual(session.history, original)


if __name__ == "__main__":
    unittest.main()
