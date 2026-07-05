"""上下文管理与记忆管理的压力/并发/不变量测试。

检验框架在极端与并发场景下仍守住核心不变量：
- 上下文：反复压缩不损坏、切割永不拆散 tool_use/tool_result 配对、
  压缩失败可恢复、极长历史稳定
- 记忆：并发读写不崩溃不丢数据（真实并行会话）、fork 后全局共享/私有隔离、
  索引渲染确定性（缓存稳定）与截断（防灌爆上下文）
"""

import tempfile
import threading
import unittest
from pathlib import Path

from mini_agent import (
    Agent, ContextManager, MemoryStore, ScriptedLLM, Session, SessionManager,
    build_default_registry,
)
from mini_agent.context.manager import _is_plain_user_message


def _tool_round(i):
    """一轮带工具配对的历史：user -> assistant(tool_use) -> user(tool_result) -> assistant(text)。"""
    return [
        {"role": "user", "content": f"问题{i}"},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": f"想{i}", "signature": "s"},
            {"type": "tool_use", "id": f"t{i}", "name": "search", "input": {"q": i}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": f"t{i}", "content": f"结果{i}"}]},
        {"role": "assistant", "content": [{"type": "text", "text": f"回答{i}"}]},
    ]


def _long_history(n):
    h = []
    for i in range(n):
        h.extend(_tool_round(i))
    return h


class TestCompactionInvariants(unittest.TestCase):
    def test_boundary_never_splits_tool_pair_across_sizes(self):
        """不同 keep_recent 下压缩，切割点永远是纯用户消息（不拆工具配对）。"""
        compacted_count = 0
        for keep in range(1, 12):
            cm = ContextManager(keep_recent_messages=keep)
            session = Session(id="c")
            session.history = _long_history(10)
            llm = ScriptedLLM([ScriptedLLM.say("摘要")])
            did = cm.compact(session, llm)
            # 不变量：若发生了压缩，保留段首条必是纯用户消息（不拆工具配对）；
            # 若没找到合法切割点则不压缩（宁可不压也不拆坏配对），两者都合法
            if did:
                compacted_count += 1
                self.assertTrue(_is_plain_user_message(session.history[1]),
                                f"keep={keep} 拆散了工具配对: {session.history[1]}")
            self._assert_tool_pairs_intact(session.history)
        self.assertGreater(compacted_count, 0, "至少应有若干 keep 值触发了压缩")

    def test_repeated_compaction_stays_consistent(self):
        """反复 压缩->增长->再压缩：始终以合法用户消息开头，不累积损坏。"""
        cm = ContextManager(max_context_tokens=500, keep_recent_messages=4)
        session = Session(id="c")
        session.history = _long_history(6)
        for cycle in range(5):
            llm = ScriptedLLM([ScriptedLLM.say(f"摘要{cycle}")])
            session.last_usage = {"input_tokens": 99_999}
            if cm.should_compact(session):
                self.assertTrue(cm.compact(session, llm))
            # 首条必是合法用户消息，工具配对完好
            self.assertEqual(session.history[0]["role"], "user")
            self._assert_tool_pairs_intact(session.history)
            # 再灌入新历史
            session.history.extend(_long_history(4))

    def test_compaction_failure_leaves_history_intact(self):
        cm = ContextManager(keep_recent_messages=4)
        session = Session(id="c")
        session.history = _long_history(6)
        snapshot = [dict(m) for m in session.history]
        llm = ScriptedLLM([])  # summarize 立即 LLMError
        self.assertFalse(cm.compact(session, llm))
        self.assertEqual(session.history, snapshot)

    def test_all_tool_results_no_clean_cut_skips(self):
        """极端：保留段之前全是工具消息、无纯用户消息 -> 不压缩、不崩溃。"""
        cm = ContextManager(keep_recent_messages=1)
        session = Session(id="c")
        # 构造开头就没有可切割点的历史（首条是 assistant）
        session.history = [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t", "name": "x", "input": {}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t", "content": "r"}]},
        ]
        llm = ScriptedLLM([ScriptedLLM.say("摘要")])
        # 不应抛异常
        cm.compact(session, llm)

    def test_very_long_history_compaction(self):
        cm = ContextManager(max_context_tokens=1000, keep_recent_messages=8)
        session = Session(id="c")
        session.history = _long_history(200)  # 800 条消息
        session.last_usage = {"input_tokens": 500_000}
        llm = ScriptedLLM([ScriptedLLM.say("大摘要")])
        original = len(session.history)
        self.assertTrue(cm.compact(session, llm))
        self.assertLess(len(session.history), original)
        self._assert_tool_pairs_intact(session.history)

    def _assert_tool_pairs_intact(self, history):
        open_ids = set()
        for msg in history:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if block.get("type") == "tool_use":
                    open_ids.add(block["id"])
                elif block.get("type") == "tool_result":
                    tid = block.get("tool_use_id")
                    self.assertIn(tid, open_ids,
                                  f"孤儿 tool_result: {tid}（其 tool_use 被压缩切走）")


class TestContextRequestSafety(unittest.TestCase):
    def test_build_request_does_not_mutate_history(self):
        cm = ContextManager()
        session = Session(id="c")
        session.history = _long_history(3)
        before = [dict(m) for m in session.history]
        cm.build_request(session, "(空)", "全局共享")
        # 打 cache_control 是在 deepcopy 上做的，原历史零污染
        self.assertEqual(session.history, before)

    def test_cache_breakpoint_on_last_block_only(self):
        cm = ContextManager()
        session = Session(id="c")
        session.history = _long_history(2)
        _, messages = cm.build_request(session, "(空)", "全局共享")
        last = messages[-1]["content"][-1]
        self.assertEqual(last["cache_control"], {"type": "ephemeral"})
        # 中间消息不带断点（只在最后一块，避免 4 断点上限被打爆）
        marked = sum(1 for m in messages if isinstance(m["content"], list)
                     for b in m["content"] if "cache_control" in b)
        self.assertEqual(marked, 1)


class TestMemoryConcurrency(unittest.TestCase):
    def test_parallel_writes_no_lost_update_no_crash(self):
        """并行会话大量写全局记忆：不崩溃、不丢数据（并发正确性回归）。"""
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "mem.json")
            n_threads, per = 8, 50
            errors = []

            def worker(tid):
                try:
                    for i in range(per):
                        store.set(f"t{tid}_k{i}", f"v{i}")
                except Exception as exc:  # 包括 RuntimeError: dict changed size
                    errors.append(exc)

            threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            self.assertEqual(errors, [], f"并发写崩溃: {errors[:3]}")
            self.assertEqual(len(store), n_threads * per, "并发写丢失了数据")
            # 磁盘文件完整可重载（原子写保证不会半截 JSON）
            reloaded = MemoryStore(Path(tmp) / "mem.json")
            self.assertEqual(len(reloaded), n_threads * per)

    def test_concurrent_read_during_write(self):
        """一个线程持续 items() 迭代，另一个持续 set() —— 不得 RuntimeError。"""
        store = MemoryStore()
        for i in range(100):
            store.set(f"k{i}", f"v{i}")
        errors = []
        stop = threading.Event()

        def reader():
            try:
                while not stop.is_set():
                    _ = store.items()
                    _ = len(store)
            except Exception as exc:
                errors.append(exc)

        def writer():
            try:
                for i in range(500):
                    store.set(f"w{i}", "x")
            finally:
                stop.set()

        r = threading.Thread(target=reader)
        w = threading.Thread(target=writer)
        r.start(); w.start(); w.join(); r.join()
        self.assertEqual(errors, [], f"并发读写崩溃: {errors[:3]}")

    def test_parallel_sessions_writing_shared_memory_via_agent(self):
        """通过真实 Agent：两个共享会话并行写全局记忆，都成功落库。"""
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "g.json")

            def agent():
                return Agent(llm=ScriptedLLM([
                    ScriptedLLM.call("memory", {"action": "save",
                                                "key": "shared", "value": "v"}),
                    ScriptedLLM.say("ok"),
                ]), registry=build_default_registry(), global_memory=memory)

            errors = []

            def run(sid, key):
                try:
                    ag = Agent(llm=ScriptedLLM([
                        ScriptedLLM.call("memory", {"action": "save",
                                                    "key": key, "value": "v"}),
                        ScriptedLLM.say("ok"),
                    ]), registry=build_default_registry(), global_memory=memory)
                    ag.run(Session(id=sid), "记住")
                except Exception as exc:
                    errors.append(exc)

            threads = [threading.Thread(target=run, args=(f"s{i}", f"key{i}"))
                       for i in range(6)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            self.assertEqual(errors, [])
            self.assertEqual(len(memory), 6)


class TestMemoryScopeAndFork(unittest.TestCase):
    def test_fork_shares_global_isolates_local(self):
        with tempfile.TemporaryDirectory() as tmp:
            service_dir = Path(tmp)
            manager = SessionManager(service_dir)
            memory = MemoryStore(service_dir / "g.json")
            registry = build_default_registry()

            shared = manager.create(use_global_memory=True)
            # 全局写：fork 后仍可见
            Agent(llm=ScriptedLLM([
                ScriptedLLM.call("memory", {"action": "save", "key": "g", "value": "1"}),
                ScriptedLLM.say("ok")]),
                registry=registry, global_memory=memory).run(shared, "记全局")

            fork = manager.fork(shared)
            self.assertEqual(memory.get("g"), "1")  # 全局对 fork 可见

            # 隔离会话：私有记忆不进全局，也不被 fork 泄漏
            iso = manager.create(use_global_memory=False)
            Agent(llm=ScriptedLLM([
                ScriptedLLM.call("memory", {"action": "save", "key": "p", "value": "x"}),
                ScriptedLLM.say("ok")]),
                registry=registry, global_memory=memory).run(iso, "记私有")
            self.assertEqual(iso.local_memory, {"p": "x"})
            self.assertIsNone(memory.get("p"))

            iso_fork = manager.fork(iso)
            self.assertEqual(iso_fork.local_memory, {"p": "x"})  # 私有记忆随 fork 深拷贝
            iso_fork.local_memory["p"] = "changed"
            self.assertEqual(iso.local_memory["p"], "x")  # 改 fork 不影响源

    def test_memory_scope_unaffected_by_provider(self):
        """切换 Provider 不改变记忆范围（记忆属于框架，不属于模型）。"""
        session = Session(id="s", use_global_memory=False)
        session.provider = "some-model"
        ctx = _make_ctx(session)
        self.assertEqual(ctx.memory_scope(), "会话私有")
        session.provider = None
        self.assertEqual(_make_ctx(session).memory_scope(), "会话私有")


class TestMemoryIndexRendering(unittest.TestCase):
    def test_deterministic_and_truncated(self):
        items = {"b": "y" * 200, "a": "short"}
        # 同输入渲染确定（缓存前缀稳定的前提）
        self.assertEqual(MemoryStore.render_index(items),
                         MemoryStore.render_index(dict(items)))
        # 超长值被截断，防止单条记忆灌爆上下文
        rendered = MemoryStore.render_index({"k": "z" * 1000}, value_limit=50)
        value_part = rendered.split("k: ", 1)[1]
        self.assertLessEqual(len(value_part), 51)
        self.assertTrue(value_part.endswith("…"))

    def test_empty_memory(self):
        self.assertEqual(MemoryStore.render_index({}), "(空)")


def _make_ctx(session):
    from mini_agent import ToolContext
    return ToolContext(session=session, global_memory=MemoryStore())


if __name__ == "__main__":
    unittest.main()
