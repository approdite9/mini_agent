"""会话/记忆的持久化与隔离测试。"""

import json
import tempfile
import unittest
from pathlib import Path

from mini_agent import (
    Agent, MemoryStore, ScriptedLLM, Session, SessionManager, build_default_registry,
)


class TestMemoryStorePersistence(unittest.TestCase):
    def test_persist_and_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mem.json"
            store = MemoryStore(path)
            store.set("城市", "北京", source="s1")
            store.set("语言", "Python")

            reloaded = MemoryStore(path)  # 新进程视角
            self.assertEqual(reloaded.items(), {"城市": "北京", "语言": "Python"})
            self.assertEqual(reloaded.get("城市"), "北京")

    def test_corrupted_file_tolerated(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mem.json"
            path.write_text("{{{ 不是 json", encoding="utf-8")
            store = MemoryStore(path)  # 不应抛异常
            self.assertEqual(len(store), 0)
            store.set("k", "v")
            self.assertEqual(MemoryStore(path).get("k"), "v")

    def test_render_index_truncates_values(self):
        index = MemoryStore.render_index({"长文": "x" * 200, "短": "ok"}, value_limit=20)
        self.assertIn("…", index)
        self.assertIn("短: ok", index)
        self.assertNotIn("x" * 30, index)


class TestSessionPersistence(unittest.TestCase):
    def test_sessions_survive_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = SessionManager(Path(tmp))
            s1 = manager.create(title="工作")
            s1.history.append({"role": "user", "content": "记住这件事"})
            s1.plan = [{"title": "步骤一", "status": "done"}]
            s1.local_memory["k"] = "v"
            s1.save()
            s2 = manager.create(title="隔离", use_global_memory=False)

            # 模拟进程重启
            manager2 = SessionManager(Path(tmp))
            loaded = manager2.get(s1.id)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.title, "工作")
            self.assertEqual(loaded.history[0]["content"], "记住这件事")
            self.assertEqual(loaded.plan[0]["status"], "done")
            self.assertEqual(loaded.local_memory, {"k": "v"})
            self.assertFalse(manager2.get(s2.id).use_global_memory)
            # 新会话 id 不与已有冲突
            s3 = manager2.create()
            self.assertNotIn(s3.id, {s1.id, s2.id})

    def test_agent_run_autosaves(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = SessionManager(Path(tmp))
            session = manager.create(title="自动保存")
            agent = Agent(
                llm=ScriptedLLM([ScriptedLLM.say("好的")]),
                registry=build_default_registry(),
            )
            agent.run(session, "hi")
            on_disk = json.loads(
                (Path(tmp) / "sessions" / f"{session.id}.json").read_text(encoding="utf-8"))
            self.assertEqual(on_disk["request_count"], 1)
            self.assertEqual(on_disk["history"][0]["content"], "hi")
            self.assertEqual(on_disk["history"][1]["role"], "assistant")


class TestMemorySharing(unittest.TestCase):
    def _agent(self, script):
        self.memory = MemoryStore()
        return Agent(
            llm=ScriptedLLM(script),
            registry=build_default_registry(),
            global_memory=self.memory,
        )

    def test_shared_memory_visible_across_sessions(self):
        agent = self._agent([
            ScriptedLLM.call("memory", {"action": "save", "key": "城市", "value": "北京"}),
            ScriptedLLM.say("记住了"),
            ScriptedLLM.say("你在北京"),
            ScriptedLLM.say("我不知道"),
        ])
        manager = SessionManager()
        s1, s2 = manager.create(), manager.create()
        s3 = manager.create(use_global_memory=False)

        agent.run(s1, "记住我在北京")
        self.assertEqual(self.memory.get("城市"), "北京")

        agent.run(s2, "我在哪？")
        s2_memory_block = agent.llm.calls[2]["system"][1]["text"]
        self.assertIn("城市: 北京", s2_memory_block)   # 共享会话可见
        self.assertIn("全局共享", s2_memory_block)

        agent.run(s3, "我在哪？")
        s3_memory_block = agent.llm.calls[3]["system"][1]["text"]
        self.assertNotIn("城市: 北京", s3_memory_block)  # 隔离会话不可见
        self.assertIn("会话私有", s3_memory_block)

    def test_isolated_memory_stays_local(self):
        agent = self._agent([
            ScriptedLLM.call("memory", {"action": "save", "key": "秘密", "value": "42"}),
            ScriptedLLM.say("好的"),
        ])
        manager = SessionManager()
        s_iso = manager.create(use_global_memory=False)
        agent.run(s_iso, "记住秘密 42")
        self.assertEqual(s_iso.local_memory, {"秘密": "42"})
        self.assertEqual(len(self.memory), 0)

    def test_histories_and_todos_are_independent(self):
        agent = self._agent([
            ScriptedLLM.call("todo", {"action": "add", "item": "买牛奶"}),
            ScriptedLLM.say("已记录"),
            ScriptedLLM.say("hi"),
        ])
        manager = SessionManager()
        s1, s2 = manager.create(), manager.create()
        agent.run(s1, "记一下买牛奶")
        agent.run(s2, "你好")
        self.assertEqual(len(s1.todos), 1)
        self.assertEqual(len(s2.todos), 0)
        self.assertNotIn("买牛奶", str(s2.history))


if __name__ == "__main__":
    unittest.main()
