"""服务层与 Web 服务端测试：AgentService 门面、并发锁、HTTP API + SSE 事件流。"""

import json
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path

from mini_agent import AgentService, ScriptedLLM, SessionBusyError
from mini_agent.server import make_server


def make_service(tmp, script):
    return AgentService(llm=ScriptedLLM(script), base_dir=Path(tmp))


class TestAgentService(unittest.TestCase):
    def test_end_to_end_with_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = make_service(tmp, [
                ScriptedLLM.call("calculator", {"expression": "2+2"}, thinking="算一下"),
                ScriptedLLM.say("等于 4"),
            ])
            session = service.create_session(title="测试")
            events = []
            result = service.send(session.id, "2+2", on_event=events.append)

            self.assertEqual(result.answer, "等于 4")
            types = [e.type for e in events]
            for expected in ["run_start", "thinking_start", "thinking_delta", "thinking_end",
                             "tool_start", "tool_result", "assistant_delta",
                             "assistant_final", "run_end"]:
                self.assertIn(expected, types)

            infos = service.list_sessions()
            self.assertEqual(infos[0]["id"], session.id)
            self.assertGreater(infos[0]["messages"], 0)
            self.assertIn("tool_start", service.trace(session.id))

    def test_session_busy_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = make_service(tmp, [ScriptedLLM.say("ok")])
            session = service.create_session()
            session.lock.acquire()  # 模拟另一个 run 占用中
            try:
                with self.assertRaises(SessionBusyError):
                    service.send(session.id, "hi")
            finally:
                session.lock.release()
            # 释放后恢复可用
            self.assertEqual(service.send(session.id, "hi").answer, "ok")

    def test_unknown_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = make_service(tmp, [])
            with self.assertRaises(KeyError):
                service.send("s999", "hi")

    def test_session_list_enhanced_fields(self):
        """会话列表增强：模型标识、运行状态、自动标题与一行任务摘要。"""
        with tempfile.TemporaryDirectory() as tmp:
            service = make_service(tmp, [ScriptedLLM.say("好的")])
            session = service.create_session()  # 默认标题 "会话N"
            service.send(session.id, "帮我规划一次北京三日游的行程")
            info = [s for s in service.list_sessions() if s["id"] == session.id][0]
            self.assertEqual(info["model"], "scripted")
            self.assertFalse(info["running"])
            self.assertIn("北京三日游", info["summary"])
            self.assertTrue(info["title"].startswith("帮我规划"))  # 首条消息自动生成标题

    def test_memory_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = make_service(tmp, [])
            service.global_memory.set("k", "v")
            session = service.create_session()
            session.local_memory["p"] = "q"
            snapshot = service.memory_snapshot(session.id)
            self.assertEqual(snapshot["global"], {"k": "v"})
            self.assertEqual(snapshot["local"], {"p": "q"})


class TestHTTPServer(unittest.TestCase):
    """对 stdlib HTTP 服务做端到端冒烟：REST + SSE 流。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.service = make_service(self.tmp.name, [
            ScriptedLLM.call("calculator", {"expression": "3*3"}, thinking="用计算器"),
            ScriptedLLM.say("答案是 9"),
        ])
        self.server = make_server(self.service, port=0)  # 随机端口
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()

    def _url(self, path):
        return f"http://127.0.0.1:{self.port}{path}"

    def _get(self, path):
        with urllib.request.urlopen(self._url(path)) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _post(self, path, payload):
        req = urllib.request.Request(
            self._url(path), data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        return urllib.request.urlopen(req)

    def test_full_http_flow(self):
        # 1. 建会话
        with self._post("/api/sessions", {"title": "web测试", "shared_memory": True}) as resp:
            created = json.loads(resp.read().decode("utf-8"))
        sid = created["id"]

        # 2. 会话列表
        sessions = self._get("/api/sessions")
        self.assertTrue(any(s["id"] == sid for s in sessions))

        # 3. 发消息，读 SSE 事件流
        events = []
        with self._post(f"/api/sessions/{sid}/messages", {"text": "3*3"}) as resp:
            self.assertIn("text/event-stream", resp.headers["Content-Type"])
            buffer = b""
            for line in resp:
                buffer += line
            for chunk in buffer.decode("utf-8").split("\n\n"):
                if chunk.startswith("data: "):
                    events.append(json.loads(chunk[6:]))

        types = [e["type"] for e in events]
        for expected in ["run_start", "tool_start", "tool_result", "run_end", "run_stats"]:
            self.assertIn(expected, types)
        run_end = [e for e in events if e["type"] == "run_end"][0]
        self.assertEqual(run_end["data"]["answer"], "答案是 9")
        # 运行审计事件附带结构化指标
        stats = [e for e in events if e["type"] == "run_stats"][0]["data"]
        self.assertEqual(stats["turns"], 2)
        self.assertEqual(stats["tools_ok"], 1)

        # 审计与 Provider 接口
        runs = self._get(f"/api/sessions/{sid}/runs")["runs"]
        self.assertEqual(len(runs), 1)
        providers = self._get("/api/providers")
        self.assertIn("available", providers)
        self.assertEqual(providers["default"], "scripted")

        # 分叉接口
        with self._post(f"/api/sessions/{sid}/fork", {}) as resp:
            forked = json.loads(resp.read().decode("utf-8"))
        self.assertTrue(forked["parent"].startswith(sid + "@"))
        fork_history = self._get(f"/api/sessions/{forked['id']}/history")
        self.assertEqual(fork_history["parent"], forked["parent"])

        # 4. 历史接口返回渲染友好的条目
        history = self._get(f"/api/sessions/{sid}/history")
        kinds = [item["kind"] for item in history["items"]]
        self.assertIn("user", kinds)
        self.assertIn("tool_call", kinds)
        self.assertIn("tool_result", kinds)
        self.assertIn("text", kinds)

        # 5. trace 与记忆接口
        self.assertIn("tool_start", self._get(f"/api/sessions/{sid}/trace")["trace"])
        self.assertIn("global", self._get(f"/api/memory?session={sid}"))

        # 结构化时间线：真实时间戳 + 事件类型
        timeline = self._get(f"/api/sessions/{sid}/timeline")["events"]
        tl_types = [e["type"] for e in timeline]
        for expected in ["user_input", "llm_call", "tool_start", "tool_result", "final_answer"]:
            self.assertIn(expected, tl_types)
        self.assertTrue(all("timestamp" in e for e in timeline))

    def test_empty_text_rejected(self):
        with self._post("/api/sessions", {}) as resp:
            sid = json.loads(resp.read().decode("utf-8"))["id"]
        try:
            self._post(f"/api/sessions/{sid}/messages", {"text": "  "})
            self.fail("应返回 400")
        except urllib.error.HTTPError as exc:
            self.assertEqual(exc.code, 400)


import urllib.error  # noqa: E402  (供 HTTPError 断言使用)

if __name__ == "__main__":
    unittest.main()
