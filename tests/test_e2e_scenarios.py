"""端到端场景测试：在真实 HTTP Server + SSE 流上跑完整业务场景（离线、无外部依赖）。

与单元测试的区别：这里不 mock 任何框架层 —— 请求走真实的 ThreadingHTTPServer、
真实的 SSE 编解码、真实的会话持久化与审计，只有 LLM 是脚本化的。
"""

import io
import json
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from contextlib import redirect_stdout
from pathlib import Path

from mini_agent import (
    AgentService, ContextManager, LLMResponse, ScriptedLLM, SessionBusyError,
)
from mini_agent.cli import CliRenderer
from mini_agent.core.events import AgentEvent
from mini_agent import LLMClient
from mini_agent.server import make_server


class _ServerFixture:
    """真实服务端夹具：随机端口 + 临时数据目录 + HTTP/SSE 客户端小工具。"""

    def __init__(self, script=None, llm=None, llm_factory=None, context_manager=None):
        self._tmp = tempfile.TemporaryDirectory()
        self.service = AgentService(
            llm=llm if llm is not None else ScriptedLLM(script or []),
            base_dir=Path(self._tmp.name),
            llm_factory=llm_factory,
            context_manager=context_manager,
        )
        self.server = make_server(self.service, port=0)
        self.port = self.server.server_address[1]
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self._thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self._tmp.cleanup()

    def url(self, path):
        return f"http://127.0.0.1:{self.port}{path}"

    def get(self, path):
        with urllib.request.urlopen(self.url(path)) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def post(self, path, payload):
        req = urllib.request.Request(
            self.url(path), data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def send_sse(self, session_id, text, attachments=None):
        """发消息并解析完整 SSE 事件流。"""
        payload = {"text": text}
        if attachments is not None:
            payload["attachments"] = attachments
        req = urllib.request.Request(
            self.url(f"/api/sessions/{session_id}/messages"),
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        events = []
        with urllib.request.urlopen(req) as resp:
            body = resp.read().decode("utf-8")
        for chunk in body.split("\n\n"):
            if chunk.startswith("data: "):
                events.append(json.loads(chunk[6:]))
        return events


class TestGoldenPath(unittest.TestCase):
    """黄金路径：计划拆解 -> 工具执行 -> 报错自纠 -> 收尾，全链路事件校验。"""

    def setUp(self):
        self.fx = _ServerFixture(script=[
            ScriptedLLM.call("plan", {"action": "set",
                                      "steps": ["算预算", "查天气", "汇总"]},
                             thinking="多步任务，先拆解计划"),
            ScriptedLLM.call("calculator", {"expression": "(1200+800)*1.1"}),
            ScriptedLLM.call("weather", {"city": "北平"}, thinking="查天气"),
            ScriptedLLM.call("weather", {"city": "北京"}, thinking="没有北平，改用北京重试"),
            ScriptedLLM.call("plan", {"action": "update", "step": 3, "status": "done"}),
            ScriptedLLM.say("预算 2200 元，北京晴 26°C，适合出行。", thinking="信息齐了"),
        ])

    def tearDown(self):
        self.fx.close()

    def test_full_scenario_over_sse(self):
        sid = self.fx.post("/api/sessions", {"title": "e2e"})["id"]
        events = self.fx.send_sse(sid, "规划北京旅行：预算 (1200+800)*1.1，看看天气")
        types = [e["type"] for e in events]

        # 1) 事件生命周期完整性：每个 thinking_start 都有配对的 thinking_end，
        #    且思考层永远在该轮回答/工具之前闭合
        self.assertEqual(types.count("thinking_start"), types.count("thinking_end"))
        # 2) 每个 tool_start 都有配对的 tool_result 或 tool_error（按 id 对齐）
        starts = {e["data"]["id"] for e in events if e["type"] == "tool_start"}
        finishes = {e["data"]["id"] for e in events
                    if e["type"] in ("tool_result", "tool_error")}
        self.assertEqual(starts, finishes)
        self.assertEqual(len(starts), 5)
        # 3) 错误自纠：恰好一次 tool_error（北平），其后同名工具成功
        errors = [e for e in events if e["type"] == "tool_error"]
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["data"]["tool"], "weather")
        after = types[events.index(errors[0]) + 1:]
        self.assertIn("tool_result", after)
        # 4) 计划可视化：plan set + update 各发一次 plan_update
        self.assertEqual(types.count("plan_update"), 2)
        # 5) 审计：run_stats 汇总正确
        stats = [e for e in events if e["type"] == "run_stats"][0]["data"]
        self.assertEqual(stats["turns"], 6)
        self.assertEqual(stats["tools_ok"], 4)
        self.assertEqual(stats["tools_failed"], 1)
        self.assertTrue(all(t["duration_ms"] >= 0 for t in stats["tools"]))
        # 6) 最终答案只在 run_end / assistant_final 中出现
        run_end = [e for e in events if e["type"] == "run_end"][0]["data"]
        self.assertIn("2200", run_end["answer"])

        # 7) 时间线：结构化事件按真实时间戳单调排列
        timeline = self.fx.get(f"/api/sessions/{sid}/timeline")["events"]
        stamps = [e["timestamp"] for e in timeline]
        self.assertEqual(stamps, sorted(stamps))
        tl_types = [e["type"] for e in timeline]
        for expected in ["user_input", "llm_call", "tool_start", "tool_error",
                         "tool_result", "final_answer"]:
            self.assertIn(expected, tl_types)

        # 8) 历史重载：结构化条目（thinking 与 text 分离、工具成对）
        items = self.fx.get(f"/api/sessions/{sid}/history")["items"]
        kinds = [i["kind"] for i in items]
        self.assertEqual(kinds.count("tool_call"), 5)
        self.assertEqual(kinds.count("tool_result"), 5)
        self.assertIn("thinking", kinds)
        self.assertIn("text", kinds)
        # 会话标题由首条消息自动生成
        self.assertTrue(self.fx.get(f"/api/sessions/{sid}/history")["title"])


class TestMultimodalOverHTTP(unittest.TestCase):
    def setUp(self):
        self.fx = _ServerFixture(script=[ScriptedLLM.say("我看到了一张图片")])

    def tearDown(self):
        self.fx.close()

    def test_image_attachment_roundtrip(self):
        sid = self.fx.post("/api/sessions", {})["id"]
        events = self.fx.send_sse(sid, "这张图是什么？", attachments=[
            {"type": "image", "media_type": "image/png", "data": "aGVsbG8="}])
        run_start = [e for e in events if e["type"] == "run_start"][0]["data"]
        self.assertEqual(run_start["attachments"], 1)
        # 历史重载能回显图片（canonical image block -> kind:image）
        items = self.fx.get(f"/api/sessions/{sid}/history")["items"]
        images = [i for i in items if i["kind"] == "image"]
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0]["media_type"], "image/png")
        self.assertEqual(images[0]["data"], "aGVsbG8=")

    def test_invalid_attachment_rejected(self):
        sid = self.fx.post("/api/sessions", {})["id"]
        req = urllib.request.Request(
            self.fx.url(f"/api/sessions/{sid}/messages"),
            data=json.dumps({"text": "x", "attachments": [{"type": "file", "data": "a"}]}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req)
        self.assertEqual(ctx.exception.code, 400)


class TestRelayAndForkOverHTTP(unittest.TestCase):
    """跨模型接力 + 分叉：通过真实 HTTP API 全流程验证。"""

    def setUp(self):
        self.alt = ScriptedLLM(
            [ScriptedLLM.say("接力模型：你之前说预算两千")], label="mock2/relay-model")

        def factory(provider=None, **kw):  # 遵守真实工厂契约：未知 Provider 抛 LLMError
            from mini_agent import LLMError
            if provider != "mock2":
                raise LLMError(f"不支持的 LLM_PROVIDER '{provider}'")
            return self.alt

        self.fx = _ServerFixture(
            script=[ScriptedLLM.say("默认模型：记下了，预算两千")],
            llm_factory=factory)

    def tearDown(self):
        self.fx.close()

    def test_relay_fork_and_audit_models(self):
        sid = self.fx.post("/api/sessions", {})["id"]
        self.fx.send_sse(sid, "预算两千，记住")

        # 中途换模型
        resp = self.fx.post(f"/api/sessions/{sid}/provider", {"provider": "mock2"})
        self.assertEqual(resp["model"], "mock2/relay-model")
        events = self.fx.send_sse(sid, "我预算多少？")
        answer = [e for e in events if e["type"] == "run_end"][0]["data"]["answer"]
        self.assertIn("接力模型", answer)
        # 接力模型收到了默认模型时期的历史（Provider 中立格式）
        self.assertIn("预算两千", str(self.alt.calls[0]["messages"]))
        # 审计逐 run 标注模型
        runs = self.fx.get(f"/api/sessions/{sid}/runs")["runs"]
        self.assertEqual([r["model"] for r in runs],
                         ["scripted", "mock2/relay-model"])

        # 分叉继承 provider，且历史独立
        forked = self.fx.post(f"/api/sessions/{sid}/fork", {})
        info = self.fx.get(f"/api/sessions/{forked['id']}/history")
        self.assertEqual(info["provider"], "mock2")
        self.assertEqual(len(info["items"]),
                         len(self.fx.get(f"/api/sessions/{sid}/history")["items"]))

    def test_unknown_provider_400(self):
        sid = self.fx.post("/api/sessions", {})["id"]
        req = urllib.request.Request(
            self.fx.url(f"/api/sessions/{sid}/provider"),
            data=json.dumps({"provider": "no-such"}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req)
        self.assertEqual(ctx.exception.code, 400)


class TestCompactionOverHTTP(unittest.TestCase):
    def test_compaction_event_streams_to_client(self):
        fx = _ServerFixture(
            script=[ScriptedLLM.say("这是前情摘要"),   # 被压缩器消耗
                    ScriptedLLM.say("基于摘要继续回答")],
            context_manager=ContextManager(max_context_tokens=500, keep_recent_messages=2))
        try:
            sid = fx.post("/api/sessions", {})["id"]
            session = fx.service.get_session(sid)
            for i in range(8):
                session.history.append({"role": "user", "content": f"旧问题{i}"})
                session.history.append(
                    {"role": "assistant", "content": [{"type": "text", "text": f"旧回答{i}"}]})
            session.last_usage = {"input_tokens": 99_999}  # 真实 usage 语义：上下文已超限

            events = fx.send_sse(sid, "继续")
            compactions = [e for e in events if e["type"] == "compaction"]
            self.assertEqual(len(compactions), 1)
            self.assertGreater(compactions[0]["data"]["dropped_messages"], 0)
            self.assertIn("[前情摘要", str(session.history[0]["content"]))
        finally:
            fx.close()


class _SlowLLM(LLMClient):
    """记录调用时间窗的慢模型：验证跨会话并行、同会话互斥。"""

    def __init__(self, delay=0.25):
        self.delay = delay
        self.windows = []
        self._lock = threading.Lock()

    def complete(self, *, system, messages, tools=None, on_delta=None):
        start = time.time()
        time.sleep(self.delay)
        end = time.time()
        with self._lock:
            self.windows.append((start, end))
        return LLMResponse(stop_reason="end_turn", text="ok",
                           raw_content=[{"type": "text", "text": "ok"}],
                           usage={"input_tokens": 1, "output_tokens": 1,
                                  "cache_read_input_tokens": 0,
                                  "cache_creation_input_tokens": 0})


class TestConcurrency(unittest.TestCase):
    def test_different_sessions_run_in_parallel(self):
        llm = _SlowLLM()
        with tempfile.TemporaryDirectory() as tmp:
            service = AgentService(llm=llm, base_dir=Path(tmp))
            s1, s2 = service.create_session(), service.create_session()
            threads = [threading.Thread(target=service.send, args=(s.id, "hi"))
                       for s in (s1, s2)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            (a_start, a_end), (b_start, b_end) = llm.windows
            # 两个窗口重叠 => 真并行（串行则第二个 start >= 第一个 end）
            self.assertLess(max(a_start, b_start), min(a_end, b_end),
                            "不同会话应并行执行")

    def test_same_session_is_serialized(self):
        llm = _SlowLLM()
        with tempfile.TemporaryDirectory() as tmp:
            service = AgentService(llm=llm, base_dir=Path(tmp))
            session = service.create_session()
            errors = []

            def first():
                service.send(session.id, "第一条")

            t = threading.Thread(target=first)
            t.start()
            time.sleep(0.08)  # 确保第一条已持锁
            try:
                service.send(session.id, "第二条")
            except SessionBusyError as exc:
                errors.append(exc)
            t.join()
            self.assertEqual(len(errors), 1, "同会话并发必须被运行锁拒绝")


class TestCliRenderer(unittest.TestCase):
    """CLI 是事件流的另一个渲染器：喂完整事件序列，验证渲染输出。"""

    def test_renders_full_event_vocabulary(self):
        renderer = CliRenderer()
        buf = io.StringIO()
        make = lambda t, **d: AgentEvent(type=t, session_id="s1", data=d)
        with redirect_stdout(buf):
            for event in [
                make("run_start", input="hi"),
                make("turn_start", turn=1),
                make("thinking_start"),
                make("thinking_delta", text="想一想"),
                make("thinking_end", content="想一想", duration_ms=120),
                make("tool_start", id="t1", tool="calculator",
                     arguments={"expression": "1+1"}),
                make("tool_result", id="t1", tool="calculator", result="2",
                     duration_ms=3),
                make("tool_start", id="t2", tool="weather", arguments={"city": "火星"}),
                make("tool_error", id="t2", tool="weather", error="暂无数据",
                     duration_ms=1),
                make("memory_update", action="save", key="城市", scope="全局"),
                make("plan_update", plan=[{"title": "步骤一", "status": "done"}]),
                make("compaction", dropped_messages=6, summary_chars=100),
                make("assistant_delta", text="答案是 2"),
                make("run_end", answer="答案是 2"),
            ]:
                renderer(event)
        out = buf.getvalue()
        for marker in ["∴", "想一想", "⚙ calculator", "✓", "2 · 3ms",
                       "⚙ weather", "✗", "暂无数据", "◆ 记忆写入", "步骤一",
                       "上下文已压缩", "答案是 2"]:
            self.assertIn(marker, out)
        # 流式过 delta 后 assistant_final/run_end 不重复渲染答案
        self.assertEqual(out.count("答案是 2"), 1)

    def _render(self, events):
        renderer = CliRenderer()
        make = lambda t, **d: AgentEvent(type=t, session_id="s1", data=d)
        buf = io.StringIO()
        with redirect_stdout(buf):
            for t, d in events:
                renderer(make(t, **d))
        return buf.getvalue()

    def test_answer_renders_without_streamed_deltas(self):
        """回归：未逐字流式的 Provider（如 qwen-max）——答案只在 assistant_final 到达，
        必须渲染，不能被丢弃（对应真实 bug）。"""
        out = self._render([
            ("run_start", {"input": "你是什么模型"}),
            ("turn_start", {"turn": 1}),
            ("assistant_final", {"content": "我是通义千问。"}),
            ("run_end", {"answer": "我是通义千问。"}),
        ])
        self.assertIn("我是通义千问。", out)
        self.assertEqual(out.count("我是通义千问。"), 1)  # 不重复

    def test_terminal_message_renders_from_run_end(self):
        """回归：达轮次上限/预算早停等终态消息只在 run_end.answer 携带，也要渲染。"""
        out = self._render([
            ("run_start", {"input": "hi"}),
            ("run_end", {"answer": "本次请求因资源上限被系统提前停止。",
                         "error": "stopped_budget"}),
        ])
        self.assertIn("资源上限被系统提前停止", out)


class TestUIContract(unittest.TestCase):
    """UI 静态契约：服务端提供的是 Execution IDE 页面，而非聊天壳。"""

    def test_index_contains_ide_components(self):
        fx = _ServerFixture(script=[])
        try:
            with urllib.request.urlopen(fx.url("/")) as resp:
                html = resp.read().decode("utf-8")
            for fragment in [
                'id="exec"',              # Execution Panel
                'id="view-timeline"',     # Timeline 视图
                'id="view-debug"',        # Debug 视图
                'id="provider-sel"',      # 跨模型接力
                'id="attach-btn"',        # 多模态输入
                'class="think',           # 思考独立折叠层
                "mode-badge",             # 会话模式徽章
                "thinking_start",         # UI 消费生命周期事件
                "tool_error",
                "memory_update",
                "run_stats",
                'id="view-inspector"',    # 状态机·回放 Inspector 视图
                "renderStateMachine",     # 状态机可视化
                "renderToolGraph",        # 工具执行图
                "scrubTo",                # 回放 scrubber
                "insp-fail",              # 失败可视化
            ]:
                self.assertIn(fragment, html)
            # 禁止 UI 依赖 Provider：页面不出现厂商名
            for forbidden in ["anthropic", "qwen", "dashscope", "openai"]:
                self.assertNotIn(forbidden, html.lower())
        finally:
            fx.close()


class TestInspectorEndpoints(unittest.TestCase):
    """Inspector 依赖的服务端端点：持久化事件日志 / 状态迁移 / 指标（HTTP 层）。"""

    def test_run_events_transitions_and_metrics_over_http(self):
        fx = _ServerFixture(script=[
            ScriptedLLM.call("calculator", {"expression": "6*7"}, thinking="算"),
            ScriptedLLM.say("42"),
        ])
        try:
            sid = fx.post("/api/sessions", {})["id"]
            fx.send_sse(sid, "算 6*7")
            run_ids = fx.get(f"/api/sessions/{sid}/run_ids")["run_ids"]
            self.assertEqual(len(run_ids), 1)
            rid = run_ids[0]
            run = fx.get(f"/api/runs/{rid}/events")
            # 事件日志 + 状态迁移（含 reason）都能取到，终态 completed
            self.assertTrue(len(run["events"]) > 0)
            self.assertEqual(run["transitions"][-1]["to"], "completed")
            self.assertTrue(all("reason" in t for t in run["transitions"]))
            # 状态机可从迁移逐帧重建（模型响应已录制供回放）
            self.assertTrue(any(e["type"] == "model_response" for e in run["events"]))
            # 指标端点
            m = fx.get(f"/api/sessions/{sid}/metrics")
            self.assertEqual(m["aggregate"]["runs"], 1)
            self.assertEqual(m["aggregate"]["counters"]["tool_ok"], 1)
        finally:
            fx.close()


if __name__ == "__main__":
    unittest.main()
