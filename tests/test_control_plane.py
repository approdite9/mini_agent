"""控制平面测试：Budget 早停、ToolRuntime 隔离（超时/重试/权限/命名空间/熔断/限流）、
记忆视图隔离、背压信号、模型降级，以及 LLM = Actor 的静态守卫。"""

import tempfile
import time
import unittest
from pathlib import Path

from mini_agent import AgentService, ScriptedLLM
from mini_agent.control.budget import Budget
from mini_agent.control.policy import DENY, ToolPolicy, ToolRule
from mini_agent.core.session import Session
from mini_agent.memory.store import MemoryStore
from mini_agent.runtime.execution_context import ExecutionContext, MemoryView
from mini_agent.runtime.state import RunState
from mini_agent.tools.registry import Tool, ToolContext, ToolError, ToolRegistry
from mini_agent.tools.runtime import ToolRuntime


def _exec_ctx(session=None, namespace=None):
    session = session or Session(id="t")
    return ExecutionContext(run_id="r", session=session, budget=Budget(),
                            global_memory=MemoryStore(), tool_namespace=namespace)


def _registry(func, name="probe", schema=None):
    reg = ToolRegistry()
    reg.register(Tool(name=name, description="test",
                      parameters=schema or {"type": "object", "properties": {}},
                      func=func))
    return reg


# ==========================================================================
# Budget
# ==========================================================================

class TestBudget(unittest.TestCase):
    def test_charge_and_exhaustion(self):
        b = Budget(max_tokens=100)
        self.assertIsNone(b.exhausted())
        b.charge({"input_tokens": 60, "output_tokens": 50})
        self.assertIsNotNone(b.exhausted())
        self.assertIn("token_budget", b.exhausted())

    def test_tool_call_budget(self):
        b = Budget(max_tool_calls=2)
        b.charge_tool_call(); self.assertIsNone(b.exhausted())
        b.charge_tool_call(); self.assertIn("tool_call_budget", b.exhausted())

    def test_cost_charge_and_pressure(self):
        b = Budget(max_cost_usd=1.0)
        pricing = {"input": 5.0, "output": 25.0}
        b.charge({"input_tokens": 100_000, "output_tokens": 20_000}, pricing)
        # 0.5 + 0.5 = 1.0
        self.assertAlmostEqual(b.spent_cost_usd, 1.0, places=4)
        self.assertGreaterEqual(b.cost_pressure(), 1.0)
        self.assertIn("cost_budget", b.exhausted())

    def test_clone_isolates_counters(self):
        tmpl = Budget(max_tokens=100)
        a, c = tmpl.clone(), tmpl.clone()
        a.charge({"input_tokens": 50})
        self.assertEqual(a.spent_tokens, 50)
        self.assertEqual(c.spent_tokens, 0)  # 各 run 计数隔离

    def test_budget_early_stop_in_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            # 极紧的工具调用预算：第 2 次工具调用后背压早停
            service = AgentService(
                llm=ScriptedLLM([ScriptedLLM.call("search", {"query": "x"})] * 6),
                base_dir=Path(tmp), budget=Budget(max_tool_calls=2, max_turns=20))
            sid = service.create_session().id
            result = service.send(sid, "一直查")
            self.assertTrue(result.error.startswith("stopped_budget"))
            self.assertEqual(result.final_state, RunState.STOPPED_BUDGET.value)


# ==========================================================================
# ToolRuntime 隔离
# ==========================================================================

class TestToolRuntimeIsolation(unittest.TestCase):
    def _run(self, registry, policy, name="probe", args=None, namespace=None):
        rt = ToolRuntime(registry, policy)
        session = Session(id="t")
        ctx = ExecutionContext(run_id="r", session=session, budget=Budget(),
                               global_memory=MemoryStore(), tool_namespace=namespace)
        tool_ctx = ToolContext(session=session, global_memory=ctx.memory_view)
        events = []
        outcome = rt.run(name, args or {}, tool_ctx, ctx,
                         lambda t, **d: events.append((t, d)), "c1")
        return outcome, events, ctx

    def test_permission_deny(self):
        reg = _registry(lambda a, c: "ok")
        policy = ToolPolicy(rules={"probe": ToolRule(permission=DENY)})
        outcome, events, _ = self._run(reg, policy)
        self.assertFalse(outcome.ok)
        self.assertTrue(outcome.denied)
        self.assertIn("tool_denied", [e[0] for e in events])

    def test_namespace_isolation(self):
        reg = _registry(lambda a, c: "ok")
        outcome, events, _ = self._run(reg, ToolPolicy(), namespace={"other_tool"})
        self.assertFalse(outcome.ok)
        self.assertTrue(outcome.denied)
        self.assertEqual(events[0][0], "tool_denied")
        self.assertEqual(events[0][1]["reason"], "out_of_namespace")

    def test_retry_recovers_flaky_tool(self):
        calls = {"n": 0}

        def flaky(args, ctx):
            calls["n"] += 1
            if calls["n"] < 3:
                raise ToolError(f"瞬态失败 #{calls['n']}")
            return "终于成功"

        reg = _registry(flaky)
        policy = ToolPolicy(rules={"probe": ToolRule(max_retries=3)})
        outcome, events, _ = self._run(reg, policy)
        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.observation, "终于成功")
        self.assertEqual(outcome.attempts, 3)
        self.assertEqual([e[0] for e in events].count("tool_retry"), 2)

    def test_retry_exhausted_reports_failure(self):
        def always_fail(args, ctx):
            raise ToolError("永久失败")
        reg = _registry(always_fail)
        policy = ToolPolicy(rules={"probe": ToolRule(max_retries=2)})
        outcome, events, _ = self._run(reg, policy)
        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.attempts, 3)  # 1 + 2 retries
        self.assertIn("永久失败", outcome.observation)

    def test_timeout(self):
        def slow(args, ctx):
            time.sleep(0.3)
            return "太慢了"
        reg = _registry(slow)
        policy = ToolPolicy(rules={"probe": ToolRule(timeout_s=0.05)})
        outcome, events, _ = self._run(reg, policy)
        self.assertFalse(outcome.ok)
        self.assertTrue(outcome.timed_out)
        self.assertIn("tool_timeout", [e[0] for e in events])

    def test_circuit_breaker_opens(self):
        def always_fail(args, ctx):
            raise ToolError("挂了")
        reg = _registry(always_fail)
        policy = ToolPolicy(rules={"probe": ToolRule(circuit_threshold=2)})
        rt = ToolRuntime(reg, policy)
        session = Session(id="t")
        ctx = ExecutionContext(run_id="r", session=session, budget=Budget(),
                               global_memory=MemoryStore())
        tool_ctx = ToolContext(session=session, global_memory=ctx.memory_view)
        emit = lambda t, **d: None
        # 连续两次失败累计到阈值
        rt.run("probe", {}, tool_ctx, ctx, emit)
        rt.run("probe", {}, tool_ctx, ctx, emit)
        events = []
        outcome = rt.run("probe", {}, tool_ctx, ctx, lambda t, **d: events.append((t, d)))
        self.assertFalse(outcome.ok)
        self.assertIn("circuit_open", [e[0] for e in events])
        self.assertIn("熔断", outcome.observation)

    def test_rate_limit(self):
        reg = _registry(lambda a, c: "ok")
        policy = ToolPolicy(rules={"probe": ToolRule(rate_per_minute=2)})
        rt = ToolRuntime(reg, policy)
        session = Session(id="t")
        ctx = ExecutionContext(run_id="r", session=session, budget=Budget(),
                               global_memory=MemoryStore())
        tool_ctx = ToolContext(session=session, global_memory=ctx.memory_view)
        emit = lambda t, **d: None
        self.assertTrue(rt.run("probe", {}, tool_ctx, ctx, emit).ok)
        self.assertTrue(rt.run("probe", {}, tool_ctx, ctx, emit).ok)
        third = rt.run("probe", {}, tool_ctx, ctx, emit)
        self.assertFalse(third.ok)
        self.assertIn("限流", third.observation)

    def test_policy_state_isolated_across_contexts(self):
        def always_fail(args, ctx):
            raise ToolError("x")
        reg = _registry(always_fail)
        policy = ToolPolicy(rules={"probe": ToolRule(circuit_threshold=1)})
        rt = ToolRuntime(reg, policy)
        session = Session(id="t")
        emit = lambda t, **d: None
        ctx_a = ExecutionContext(run_id="a", session=session, budget=Budget(),
                                 global_memory=MemoryStore())
        tool_ctx = ToolContext(session=session, global_memory=ctx_a.memory_view)
        rt.run("probe", {}, tool_ctx, ctx_a, emit)  # ctx_a 熔断计数 +1
        # 新 run 的 ExecutionContext 有独立 policy_state：不受 ctx_a 影响
        ctx_b = ExecutionContext(run_id="b", session=session, budget=Budget(),
                                 global_memory=MemoryStore())
        self.assertEqual(ctx_b.policy_state, {})


# ==========================================================================
# 记忆视图隔离
# ==========================================================================

class TestMemoryView(unittest.TestCase):
    def test_read_your_writes_but_deferred_commit(self):
        store = MemoryStore()
        store.set("a", "1")
        view = MemoryView(store)
        view.set("b", "2")
        # 视图内读到自己的写 + 起始快照
        self.assertEqual(view.get("b"), "2")
        self.assertEqual(view.items(), {"a": "1", "b": "2"})
        # 未提交前真实库不可见
        self.assertIsNone(store.get("b"))
        # 提交后落库
        self.assertEqual(view.commit(source="s"), 1)
        self.assertEqual(store.get("b"), "2")

    def test_concurrent_runs_isolated_until_commit(self):
        store = MemoryStore()
        v1, v2 = MemoryView(store), MemoryView(store)
        v1.set("x", "from1")
        # v2 起始快照早于 v1 提交，看不到 v1 未提交的写
        self.assertIsNone(v2.get("x"))
        v1.commit(source="s1")
        v2.set("y", "from2")
        v2.commit(source="s2")
        self.assertEqual(store.get("x"), "from1")
        self.assertEqual(store.get("y"), "from2")


# ==========================================================================
# LLM = Actor 静态守卫
# ==========================================================================

class TestLLMIsActorOnly(unittest.TestCase):
    """LLM 层只负责推理/工具选择/最终答案，绝不含调度/重试/流程控制词汇。"""

    CONTROL_VOCAB = ["retry", "schedule", "backpressure", "budget", "circuit",
                     "rate_limit", "early_stop", "downgrade", "compact",
                     "max_turns", "should_compact"]

    def test_llm_layer_has_no_control_vocabulary(self):
        pkg = Path(__file__).resolve().parent.parent / "mini_agent"
        violations = []
        for rel in ("core/llm.py", "core/llm_adapters.py"):
            text = (pkg / rel).read_text(encoding="utf-8").lower()
            for word in self.CONTROL_VOCAB:
                if word in text:
                    violations.append(f"{rel} 含控制词 '{word}'")
        self.assertEqual(violations, [],
                         "LLM 层泄漏了控制平面职责:\n" + "\n".join(violations))

    def test_llm_interface_is_minimal(self):
        from mini_agent.core.llm import LLMClient
        public = {m for m in dir(LLMClient) if not m.startswith("_")}
        # LLMClient 只暴露：complete（推理）+ summarize（内部一次性调用）+ describe/pricing
        self.assertEqual(public, {"complete", "summarize", "describe", "pricing"})

    def test_control_flow_lives_in_runtime_and_control(self):
        """压缩/预算/熔断等决策代码只应出现在 runtime/ 与 control/。"""
        pkg = Path(__file__).resolve().parent.parent / "mini_agent"
        # tools/builtin（工具实现）不应包含控制决策词
        text = (pkg / "tools" / "builtin.py").read_text(encoding="utf-8").lower()
        for word in ("backpressure", "should_compact", "early_stop", "circuit_threshold"):
            self.assertNotIn(word, text)


if __name__ == "__main__":
    unittest.main()
