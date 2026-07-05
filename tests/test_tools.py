"""工具本体 + 注册机制（schema 校验）+ plan 长任务拆解工具的测试。"""

import unittest

from mini_agent import MemoryStore, Session, ToolContext, ToolError, build_default_registry


def make_ctx(use_global_memory: bool = True):
    session = Session(id="t1", use_global_memory=use_global_memory)
    return session, ToolContext(session=session, global_memory=MemoryStore())


class TestCalculator(unittest.TestCase):
    def setUp(self):
        self.registry = build_default_registry()
        _, self.ctx = make_ctx()

    def calc(self, expr):
        return self.registry.execute("calculator", {"expression": expr}, self.ctx)

    def test_basic_arithmetic(self):
        self.assertEqual(self.calc("(3+5)*2"), "16")
        self.assertEqual(self.calc("2**10"), "1024")
        self.assertEqual(self.calc("7//2 + 7%2"), "4")
        self.assertEqual(self.calc("-5 + 1.5"), "-3.5")

    def test_division_result_normalized(self):
        self.assertEqual(self.calc("126/7"), "18")

    def test_divide_by_zero(self):
        with self.assertRaisesRegex(ToolError, "除数不能为零"):
            self.calc("1/0")

    def test_syntax_error(self):
        with self.assertRaisesRegex(ToolError, "语法错误"):
            self.calc("3 +* 5")

    def test_unsafe_expression_rejected(self):
        with self.assertRaisesRegex(ToolError, "不支持的语法"):
            self.calc("__import__('os').system('ls')")


class TestSearchAndDocs(unittest.TestCase):
    def setUp(self):
        self.registry = build_default_registry()
        _, self.ctx = make_ctx()

    def test_search_hit(self):
        self.assertIn("ReAct", self.registry.execute("search", {"query": "ReAct"}, self.ctx))

    def test_search_miss(self):
        self.assertIn("未找到",
                      self.registry.execute("search", {"query": "量子引力乱码xyz"}, self.ctx))

    def test_search_empty_query(self):
        with self.assertRaisesRegex(ToolError, "不能为空"):
            self.registry.execute("search", {"query": "   "}, self.ctx)

    def test_read_docs_list(self):
        self.assertIn("agent_design.md", self.registry.execute("read_docs", {}, self.ctx))

    def test_read_docs_content(self):
        result = self.registry.execute("read_docs", {"doc_name": "agent_design.md"}, self.ctx)
        self.assertIn("ReAct", result)

    def test_read_docs_missing(self):
        with self.assertRaisesRegex(ToolError, "不存在"):
            self.registry.execute("read_docs", {"doc_name": "nope.md"}, self.ctx)

    def test_read_docs_path_traversal_blocked(self):
        with self.assertRaises(ToolError):
            self.registry.execute("read_docs", {"doc_name": "../demo.py"}, self.ctx)


class TestTodoWeatherMemory(unittest.TestCase):
    def setUp(self):
        self.registry = build_default_registry()
        self.session, self.ctx = make_ctx()

    def test_todo_lifecycle(self):
        run = lambda args: self.registry.execute("todo", args, self.ctx)
        self.assertIn("已添加", run({"action": "add", "item": "写周报"}))
        self.assertIn("写周报", run({"action": "list"}))
        self.assertIn("已完成", run({"action": "done", "item": "写周报"}))
        self.assertIn("[x] 写周报", run({"action": "list"}))
        self.assertIn("已删除", run({"action": "remove", "item": "写周报"}))
        self.assertIn("为空", run({"action": "list"}))

    def test_weather_known_and_unknown(self):
        self.assertIn("北京", self.registry.execute("weather", {"city": "北京"}, self.ctx))
        with self.assertRaisesRegex(ToolError, "暂无"):
            self.registry.execute("weather", {"city": "火星"}, self.ctx)

    def test_memory_global_scope(self):
        run = lambda args: self.registry.execute("memory", args, self.ctx)
        run({"action": "save", "key": "昵称", "value": "小明"})
        self.assertIn("小明", run({"action": "get", "key": "昵称"}))
        self.assertIn("昵称", run({"action": "list"}))
        self.assertEqual(self.ctx.global_memory.get("昵称"), "小明")
        self.assertEqual(self.session.local_memory, {})

    def test_memory_isolated_scope(self):
        session, ctx = make_ctx(use_global_memory=False)
        self.registry.execute("memory", {"action": "save", "key": "k", "value": "v"}, ctx)
        self.assertEqual(session.local_memory, {"k": "v"})
        self.assertEqual(len(ctx.global_memory), 0)

    def test_memory_get_missing(self):
        with self.assertRaisesRegex(ToolError, "记忆中没有"):
            self.registry.execute("memory", {"action": "get", "key": "无"}, self.ctx)


class TestPlanTool(unittest.TestCase):
    """长流程任务拆解：plan 工具。"""

    def setUp(self):
        self.registry = build_default_registry()
        self.session, self.ctx = make_ctx()
        self.emitted = []
        self.ctx.emit = lambda etype, **data: self.emitted.append((etype, data))

    def run_plan(self, args):
        return self.registry.execute("plan", args, self.ctx)

    def test_set_and_update_flow(self):
        result = self.run_plan({"action": "set", "steps": ["调研", "实现", "测试"]})
        self.assertIn("计划已创建", result)
        self.assertEqual(len(self.session.plan), 3)
        self.assertTrue(all(s["status"] == "pending" for s in self.session.plan))

        self.run_plan({"action": "update", "step": 1, "status": "in_progress"})
        self.assertEqual(self.session.plan[0]["status"], "in_progress")
        self.run_plan({"action": "update", "step": 1, "status": "done"})
        self.assertEqual(self.session.plan[0]["status"], "done")

        # 每次变更都上抛 plan_update 事件（前端实时渲染进度条）
        plan_events = [e for e in self.emitted if e[0] == "plan_update"]
        self.assertEqual(len(plan_events), 3)

    def test_show(self):
        self.run_plan({"action": "set", "steps": ["a", "b"]})
        shown = self.run_plan({"action": "show"})
        self.assertIn("1.", shown)
        self.assertIn("b", shown)

    def test_update_without_plan(self):
        with self.assertRaisesRegex(ToolError, "尚无计划"):
            self.run_plan({"action": "update", "step": 1, "status": "done"})

    def test_invalid_step_and_status(self):
        self.run_plan({"action": "set", "steps": ["a"]})
        with self.assertRaisesRegex(ToolError, "step 必须是"):
            self.run_plan({"action": "update", "step": 9, "status": "done"})
        with self.assertRaisesRegex(ToolError, "取值必须是"):  # enum 在 schema 层就拦截
            self.run_plan({"action": "update", "step": 1, "status": "flying"})

    def test_set_requires_nonempty_steps(self):
        with self.assertRaisesRegex(ToolError, "非空字符串数组"):
            self.run_plan({"action": "set", "steps": []})


class TestRegistryValidation(unittest.TestCase):
    def setUp(self):
        self.registry = build_default_registry()
        _, self.ctx = make_ctx()

    def test_unknown_tool(self):
        with self.assertRaisesRegex(ToolError, "未知工具"):
            self.registry.execute("no_such_tool", {}, self.ctx)

    def test_missing_required_param(self):
        with self.assertRaisesRegex(ToolError, "必填参数"):
            self.registry.execute("calculator", {}, self.ctx)

    def test_wrong_param_type(self):
        with self.assertRaisesRegex(ToolError, "类型错误"):
            self.registry.execute("calculator", {"expression": 123}, self.ctx)

    def test_unexpected_param(self):
        with self.assertRaisesRegex(ToolError, "不支持参数"):
            self.registry.execute("weather", {"city": "北京", "foo": 1}, self.ctx)

    def test_enum_violation(self):
        with self.assertRaisesRegex(ToolError, "取值必须是"):
            self.registry.execute("todo", {"action": "fly"}, self.ctx)

    def test_bool_not_accepted_as_integer(self):
        with self.assertRaisesRegex(ToolError, "类型错误"):
            self.registry.validate_args("plan", {"action": "update", "step": True, "status": "done"})

    def test_tool_specs_unified_schema(self):
        tools = self.registry.tool_specs()
        names = [t["name"] for t in tools]
        for expected in ["calculator", "search", "read_docs", "todo", "weather", "memory", "plan"]:
            self.assertIn(expected, names)
        # 框架统一 ToolSpec：input_schema 字段 + 按名称排序（字节级稳定，缓存友好）
        self.assertTrue(all("input_schema" in t for t in tools))
        self.assertEqual(names, sorted(names))


if __name__ == "__main__":
    unittest.main()
