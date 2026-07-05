"""内置工具及使用场景：

- plan       : 长流程任务拆解——先制定分步计划，边执行边更新状态（前端实时渲染进度）
- calculator : 数学计算（避免模型口算出错）
- search     : 内置知识库检索
- read_docs  : 阅读本地 docs/ 文档
- todo       : 会话级待办管理
- weather    : 城市天气查询（mock 外部 API）
- memory     : 读写持久化记忆（范围由会话设置决定：全局共享 / 会话私有）
"""

from __future__ import annotations

import ast
import operator
from pathlib import Path
from typing import Any, Dict, List

from .registry import Tool, ToolContext, ToolError, ToolRegistry


# --------------------------------------------------------------------------
# plan —— 长流程任务拆解
# --------------------------------------------------------------------------

_PLAN_STATUSES = ["pending", "in_progress", "done", "skipped"]
_STATUS_ICON = {"pending": "○", "in_progress": "◐", "done": "●", "skipped": "⊘"}


def _render_plan(plan: List[Dict[str, Any]]) -> str:
    if not plan:
        return "当前没有计划"
    return "\n".join(
        f"{i + 1}. {_STATUS_ICON.get(s['status'], '?')} [{s['status']}] {s['title']}"
        for i, s in enumerate(plan)
    )


def _plan(args: Dict[str, Any], ctx: ToolContext) -> str:
    action = args["action"]
    if action == "set":
        steps = args.get("steps") or []
        if not steps or not all(isinstance(s, str) and s.strip() for s in steps):
            raise ToolError("set 操作需要 steps: 非空字符串数组")
        ctx.session.plan = [{"title": s.strip(), "status": "pending"} for s in steps]
        ctx.session.save()
        ctx.emit("plan_update", plan=list(ctx.session.plan))
        return "计划已创建:\n" + _render_plan(ctx.session.plan)
    if action == "update":
        plan = ctx.session.plan
        if not plan:
            raise ToolError("尚无计划，请先用 action=set 创建")
        step, status = args.get("step"), args.get("status")
        if not isinstance(step, int) or not (1 <= step <= len(plan)):
            raise ToolError(f"step 必须是 1~{len(plan)} 的整数")
        if status not in _PLAN_STATUSES:
            raise ToolError(f"status 必须是 {_PLAN_STATUSES} 之一")
        plan[step - 1]["status"] = status
        ctx.session.save()
        ctx.emit("plan_update", plan=list(plan))
        return "计划已更新:\n" + _render_plan(plan)
    if action == "show":
        return _render_plan(ctx.session.plan)
    raise ToolError(f"不支持的 action: {action}")


# --------------------------------------------------------------------------
# calculator —— AST 白名单安全求值
# --------------------------------------------------------------------------

_MAX_POW_RESULT_BITS = 4096  # 幂结果位数上限，防天文数字整数拖垮进程（DoS）


def _guarded_pow(base, exp):
    # 整数幂在指数很大时会生成巨型整数（如 9**9**9），CPU/内存双爆。
    # 估算结果位数并在超限时拒绝；浮点幂会溢出为 inf，不构成算力炸弹，放行。
    if isinstance(base, int) and isinstance(exp, int) and exp > 0 and base not in (0, 1, -1):
        if base.bit_length() * exp > _MAX_POW_RESULT_BITS:
            raise ToolError("幂运算结果过大，已拒绝（防止拒绝服务）")
    return operator.pow(base, exp)


_ALLOWED_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: _guarded_pow,
}
_ALLOWED_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
        return _ALLOWED_UNARY[type(node.op)](_safe_eval(node.operand))
    raise ToolError(f"表达式包含不支持的语法: {ast.dump(node)[:60]}")


def _calculator(args: Dict[str, Any], ctx: ToolContext) -> str:
    expr = args["expression"]
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"表达式语法错误: {expr!r}") from exc
    try:
        result = _safe_eval(tree)
    except ZeroDivisionError as exc:
        raise ToolError("除数不能为零") from exc
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)


# --------------------------------------------------------------------------
# search / read_docs
# --------------------------------------------------------------------------

_SEARCH_CORPUS: Dict[str, str] = {
    "ReAct 模式": "ReAct 是一种 Agent 范式：模型交替进行 Reasoning(思考) 与 Acting(调用工具)，"
                  "根据工具返回的观察结果决定下一步，直到得出最终答案。",
    "Function Calling": "Function calling 指 LLM 根据工具的名称、描述与参数 schema，"
                        "输出结构化的调用请求，由外部程序执行后把结果回传给模型。",
    "Python GIL": "GIL(全局解释器锁)使 CPython 同一时刻只有一个线程执行字节码，"
                  "CPU 密集任务应使用多进程，IO 密集任务可用多线程或 asyncio。",
    "Transformer": "Transformer 是基于自注意力机制的神经网络架构，"
                   "是现代大语言模型的基础结构。",
    "北京": "北京是中国的首都，著名景点包括故宫、长城、颐和园，最佳旅游季节为春秋两季。",
    "上海": "上海是中国的经济中心，地标有外滩、东方明珠、豫园，夏季湿热、冬季湿冷。",
}


def _search(args: Dict[str, Any], ctx: ToolContext) -> str:
    query = args["query"].strip()
    if not query:
        raise ToolError("query 不能为空")
    terms = [t for t in query.replace("，", " ").split() if t] or [query]
    scored = []
    for title, text in _SEARCH_CORPUS.items():
        haystack = (title + text).lower()
        score = sum(1 for t in terms if t.lower() in haystack)
        if query.lower() in haystack:
            score += 2
        if score > 0:
            scored.append((score, title, text))
    if not scored:
        return f"未找到与 '{query}' 相关的结果"
    scored.sort(key=lambda x: -x[0])
    return "\n".join(f"[{title}] {text}" for _, title, text in scored[:3])


# builtin.py 位于 mini_agent/tools/ 下，上溯三层到项目根再进 docs/
DOCS_DIR = Path(__file__).resolve().parent.parent.parent / "docs"


def _read_docs(args: Dict[str, Any], ctx: ToolContext) -> str:
    name = args.get("doc_name")
    if not DOCS_DIR.is_dir():
        raise ToolError(f"文档目录不存在: {DOCS_DIR}")
    available = sorted(p.name for p in DOCS_DIR.glob("*.md"))
    if not name:
        return "可用文档: " + (", ".join(available) or "(空)")
    target = (DOCS_DIR / name).resolve()
    if DOCS_DIR.resolve() not in target.parents:
        raise ToolError(f"非法文档路径: {name}")
    if not target.is_file():
        raise ToolError(f"文档 '{name}' 不存在，可用文档: {', '.join(available)}")
    return target.read_text(encoding="utf-8")[:4000]


# --------------------------------------------------------------------------
# todo / weather / memory
# --------------------------------------------------------------------------

def _todo(args: Dict[str, Any], ctx: ToolContext) -> str:
    action = args["action"]
    todos = ctx.session.todos
    if action == "add":
        item = args.get("item", "").strip()
        if not item:
            raise ToolError("add 操作需要非空的 item 参数")
        todos.append({"item": item, "done": False})
        ctx.session.save()
        return f"已添加待办: {item}（当前共 {len(todos)} 项）"
    if action == "list":
        if not todos:
            return "待办列表为空"
        return "\n".join(
            f"{i + 1}. [{'x' if t['done'] else ' '}] {t['item']}"
            for i, t in enumerate(todos))
    if action == "done":
        item = args.get("item", "").strip()
        for t in todos:
            if t["item"] == item:
                t["done"] = True
                ctx.session.save()
                return f"已完成待办: {item}"
        raise ToolError(f"待办 '{item}' 不存在")
    if action == "remove":
        item = args.get("item", "").strip()
        for i, t in enumerate(todos):
            if t["item"] == item:
                todos.pop(i)
                ctx.session.save()
                return f"已删除待办: {item}"
        raise ToolError(f"待办 '{item}' 不存在")
    raise ToolError(f"不支持的 action: {action}")


_WEATHER_DATA = {
    "北京": "晴，26°C，西北风 3 级，空气质量良",
    "上海": "多云转小雨，29°C，湿度 78%",
    "广州": "雷阵雨，31°C，注意防雷防雨",
    "深圳": "阴，30°C，湿度 85%",
    "杭州": "晴，28°C，微风",
}


def _weather(args: Dict[str, Any], ctx: ToolContext) -> str:
    city = args["city"].strip()
    if city in _WEATHER_DATA:
        return f"{city}今日天气: {_WEATHER_DATA[city]}"
    raise ToolError(f"暂无 '{city}' 的天气数据，支持的城市: {', '.join(_WEATHER_DATA)}")


def _memory(args: Dict[str, Any], ctx: ToolContext) -> str:
    action = args["action"]
    if action == "save":
        key, value = args.get("key", "").strip(), args.get("value", "").strip()
        if not key or not value:
            raise ToolError("save 操作需要非空的 key 和 value 参数")
        ctx.memory_set(key, value)
        ctx.emit("memory_update", action="save", key=key, scope=ctx.memory_scope())
        return f"已保存到{ctx.memory_scope()}记忆: {key} = {value}"
    items = ctx.memory_items()
    if action == "get":
        key = args.get("key", "").strip()
        if key not in items:
            raise ToolError(f"记忆中没有 '{key}'，已有键: {', '.join(items) or '(空)'}")
        return f"{key} = {items[key]}"
    if action == "list":
        if not items:
            return f"{ctx.memory_scope()}记忆为空"
        return "\n".join(f"{k} = {v}" for k, v in items.items())
    raise ToolError(f"不支持的 action: {action}")


# --------------------------------------------------------------------------

def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(Tool(
        name="plan",
        description="任务计划管理。遇到需要 3 步以上的复杂任务时，先用 action=set 制定分步计划，"
                    "开始执行某步前将其置为 in_progress，完成后置为 done。"
                    "简单的一两步任务不要使用本工具。",
        parameters={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["set", "update", "show"]},
                "steps": {"type": "array", "items": {"type": "string"},
                          "description": "set 时必填：分步计划，每项一句话"},
                "step": {"type": "integer", "description": "update 时必填：第几步（从 1 开始）"},
                "status": {"type": "string", "enum": _PLAN_STATUSES,
                           "description": "update 时必填：该步骤的新状态"},
            },
            "required": ["action"],
        },
        func=_plan,
    ))
    registry.register(Tool(
        name="calculator",
        description="计算数学表达式。当问题涉及加减乘除、幂、取余等数值计算时调用，不要口算。"
                    "支持 + - * / // % ** 和括号。",
        parameters={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "如 '(3+5)*2/4'"},
            },
            "required": ["expression"],
        },
        func=_calculator,
    ))
    registry.register(Tool(
        name="search",
        description="在内置知识库中检索资料。当用户询问概念、常识、城市介绍等事实类问题时，"
                    "先调用本工具获取依据再回答。",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "检索关键词"}},
            "required": ["query"],
        },
        func=_search,
    ))
    registry.register(Tool(
        name="read_docs",
        description="读取本地 docs/ 目录下的 markdown 文档。当用户询问'文档里怎么写的'"
                    "或需要引用项目文档时调用；不传 doc_name 时返回可用文档列表。",
        parameters={
            "type": "object",
            "properties": {
                "doc_name": {"type": "string", "description": "文档文件名，如 'agent_design.md'"},
            },
            "required": [],
        },
        func=_read_docs,
    ))
    registry.register(Tool(
        name="todo",
        description="管理当前会话的待办事项（按会话隔离）。当用户要求记录任务、查看任务列表、"
                    "标记完成或删除任务时调用。",
        parameters={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["add", "list", "done", "remove"]},
                "item": {"type": "string", "description": "待办内容，add/done/remove 时必填"},
            },
            "required": ["action"],
        },
        func=_todo,
    ))
    registry.register(Tool(
        name="weather",
        description="查询指定城市的今日天气。当用户询问天气、出行穿衣建议时调用。",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string", "description": "城市名，如 '北京'"}},
            "required": ["city"],
        },
        func=_weather,
    ))
    registry.register(Tool(
        name="memory",
        description="读写持久化的用户记忆（偏好、事实等，跨进程保存）。用户让你'记住'某事时用 save；"
                    "system prompt 的记忆索引只含摘要，需要完整内容时用 get。"
                    "记忆范围由会话设置决定：共享全局或会话私有。",
        parameters={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["save", "get", "list"]},
                "key": {"type": "string", "description": "记忆键，如 '用户昵称'"},
                "value": {"type": "string", "description": "记忆值，save 时必填"},
            },
            "required": ["action"],
        },
        func=_memory,
    ))
    return registry
