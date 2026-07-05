"""工具注册机制（名称 + 描述 + JSON Schema）。

注册表负责：
- tool_specs()     : 渲染为框架统一工具规格（ToolSpec），各 LLM Provider 自行转换为其 API 格式
- validate_args()  : 执行前按 schema 校验（必填/类型/enum/未知参数）
- execute()        : 统一执行入口，一切异常收敛为 ToolError（不击穿控制器循环）

注意：本模块只负责“工具本身”的注册与裸执行。执行隔离（timeout/retry/permission/
rate-limit/circuit-breaker）由控制层的 tools/runtime.py::ToolRuntime 在其之上包裹。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..memory.store import MemoryStore
    from ..core.session import Session


class ToolError(Exception):
    """工具校验失败或执行失败。message 会作为 tool_result(is_error) 反馈给模型。"""


@dataclass
class ToolContext:
    """工具执行时可访问的上下文：会话状态、记忆、事件上抛通道。"""

    session: "Session"
    global_memory: "MemoryStore"
    emit: Callable[..., None] = lambda event_type, **data: None  # 控制器注入

    # ---- 记忆读写的统一入口（自动路由到全局 / 会话私有存储） ----
    def _use_global(self) -> bool:
        return self.session.use_global_memory

    def memory_scope(self) -> str:
        return "全局(跨会话共享)" if self._use_global() else "会话私有"

    def memory_set(self, key: str, value: str) -> None:
        if self._use_global():
            self.global_memory.set(key, value, source=self.session.id)
        else:
            self.session.local_memory[key] = value
            self.session.save()

    def memory_items(self) -> Dict[str, str]:
        if self._use_global():
            return self.global_memory.items()
        return dict(self.session.local_memory)


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema
    func: Callable[[Dict[str, Any], ToolContext], str]


class ToolRegistry:
    _TYPE_MAP = {
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"工具重复注册: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise ToolError(f"未知工具 '{name}'，可用工具: {', '.join(sorted(self._tools))}")
        return self._tools[name]

    def names(self) -> List[str]:
        return sorted(self._tools)

    def tool_specs(self, allowed: "set[str] | None" = None) -> List[Dict[str, Any]]:
        """渲染为框架统一 ToolSpec: {name, description, input_schema(JSON Schema)}。
        按名称排序保证字节级稳定（支持 prompt cache 的 Provider 依赖这一点）。
        allowed 非 None 时只渲染该子集（供 ExecutionContext 的工具命名空间隔离）。"""
        names = [n for n in self.names() if allowed is None or n in allowed]
        return [
            {
                "name": self._tools[n].name,
                "description": self._tools[n].description,
                "input_schema": self._tools[n].parameters,
            }
            for n in names
        ]

    def validate_args(self, name: str, args: Any) -> Dict[str, Any]:
        tool = self.get(name)
        if not isinstance(args, dict):
            raise ToolError(f"工具 '{name}' 的 arguments 必须是 JSON 对象")
        schema = tool.parameters
        props: Dict[str, Any] = schema.get("properties", {})
        for required_key in schema.get("required", []):
            if required_key not in args:
                raise ToolError(f"工具 '{name}' 缺少必填参数 '{required_key}'")
        for key, value in args.items():
            if key not in props:
                raise ToolError(f"工具 '{name}' 不支持参数 '{key}'")
            spec = props[key]
            expected = self._TYPE_MAP.get(spec.get("type", ""))
            if expected is not None and not isinstance(value, expected):
                raise ToolError(
                    f"参数 '{key}' 类型错误，期望 {spec['type']}，实际 {type(value).__name__}")
            if spec.get("type") in ("number", "integer") and isinstance(value, bool):
                raise ToolError(f"参数 '{key}' 类型错误，期望 {spec['type']}")
            if "enum" in spec and value not in spec["enum"]:
                raise ToolError(f"参数 '{key}' 取值必须是 {spec['enum']} 之一，实际 '{value}'")
        return args

    def execute(self, name: str, args: Dict[str, Any], ctx: ToolContext) -> str:
        tool = self.get(name)
        validated = self.validate_args(name, args)
        try:
            return tool.func(validated, ctx)
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(f"工具 '{name}' 执行异常: {exc}") from exc
