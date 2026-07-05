"""LLM Provider 抽象层 —— 框架与具体模型之间的唯一边界。

设计原则：
- LLM 只负责推理与工具选择；ReAct 调度、上下文、记忆、工具运行时、会话、
  事件流、trace 全部由框架实现，不依赖任何 Provider
- Agent 永远只消费 LLMResponse 这一个统一协议；Provider 负责把自家 API 的
  返回转换成它，把框架的请求转换成自家格式
- Provider 不支持的能力返回空值（如 thinking=""），禁止让 Agent 做
  "if provider == xxx" 判断 —— 任何 Provider 名称不得出现在框架内核中

框架侧的统一约定（Provider 必须双向转换）：
- 工具规格 ToolSpec: {"name", "description", "input_schema": <JSON Schema>}
  （由 ToolRegistry.tool_specs() 产出）
- 对话历史 canonical blocks: 消息 content 为 str 或 block 列表，block 类型为
  text / thinking / tool_use / tool_result / image —— 这是框架定义的持久化与
  上下文管理格式，不是某家厂商的 wire format。raw_content 必须用它表达，
  这样历史才能经同一 Provider 无损回放，也能被 ContextManager 统一压缩
- 多模态：canonical image block 形如
  {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}}
  各 Provider 自行转换；不支持视觉的模型在 API 层报错并以 LLMError 上抛，
  内核不做能力判断
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


class LLMError(Exception):
    """LLM 调用失败（配置缺失、网络、鉴权、限流、安全拒绝、脚本耗尽等）。"""


# 流式增量回调：(事件类型, 数据)，事件类型为 "thinking_delta" | "text_delta"
StreamCallback = Callable[[str, Dict[str, Any]], None]


@dataclass
class ToolCallRequest:
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class LLMResponse:
    """Agent 消费的唯一协议。所有 Provider 的返回都收敛到这里。"""

    stop_reason: str  # "end_turn" | "tool_use" | "max_tokens" | ...
    text: str = ""
    thinking: str = ""  # Provider 不支持思考输出时为空串
    tool_calls: List[ToolCallRequest] = field(default_factory=list)
    # 框架 canonical blocks（纯 dict）：作为 assistant 消息回写历史并持久化
    raw_content: List[Dict[str, Any]] = field(default_factory=list)
    # 统一 usage：input/output/cache_read_input/cache_creation_input tokens
    usage: Dict[str, int] = field(default_factory=dict)

    @property
    def wants_tools(self) -> bool:
        return self.stop_reason == "tool_use" and bool(self.tool_calls)


class LLMClient(ABC):
    #: 可选的定价元数据（供框架审计层估算成本，纯数据、无 Provider 逻辑）：
    #: {"input": $/MTok, "output": $/MTok,
    #:  "cache_read_multiplier": 命中缓存的输入折扣, "cache_write_multiplier": 写缓存溢价,
    #:  "currency": "USD"}；未知定价时为 None，审计层只展示 token 数
    pricing: Optional[Dict[str, Any]] = None

    def describe(self) -> str:
        """人类可读的模型标识（如 'provider/model'），用于审计与前端展示。"""
        return type(self).__name__

    @abstractmethod
    def complete(
        self,
        *,
        system: List[Dict[str, Any]],
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        on_delta: Optional[StreamCallback] = None,
    ) -> LLMResponse:
        """system 为框架 text block 列表（可带 cache_control 提示，Provider
        不支持缓存时忽略即可）；messages 为框架 canonical 消息；tools 为
        ToolSpec 列表。流式增量经 on_delta 上抛。"""

    def summarize(self, prompt: str) -> str:
        """无工具、无历史的一次性调用，供框架内部任务（如上下文压缩）使用。"""
        response = self.complete(
            system=[{"type": "text", "text": "你是精炼的文本压缩助手，只输出结果本身。"}],
            messages=[{"role": "user", "content": prompt}],
        )
        return response.text
