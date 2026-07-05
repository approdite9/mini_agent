"""上下文与性能管理。

两个核心机制：

1. Prompt Cache 断点设计（省钱 + 提速）
   请求的渲染顺序是 tools -> system -> messages，缓存是前缀匹配。因此：
   - 工具 schema 按名称排序（tools.tool_specs），字节级稳定
   - system 拆成两块：块 1 = 冻结的核心 prompt（行为准则，永不变），
     在块 1 上打 cache_control 断点 —— tools + 核心 prompt 一起命中缓存；
     块 2 = 记忆索引（会变），放在断点之后，它的变化不打穿前缀缓存
   - 最后一条消息的最后一个 block 再打一个断点，多轮对话增量命中

2. Usage 驱动的上下文压缩（compaction）
   不用估算 token —— 直接用上一轮 API 返回的真实 usage 判断上下文规模。
   超过阈值时，把较早的对话用 LLM 压缩成一段摘要，替换为一条前情提要消息。
   压缩边界必须落在"纯用户消息"上，绝不能拆开 tool_use/tool_result 配对
   （拆开会被 API 直接拒绝）。
"""

from __future__ import annotations

import copy
import json
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..core.llm import LLMClient, LLMError
from ..core.session import Session

CORE_SYSTEM_PROMPT = """\
你是一个高效可靠的中文 AI 助手，通过调用工具完成用户任务。

工作准则：
- 数值计算必须使用 calculator，不要口算；事实类问题优先用 search/read_docs 查证
- 需要 3 步以上的复杂任务，先用 plan 工具拆解为分步计划，执行中逐步更新状态
- 用户要求"记住"的信息用 memory 工具持久化；system 中的记忆索引只是摘要，需要完整内容时用 memory get
- 工具报错时分析原因：能修正参数就重试，不能就向用户如实说明
- 回答简洁直接，先给结论再给必要的支撑信息
"""
# 注意：这个 prompt 是冻结的 —— 不要往里插入时间戳、会话 ID 等易变内容，
# 否则每个请求都会打穿 prompt cache。易变内容一律放记忆块或消息里。


def _flatten_message(msg: Dict[str, Any]) -> str:
    """把一条原生消息压平为纯文本，供压缩摘要使用。"""
    content = msg.get("content", "")
    if isinstance(content, str):
        return f"{msg['role']}: {content}"
    parts = []
    for block in content:
        btype = block.get("type")
        if btype == "text":
            parts.append(block.get("text", ""))
        elif btype == "tool_use":
            parts.append(f"(调用工具 {block.get('name')} 参数 "
                         f"{json.dumps(block.get('input', {}), ensure_ascii=False)[:200]})")
        elif btype == "tool_result":
            result = block.get("content", "")
            if isinstance(result, list):
                result = " ".join(str(b.get("text", "")) for b in result if isinstance(b, dict))
            parts.append(f"(工具结果: {str(result)[:200]})")
        elif btype == "image":
            parts.append("(用户发送了一张图片)")
        # thinking block 不进摘要 —— 内部推理无需保留
    return f"{msg['role']}: {' '.join(p for p in parts if p)}"


def _is_plain_user_message(msg: Dict[str, Any]) -> bool:
    """是否为不含 tool_result 的用户消息 —— 唯一合法的压缩切割点。"""
    if msg.get("role") != "user":
        return False
    content = msg.get("content", "")
    if isinstance(content, str):
        return True
    return all(b.get("type") != "tool_result" for b in content)


class ContextManager:
    def __init__(
        self,
        max_context_tokens: int = 60_000,   # 超过即触发压缩（远低于模型上限，留足余量）
        keep_recent_messages: int = 12,     # 压缩时至少保留的近期消息数
        summary_max_chars: int = 12_000,    # 送去压缩的原文上限，防摘要请求本身过大
    ):
        self.max_context_tokens = max_context_tokens
        self.keep_recent_messages = keep_recent_messages
        self.summary_max_chars = summary_max_chars

    # ------------------------------------------------------------------
    def build_request(
        self,
        session: Session,
        memory_index: str,
        memory_scope: str,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """组装 (system blocks, messages)，按缓存友好的方式放置 cache_control。"""
        system = [
            {
                "type": "text",
                "text": CORE_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},  # 断点1：tools+核心prompt
            },
            {
                "type": "text",
                "text": f"## 持久记忆索引（范围：{memory_scope}）\n{memory_index}",
            },
        ]

        # 深拷贝后在最后一条消息上打断点，绝不污染 session.history
        messages = copy.deepcopy(session.history)
        if messages:
            last = messages[-1]
            if isinstance(last.get("content"), str):
                last["content"] = [{"type": "text", "text": last["content"]}]
            if isinstance(last["content"], list) and last["content"]:
                last["content"][-1]["cache_control"] = {"type": "ephemeral"}  # 断点2：多轮增量
        return system, messages

    # ------------------------------------------------------------------
    def should_compact(self, session: Session) -> bool:
        return session.context_tokens() > self.max_context_tokens

    def compact(
        self,
        session: Session,
        llm: LLMClient,
        on_event: Optional[Callable[..., None]] = None,
    ) -> bool:
        """把较早的历史压缩为一条前情摘要。返回是否实际发生了压缩。

        切割点选择：从"保留最近 keep_recent_messages 条"的位置向后找
        第一条纯用户消息，保证 tool_use/tool_result 配对不被拆散。
        """
        history = session.history
        desired_cut = len(history) - self.keep_recent_messages
        if desired_cut < 2:
            return False  # 历史太短，压了也省不了多少

        cut = None
        for i in range(desired_cut, len(history)):
            if _is_plain_user_message(history[i]):
                cut = i
                break
        if cut is None or cut < 2:
            return False

        old_text = "\n".join(_flatten_message(m) for m in history[:cut])
        old_text = old_text[-self.summary_max_chars:]
        try:
            summary = llm.summarize(
                "请把下面这段人机对话压缩成简明的中文摘要，保留：用户的关键信息与偏好、"
                "已完成的任务与结论、未完成的事项。丢弃寒暄与过程细节。\n\n" + old_text)
        except LLMError:
            return False  # 压缩失败不致命：本轮先照常进行，下轮再试

        dropped = cut
        session.history = [
            {"role": "user",
             "content": f"[前情摘要，由更早的 {dropped} 条消息压缩而来]\n{summary}"},
            *history[cut:],
        ]
        session.save()
        if on_event:
            on_event("compaction", dropped_messages=dropped,
                     summary_chars=len(summary))
        return True
