"""LLM Provider 适配器（仿 HelloAgents 的 core/llm_adapters.py，三家收敛到一处）。

- AnthropicLLM : 官方 anthropic SDK（streaming/adaptive thinking/原生工具/prompt cache）
- QwenLLM      : DashScope OpenAI 兼容接口（openai SDK）+ 请求/响应双向转换纯函数
- ScriptedLLM  : 离线测试/演示后端

三家互不共享 SDK 逻辑，各自把自家返回收敛到统一的 LLMResponse。
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from .llm import LLMClient, LLMError, LLMResponse, StreamCallback, ToolCallRequest


# ==========================================================================
# Anthropic
# ==========================================================================

_ANTHROPIC_PRICING = {
    "claude-fable-5": {"input": 10.0, "output": 50.0},
    "claude-opus-4-8": {"input": 5.0, "output": 25.0},
    "claude-opus-4-7": {"input": 5.0, "output": 25.0},
    "claude-opus-4-6": {"input": 5.0, "output": 25.0},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5": {"input": 1.0, "output": 5.0},
}


class AnthropicLLM(LLMClient):
    def __init__(self, model: str = "claude-opus-4-8", max_tokens: int = 16000):
        self.model = model
        self.max_tokens = max_tokens
        self._client = None
        base = _ANTHROPIC_PRICING.get(model)
        self.pricing = (
            {**base, "cache_read_multiplier": 0.1, "cache_write_multiplier": 1.25,
             "currency": "USD"} if base else None)

    def describe(self) -> str:
        return f"anthropic/{self.model}"

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic  # noqa: PLC0415
            except ImportError as exc:
                raise LLMError("未安装 anthropic SDK，请先执行: pip install anthropic") from exc
            self._client = anthropic.Anthropic()
        return self._client

    def complete(self, *, system, messages, tools=None, on_delta=None) -> LLMResponse:
        import anthropic  # noqa: PLC0415

        client = self._get_client()
        kwargs: Dict[str, Any] = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
            thinking={"type": "adaptive", "display": "summarized"},
        )
        if tools:
            kwargs["tools"] = tools  # ToolSpec 与本 Provider 格式一致，直接透传

        try:
            with client.messages.stream(**kwargs) as stream:
                for event in stream:
                    if on_delta is None or event.type != "content_block_delta":
                        continue
                    delta = event.delta
                    if delta.type == "text_delta":
                        on_delta("text_delta", {"text": delta.text})
                    elif delta.type == "thinking_delta":
                        on_delta("thinking_delta", {"text": delta.thinking})
                final = stream.get_final_message()
        except anthropic.RateLimitError as exc:
            raise LLMError(f"触发限流(429)，请稍后重试: {exc}") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"API 错误({exc.status_code}): {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"网络连接失败: {exc}") from exc

        if final.stop_reason == "refusal":
            raise LLMError("请求被模型安全策略拒绝(stop_reason=refusal)")

        return self._to_response(final)

    @staticmethod
    def _to_response(final) -> LLMResponse:
        """SDK Message -> 统一 LLMResponse（raw_content 为框架 canonical blocks）。"""
        raw_content: List[Dict[str, Any]] = []
        text_parts: List[str] = []
        thinking_parts: List[str] = []
        tool_calls: List[ToolCallRequest] = []

        for block in final.content:
            if block.type == "text":
                raw_content.append({"type": "text", "text": block.text})
                text_parts.append(block.text)
            elif block.type == "thinking":
                raw_content.append({
                    "type": "thinking",
                    "thinking": block.thinking,
                    "signature": block.signature,
                })
                if block.thinking:
                    thinking_parts.append(block.thinking)
            elif block.type == "redacted_thinking":
                raw_content.append({"type": "redacted_thinking", "data": block.data})
            elif block.type == "tool_use":
                raw_content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
                tool_calls.append(ToolCallRequest(
                    id=block.id, name=block.name, arguments=dict(block.input or {})))
            else:
                raw_content.append(block.model_dump())

        usage = final.usage
        return LLMResponse(
            stop_reason=final.stop_reason or "end_turn",
            text="".join(text_parts),
            thinking="\n".join(thinking_parts),
            tool_calls=tool_calls,
            raw_content=raw_content,
            usage={
                "input_tokens": getattr(usage, "input_tokens", 0) or 0,
                "output_tokens": getattr(usage, "output_tokens", 0) or 0,
                "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
                "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
            },
        )


# ==========================================================================
# Qwen —— DashScope OpenAI 兼容接口 + 双向转换纯函数（可离线单测）
# ==========================================================================

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

_STOP_REASON_MAP = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
}


def _system_text(system: List[Dict[str, Any]]) -> str:
    return "\n\n".join(
        b.get("text", "") for b in system if b.get("type") == "text"
    ).strip()


def _result_text(block: Dict[str, Any]) -> str:
    content = block.get("content", "")
    if isinstance(content, list):
        content = " ".join(
            str(b.get("text", "")) for b in content if isinstance(b, dict))
    text = str(content)
    return f"[error] {text}" if block.get("is_error") else text


def to_provider_messages(system: List[Dict[str, Any]],
                         messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """cache_control 等缓存提示直接忽略；thinking block 对本 Provider 无意义，跳过。"""
    out: List[Dict[str, Any]] = []
    text = _system_text(system)
    if text:
        out.append({"role": "system", "content": text})

    for msg in messages:
        role, content = msg["role"], msg["content"]
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue

        if role == "user":
            texts: List[str] = []
            images: List[Dict[str, Any]] = []
            for block in content:
                btype = block.get("type")
                if btype == "tool_result":
                    out.append({
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id", ""),
                        "content": _result_text(block),
                    })
                elif btype == "text":
                    texts.append(block.get("text", ""))
                elif btype == "image":
                    source = block.get("source", {})
                    media = source.get("media_type", "image/png")
                    images.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{media};base64,{source.get('data', '')}"},
                    })
            if images:
                parts: List[Dict[str, Any]] = images[:]
                if texts:
                    parts.append({"type": "text", "text": "\n".join(texts)})
                out.append({"role": "user", "content": parts})
            elif texts:
                out.append({"role": "user", "content": "\n".join(texts)})
            continue

        text_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        for block in content:
            btype = block.get("type")
            if btype == "text":
                text_parts.append(block.get("text", ""))
            elif btype == "tool_use":
                tool_calls.append({
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {}), ensure_ascii=False),
                    },
                })
        entry: Dict[str, Any] = {"role": "assistant",
                                 "content": "".join(text_parts) or None}
        if tool_calls:
            entry["tool_calls"] = tool_calls
        out.append(entry)
    return out


def to_provider_tools(specs: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": spec["name"],
                "description": spec.get("description", ""),
                "parameters": spec.get("input_schema", {"type": "object"}),
            },
        }
        for spec in (specs or [])
    ]


class StreamAccumulator:
    """chat.completions 流式状态机：吃 chunk，吐统一 LLMResponse。"""

    def __init__(self, on_delta: Optional[StreamCallback] = None):
        self._on_delta = on_delta
        self._text: List[str] = []
        self._reasoning: List[str] = []
        self._tool_slots: Dict[int, Dict[str, str]] = {}
        self._finish_reason: Optional[str] = None
        self._usage = None

    def feed(self, chunk: Any) -> None:
        if getattr(chunk, "usage", None) is not None:
            self._usage = chunk.usage
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            return
        choice = choices[0]
        delta = getattr(choice, "delta", None)
        if delta is not None:
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                self._reasoning.append(reasoning)
                if self._on_delta:
                    self._on_delta("thinking_delta", {"text": reasoning})
            content = getattr(delta, "content", None)
            if content:
                self._text.append(content)
                if self._on_delta:
                    self._on_delta("text_delta", {"text": content})
            for tc in getattr(delta, "tool_calls", None) or []:
                slot = self._tool_slots.setdefault(
                    tc.index, {"id": "", "name": "", "arguments": ""})
                if getattr(tc, "id", None):
                    slot["id"] = tc.id
                fn = getattr(tc, "function", None)
                if fn is not None:
                    if getattr(fn, "name", None):
                        slot["name"] += fn.name
                    if getattr(fn, "arguments", None):
                        slot["arguments"] += fn.arguments
        if getattr(choice, "finish_reason", None):
            self._finish_reason = choice.finish_reason

    def finalize(self) -> LLMResponse:
        text = "".join(self._text)
        thinking = "".join(self._reasoning)

        raw_content: List[Dict[str, Any]] = []
        if text:
            raw_content.append({"type": "text", "text": text})

        tool_calls: List[ToolCallRequest] = []
        for index in sorted(self._tool_slots):
            slot = self._tool_slots[index]
            call_id = slot["id"] or f"call_{index}"
            try:
                arguments = json.loads(slot["arguments"]) if slot["arguments"] else {}
                if not isinstance(arguments, dict):
                    arguments = {}
            except json.JSONDecodeError:
                arguments = {}
            raw_content.append({
                "type": "tool_use", "id": call_id,
                "name": slot["name"], "input": arguments,
            })
            tool_calls.append(ToolCallRequest(
                id=call_id, name=slot["name"], arguments=arguments))

        stop_reason = _STOP_REASON_MAP.get(
            self._finish_reason or "", self._finish_reason or "end_turn")
        if tool_calls:
            stop_reason = "tool_use"

        usage = self._usage
        cached = 0
        if usage is not None:
            details = getattr(usage, "prompt_tokens_details", None)
            cached = getattr(details, "cached_tokens", 0) or 0 if details else 0
        return LLMResponse(
            stop_reason=stop_reason,
            text=text,
            thinking=thinking,
            tool_calls=tool_calls,
            raw_content=raw_content,
            usage={
                "input_tokens": (getattr(usage, "prompt_tokens", 0) or 0) if usage else 0,
                "output_tokens": (getattr(usage, "completion_tokens", 0) or 0) if usage else 0,
                "cache_read_input_tokens": cached,
                "cache_creation_input_tokens": 0,
            },
        )


_QWEN_PRICING = {
    "qwen-max": {"input": 1.6, "output": 6.4},
    "qwen-plus": {"input": 0.4, "output": 1.2},
    "qwen-turbo": {"input": 0.05, "output": 0.2},
}


class QwenLLM(LLMClient):
    def __init__(self, model: str = "qwen-plus", max_tokens: int = 8192,
                 base_url: str = DEFAULT_BASE_URL, api_key: Optional[str] = None):
        self.model = model
        self.max_tokens = max_tokens
        self.base_url = base_url
        self.api_key = api_key
        self._client = None
        base = _QWEN_PRICING.get(model)
        self.pricing = (
            {**base, "cache_read_multiplier": 0.4, "cache_write_multiplier": 1.0,
             "currency": "USD"} if base else None)

    def describe(self) -> str:
        return f"qwen/{self.model}"

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI  # noqa: PLC0415
            except ImportError as exc:
                raise LLMError("未安装 openai SDK，请先执行: pip install openai") from exc
            key = self.api_key or os.environ.get("DASHSCOPE_API_KEY")
            if not key:
                raise LLMError("缺少 DASHSCOPE_API_KEY，请先 export DASHSCOPE_API_KEY=sk-...")
            self._client = OpenAI(api_key=key, base_url=self.base_url)
        return self._client

    def complete(self, *, system, messages, tools=None, on_delta=None) -> LLMResponse:
        client = self._get_client()
        import openai  # noqa: PLC0415

        kwargs: Dict[str, Any] = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=to_provider_messages(system, messages),
            stream=True,
            stream_options={"include_usage": True},
        )
        provider_tools = to_provider_tools(tools)
        if provider_tools:
            kwargs["tools"] = provider_tools

        accumulator = StreamAccumulator(on_delta)
        try:
            for chunk in client.chat.completions.create(**kwargs):
                accumulator.feed(chunk)
        except openai.RateLimitError as exc:
            raise LLMError(f"触发限流(429)，请稍后重试: {exc}") from exc
        except openai.APIStatusError as exc:
            raise LLMError(f"API 错误({exc.status_code}): {exc}") from exc
        except openai.APIConnectionError as exc:
            raise LLMError(f"网络连接失败: {exc}") from exc

        return accumulator.finalize()


# ==========================================================================
# Scripted —— 离线测试/演示
# ==========================================================================

class ScriptedLLM(LLMClient):
    """用 ScriptedLLM.say() / ScriptedLLM.call() 构造脚本条目；
    记录每次 complete 收到的输入，供测试断言。"""

    def __init__(self, script: List[Dict[str, Any]], label: str = "scripted"):
        self._script = list(script)
        self._call_counter = 0
        self._label = label
        self.calls: List[Dict[str, Any]] = []

    def describe(self) -> str:
        return self._label

    @staticmethod
    def say(text: str, thinking: str = "", usage: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        return {"text": text, "thinking": thinking, "tool_calls": [], "usage": usage}

    @staticmethod
    def call(name: str, arguments: Dict[str, Any], thinking: str = "",
             text: str = "", usage: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        return {"text": text, "thinking": thinking,
                "tool_calls": [{"name": name, "arguments": arguments}], "usage": usage}

    def complete(self, *, system, messages, tools=None, on_delta=None) -> LLMResponse:
        self.calls.append({
            "system": [dict(b) for b in system],
            "messages": [dict(m) for m in messages],
            "tools": [dict(t) for t in (tools or [])],
        })
        if not self._script:
            raise LLMError("ScriptedLLM 预设回复已用尽")
        entry = self._script.pop(0)

        thinking = entry.get("thinking", "")
        text = entry.get("text", "")
        if on_delta:
            if thinking:
                on_delta("thinking_delta", {"text": thinking})
            if text:
                on_delta("text_delta", {"text": text})

        raw_content: List[Dict[str, Any]] = []
        if thinking:
            raw_content.append({"type": "thinking", "thinking": thinking, "signature": "scripted"})
        if text:
            raw_content.append({"type": "text", "text": text})

        tool_calls: List[ToolCallRequest] = []
        for spec in entry.get("tool_calls", []):
            self._call_counter += 1
            call_id = f"toolu_scripted_{self._call_counter}"
            raw_content.append({
                "type": "tool_use", "id": call_id,
                "name": spec["name"], "input": spec["arguments"],
            })
            tool_calls.append(ToolCallRequest(
                id=call_id, name=spec["name"], arguments=spec["arguments"]))

        usage = entry.get("usage") or {
            "input_tokens": sum(len(str(m.get("content", ""))) for m in messages) // 3,
            "output_tokens": (len(text) + len(thinking)) // 3,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }
        return LLMResponse(
            stop_reason="tool_use" if tool_calls else "end_turn",
            text=text,
            thinking=thinking,
            tool_calls=tool_calls,
            raw_content=raw_content,
            usage=usage,
        )
