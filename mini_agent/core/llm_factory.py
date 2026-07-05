"""Provider 工厂 —— 框架内核唯一的 LLM 获取入口。

按 LLM_PROVIDER 环境变量（或显式参数）选择 Provider：

    export LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=sk-...
    export LLM_PROVIDER=qwen      DASHSCOPE_API_KEY=sk-...
    export LLM_MODEL=qwen-max     # 可选，覆盖 Provider 默认模型

新增模型 = 新增一个 Provider 文件 + 在 _REGISTRY 注册一行，
Agent / Tool / Memory / Context / Event 全部零改动。
"""

from __future__ import annotations

import os
from typing import Callable, Dict, Optional, Tuple

from .llm import LLMClient, LLMError
from .llm_adapters import AnthropicLLM, QwenLLM

# provider 名 -> (构造函数, 必需的环境变量)
_REGISTRY: Dict[str, Tuple[Callable[..., LLMClient], str]] = {
    "anthropic": (AnthropicLLM, "ANTHROPIC_API_KEY"),
    "qwen": (QwenLLM, "DASHSCOPE_API_KEY"),
}

DEFAULT_PROVIDER = "anthropic"


def available_providers() -> list:
    """已注册的 Provider 名称列表（供前端展示与切换）。"""
    return sorted(_REGISTRY)


def create_llm(provider: Optional[str] = None, model: Optional[str] = None,
               **kwargs) -> LLMClient:
    name = (provider or os.environ.get("LLM_PROVIDER") or DEFAULT_PROVIDER).strip().lower()
    if name not in _REGISTRY:
        raise LLMError(
            f"不支持的 LLM_PROVIDER '{name}'，可选: {', '.join(sorted(_REGISTRY))}")

    factory, env_key = _REGISTRY[name]
    if not os.environ.get(env_key) and "api_key" not in kwargs:
        raise LLMError(
            f"Provider '{name}' 需要环境变量 {env_key}。示例:\n"
            f"  export LLM_PROVIDER={name} {env_key}=sk-...")

    model = model or os.environ.get("LLM_MODEL") or None
    if model:
        kwargs["model"] = model
    return factory(**kwargs)
