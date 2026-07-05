"""流适配器：Provider 原始 delta -> 带生命周期边界的语义事件。

Provider 只上抛裸增量（thinking_delta / text_delta）；本适配器把它合成
thinking_start → thinking_delta* → thinking_end（携带完整内容与耗时）与
assistant_delta*，使前端/replay 永远不需要自己猜"思考到哪结束、回答从哪开始"。
每轮 LLM 调用创建一个实例，close() 收口当前层。
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional


class TurnStream:
    def __init__(self, emit):
        self._emit = emit
        self._mode: Optional[str] = None  # None | "thinking" | "text"
        self._thinking_buf: List[str] = []
        self._thinking_started = 0.0

    def __call__(self, etype: str, data: Dict[str, Any]) -> None:
        if etype == "thinking_delta":
            if self._mode != "thinking":
                self.close()
                self._mode = "thinking"
                self._thinking_started = time.time()
                self._thinking_buf = []
                self._emit("thinking_start")
            self._thinking_buf.append(data.get("text", ""))
            self._emit("thinking_delta", **data)
        elif etype == "text_delta":
            if self._mode != "text":
                self.close()
                self._mode = "text"
            self._emit("assistant_delta", **data)

    def close(self) -> None:
        """结束当前层（思考层收口时携带完整内容与耗时）。"""
        if self._mode == "thinking":
            self._emit(
                "thinking_end",
                content="".join(self._thinking_buf),
                duration_ms=int((time.time() - self._thinking_started) * 1000))
        self._mode = None


# 向后兼容别名
_TurnStream = TurnStream
