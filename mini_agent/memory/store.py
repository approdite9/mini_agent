"""分层记忆设计。

三层记忆，各司其职：
1. 工作记忆   = 会话历史（session.history），随上下文窗口滚动，由 ContextManager 压缩
2. 会话记忆   = session.local_memory，隔离会话的私有键值，随会话文件持久化
3. 全局记忆   = MemoryStore（本文件），跨会话共享、磁盘 JSON 持久化、带更新时间与来源

注入策略（性能考量）：
- system prompt 只注入"记忆索引"（键 + 截断摘要），保持 prompt 精简；
- 完整值由模型通过 memory 工具按需 get —— 避免记忆膨胀撑爆上下文，
  同时记忆变化只影响 system 中独立的记忆块，不打穿核心 prompt 的缓存前缀。
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Dict, Optional


class MemoryStore:
    """磁盘 JSON 持久化的键值记忆，线程安全。path=None 时为纯内存（测试用）。

    全局记忆被多个会话共享，而不同会话可并行运行，因此所有读写都必须加锁：
    否则一个会话 items() 迭代 dict 时另一个会话 set() 改 dict 会
    RuntimeError 崩溃。落盘用 临时文件 + os.replace 原子替换，避免并发写
    产生半截 JSON 导致下次启动记忆全丢。
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else None
        self._entries: Dict[str, Dict[str, object]] = {}
        self._lock = threading.RLock()
        self._load()

    def _load(self) -> None:
        if self.path and self.path.is_file():
            try:
                self._entries = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._entries = {}  # 损坏的记忆文件不应导致启动失败

    def _persist(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self._entries, ensure_ascii=False, indent=2)
        # 原子写：先写临时文件再 rename，读者永远看到完整的旧或新版本
        tmp = self.path.with_suffix(self.path.suffix + f".{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, self.path)

    # ------------------------------------------------------------------
    def set(self, key: str, value: str, source: str = "") -> None:
        with self._lock:
            self._entries[key] = {
                "value": value,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "source": source,
            }
            self._persist()

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            entry = self._entries.get(key)
            return str(entry["value"]) if entry else None

    def items(self) -> Dict[str, str]:
        with self._lock:
            return {k: str(v["value"]) for k, v in self._entries.items()}

    def clear(self) -> None:
        with self._lock:
            self._entries = {}
            self._persist()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    # ------------------------------------------------------------------
    @staticmethod
    def render_index(items: Dict[str, str], value_limit: int = 60) -> str:
        """渲染记忆索引：键 + 截断的值。完整值靠 memory 工具 get。"""
        if not items:
            return "(空)"
        lines = []
        for key, value in items.items():
            short = value if len(value) <= value_limit else value[: value_limit - 1] + "…"
            lines.append(f"- {key}: {short}")
        return "\n".join(lines)
