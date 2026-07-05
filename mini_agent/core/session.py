"""会话（窗口）与会话运行时。

- 每个会话有独立的：对话历史（原生 content blocks）、待办、任务计划、
  私有记忆、token 用量统计
- 持久化：会话保存为 base_dir/sessions/<id>.json，进程重启后可恢复继续对话
- 运行时：每个会话持有一把锁，同一会话同时只允许一个 run（Web 端并发防护）；
  不同会话之间可并行
"""

from __future__ import annotations

import copy
import itertools
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Session:
    id: str
    title: str = ""
    use_global_memory: bool = True
    history: List[Dict[str, Any]] = field(default_factory=list)   # 原生消息（content 为 str 或 block 列表）
    todos: List[Dict[str, Any]] = field(default_factory=list)
    plan: List[Dict[str, Any]] = field(default_factory=list)      # [{"title", "status"}]
    local_memory: Dict[str, str] = field(default_factory=dict)
    request_count: int = 0
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    last_usage: Dict[str, int] = field(default_factory=dict)      # 最近一轮 LLM 的 usage，驱动压缩决策
    total_usage: Dict[str, int] = field(default_factory=dict)     # 累计 usage
    runs: List[Dict[str, Any]] = field(default_factory=list)      # 运行审计记录（Run Inspector）
    provider: Optional[str] = None                                # 会话级模型 Provider 覆盖（None=默认）
    parent: Optional[str] = None                                  # 分叉来源，如 "s1@6"

    _dir: Optional[Path] = field(default=None, repr=False, compare=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    # ------------------------------------------------------------------
    @property
    def lock(self) -> threading.Lock:
        return self._lock

    def context_tokens(self) -> int:
        """最近一轮请求的实际上下文规模（input + cache 读写都算进上下文）。"""
        u = self.last_usage
        return (u.get("input_tokens", 0)
                + u.get("cache_read_input_tokens", 0)
                + u.get("cache_creation_input_tokens", 0))

    def record_usage(self, usage: Dict[str, int]) -> None:
        self.last_usage = dict(usage)
        for key, value in usage.items():
            self.total_usage[key] = self.total_usage.get(key, 0) + int(value or 0)

    # ---- 持久化 -------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "use_global_memory": self.use_global_memory,
            "history": self.history,
            "todos": self.todos,
            "plan": self.plan,
            "local_memory": self.local_memory,
            "request_count": self.request_count,
            "created_at": self.created_at,
            "last_usage": self.last_usage,
            "total_usage": self.total_usage,
            "runs": self.runs,
            "provider": self.provider,
            "parent": self.parent,
        }

    def save(self) -> None:
        if self._dir is None:
            return  # 未挂载持久化目录（测试场景）
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / f"{self.id}.json"
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8")

    @classmethod
    def from_dict(cls, data: Dict[str, Any], directory: Optional[Path] = None) -> "Session":
        session = cls(
            id=data["id"],
            title=data.get("title", ""),
            use_global_memory=data.get("use_global_memory", True),
            history=data.get("history", []),
            todos=data.get("todos", []),
            plan=data.get("plan", []),
            local_memory=data.get("local_memory", {}),
            request_count=data.get("request_count", 0),
            created_at=data.get("created_at", ""),
            last_usage=data.get("last_usage", {}),
            total_usage=data.get("total_usage", {}),
            runs=data.get("runs", []),
            provider=data.get("provider"),
            parent=data.get("parent"),
        )
        session._dir = directory
        return session


def _is_clean_cut(history: List[Dict[str, Any]]) -> bool:
    """分叉切割合法性：历史为空，或最后一条是不含未回填 tool_use 的 assistant 消息。"""
    if not history:
        return True
    last = history[-1]
    if last.get("role") != "assistant":
        return False
    content = last.get("content", "")
    if isinstance(content, str):
        return True
    return all(b.get("type") != "tool_use" for b in content)


class SessionManager:
    """会话的创建 / 加载 / 列表。base_dir=None 时纯内存（测试用）。"""

    def __init__(self, base_dir: Optional[Path] = None):
        self._dir = Path(base_dir) / "sessions" if base_dir else None
        self._sessions: Dict[str, Session] = {}
        self._counter = itertools.count(1)
        self._load_existing()

    def _load_existing(self) -> None:
        if self._dir is None or not self._dir.is_dir():
            return
        max_num = 0
        for path in sorted(self._dir.glob("s*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                session = Session.from_dict(data, directory=self._dir)
            except (json.JSONDecodeError, KeyError, OSError):
                continue  # 跳过损坏文件
            self._sessions[session.id] = session
            try:
                max_num = max(max_num, int(session.id.lstrip("s")))
            except ValueError:
                pass
        self._counter = itertools.count(max_num + 1)

    def create(self, title: str = "", use_global_memory: bool = True) -> Session:
        sid = f"s{next(self._counter)}"
        session = Session(
            id=sid,
            title=title or f"会话{sid[1:]}",
            use_global_memory=use_global_memory,
        )
        session._dir = self._dir
        self._sessions[sid] = session
        session.save()
        return session

    def get(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> bool:
        """删除会话：移出内存并删除落盘文件。返回是否存在过。"""
        session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        if self._dir is not None:
            try:
                (self._dir / f"{session_id}.json").unlink(missing_ok=True)
            except OSError:
                pass  # 文件删除失败不影响内存态移除
        return True

    def list(self) -> List[Session]:
        return list(self._sessions.values())

    def fork(self, source: Session, at_message: Optional[int] = None,
             provider: Optional[str] = None, title: str = "") -> Session:
        """从 source 分叉平行会话：复制到 at_message（默认全量）为止的历史与
        工作状态，可换 Provider 接力。切割点必须完整（不能拆散 tool_use 配对）。"""
        history = source.history if at_message is None else source.history[:at_message]
        if not _is_clean_cut(history):
            raise ValueError(
                f"非法分叉点 {at_message}：切割处必须是完整的 assistant 回答"
                f"（不能悬空 tool_use），请换一个位置")
        fork = self.create(
            title=title or f"{source.title}·fork",
            use_global_memory=source.use_global_memory)
        fork.history = copy.deepcopy(history)
        fork.todos = copy.deepcopy(source.todos)
        fork.plan = copy.deepcopy(source.plan)
        fork.local_memory = dict(source.local_memory)
        fork.provider = provider if provider is not None else source.provider
        fork.parent = f"{source.id}@{len(history)}"
        fork.save()
        return fork
