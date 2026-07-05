"""ExecutionContext —— 每个 run 的隔离沙箱（Execution Context Isolation）。

一次 run 独占：
- budget          : 独立的资源上限与记账（control/budget.py）
- tool_namespace  : 本 run 可见的工具子集（None=全部）
- memory_view     : 全局记忆的隔离视图（run 起始快照 + 写回缓冲，提交时才落库）
- policy_state    : 工具隔离层的计数（熔断/限流），run 内有效，不跨 run 泄漏
- fallback_llm    : 背压降级时切换到的更便宜模型（可选）

隔离的意义：并发的多个 run 读到一致的记忆快照、各自的预算与熔断计数互不影响；
一个 run 的失控不会污染其它 run 的策略状态。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional, Set

from ..memory.store import MemoryStore
from ..control.budget import Budget

if TYPE_CHECKING:
    from ..core.llm import LLMClient
    from ..core.session import Session


class MemoryView:
    """全局记忆的 run 级隔离视图：读 = 起始快照 ∪ 本 run 缓冲（read-your-writes）；
    写进缓冲；commit() 时才真正落全局库。未提交的写对其它并发 run 不可见。"""

    def __init__(self, store: MemoryStore):
        self._store = store
        self._snapshot: Dict[str, str] = store.items()
        self._buffer: Dict[str, str] = {}

    def get(self, key: str) -> Optional[str]:
        if key in self._buffer:
            return self._buffer[key]
        return self._snapshot.get(key)

    def items(self) -> Dict[str, str]:
        merged = dict(self._snapshot)
        merged.update(self._buffer)
        return merged

    def set(self, key: str, value: str, source: str = "") -> None:
        self._buffer[key] = value

    def commit(self, source: str = "") -> int:
        """把缓冲写回真实全局库（原子、加锁）。返回提交的条目数。"""
        for key, value in self._buffer.items():
            self._store.set(key, value, source=source)
        n = len(self._buffer)
        self._buffer.clear()
        return n


@dataclass
class ExecutionContext:
    run_id: str
    session: "Session"
    budget: Budget
    global_memory: MemoryStore
    tool_namespace: Optional[Set[str]] = None
    fallback_llm: Optional["LLMClient"] = None
    policy_state: Dict[str, Any] = field(default_factory=dict)
    memory_view: MemoryView = field(init=False)

    def __post_init__(self) -> None:
        self.memory_view = MemoryView(self.global_memory)

    def commit_memory(self) -> int:
        return self.memory_view.commit(source=self.session.id)

    def tool_visible(self, name: str) -> bool:
        return self.tool_namespace is None or name in self.tool_namespace

    def snapshot(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "budget": self.budget.snapshot(),
            "tool_namespace": sorted(self.tool_namespace) if self.tool_namespace else None,
            "policy_state_keys": sorted(self.policy_state),
        }
