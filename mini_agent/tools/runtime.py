"""ToolRuntime —— 工具执行隔离层（Tool Runtime Isolation）。

在 ToolRegistry.execute 之上包裹一层控制：
- permission   : deny/confirm 直接拒绝（confirm 无人值守按 deny 处理）
- namespace    : 不在本 run 工具命名空间内的调用被拒
- circuit      : 连续失败达阈值后打开熔断，快速失败（不再真执行）
- rate-limit   : 每分钟调用上限（令牌桶/时间窗）
- timeout      : 单次执行超时（best-effort：子线程 + join 超时；进程内不能强杀，
                 超时后后台线程自然结束，这是诚实的进程内隔离，非 OS 级沙箱）
- retry        : 失败/超时后按策略重试（error_recovery 场景据此自愈）

隔离计数（熔断/限流）存放在每个 run 独占的 ExecutionContext.policy_state，不跨 run 泄漏。
额外的控制动作以语义事件上抛：tool_denied / tool_retry / tool_timeout / circuit_open。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

from ..control.policy import CONFIRM, DENY, ToolPolicy, ToolRule
from .registry import ToolContext, ToolError, ToolRegistry

if TYPE_CHECKING:
    from ..runtime.execution_context import ExecutionContext


class _Timeout(Exception):
    pass


def _run_with_timeout(fn: Callable, args: tuple, timeout: Optional[float]):
    if not timeout:
        return fn(*args)
    box: Dict[str, Any] = {}

    def worker():
        try:
            box["result"] = fn(*args)
        except BaseException as exc:  # noqa: BLE001 - 转交主线程重新抛出
            box["error"] = exc

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise _Timeout()
    if "error" in box:
        raise box["error"]
    return box["result"]


@dataclass
class ToolOutcome:
    observation: str
    ok: bool
    attempts: int = 1
    timed_out: bool = False
    denied: bool = False


class ToolRuntime:
    def __init__(self, registry: ToolRegistry, policy: Optional[ToolPolicy] = None):
        self.registry = registry
        self.policy = policy or ToolPolicy()

    # ------------------------------------------------------------------
    def run(self, name: str, args: Dict[str, Any], tool_ctx: ToolContext,
            exec_ctx: "ExecutionContext", emit: Callable[..., None],
            call_id: str = "") -> ToolOutcome:
        rule = self.policy.rule_for(name)
        st = exec_ctx.policy_state

        # 1) 工具命名空间隔离
        if not exec_ctx.tool_visible(name):
            emit("tool_denied", id=call_id, tool=name, reason="out_of_namespace")
            return ToolOutcome(f"工具 '{name}' 不在本 run 的可用工具集内", False, denied=True)

        # 2) 权限
        if rule.permission in (DENY, CONFIRM):
            reason = "denied" if rule.permission == DENY else "requires_confirmation"
            emit("tool_denied", id=call_id, tool=name, reason=reason)
            return ToolOutcome(f"工具 '{name}' 被策略拒绝（{reason}）", False, denied=True)

        # 3) 熔断
        if rule.circuit_threshold and self._circuit_open(st, name, rule):
            emit("circuit_open", id=call_id, tool=name,
                 failures=st.get(_ckey(name), 0), threshold=rule.circuit_threshold)
            return ToolOutcome(f"工具 '{name}' 已熔断（连续失败过多），快速失败", False)

        # 4) 限流
        if rule.rate_per_minute and self._rate_exceeded(st, name, rule):
            return ToolOutcome(f"工具 '{name}' 触发限流（>{rule.rate_per_minute}/min）", False)

        # 5) 执行 + 超时 + 重试
        last_error = ""
        timed_out = False
        attempts = 0
        for attempt in range(1, rule.max_retries + 2):
            attempts = attempt
            try:
                result = _run_with_timeout(
                    self.registry.execute, (name, args, tool_ctx), rule.timeout_s)
                self._record_success(st, name)
                return ToolOutcome(result, True, attempts=attempt)
            except _Timeout:
                timed_out = True
                last_error = f"工具 '{name}' 执行超时(>{rule.timeout_s}s)"
                emit("tool_timeout", id=call_id, tool=name, timeout_s=rule.timeout_s,
                     attempt=attempt)
            except ToolError as exc:
                last_error = str(exc)
            if attempt <= rule.max_retries:
                emit("tool_retry", id=call_id, tool=name, attempt=attempt,
                     next_attempt=attempt + 1, error=last_error)
        self._record_failure(st, name)
        return ToolOutcome(last_error, False, attempts=attempts, timed_out=timed_out)

    # ---- 熔断 / 限流状态（存 per-run policy_state，隔离）----
    @staticmethod
    def _circuit_open(st: Dict[str, Any], name: str, rule: ToolRule) -> bool:
        return st.get(_ckey(name), 0) >= rule.circuit_threshold

    @staticmethod
    def _record_failure(st: Dict[str, Any], name: str) -> None:
        st[_ckey(name)] = st.get(_ckey(name), 0) + 1

    @staticmethod
    def _record_success(st: Dict[str, Any], name: str) -> None:
        st[_ckey(name)] = 0  # 成功重置连续失败计数

    @staticmethod
    def _rate_exceeded(st: Dict[str, Any], name: str, rule: ToolRule) -> bool:
        now = time.time()
        key = _rkey(name)
        window = [ts for ts in st.get(key, []) if now - ts < 60.0]
        if len(window) >= rule.rate_per_minute:
            st[key] = window
            return True
        window.append(now)
        st[key] = window
        return False


def _ckey(name: str) -> str:
    return f"circuit:{name}"


def _rkey(name: str) -> str:
    return f"rate:{name}"
