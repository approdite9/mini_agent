"""AgentService —— 前端无关的服务门面。

CLI 与 Web 服务端都只依赖这一层：会话的创建/恢复/分叉、发送消息（拿事件流）、
运行审计、模型切换、trace 与记忆查询。

两个旗舰能力在此落地：
- 运行审计：send() 用 RunCollector 旁路收集事件，每次 run 沉淀一条审计记录
  （轮次/工具耗时/usage/缓存命中率/估算成本），随会话持久化
- 跨模型接力：会话可携带 provider 覆盖（session.provider），send() 按会话
  路由到对应 LLM 实例 —— 因为历史是 Provider 中立的 canonical 格式，
  任何模型都能接着上一个模型的对话继续干活

数据目录（默认 ~/.mini_agent，可用 MINI_AGENT_HOME 覆盖）：
    sessions/<id>.json     会话（历史/计划/待办/私有记忆/usage/审计记录）
    global_memory.json     全局记忆
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .runtime.controller import Agent, AgentResult
from .runtime.replay import Replayer
from .control.budget import Budget
from .control.policy import ToolPolicy
from .context.manager import ContextManager
from .core.events import EventCallback
from .observability.inspector import RunCollector
from .observability.metrics import aggregate_metrics, compute_run_metrics
from .core.llm import LLMClient
from .core.llm_factory import available_providers, create_llm
from .memory.store import MemoryStore
from .core.session import Session, SessionManager
from .tools.registry import ToolRegistry
from .tools.builtin import build_default_registry
from .observability.tracer import Tracer

_DEFAULT = "__default__"
_MAX_RUN_RECORDS = 50  # 每会话保留的审计记录条数


def _session_summary(session: Session, limit: int = 40) -> str:
    """一行任务摘要：取最近一条纯文本用户消息（无 LLM 调用成本）。"""
    for msg in reversed(session.history):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            text = content.strip()
        else:
            texts = [b.get("text", "") for b in content
                     if isinstance(b, dict) and b.get("type") == "text"]
            if any(b.get("type") == "tool_result" for b in content if isinstance(b, dict)):
                continue  # 工具结果回写不是用户任务
            text = " ".join(texts).strip()
        if text:
            return text if len(text) <= limit else text[: limit - 1] + "…"
    return ""


class SessionBusyError(Exception):
    """同一会话上已有一个 run 在进行中。"""


def default_home() -> Path:
    return Path(os.environ.get("MINI_AGENT_HOME", Path.home() / ".mini_agent"))


class AgentService:
    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        base_dir: Optional[Path] = None,
        registry: Optional[ToolRegistry] = None,
        context_manager: Optional[ContextManager] = None,
        max_tool_turns: int = 16,
        llm_factory: Optional[Callable[..., LLMClient]] = None,
        budget: Optional[Budget] = None,
        tool_policy: Optional[ToolPolicy] = None,
    ):
        self.base_dir = Path(base_dir) if base_dir else default_home()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.tracer = Tracer()
        self.global_memory = MemoryStore(self.base_dir / "global_memory.json")
        self.sessions = SessionManager(self.base_dir)
        self.registry = registry or build_default_registry()
        self.context = context_manager or ContextManager()
        self.max_tool_turns = max_tool_turns
        self.budget = budget            # 预算模板（None=默认，仅 max_turns 限制）
        self.tool_policy = tool_policy   # 工具策略（None=全 allow，行为同重构前）

        self._llm_factory = llm_factory or create_llm
        self._llms: Dict[str, LLMClient] = {
            _DEFAULT: llm if llm is not None else self._llm_factory()}
        self._agents: Dict[str, Agent] = {}

    # ---- LLM / Agent 路由 ----------------------------------------------
    def _llm_for(self, provider: Optional[str]) -> LLMClient:
        key = provider or _DEFAULT
        if key not in self._llms:
            self._llms[key] = self._llm_factory(provider=key)
        return self._llms[key]

    def _agent_for(self, session: Session) -> Agent:
        key = session.provider or _DEFAULT
        if key not in self._agents:
            self._agents[key] = Agent(
                llm=self._llm_for(session.provider),
                registry=self.registry,
                global_memory=self.global_memory,
                tracer=self.tracer,
                context_manager=self.context,
                max_tool_turns=self.max_tool_turns,
                runs_dir=self.base_dir / "runs",  # 每 run 的事件日志落盘（可复现）
                budget=self.budget,
                tool_policy=self.tool_policy,
            )
        return self._agents[key]

    def providers(self) -> Dict[str, Any]:
        return {
            "default": self._llms[_DEFAULT].describe(),
            "available": available_providers(),
        }

    def set_provider(self, session_id: str, provider: Optional[str]) -> str:
        """切换会话使用的模型 Provider（跨模型接力）。返回生效的模型标识。
        provider 为 None/'default' 时回到默认模型。校验失败抛 LLMError。"""
        session = self.get_session(session_id)
        if provider in (None, "", "default"):
            session.provider = None
        else:
            self._llm_for(provider)  # 先实例化以校验配置（缺 key 等在此报错）
            session.provider = provider
        session.save()
        return self._llm_for(session.provider).describe()

    # ---- 会话 ------------------------------------------------------------
    def create_session(self, title: str = "", shared_memory: bool = True,
                       provider: Optional[str] = None) -> Session:
        session = self.sessions.create(title=title, use_global_memory=shared_memory)
        if provider:
            self._llm_for(provider)
            session.provider = provider
            session.save()
        return session

    def fork_session(self, session_id: str, at_message: Optional[int] = None,
                     provider: Optional[str] = None, title: str = "") -> Session:
        """分叉平行会话，可换 Provider 接力（历史是 Provider 中立格式，直接可用）。"""
        source = self.get_session(session_id)
        if provider:
            self._llm_for(provider)
        return self.sessions.fork(source, at_message=at_message,
                                  provider=provider, title=title)

    def delete_session(self, session_id: str) -> None:
        """删除会话。运行中的会话拒绝删除（避免撕裂正在写入的状态）。"""
        session = self.get_session(session_id)
        if session.lock.locked():
            raise SessionBusyError(f"会话 {session_id} 正在运行，无法删除")
        self.sessions.delete(session_id)

    def rename_session(self, session_id: str, title: str) -> Session:
        title = title.strip()
        if not title:
            raise ValueError("标题不能为空")
        session = self.get_session(session_id)
        session.title = title[:60]
        session.save()
        return session

    def get_session(self, session_id: str) -> Session:
        session = self.sessions.get(session_id)
        if session is None:
            raise KeyError(f"会话不存在: {session_id}")
        return session

    def list_sessions(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": s.id,
                "title": s.title,
                "shared_memory": s.use_global_memory,
                "messages": len(s.history),
                "requests": s.request_count,
                "created_at": s.created_at,
                "context_tokens": s.context_tokens(),
                "plan": s.plan,
                "provider": s.provider,
                "parent": s.parent,
                "running": s.lock.locked(),
                "model": self._llm_for(s.provider).describe() if (
                    (s.provider or _DEFAULT) in self._llms or s.provider is None
                ) else s.provider,
                "summary": _session_summary(s),
            }
            for s in self.sessions.list()
        ]

    # ---- 核心：发送消息（事件流出 + 审计沉淀）---------------------------
    def send(self, session_id: str, text: str,
             on_event: Optional[EventCallback] = None,
             attachments: Optional[List[Dict[str, Any]]] = None) -> AgentResult:
        """attachments 为框架 canonical 媒体块（见 llm/base.py），多模态接口。"""
        session = self.get_session(session_id)
        agent = self._agent_for(session)
        if not session.lock.acquire(blocking=False):
            raise SessionBusyError(f"会话 {session_id} 正在处理上一条消息")
        try:
            # 首条消息自动生成会话标题（无 LLM 成本）
            if session.request_count == 0 and session.title.startswith("会话"):
                session.title = text.strip()[:20] or session.title
            collector = RunCollector(on_event)
            result = agent.run(session, text, on_event=collector,
                               attachments=attachments)
            record = collector.build(result, user_input=text,
                                     model=agent.llm.describe(),
                                     pricing=agent.llm.pricing)
            session.runs.append(record)
            session.runs = session.runs[-_MAX_RUN_RECORDS:]
            session.save()
            return result
        finally:
            session.lock.release()

    # ---- 可观测性 ---------------------------------------------------------
    def runs(self, session_id: str) -> List[Dict[str, Any]]:
        return list(self.get_session(session_id).runs)

    def trace(self, session_id: str) -> str:
        return self.tracer.dump(session_id)

    def timeline(self, session_id: str) -> List[Dict[str, Any]]:
        """结构化时间线：本进程内该会话的全部 trace 事件（含真实时间戳）。"""
        self.get_session(session_id)  # 校验存在性
        return [e.to_dict() for e in self.tracer.for_session(session_id)]

    # ---- 持久化事件日志 / 回放 / 指标（进程重启后仍可读）----
    @property
    def runs_dir(self) -> Path:
        return self.base_dir / "runs"

    def list_runs(self, session_id: str) -> List[str]:
        """该会话已落盘的 run_id 列表（来自持久化事件日志，重启后仍在）。"""
        self.get_session(session_id)
        d = self.runs_dir
        if not d.is_dir():
            return []
        return sorted(p.stem for p in d.glob(f"{session_id}-r*.jsonl"))

    def run_events(self, run_id: str) -> List[Dict[str, Any]]:
        """某个 run 的完整事件日志（供前端 replay / 逐帧重建）。"""
        return Replayer.from_run(run_id, self.runs_dir).events

    def run_transitions(self, run_id: str) -> List[Dict[str, Any]]:
        return Replayer.from_run(run_id, self.runs_dir).transitions()

    def metrics(self, session_id: str) -> Dict[str, Any]:
        """会话级系统指标：逐 run 指标 + 聚合（从持久化日志复算）。"""
        per_run = [compute_run_metrics(self.run_events(rid))
                   for rid in self.list_runs(session_id)]
        return {"runs": per_run, "aggregate": aggregate_metrics(per_run)}

    def memory_snapshot(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        snapshot: Dict[str, Any] = {"global": self.global_memory.items()}
        if session_id:
            session = self.get_session(session_id)
            snapshot["local"] = dict(session.local_memory)
        return snapshot
