"""评测核心抽象（AgentBench 风格，但面向本 Runtime 的执行环境）。

区别于离线 QA benchmark：
- Task 不是"问题→标准答案"，而是"目标 + 环境 + 状态级判分器"。
- 判分基于**执行后的真实状态**（事件日志、会话/记忆状态、终态、ExecutionContext），
  而不是对文本答案做字符串匹配。
- 运行在**真实 API 执行环境**：Runner 用真实 LLM 驱动完整 Controller；离线（无 key）
  时用每个 Task 自带的 scripted 轨迹校验判分器与环境本身是否正确（结构自检）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from .environment import Environment


@dataclass
class ScoreResult:
    passed: bool
    score: float
    detail: str = ""

    @classmethod
    def from_checks(cls, checks: List[tuple],
                    pass_threshold: float = 0.999) -> "ScoreResult":
        """部分得分判分：checks 为 [(名称, 权重, 是否通过)]。
        score = 加权通过比例（0~1 连续分），passed = score ≥ pass_threshold
        （默认要求全部检查项通过，与旧的二值判分兼容）。"""
        total = sum(w for _, w, _ in checks)
        got = sum(w for _, w, ok in checks if ok)
        score = round(got / total, 4) if total else 0.0
        detail = ", ".join(f"{name}={'✓' if ok else '✗'}" for name, _, ok in checks)
        return cls(score >= pass_threshold, score, detail)


@dataclass
class EvalContext:
    """判分器可访问的执行后状态：全部来自真实运行的持久化产物。"""

    service: Any            # AgentService
    session_id: str
    result: Any             # AgentResult
    environment: "Environment"

    # ---- 便捷读取器（状态级，非文本匹配）----
    def events(self) -> List[Dict[str, Any]]:
        return self.service.run_events(self.result.run_id)

    def metrics(self) -> Dict[str, Any]:
        from mini_agent.observability.metrics import compute_run_metrics
        return compute_run_metrics(self.events())

    def counter(self, key: str) -> int:
        return self.metrics()["counters"].get(key, 0)

    def final_state(self) -> str:
        return self.result.final_state

    def answer(self) -> str:
        return self.result.answer or ""

    def tool_results(self, name: str) -> List[str]:
        return [e["data"].get("result", "") for e in self.events()
                if e.get("type") == "tool_result" and e["data"].get("tool") == name]

    def tool_errors(self, name: Optional[str] = None) -> List[str]:
        return [e["data"].get("error", "") for e in self.events()
                if e.get("type") == "tool_error"
                and (name is None or e["data"].get("tool") == name)]

    def session_state(self):
        return self.service.get_session(self.session_id)

    def memory_get(self, key: str) -> Optional[str]:
        session = self.session_state()
        if session.use_global_memory:
            return self.service.global_memory.get(key)
        return session.local_memory.get(key)


Scorer = Callable[[EvalContext], ScoreResult]


@dataclass
class Task:
    id: str
    family: str
    goal: str                                  # 交给 Agent 的用户目标
    build_env: Callable[[], "Environment"]      # 构造隔离执行环境
    scorer: Scorer                             # 状态级判分器
    scripted: List[Dict[str, Any]] = field(default_factory=list)  # 离线结构自检用的模型轨迹
    weight: float = 1.0                        # 综合评分中的任务权重（难任务加权）
    ideal_turns: Optional[int] = None          # 理想轮次（效率分基准；None=不计效率）
    followups: List[str] = field(default_factory=list)  # 多轮任务：goal 之后追加的用户消息


@dataclass
class TaskResult:
    task_id: str
    family: str
    passed: bool                               # 严格通过：k 次尝试全部通过
    score: float                               # 0~1 连续分（k 次尝试的均值）
    detail: str
    final_state: str
    run_id: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    weight: float = 1.0
    efficiency: Optional[float] = None         # 0~1：ideal_turns/实际轮次（None=不适用）
    stability: float = 1.0                     # 0~1：k 次尝试得分的一致性（k=1 恒为 1）
    pass_any: bool = True                      # pass@k：任意一次通过
    attempts: List[Dict[str, Any]] = field(default_factory=list)  # 逐次尝试明细

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id, "family": self.family, "passed": self.passed,
            "score": self.score, "detail": self.detail, "final_state": self.final_state,
            "run_id": self.run_id, "metrics": self.metrics, "error": self.error,
            "weight": self.weight, "efficiency": self.efficiency,
            "stability": self.stability, "pass_any": self.pass_any,
            "attempts": self.attempts,
        }
