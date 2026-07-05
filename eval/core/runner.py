"""EvalRunner —— 用真实 LLM 在每个任务的隔离环境上跑完整 Controller 并判分。

每个任务独占一个临时工作目录（会话/事件日志/记忆隔离）。llm_factory 决定用什么模型：
- 真实模式：create_llm()（真实 API，MINI_AGENT_LIVE + key 时）
- 离线自检：ScriptedLLM(task.scripted)（验证判分器与环境正确性，不产生费用）

重复采样（repeat>1）：每个任务独立重跑 k 次（环境与 LLM 每次全新构建），
聚合出 均值分 / pass@k / 稳定性，区分"侥幸通过"与"稳定通过"。
"""

from __future__ import annotations

import statistics
import tempfile
from pathlib import Path
from typing import Callable, List, Optional

from .task import EvalContext, ScoreResult, Task, TaskResult


class EvalRunner:
    def __init__(self, llm_factory: Callable[[Task], object]):
        self.llm_factory = llm_factory

    def run_once(self, task: Task, workdir: Path) -> TaskResult:
        """单次尝试：构建全新环境与 LLM，发送 goal（及多轮 followups）后状态级判分。"""
        env = task.build_env()
        llm = self.llm_factory(task)
        service, sid = env.build(workdir, llm)
        try:
            result = service.send(sid, task.goal)
            for followup in task.followups:   # 多轮任务：延续同一会话
                result = service.send(sid, followup)
        except Exception as exc:  # 环境/运行异常也是一种失败结果，不中断整批
            return TaskResult(task.id, task.family, False, 0.0,
                              f"run 异常: {type(exc).__name__}: {exc}",
                              final_state="error", run_id="", error=str(exc),
                              weight=task.weight, pass_any=False)

        ctx = EvalContext(service=service, session_id=sid, result=result, environment=env)
        try:
            score: ScoreResult = task.scorer(ctx)
        except Exception as exc:
            score = ScoreResult(False, 0.0, f"判分器异常: {type(exc).__name__}: {exc}")

        try:
            metrics = ctx.metrics()
        except Exception:
            metrics = {}
        efficiency = None
        turns = (metrics.get("counters") or {}).get("turns", 0)
        if task.ideal_turns and turns > 0:
            efficiency = round(min(1.0, task.ideal_turns / turns), 4)
        return TaskResult(
            task_id=task.id, family=task.family, passed=score.passed, score=score.score,
            detail=score.detail, final_state=result.final_state, run_id=result.run_id,
            metrics=metrics, weight=task.weight, efficiency=efficiency,
            pass_any=score.passed)

    def run_task(self, task: Task, workdir: Path, repeat: int = 1) -> TaskResult:
        """跑 k 次并聚合。repeat=1 时目录布局与单次完全相同（向后兼容）。"""
        repeat = max(1, int(repeat))
        attempts: List[TaskResult] = []
        for i in range(1, repeat + 1):
            attempt_dir = workdir if repeat == 1 else workdir / f"attempt-{i}"
            attempts.append(self.run_once(task, attempt_dir))
        if repeat == 1:
            return attempts[0]

        scores = [a.score for a in attempts]
        passes = [a.passed for a in attempts]
        effs = [a.efficiency for a in attempts if a.efficiency is not None]
        # 稳定性：得分离散度越大越不稳（二值 0/1 各半时 pstdev=0.5 → 0 分）
        stability = round(max(0.0, 1.0 - 2.0 * statistics.pstdev(scores)), 4)
        last = attempts[-1]
        return TaskResult(
            task_id=task.id, family=task.family,
            passed=all(passes),
            score=round(sum(scores) / len(scores), 4),
            detail=f"{sum(passes)}/{repeat} 次通过 · 末次: {last.detail}",
            final_state=last.final_state, run_id=last.run_id,
            metrics=last.metrics, error=last.error, weight=task.weight,
            efficiency=round(sum(effs) / len(effs), 4) if effs else None,
            stability=stability,
            pass_any=any(passes),
            attempts=[{
                "attempt": i + 1, "passed": a.passed, "score": a.score,
                "final_state": a.final_state, "run_id": a.run_id,
                "detail": a.detail, "efficiency": a.efficiency, "error": a.error,
            } for i, a in enumerate(attempts)])

    def run_all(self, tasks: List[Task], base_dir: Optional[Path] = None,
                repeat: int = 1) -> List[TaskResult]:
        results: List[TaskResult] = []
        root = Path(base_dir) if base_dir else Path(tempfile.mkdtemp(prefix="mini_agent_eval_"))
        for task in tasks:
            workdir = root / task.id
            results.append(self.run_task(task, workdir, repeat=repeat))
        return results
