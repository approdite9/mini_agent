"""评测报告：按任务族聚合通过率 + 逐任务明细（console + JSON）+ 评测集清单。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

from .task import Task, TaskResult

_G, _R, _DIM, _RESET = "\x1b[32m", "\x1b[31m", "\x1b[2m", "\x1b[0m"


def _weighted(pairs: List[tuple]) -> float:
    """pairs: [(value, weight)] → 加权平均；空时返回 0。"""
    total_w = sum(w for _, w in pairs)
    return sum(v * w for v, w in pairs) / total_w if total_w else 0.0


# 综合评分维度权重：质量（判分得分）为主，效率与稳定性为辅
_DIM_WEIGHTS = {"quality": 0.70, "efficiency": 0.15, "stability": 0.15}


def composite_scores(results: List[TaskResult]) -> Dict[str, Any]:
    """0-100 综合评分：按任务权重加权的 质量/效率/稳定性 三维合成。

    - 质量分   = Σ w·score / Σ w（判分器的 0~1 连续分）
    - 效率分   = Σ w·efficiency / Σ w（仅统计声明了 ideal_turns 的任务）
    - 稳定性分 = Σ w·stability / Σ w（重复采样的得分一致性；k=1 恒为 1）
    - 综合分   = 100 × (0.70×质量 + 0.15×效率 + 0.15×稳定性)，
                无效率数据的维度按剩余权重归一化，不惩罚未声明的任务。
    """
    if not results:
        return {"quality": None, "efficiency": None, "stability": None,
                "composite": None, "dim_weights": _DIM_WEIGHTS}
    dims: Dict[str, Any] = {
        "quality": _weighted([(r.score, r.weight) for r in results]),
        "stability": _weighted([(r.stability, r.weight) for r in results]),
    }
    eff_pairs = [(r.efficiency, r.weight) for r in results if r.efficiency is not None]
    dims["efficiency"] = _weighted(eff_pairs) if eff_pairs else None
    avail = {k: v for k, v in dims.items() if v is not None}
    wsum = sum(_DIM_WEIGHTS[k] for k in avail)
    composite = sum(v * _DIM_WEIGHTS[k] for k, v in avail.items()) / wsum if wsum else 0.0
    return {
        **{k: (round(v * 100, 1) if v is not None else None) for k, v in dims.items()},
        "composite": round(composite * 100, 1),
        "dim_weights": _DIM_WEIGHTS,
    }


def summarize(results: List[TaskResult]) -> Dict[str, Any]:
    families: Dict[str, Dict[str, Any]] = {}
    for r in results:
        f = families.setdefault(r.family, {"passed": 0, "total": 0, "_scores": []})
        f["total"] += 1
        f["passed"] += 1 if r.passed else 0
        f["_scores"].append((r.score, r.weight))
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    return {
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / total, 4) if total else None,
        "pass_any": sum(1 for r in results if r.pass_any),
        "scores": composite_scores(results),
        "by_family": {
            f: {"passed": v["passed"], "total": v["total"],
                "pass_rate": round(v["passed"] / v["total"], 4),
                "avg_score": round(_weighted(v["_scores"]), 4)}
            for f, v in families.items()
        },
        "results": [r.to_dict() for r in results],
    }


def dataset_manifest(tasks: List[Task], results: List[TaskResult],
                     root: Path, info: Dict[str, Any]) -> Dict[str, Any]:
    """评测集清单：把一次评测的输入（goal/环境信息）、判分结果与落盘产物
    （会话/事件日志/记忆终态的相对路径）绑成一份自包含的留痕数据集。

    manifest.json 与各任务工作目录同放于 root 下，事后可凭 run_events
    在「状态机·回放」里逐帧复盘任何一个任务。"""
    by_id = {t.id: t for t in tasks}
    entries = []
    for r in results:
        tdir = root / r.task_id
        artifacts: Dict[str, List[str]] = {}
        if tdir.is_dir():
            def _rel(*patterns):  # 单次布局 + 重复采样的 attempt-N/ 布局都收
                out = []
                for pat in patterns:
                    out.extend(str(p.relative_to(root)) for p in tdir.glob(pat))
                return sorted(out)
            artifacts = {
                "sessions": _rel("sessions/*.json", "attempt-*/sessions/*.json"),
                "run_events": _rel("runs/*.jsonl", "attempt-*/runs/*.jsonl"),
                "memory": _rel("global_memory.json", "attempt-*/global_memory.json"),
            }
        task = by_id.get(r.task_id)
        entries.append({
            **r.to_dict(),
            "goal": task.goal if task else "",
            "workdir": r.task_id,
            "artifacts": artifacts,
        })
    summary = {k: v for k, v in summarize(results).items() if k != "results"}
    return {
        "kind": "mini_agent_eval_dataset",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        **info,
        "summary": summary,
        "tasks": entries,
    }


def to_json(results: List[TaskResult]) -> str:
    return json.dumps(summarize(results), ensure_ascii=False, indent=2)


def print_console(results: List[TaskResult]) -> None:
    summary = summarize(results)
    print(f"\n{'=' * 64}\nmini_agent 系统级评测报告（基于真实执行环境，状态级判分）\n{'=' * 64}")
    repeated = any(len(r.attempts) > 1 for r in results)
    for r in results:
        mark = f"{_G}PASS{_RESET}" if r.passed else f"{_R}FAIL{_RESET}"
        extra = f" 分={r.score:.2f}"
        if len(r.attempts) > 1:
            extra += f" pass@{len(r.attempts)}={'✓' if r.pass_any else '✗'} 稳定={r.stability:.2f}"
        if r.efficiency is not None:
            extra += f" 效率={r.efficiency:.2f}"
        print(f"  [{mark}] {r.family:<18} {r.task_id:<24}{extra}")
        print(f"         {_DIM}终态={r.final_state} {r.detail}{_RESET}")
    print(f"{'-' * 64}")
    for family, v in summary["by_family"].items():
        rate = v["pass_rate"] * 100
        print(f"  {family:<20} {v['passed']}/{v['total']}  ({rate:.0f}%)  均分 {v['avg_score']:.2f}")
    s = summary["scores"]
    total_rate = (summary["pass_rate"] or 0) * 100
    print(f"{'-' * 64}")
    print(f"  通过: {summary['passed']}/{summary['total']} ({total_rate:.0f}%)"
          + (f"  pass@k: {summary['pass_any']}/{summary['total']}" if repeated else ""))
    eff_txt = f"{s['efficiency']}" if s["efficiency"] is not None else "—"
    print(f"  质量 {s['quality']} · 效率 {eff_txt} · 稳定性 {s['stability']}"
          f"  →  {_G}综合评分 {s['composite']}/100{_RESET}"
          f"  {_DIM}(权重 0.70/0.15/0.15){_RESET}\n")
