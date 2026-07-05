"""评测任务族的收集入口。"""

from __future__ import annotations

from typing import Dict, List

from ..core.task import Task
from . import (budget_adherence, error_recovery, memory_multiturn, multi_step,
               policy_enforcement, tool_correctness)

_MODULES = [tool_correctness, multi_step, memory_multiturn,
            error_recovery, budget_adherence, policy_enforcement]


def all_tasks() -> List[Task]:
    tasks: List[Task] = []
    for mod in _MODULES:
        tasks.extend(mod.TASKS)
    return tasks


def tasks_by_family() -> Dict[str, List[Task]]:
    out: Dict[str, List[Task]] = {}
    for task in all_tasks():
        out.setdefault(task.family, []).append(task)
    return out


def select(families: List[str] = None) -> List[Task]:
    if not families:
        return all_tasks()
    fset = set(families)
    return [t for t in all_tasks() if t.family in fset]
