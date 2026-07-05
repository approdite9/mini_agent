"""任务族 5：memory_multiturn —— 跨轮记忆利用。

多轮对话任务（goal + followups 延续同一会话）：第一轮让 Agent 记住信息，
第二轮在不复述的前提下要求取回。判分看真实记忆状态里是否落了盘、
以及最后一轮的答案是否用上了它——考察"记忆不是摆设"。
"""

from __future__ import annotations

from eval.core.environment import Environment
from eval.core.task import EvalContext, ScoreResult, Task
from mini_agent import ScriptedLLM

_SECRET = "R2D2-77"


def _score_recall(ctx: EvalContext) -> ScoreResult:
    snap = ctx.service.memory_snapshot(ctx.session_id)
    all_values = list(snap.get("global", {}).values()) + list(snap.get("local", {}).values())
    return ScoreResult.from_checks([
        ("记忆中落盘了密码", 0.4, any(_SECRET in v for v in all_values)),
        ("第二轮答案含密码", 0.5, _SECRET in ctx.answer()),
        ("完成", 0.1, ctx.final_state() == "completed"),
    ])


TASKS = [
    Task(
        id="mm_cross_turn_recall",
        family="memory_multiturn",
        goal=f"请记住：我的会议室密码是 {_SECRET}。",
        followups=["我刚才让你记的会议室密码是多少？请直接告诉我。"],
        build_env=lambda: Environment(),
        scorer=_score_recall,
        weight=1.5,
        # scripted 覆盖两轮：第一轮 存→答；第二轮 直接答（记忆已在）
        scripted=[
            ScriptedLLM.call("memory", {"action": "save",
                                        "key": "会议室密码", "value": _SECRET}),
            ScriptedLLM.say(f"好的，已记住会议室密码 {_SECRET}。"),
            ScriptedLLM.call("memory", {"action": "get", "key": "会议室密码"},
                             thinking="从记忆取回"),
            ScriptedLLM.say(f"你的会议室密码是 {_SECRET}。"),
        ],
    ),
]
