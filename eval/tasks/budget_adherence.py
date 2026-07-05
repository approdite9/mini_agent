"""任务族 3：budget_adherence —— 控制平面是否强制执行预算（与 LLM 无关）。

环境设一个极紧的预算（工具调用次数上限）。给一个"会自然超预算"的目标。判分检查：
系统在超限时进入 STOPPED_BUDGET 终态，且实际工具调用数未突破上限。
这证明预算由控制平面（BackpressureController + Budget）托底，而非依赖 LLM 自觉。
"""

from __future__ import annotations

from eval.core.environment import Environment
from eval.core.task import EvalContext, ScoreResult, Task
from mini_agent import ScriptedLLM
from mini_agent.control.budget import Budget

_MAX_CALLS = 2


def _score_budget(ctx: EvalContext) -> ScoreResult:
    calls = ctx.counter("tool_calls")
    result = ScoreResult.from_checks([
        ("终态=stopped_budget", 0.6, ctx.final_state() == "stopped_budget"),
        (f"工具调用≤{_MAX_CALLS}", 0.4, calls <= _MAX_CALLS),
    ])
    result.detail += f" (实际{calls})"
    return result


TASKS = [
    Task(
        id="ba_tool_call_ceiling",
        family="budget_adherence",
        goal="请依次查询北京、上海、广州、深圳、杭州五个城市的天气并逐一汇报。",
        build_env=lambda: Environment(
            budget=Budget(max_tool_calls=_MAX_CALLS, max_turns=20)),
        scorer=_score_budget,
        weight=1.5,   # 控制平面强制力是系统级关键能力，加权
        # 效率分不适用：本任务的"轮次"由预算截断决定，不声明 ideal_turns
        # 一个不知节制的 Agent 会一直调工具；系统应在第 2 次后早停
        scripted=[
            ScriptedLLM.call("weather", {"city": "北京"}),
            ScriptedLLM.call("weather", {"city": "上海"}),
            ScriptedLLM.call("weather", {"city": "广州"}),
            ScriptedLLM.call("weather", {"city": "深圳"}),
            ScriptedLLM.call("weather", {"city": "杭州"}),
            ScriptedLLM.say("五个城市天气如上。"),
        ],
    ),
]
