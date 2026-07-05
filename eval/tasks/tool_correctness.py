"""任务族 1：tool_correctness —— Agent 是否调对工具/参数并使最终状态正确。

判分基于执行后的真实状态（工具结果、记忆状态、终态），而非文本答案匹配。
"""

from __future__ import annotations

from eval.core.environment import Environment
from eval.core.task import EvalContext, ScoreResult, Task
from mini_agent import ScriptedLLM


def _score_calc(ctx: EvalContext) -> ScoreResult:
    return ScoreResult.from_checks([
        ("calculator→2200", 0.5, any(r == "2200" for r in ctx.tool_results("calculator"))),
        ("答案含2200", 0.3, "2200" in ctx.answer()),
        ("完成", 0.2, ctx.final_state() == "completed"),
    ])


def _score_memory(ctx: EvalContext) -> ScoreResult:
    return ScoreResult.from_checks([
        ("记忆[项目代号]=蓝鲸七号", 0.8, ctx.memory_get("项目代号") == "蓝鲸七号"),
        ("完成", 0.2, ctx.final_state() == "completed"),
    ])


def _score_weather(ctx: EvalContext) -> ScoreResult:
    return ScoreResult.from_checks([
        ("weather命中北京", 0.5, any("北京" in r for r in ctx.tool_results("weather"))),
        ("答案含北京", 0.3, "北京" in ctx.answer()),
        ("完成", 0.2, ctx.final_state() == "completed"),
    ])


TASKS = [
    Task(
        id="tc_calculator",
        family="tool_correctness",
        goal="请用计算器算出 (1200+800)*1.1 的结果，并把最终数值告诉我。",
        build_env=lambda: Environment(),
        scorer=_score_calc,
        ideal_turns=2,
        scripted=[
            ScriptedLLM.call("calculator", {"expression": "(1200+800)*1.1"}, thinking="算一下"),
            ScriptedLLM.say("(1200+800)*1.1 = 2200"),
        ],
    ),
    Task(
        id="tc_memory_write",
        family="tool_correctness",
        goal="请记住：我的项目代号是蓝鲸七号。",
        build_env=lambda: Environment(),
        scorer=_score_memory,
        ideal_turns=2,
        scripted=[
            ScriptedLLM.call("memory", {"action": "save", "key": "项目代号", "value": "蓝鲸七号"}),
            ScriptedLLM.say("好的，已记住你的项目代号是蓝鲸七号。"),
        ],
    ),
    Task(
        id="tc_weather_lookup",
        family="tool_correctness",
        goal="北京今天天气怎么样？",
        build_env=lambda: Environment(),
        scorer=_score_weather,
        ideal_turns=2,
        scripted=[
            ScriptedLLM.call("weather", {"city": "北京"}, thinking="查天气"),
            ScriptedLLM.say("北京今天晴，26°C，适合出行。"),
        ],
    ),
]
