"""任务族 4：multi_step —— 多步工具链的编排能力。

单个目标需要串联多个不同工具（查询→计算→落盘/计划管理），判分逐环节给部分分：
既看每一步的真实状态产物，也看最终整体完成。这类任务区分"会调一个工具"和
"能把一串工具组织成正确流程"。
"""

from __future__ import annotations

from eval.core.environment import Environment
from eval.core.task import EvalContext, ScoreResult, Task
from mini_agent import ScriptedLLM


def _score_chain(ctx: EvalContext) -> ScoreResult:
    saved = ctx.memory_get("演算结果") or ""
    return ScoreResult.from_checks([
        ("weather命中北京", 0.25, any("北京" in r for r in ctx.tool_results("weather"))),
        ("calculator→10120", 0.35, any(r == "10120" for r in ctx.tool_results("calculator"))),
        ("记忆[演算结果]含10120", 0.3, "10120" in saved),
        ("完成", 0.1, ctx.final_state() == "completed"),
    ])


def _score_plan_exec(ctx: EvalContext) -> ScoreResult:
    plan = ctx.session_state().plan
    saved = ctx.memory_get("巡检状态") or ""
    return ScoreResult.from_checks([
        ("计划已创建(≥2步)", 0.2, len(plan) >= 2),
        ("计划全部done", 0.2, bool(plan) and all(s["status"] == "done" for s in plan)),
        ("weather命中上海", 0.2, any("上海" in r for r in ctx.tool_results("weather"))),
        ("记忆[巡检状态]已写", 0.3, "上海已查" in saved),
        ("完成", 0.1, ctx.final_state() == "completed"),
    ])


TASKS = [
    Task(
        id="ms_weather_calc_memory",
        family="multi_step",
        goal=("请依次完成三步：1) 查询北京天气；2) 用计算器算出 88*115 的结果；"
              "3) 把第 2 步的计算结果（纯数字）用 memory 工具保存，key 为 演算结果。"
              "全部完成后汇报每一步的结果。"),
        build_env=lambda: Environment(),
        scorer=_score_chain,
        weight=1.5,
        ideal_turns=4,   # 三次工具 + 一次汇报
        scripted=[
            ScriptedLLM.call("weather", {"city": "北京"}, thinking="第一步查天气"),
            ScriptedLLM.call("calculator", {"expression": "88*115"}, thinking="第二步计算"),
            ScriptedLLM.call("memory", {"action": "save", "key": "演算结果", "value": "10120"}),
            ScriptedLLM.say("三步完成：北京晴；88*115=10120；已存入记忆[演算结果]。"),
        ],
    ),
    Task(
        id="ms_plan_and_execute",
        family="multi_step",
        goal=("请用 plan 工具制定一个两步计划：① 查询上海天气，② 把'上海已查'"
              "保存到记忆（key 为 巡检状态）。然后逐步执行，每完成一步就把该步"
              "标记为 done，最后汇报。"),
        build_env=lambda: Environment(),
        scorer=_score_plan_exec,
        weight=2.0,      # 计划-执行-回写闭环，最复杂任务
        ideal_turns=6,   # set + 2×(执行+update) + 汇报
        scripted=[
            ScriptedLLM.call("plan", {"action": "set",
                                      "steps": ["查询上海天气", "保存巡检状态到记忆"]}),
            ScriptedLLM.call("weather", {"city": "上海"}),
            ScriptedLLM.call("plan", {"action": "update", "step": 1, "status": "done"}),
            ScriptedLLM.call("memory", {"action": "save", "key": "巡检状态", "value": "上海已查"}),
            ScriptedLLM.call("plan", {"action": "update", "step": 2, "status": "done"}),
            ScriptedLLM.say("计划执行完毕：上海多云；巡检状态已入记忆。"),
        ],
    ),
]
