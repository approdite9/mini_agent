"""任务族 2：error_recovery —— 工具报错后 Agent 能否自愈并最终成功。

环境注入一个"前若干次失败、之后成功"的 flaky 工具。判分检查事件日志中出现过
tool_error，且随后同一工具产生了 tool_result（恢复），且任务成功完成。
考察的是"看到失败观察后重试"的 Agent 级恢复能力（真实 API 下模型自发重试）。
"""

from __future__ import annotations

from eval.core.environment import Environment, flaky_tool
from eval.core.task import EvalContext, ScoreResult, Task
from mini_agent import ScriptedLLM

def _fresh_flaky():
    """每次构建环境都造一个全新的 flaky 工具，其失败计数不跨 run 泄漏。"""
    return flaky_tool(
        name="flaky_fetch",
        description="获取远端配置值。可能偶发失败并提示重试。",
        succeed_on_attempt=2,
        success_value="配置值=OK-42",
    )


def _score_recovery(ctx: EvalContext) -> ScoreResult:
    return ScoreResult.from_checks([
        ("出现失败", 0.3, ctx.counter("tool_error") >= 1),
        ("恢复成功", 0.4, any("OK-42" in r for r in ctx.tool_results("flaky_fetch"))),
        ("完成", 0.3, ctx.final_state() == "completed"),
    ])


TASKS = [
    Task(
        id="er_flaky_retry",
        family="error_recovery",
        goal="请用 flaky_fetch 工具获取远端配置值并告诉我；如果失败就重试。",
        build_env=lambda: Environment(extra_tools=[_fresh_flaky()]),
        scorer=_score_recovery,
        weight=1.5,       # 自愈能力比单步调用更难，加权
        ideal_turns=3,    # 失败→重试→汇报
        scripted=[
            # 第一次调用失败（flaky 工具前一次必失败）
            ScriptedLLM.call("flaky_fetch", {}, thinking="先取一次配置"),
            # 看到 [tool_error] 观察后重试
            ScriptedLLM.call("flaky_fetch", {}, thinking="上次失败了，重试一次"),
            ScriptedLLM.say("获取成功：配置值=OK-42。"),
        ],
    ),
]
