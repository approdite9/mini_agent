"""评测框架测试：离线自检全通过，且判分器对错误行为有判别力（不是永远 PASS）。"""

import tempfile
import unittest
from pathlib import Path

from eval.core.environment import Environment, flaky_tool
from eval.core.runner import EvalRunner
from eval.core.task import Task
from eval.tasks import all_tasks, select, tasks_by_family
from eval.tasks.budget_adherence import _score_budget
from eval.tasks.error_recovery import _score_recovery
from eval.tasks.tool_correctness import _score_calc
from mini_agent import ScriptedLLM
from mini_agent.control.budget import Budget


def _offline_runner():
    return EvalRunner(lambda task: ScriptedLLM(task.scripted))


class TestHarnessSelfCheck(unittest.TestCase):
    def test_all_tasks_pass_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = _offline_runner().run_all(all_tasks(), base_dir=Path(tmp))
            failed = [(r.task_id, r.detail) for r in results if not r.passed]
            self.assertEqual(failed, [], f"离线自检有失败: {failed}")
            # 覆盖六个族
            self.assertEqual(set(tasks_by_family()), {
                "tool_correctness", "multi_step", "memory_multiturn",
                "error_recovery", "budget_adherence", "policy_enforcement"})

    def test_select_by_family(self):
        only = select(["budget_adherence"])
        self.assertTrue(all(t.family == "budget_adherence" for t in only))
        self.assertEqual(len(only), len(tasks_by_family()["budget_adherence"]))


class TestScorersDiscriminate(unittest.TestCase):
    """判分器必须能判负 —— 否则评测无意义。"""

    def _run_task(self, task):
        with tempfile.TemporaryDirectory() as tmp:
            return _offline_runner().run_task(task, Path(tmp) / task.id)

    def test_budget_scorer_fails_when_budget_not_enforced(self):
        # 同样的"狂调工具"脚本，但预算宽松 -> 会正常 completed，而非 stopped_budget
        task = Task(
            id="ba_negative", family="budget_adherence",
            goal="查五个城市天气",
            build_env=lambda: Environment(budget=Budget(max_turns=20)),  # 无工具调用上限
            scorer=_score_budget,
            scripted=[
                ScriptedLLM.call("weather", {"city": "北京"}),
                ScriptedLLM.call("weather", {"city": "上海"}),
                ScriptedLLM.say("完成"),
            ],
        )
        r = self._run_task(task)
        self.assertFalse(r.passed)  # 未强制预算 -> 判负
        self.assertEqual(r.final_state, "completed")

    def test_recovery_scorer_fails_without_error(self):
        # flaky 首次即成功（succeed_on_attempt=1）-> 没有 tool_error -> 不算"恢复"
        good = flaky_tool("flaky_fetch", "always ok", succeed_on_attempt=1,
                          success_value="配置值=OK-42")
        task = Task(
            id="er_negative", family="error_recovery", goal="取配置",
            build_env=lambda: Environment(extra_tools=[good]),
            scorer=_score_recovery,
            scripted=[ScriptedLLM.call("flaky_fetch", {}), ScriptedLLM.say("配置值=OK-42")],
        )
        r = self._run_task(task)
        self.assertFalse(r.passed)  # 没经历失败即成功 -> 不判为"错误恢复"

    def test_tool_correctness_scorer_fails_on_wrong_result(self):
        # 不调用 calculator、直接口算错误答案 -> 判负
        task = Task(
            id="tc_negative", family="tool_correctness",
            goal="算 (1200+800)*1.1",
            build_env=lambda: Environment(),
            scorer=_score_calc,
            scripted=[ScriptedLLM.say("大概是 2000 吧")],  # 没调工具、答案错
        )
        r = self._run_task(task)
        self.assertFalse(r.passed)


class TestReport(unittest.TestCase):
    def test_summarize_by_family(self):
        from eval.core.report import summarize
        with tempfile.TemporaryDirectory() as tmp:
            results = _offline_runner().run_all(all_tasks(), base_dir=Path(tmp))
        summary = summarize(results)
        self.assertEqual(summary["passed"], summary["total"])
        self.assertIn("tool_correctness", summary["by_family"])
        self.assertEqual(summary["by_family"]["budget_adherence"]["pass_rate"], 1.0)

    def test_composite_scores_full_marks_offline(self):
        # 离线自检全对时：质量/稳定性=100，综合=100
        from eval.core.report import summarize
        with tempfile.TemporaryDirectory() as tmp:
            results = _offline_runner().run_all(all_tasks(), base_dir=Path(tmp))
        s = summarize(results)["scores"]
        self.assertEqual(s["quality"], 100.0)
        self.assertEqual(s["stability"], 100.0)
        self.assertEqual(s["composite"], 100.0)

    def test_composite_penalizes_failure(self):
        from eval.core.report import composite_scores
        from eval.core.task import TaskResult
        good = TaskResult("a", "f", True, 1.0, "", "completed", "r1", weight=1.0)
        bad = TaskResult("b", "f", False, 0.0, "", "failed", "r2", weight=1.0)
        s = composite_scores([good, bad])
        self.assertEqual(s["quality"], 50.0)
        self.assertLess(s["composite"], 100.0)


class TestPartialCredit(unittest.TestCase):
    def test_from_checks_partial(self):
        from eval.core.task import ScoreResult
        r = ScoreResult.from_checks([("a", 0.5, True), ("b", 0.3, False), ("c", 0.2, True)])
        self.assertFalse(r.passed)          # 有检查项失败 -> 不通过
        self.assertAlmostEqual(r.score, 0.7)  # 但拿到部分分
        self.assertIn("b=✗", r.detail)

    def test_from_checks_all_pass(self):
        from eval.core.task import ScoreResult
        r = ScoreResult.from_checks([("a", 1.0, True)])
        self.assertTrue(r.passed)
        self.assertEqual(r.score, 1.0)


class TestRepeatSampling(unittest.TestCase):
    def test_repeat_aggregation_stable_pass(self):
        # 同一任务重复 3 次（scripted 每次重放同一轨迹）：全过 -> 稳定性=1
        task = all_tasks()[0]
        with tempfile.TemporaryDirectory() as tmp:
            r = _offline_runner().run_task(task, Path(tmp) / task.id, repeat=3)
        self.assertTrue(r.passed)
        self.assertTrue(r.pass_any)
        self.assertEqual(r.stability, 1.0)
        self.assertEqual(len(r.attempts), 3)
        self.assertEqual(r.score, 1.0)

    def test_repeat_aggregation_flaky_task(self):
        # 制造"第 1 次过、第 2 次挂"的不稳定任务：验证 pass@k 与稳定性判别力
        from eval.core.task import ScoreResult, Task
        calls = {"n": 0}

        def scorer(ctx):
            calls["n"] += 1
            ok = calls["n"] == 1
            return ScoreResult(ok, 1.0 if ok else 0.0, f"attempt{calls['n']}")

        task = Task(
            id="flaky_meta", family="tool_correctness", goal="说句话",
            build_env=lambda: Environment(), scorer=scorer,
            scripted=[ScriptedLLM.say("好")],
        )
        with tempfile.TemporaryDirectory() as tmp:
            r = _offline_runner().run_task(task, Path(tmp) / task.id, repeat=2)
        self.assertFalse(r.passed)      # 严格口径：非全过
        self.assertTrue(r.pass_any)     # pass@2 = 通过
        self.assertEqual(r.score, 0.5)
        self.assertEqual(r.stability, 0.0)  # 0/1 各半 -> 最不稳定


class TestMultiTurn(unittest.TestCase):
    def test_followups_run_in_same_session(self):
        from eval.tasks.memory_multiturn import TASKS as MM
        task = MM[0]
        with tempfile.TemporaryDirectory() as tmp:
            r = _offline_runner().run_task(task, Path(tmp) / task.id)
        self.assertTrue(r.passed, r.detail)


if __name__ == "__main__":
    unittest.main()
