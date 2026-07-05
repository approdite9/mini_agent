# mini_agent 系统级评测（AgentBench 风格）

不是离线 QA benchmark，而是**基于真实 API 执行环境**、对 Agent Runtime 做**状态级**评测：
每个任务给一个目标 + 一个隔离环境（可注入故障/预算/策略），用真实 LLM 驱动**完整
Controller** 跑一遍，再检查**执行后的真实状态**（事件日志、会话/记忆状态、终态、指标）判分
——而非对文本答案做字符串匹配。

## 运行

```bash
# 真实 API 模式（需 key）：真实模型驱动完整运行时
export MINI_AGENT_LIVE=1 ANTHROPIC_API_KEY=sk-...      # 或 LLM_PROVIDER=qwen DASHSCOPE_API_KEY=...
python -m eval.runner --config eval/configs/minimal.json --json report.json

# 离线自检（无 key / CI）：用每个任务自带的 scripted 轨迹校验判分器与环境正确性
python -m eval.runner --offline

# 留痕：--workdir 把评测生成的数据构成评测集保存（不给则跑完即清理）
python -m eval.runner --config eval/configs/minimal.json --json report.json --workdir eval_runs

# 重复采样：每任务独立重跑 k 次，报告 pass@k 与稳定性（费用×k）
python -m eval.runner --repeat 3 --workdir eval_runs
```

## 评测集留痕（--workdir）

每次评测在 `DIR/<时间戳>/` 下保留一份自包含评测集：

```
eval_runs/20260705-153000/
  manifest.json            # 清单：mode/provider/goal/判分结果/产物相对路径索引
  report.json              # 汇总报告（同 --json 内容）
  <task_id>/               # 每任务隔离工作目录
    sessions/s1.json       # 会话完整历史（含工具调用与结果）
    runs/s1-r1.jsonl       # 逐事件日志（可在「状态机·回放」逐帧复盘）
    global_memory.json     # 记忆终态
```

## 六个任务族（状态级判分，部分得分制）

| 族 | 考察什么 | 判分依据（真实状态） |
|---|---|---|
| **tool_correctness** | 是否调对工具/参数并使最终状态正确 | 事件日志中工具结果正确（如 calculator→2200）、记忆写对、终态 completed |
| **multi_step** | 多步工具链编排（查询→计算→落盘、计划-执行-回写） | 每一环节的真实产物逐项给分：中间工具结果、plan 状态全 done、记忆终态 |
| **memory_multiturn** | 跨轮记忆利用（多轮对话延续同一会话） | 第一轮落盘、第二轮不复述也能取回：记忆状态 + 末轮答案 |
| **error_recovery** | 工具报错后能否自愈 | 环境注入前一次必失败的 flaky 工具；日志需出现 tool_error 且随后同工具 tool_result 恢复、任务完成 |
| **budget_adherence** | 控制平面是否强制预算（与 LLM 无关） | 设极紧预算（工具调用上限），验证系统在超限时进入 STOPPED_BUDGET 终态且实际调用未突破上限 |
| **policy_enforcement** | 工具权限拦截下的正确行为 | weather 被策略 deny：日志出现 tool_denied、无结果泄漏、Agent 体面完成 |

## 综合评分（0-100）

判分器返回 0~1 连续分（`ScoreResult.from_checks` 加权检查项，部分正确拿部分分），
按任务权重（难任务 weight 1.5~2.0）聚合成三个维度再合成：

```
质量分   = Σ w·score / Σ w                （判分得分，权重 0.70）
效率分   = Σ w·(ideal_turns/实际轮次) / Σ w（声明了 ideal_turns 的任务，权重 0.15）
稳定性分 = Σ w·stability / Σ w             （--repeat k 次得分一致性，权重 0.15）
综合分   = 100 × 加权合成（缺失维度按剩余权重归一化）
```

`--repeat k` 时逐任务另报 `pass@k`（任一次通过）与严格通过（k 次全过）；
`passed` 采用严格口径，避免把侥幸通过计入回归基线。

## 结构

```
eval/
  core/
    task.py         # Task / TaskResult / ScoreResult / EvalContext(状态级读取器)
    environment.py   # Environment(工具/故障/预算/策略/初始记忆) + flaky_tool 注入器
    runner.py        # EvalRunner：在隔离工作目录用真实 LLM 跑完整 Controller 并判分
    report.py        # 按族聚合通过率 + 逐任务明细（console + JSON）
  tasks/
    tool_correctness.py / multi_step.py / memory_multiturn.py
    error_recovery.py / budget_adherence.py / policy_enforcement.py
  configs/minimal.json
  runner.py          # CLI 入口（python -m eval.runner）
```

> 配置用 JSON 而非 YAML，避免引入 PyYAML 依赖（框架只允许 anthropic / openai 两个官方 SDK）。
