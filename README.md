# mini_agent — 无框架 Agent Runtime（AgentOS）

不依赖 LangGraph / AutoGen / CrewAI 等任何 Agent 框架，内核纯 Python 标准库（仅
官方 `anthropic` / `openai` 两个 SDK 用于真实调用）。这不只是一个 Agent，而是一个
**完全受控的 Agent Runtime**：

- **LLM 只作为 Actor**（只输出 reasoning / 工具选择 / 最终答案），调度 / retry /
  recovery / 流程控制全部由系统的 **控制平面（Control Plane）** 承担，被测试证明为
  Actor-only。
- **执行语义显式化**：一次 run 是一个显式**状态机**（`RunState`），事件不只是日志，
  而承载 `from_state`/`to_state`/`reason`（为什么发生每次迁移）+ 全程落 append-only
  事件日志，任意 run **可从日志完全复现**（record & replay）。
- **可插拔推理引擎**：Provider 层支持 Claude 与 Qwen，切换只改环境变量，内核零改动。
- 一套内核，多个前端：终端 CLI + Execution Inspector Web 客户端（状态机可视化/工具
  执行图/回放）；以及 **AgentBench 风格的系统级评测**（6 任务族 · 0-100 综合评分 ·
  重复采样 · 评测集留痕，基于真实 API 执行环境，详见 [eval/README.md](eval/README.md)）。

## 架构（仿 HelloAgents 分包）

```
mini_agent/
  core/          llm 抽象 + adapters(Claude/Qwen/Scripted) + factory、events、session
  runtime/       ★ 执行语义层：state(状态机) / controller / stream_adapter /
                    execution_context(per-run 隔离) / replay(EventLog + Replayer + ReplayLLM)
  control/       ★ 控制平面：backpressure / budget / policy（系统拥有，LLM 不参与）
  tools/         registry + builtin + runtime(隔离层: timeout/retry/permission/ratelimit/circuit)
  context/ memory/ observability(tracer/metrics/inspector)
  service · cli · server · web/   门面与 Execution Inspector 前端
eval/            ★ AgentBench 风格系统级评测（6 族 9 任务，真实 API 门控，状态级判分，
                    综合评分 + pass@k + 评测集留痕）
```

## 快速开始

```bash
# 离线体验（无需 API key）
python demo.py        # 终端演示：脚本化 Mock LLM 走真实内核
python ui_demo.py     # Execution Inspector Web 演示（http://127.0.0.1:8899）
                      # 发消息后切到"状态机·回放"标签：状态机时间线 / 工具执行图 / 逐帧回放

# 全部测试（188 个离线用例，无需网络；另有 5 个 Live API 用例按需门控）
python -m unittest discover -s tests

# 系统级评测（AgentBench 风格，详见 eval/README.md）
python -m eval.runner --offline                          # 离线自检（校验判分器/环境）
MINI_AGENT_LIVE=1 DASHSCOPE_API_KEY=sk-... \
  python -m eval.runner --repeat 3 --workdir eval_runs   # 真实 API + 重复采样 + 评测集留痕

# 确定性回放（从持久化事件日志复现任意 run，0 次真实 API）
python -m mini_agent.replay <run_id> --home ~/.mini_agent [--steps]

# 真实 LLM API 集成测试（按需）
MINI_AGENT_LIVE=1 ANTHROPIC_API_KEY=sk-... DASHSCOPE_API_KEY=sk-... \
  python -m unittest tests.test_live_llm -v

# 真实模型 —— 用环境变量选择 Provider
pip install anthropic                # 或 pip install openai
export LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=sk-...
# 或:
export LLM_PROVIDER=qwen DASHSCOPE_API_KEY=sk-...
export LLM_MODEL=qwen-max            # 可选，覆盖 Provider 默认模型

python -m mini_agent.cli             # CLI 前端
python -m mini_agent.server          # Web 前端 → http://127.0.0.1:8642
```

CLI 与 Web 共享数据目录（默认 `~/.mini_agent`，可用 `MINI_AGENT_HOME` 覆盖）：
在 CLI 里聊到一半，打开网页能看到同一批会话并继续。

## 架构：事件流内核 + 可替换前端

```
                 ┌────────────────────── AgentService（前端无关门面）─────────────────────┐
   CLI 前端 ────►│                                                                        │
  (cli.py)       │   Agent.run() ─ ReAct 循环                                             │
                 │     │  ①(必要时)压缩历史 ── ContextManager（usage 驱动）               │
   Web 前端 ────►│     │  ②组装请求：冻结 system + 记忆索引 + 历史（含 cache 断点）       │
  (server.py     │     │  ③流式调 LLM ── AnthropicLLM（原生 thinking/tool_use/stream）    │
   + SSE         │     │  ④stop_reason == tool_use ?                                     │
   + index.html) │     │      是 → 校验并执行全部工具调用（结果合并进一条 user 消息）→ ②  │
                 │     │      否 → 最终回答，结束                                          │
                 │     ▼                                                                  │
                 │   事件流: thinking_delta / text_delta / tool_call / tool_result /      │
                 │           plan_update / compaction / usage / error / run_end           │
                 └────────────────────────────────────────────────────────────────────────┘
                       ▲ 持久化：sessions/<id>.json + global_memory.json + Tracer 全链路日志
```

**前后端可切换的本质**：内核只产出**带生命周期边界**的类型化事件流（`events.py`）——
`thinking_start/delta/end`、`assistant_delta/final`、`tool_start/result/error`（带
latency）、`plan_update`、`memory_update`、`usage`、`compaction`、`run_stats`。
UI = Event Renderer：只按事件类型渲染，禁止解析原始 LLM 文本。CLI 的 ANSI
渲染器、Web 的 SSE + DOM 渲染器、测试的事件断言，都是同一事件流的消费者。

**Web 端是 Execution IDE，不是聊天窗**：
- 思考层独立折叠块（默认折叠、点击展开、显示耗时与字数，不与回答混排）
- 工具调用是执行单元卡片（名称/参数/running→✓✗ 状态/延迟/完整输出）
- 会话模式三重区分：🌐 共享 / 🔒 隔离（图标 + 左侧色条 + tooltip），
  列表每项显示当前模型、运行状态、一行任务摘要（首条消息自动生成标题）
- 右侧 Execution Panel：计划状态、工具执行链、记忆访问、token/缓存命中/延迟/累计成本
- 三视图：对话 / **时间线**（按真实时间戳的 `+0.42s ⚙ calculator ✓ 3ms` 执行轨迹，
  数据来自 Tracer 而非前端拼凑）/ 调试（原始 trace）

**多模态输入（接口层）**：`service.send(attachments=[canonical image block])` →
框架统一格式进历史 → 各 Provider 自行转换（透传 / `image_url` 分片），
不支持视觉的模型在 API 层报错并以 LLMError 统一上抛，内核不做能力判断；
Web 端 📎 附图。

## 目录结构

| 模块 | 职责 |
|---|---|
| `mini_agent/core/llm.py` | **唯一 LLM 协议**：LLMClient / LLMResponse（text · thinking · tool_calls · usage · stop_reason） |
| `mini_agent/core/llm_adapters.py` | Provider 适配器：AnthropicLLM / QwenLLM / ScriptedLLM（双向转换 + streaming，互不共享 SDK 逻辑） |
| `mini_agent/core/llm_factory.py` | Provider 工厂：按 `LLM_PROVIDER` 环境变量或显式参数选择，内核零 if-provider |
| `mini_agent/core/events.py` | 事件流词汇表 —— 前后端解耦的关键抽象 |
| `mini_agent/core/session.py` | 会话运行时：持久化可恢复、运行锁、usage 统计、分叉切割校验、删除/重命名 |
| `mini_agent/runtime/state.py` | run 的显式状态机（RunState + 迁移原因） |
| `mini_agent/runtime/controller.py` | Agent 控制器：ReAct 循环、并行工具调用合并回写、轮次上限 |
| `mini_agent/runtime/execution_context.py` | per-run 隔离的执行上下文（策略计数不跨 run 泄漏） |
| `mini_agent/runtime/replay.py` | append-only 事件日志（EventLog）+ Replayer/ReplayLLM，任意 run 可确定性复现 |
| `mini_agent/control/` | **控制平面**（系统拥有，LLM 不参与）：backpressure / budget（预算早停）/ policy（allow/deny/超时/重试/限流/熔断） |
| `mini_agent/tools/` | registry（JSON Schema 校验）+ builtin（7 内置工具，含 plan 长任务拆解）+ runtime（隔离执行层） |
| `mini_agent/context/manager.py` | 上下文管理：prompt cache 断点设计 + usage 驱动的历史压缩 |
| `mini_agent/memory/store.py` | 分层记忆：磁盘持久化（原子写 + 锁）+ 索引注入策略 |
| `mini_agent/observability/` | tracer（全链路日志）/ metrics（事件日志复算指标）/ inspector（运行审计、成本估算） |
| `mini_agent/service.py` | AgentService：前端无关的服务门面 |
| `mini_agent/cli.py` · `server.py` · `web/index.html` | 终端前端；纯标准库 HTTP+SSE 后端；Execution IDE 单页前端 |
| `mini_agent/replay.py` | 回放 CLI 入口（`python -m mini_agent.replay <run_id>`） |
| `eval/` | 系统级评测：core（task/environment/runner/report）+ tasks（6 族）+ configs |
| `docs/` | `read_docs` 内置工具的演示素材（运行时资产，非项目文档） |

## 两个旗舰能力（不只是又一个聊天界面）

### ⏱ 运行审计驾驶舱（Run Inspector）—— Agent 的飞行记录仪

每次 run 自动沉淀一条结构化审计记录，随会话持久化：

```
↳ anthropic/claude-opus-4-8 · 3轮 · 工具4✓1✗ · 12.4s · tok 1.2k+18.6kc/860 · cache 94% · ≈$0.0311
```

- 逐工具耗时与成败、真实 token 用量、**缓存命中率**、按 Provider 声明的定价
  元数据**估算成本**、上下文压缩次数
- CLI `/inspect` 回放全部运行；Web 端每次回答后附指标条 + 右侧审计抽屉
- 实现是事件流架构的又一次收益：`RunCollector` 只是 on_event 的中间消费者
  （透传给前端的同时旁路记账），**Agent 循环零改动**
- 这层可观测性在 LangChain 里是拆出去卖的（LangSmith）；这里是框架内建的

### ⑂ 会话分叉 + 跨模型接力（Fork & Relay）—— Provider 抽象的价值兑现

因为对话历史是 Provider 中立的 canonical 格式，所以：

- **中途换模型接着聊**：CLI `/model qwen`、Web 顶部下拉框——同一份历史、
  记忆、计划、待办，换个推理引擎继续干活。典型用法：便宜模型起草，
  强模型接力收尾；审计记录会标注每次 run 用的是哪个模型，成本对比一目了然
- **从任意位置分叉平行会话**：`/fork [n]` / Web 分叉按钮，复制历史与工作状态
  开新路线（坏轨迹逃生 / 同题不同模型 A/B）。分叉边界有校验：
  不能拆散 tool_use/tool_result 配对
- 这不是额外开发的功能，而是"历史格式不属于任何厂商"这一设计决策的自然推论
  ——在把历史存成厂商私有格式的框架里，这件事做不到

## 关键设计与思考

### 0. LLM Provider 可插拔：模型不是框架的组成部分

```
Agent ──► LLMClient（唯一抽象）──► AnthropicLLM / QwenLLM / ScriptedLLM / ...
                │
        LLMResponse（唯一协议）: text · thinking · tool_calls · usage · stop_reason
```

- **框架定义协议，Provider 做翻译**：工具规格是框架的 ToolSpec（name + description +
  JSON Schema），对话历史是框架的 canonical blocks（text/thinking/tool_use/tool_result）。
  每个 Provider 只负责双向转换 + API 调用 + streaming，互相不共享任何 SDK 逻辑
- **能力缺失返回空值，而不是让 Agent 判断**：Qwen 不产出思考时 `thinking=""`，
  不支持缓存断点时忽略 `cache_control`——内核永远没有 `if provider == xxx`
- **解耦是被测试守护的**：`test_providers.py` 里的守卫测试扫描全部内核文件，
  出现任何 Provider 名称（anthropic/qwen/openai/...）即失败
- 新增 GPT/Gemini/本地模型 = 新增一个 Provider 文件 + 工厂注册一行，
  Agent/Tool/Memory/Context/Event 全部零改动

### 1. 上下文与性能：缓存前缀是一等公民

Claude 的 prompt cache 是**前缀匹配**——前缀里任何一个字节变了，之后全部失效。
因此请求组装严格遵循"稳定在前、易变在后"：

- 工具 schema 按名称排序，字节级稳定（渲染顺序 tools → system → messages）
- system 拆两块：**块 1 = 冻结的核心 prompt**（不含时间戳/会话 ID 等任何易变内容），
  在此打第一个 `cache_control` 断点，tools + 核心 prompt 一起命中缓存；
  **块 2 = 记忆索引**（会变），放在断点之后——记忆变化不打穿前缀缓存
- 最后一条消息再打第二个断点，多轮对话增量命中
- 每轮的 `usage`（含 cache_read/cache_write）实时上抛到前端，缓存效果可观测

### 2. 上下文压缩：用真实 usage，不做本地估算

token 数不靠字符数猜——上一轮 API 返回的 `usage` 就是上下文规模的精确事实。
超过阈值（默认 60K，远低于模型上限留足余量）即触发压缩：把较早的历史用 LLM
总结成一条"前情摘要"消息。两个工程细节：

- **切割边界必须落在纯用户消息上**——`tool_use`/`tool_result` 是 API 强制配对的，
  拆开直接报 400；边界从期望位置向后搜索第一条合法消息
- **压缩失败不致命**——LLM 故障时保留原历史，本轮照常，下轮再试

### 3. 记忆设计：分层 + 索引注入（召回时机与放置方式）

三层记忆：
- **工作记忆** = 会话历史，随上下文滚动，由压缩机制管理
- **会话记忆 / 全局记忆** = 磁盘 JSON（会话私有 / 跨会话共享，由窗口创建时决定），
  跨进程、跨前端存活

**放置方式**：
- system 块 2 只放"记忆索引"（键 + 60 字截断值 + **条数上限 50**，超出时索引首行
  提示用 `memory list` 看全量）——索引体积从值与条数两个维度都有界
- 索引放在 cache 断点**之后**：记忆变化不打穿 tools + 核心 prompt 的缓存前缀
- 完整值不进 system：由模型 `memory get` 按需读取，结果以 tool_result 进入对话历史

**召回时机**：
- **被动召回（每轮）**：ReAct 循环每轮组装请求时重新渲染索引——run 中途的写入
  （包括其他并行会话写共享记忆）下一轮即可见
- **主动召回（按需）**：模型对照索引判断需要哪条的完整值再调 `memory get`；
  get 不存在的键会报错并列出已有键，引导模型自我修正
- **写入时机**：用户要求"记住"的信息由模型调用 `memory save` 落盘
  （core prompt 行为准则），写入即发 `memory_update` 事件供前端展示

这是"记忆规模"与"上下文成本"的解耦点：记忆再多，每轮注入的索引成本也是常数级。

### 4. 长流程任务：plan 工具

模型遇到多步任务时先 `plan set` 拆解，执行中逐步 `update` 状态。计划持久化在会话里，
每次变更发 `plan_update` 事件——CLI 画进度框、Web 端渲染置顶计划面板，
用户随时看到"做到哪一步了"。这与 Claude Code 的 TODO 机制同构。

### 5. Tools 与 Session 运行时

- 工具执行拿到 `ToolContext`：会话状态（待办/计划/私有记忆）、全局记忆、
  事件上抛通道 `emit`——工具也能驱动前端（如 plan 的进度渲染）
- 一轮响应里的**多个并行工具调用**全部执行后，结果合并进**同一条** user 消息
  （API 语义要求，拆开会破坏模型的并行调用行为）
- 所有工具异常收敛为 `ToolError` → `tool_result(is_error=true)` 反馈给模型
  自我修正，永不击穿循环
- 会话即运行时：每会话一把锁（同会话串行、跨会话并行），历史/计划/usage
  全部落盘，进程重启无损恢复

### 6. 异常处理与可观测性

- LLM 故障分类捕获（限流/网络/安全拒绝），会话保持可用
- `Tracer` 记录全链路（输入/思考/工具/结果/压缩/usage/错误），CLI `/trace` 与
  Web `/api/.../trace` 随时回放
- SSE 客户端断开不中断 Agent：跑完落盘，刷新页面历史完整

## 系统级评测（eval/，AgentBench 风格）

- **6 任务族 9 任务**：tool_correctness（调对工具）/ multi_step（多步工具链、计划-执行
  闭环）/ memory_multiturn（跨轮记忆利用，多轮同会话）/ error_recovery（注入故障后
  自愈）/ budget_adherence（预算强制早停）/ policy_enforcement（权限拦截下体面收场）
- **状态级判分**：检查执行后的真实状态（事件日志/记忆/终态），不做文本答案字符串匹配；
  判分器**部分得分制**（`ScoreResult.from_checks` 加权检查项）
- **0-100 综合评分**：质量 0.70 + 效率 0.15 + 稳定性 0.15，按任务权重加权合成
- **重复采样** `--repeat k`：区分 pass@k 与严格通过（k 次全过），稳定性入综合分
- **评测集留痕** `--workdir`：每次评测沉淀自包含数据集（manifest.json + 逐任务会话/
  可回放事件日志/记忆终态），可事后逐帧复盘
- **离线自检** `--offline`：无 key 用 scripted 轨迹校验判分器与环境本身，CI 零费用

用法与判分细节见 [eval/README.md](eval/README.md)。

## Web 界面（Claude Code 风格）

- 左侧会话侧栏：新建**共享窗口**（互通全局记忆）/ **隔离窗口**（互不影响），一键切换；
  会话可删除（hover ✕，二次确认）、可重命名（双击标题）
- 思考过程：灰色斜体流式输出，完成后自动折叠，点击展开
- 工具调用卡片：名称 + 参数摘要（`key: value` 截断）→ 展开看格式化参数与完整输出，
  运行中脉冲动画 → ✓/✗ 结果，一键复制输出
- 任务计划面板：置顶实时进度（○ 待办 / ◐ 进行中 / ● 完成）
- 底部状态：每轮 token 用量与缓存命中；头部显示当前上下文规模
- 交互细节：平滑滚动 + 上翻不被新消息拉回（回到底部悬浮按钮）、拖拽/📎 附图、
  `?` 快捷键面板、↑ 输入历史、空会话欢迎页与示例引导、右上角 toast 通知

## CLI 命令

`/new` `/new!`（隔离） `/sessions` `/switch <id>` `/model [名称]`（跨模型接力）
`/fork [n]`（分叉） `/inspect`（运行审计） `/history` `/plan` `/trace` `/memory` `/usage` `/quit`

## 安全与健壮性（专业测试主动发现并修复的真实缺陷）

`test_security.py` 与 `test_context_memory_stress.py` 不是刷绿灯，而是照着真实
攻击面/并发边界找漏洞。这一轮发现并修复了三个真问题：

| 缺陷 | 影响 | 修复 |
|---|---|---|
| calculator 幂运算无界 | `9**9**9` 生成天文数字整数，冻住进程（DoS） | 幂结果位数上限，超限即拒（瞬间拦截） |
| MemoryStore 无锁 | 并行会话共享全局记忆时 `items()` 迭代与 `set()` 改写并发 → `RuntimeError` 崩溃；非原子写导致文件损坏、下次启动记忆全丢 | 加 RLock + 临时文件 `os.replace` 原子写 |
| server 请求体无上限 | 超大 `Content-Length` 一次性 `read()` 撑爆内存 | 8MB 上限 + 有界排空后返回 413 |

守住的核心安全属性（不依赖 LLM 判断力，框架层强保证）：
- **结构化边界**：工具结果/记忆值/用户输入始终封装为独立 canonical block，
  注入载荷（越狱话术、伪造 `"role":"system"`、伪 `</tool_result>` 标签）经
  Anthropic/Qwen 转换后仍是惰性字符串，无法越权制造新消息/角色/工具调用
- **冻结前缀不可污染**：无论用户/记忆里塞什么，`system[0]` 核心指令字节级不变
  —— 既是 prompt cache 命中前提，也是 prompt-injection 的 operator 安全边界
- **工具沙箱**：calculator 表达式白名单（拒 import/lambda/推导式/属性访问/DoS）、
  read_docs 路径穿越防护（`../`、绝对路径、URL 编码、符号链接）
- **输入加固**：畸形 JSON 不崩服务、非法附件 400、会话 id 客户端不可操纵

## 测试覆盖（188 离线用例 + 5 个 Live API 用例）

四层测试策略：
1. **单元测试**：内核各模块的正常/异常路径（下列 test_* 文件）
2. **安全/压力测试**：`test_security`（提示词注入抵御、工具沙箱、输入加固，见上）
   + `test_context_memory_stress`（反复压缩不损坏且永不拆工具配对、压缩失败可恢复、
   极长历史、**记忆并发读写不崩不丢**、fork 全局共享 vs 私有隔离、索引确定性与截断）
3. **端到端场景测试**（`test_e2e_scenarios`）：真实 HTTP Server + SSE 上跑完整业务
   场景——黄金路径（计划→工具→报错自纠→收尾，校验事件生命周期配对/时间线单调/
   审计汇总）、多模态附件往返、跨模型接力+分叉、压缩事件、**跨会话并行/同会话互斥**
   （用时间窗重叠证明）、CLI 渲染器全事件词汇表、UI 静态契约（IDE 组件齐全 +
   页面禁含 Provider 名）
4. **Live API 测试**（`test_live_llm`，`MINI_AGENT_LIVE=1` + key 时运行，否则自动
   skip）：真实 Claude/Qwen 的流式增量、原生工具调用决策（模型自主调 calculator
   得 548）、真实 usage、**真实跨厂商接力**（Claude 记住的事实换 Qwen 能接着答）

- **test_tools**：7 工具正常/异常路径、schema 校验（必填/类型/enum/bool≠int/未知参数）、安全（表达式注入、路径穿越）、plan 全流程与事件
- **test_context**：缓存断点位置、核心 prompt 字节级冻结、历史零污染、usage 驱动压缩、切割边界不拆工具配对、压缩失败兜底
- **test_agent**：直接回答、工具循环、**并行调用合并回写**、错误自纠、未知工具、轮次上限、LLM 故障、追问上下文、run 内自动压缩、usage 累计、事件流完整性
- **test_session_memory**：记忆/会话磁盘持久化与重启恢复、损坏文件容错、共享 vs 隔离
- **test_service_server**：服务门面端到端、会话忙锁、HTTP API + SSE 事件流冒烟、错误码
- **test_providers**：工厂按环境变量选择/缺 key 报错/未知 Provider；Qwen 双向转换（消息/工具/流式聚合/并行调用/坏参数降级/usage 映射，全部离线）；**解耦守卫**（内核文件禁含 Provider 名称）
- **test_inspector_fork**：成本/缓存命中率计算、审计记录生成与持久化、事件透传不丢失；分叉深拷贝与边界校验、**跨模型接力**（换 Provider 后新模型收到同一份历史、审计标注各自模型）
- **test_security**：calculator 沙箱（代码执行向量/幂 DoS/正常放行）、read_docs 路径穿越、工具结果结构化注入惰性（Anthropic+Qwen）、冻结前缀不可污染、记忆投毒被隔离、服务端输入加固（超大体 413/畸形 JSON/非法附件/会话 id 不可控）
- **test_context_memory_stress**：压缩不变量（跨 keep 值不拆配对/反复压缩一致/失败可恢复/极长历史）、build_request 零污染与断点唯一、**记忆并发**（8 线程无丢失、读写并发不崩、Agent 并行写全局）、fork 记忆作用域、索引确定性与截断
- **test_runtime / test_control_plane**：状态机迁移与终态、执行上下文 per-run 隔离；背压早停、预算强制、策略（deny/超时/重试/限流/熔断）
- **test_replay_metrics**：事件日志落盘、Replayer 确定性复现、指标离线复算
- **test_eval**：评测框架自检（离线全过 + 六族覆盖）、**判分器判别力**（负样本必须判负）、部分得分、重复采样聚合（pass@k/稳定性）、综合评分、多轮任务
