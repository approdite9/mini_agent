"""离线 demo：用 ScriptedLLM 模拟模型行为，走真实内核（事件流/工具/记忆/压缩），
完整演示 计划拆解 -> 工具调用 -> 错误自纠 -> 跨会话记忆 全流程（无需 API key）。

真实模型请用: python -m mini_agent.cli 或 python -m mini_agent.server
运行: python demo.py
"""

import tempfile
from pathlib import Path

from mini_agent import AgentService, ScriptedLLM
from mini_agent.cli import CliRenderer
from mini_agent.observability.inspector import format_run


def main() -> None:
    script = [
        # ---- 请求 1（s1）：复杂任务 -> 先 plan 拆解，再逐步执行 ----
        ScriptedLLM.call("plan", {"action": "set", "steps": [
            "计算旅行预算 (1200+800)*1.1", "查询北京天气", "汇总建议"]},
            thinking="这是个多步任务，先制定计划。"),
        ScriptedLLM.call("plan", {"action": "update", "step": 1, "status": "in_progress"}),
        ScriptedLLM.call("calculator", {"expression": "(1200+800)*1.1"}),
        ScriptedLLM.call("plan", {"action": "update", "step": 1, "status": "done"}),
        ScriptedLLM.call("weather", {"city": "北平"}, thinking="查天气。"),
        ScriptedLLM.call("weather", {"city": "北京"},
                         thinking="工具报错没有'北平'，改用'北京'重试。"),
        ScriptedLLM.call("plan", {"action": "update", "step": 2, "status": "done"}),
        ScriptedLLM.say("预算约 2200 元；北京今天晴 26°C，适合出行。建议轻装+防晒。",
                        thinking="信息齐了，汇总回答。"),
        # ---- 请求 2（s1）：记住偏好（写入全局记忆，磁盘持久化）----
        ScriptedLLM.call("memory", {"action": "save", "key": "常驻城市", "value": "北京"}),
        ScriptedLLM.say("好的，已记住你的常驻城市是北京。"),
        # ---- 请求 3（s2 共享窗口）：能看到全局记忆 ----
        ScriptedLLM.say("你的常驻城市是北京（来自全局记忆索引）。",
                        thinking="system 记忆索引里有 常驻城市: 北京。"),
        # ---- 请求 4（s3 隔离窗口）：看不到 ----
        ScriptedLLM.say("这个窗口是隔离的，我这里没有你的城市记录。"),
    ]

    with tempfile.TemporaryDirectory() as tmp:
        service = AgentService(llm=ScriptedLLM(script), base_dir=Path(tmp))
        renderer = CliRenderer()

        s1 = service.create_session(title="旅行规划")
        s2 = service.create_session(title="共享窗口")
        s3 = service.create_session(title="隐私窗口", shared_memory=False)

        steps = [
            (s1, "帮我规划北京旅行：预算 (1200+800)*1.1，再看看天气给点建议"),
            (s1, "记住：我常驻北京"),
            (s2, "我常驻哪个城市？"),
            (s3, "我常驻哪个城市？"),
        ]
        for session, text in steps:
            print(f"\n{'=' * 64}\n[{session.id}·{session.title}] › {text}")
            service.send(session.id, text, on_event=renderer)
            print(f"\n  ↳ 审计: {format_run(session.runs[-1])}")

        # 旗舰功能：分叉 s1 —— 复制历史与工作状态，开一条平行路线（可换模型接力）
        fork = service.fork_session(s1.id)
        print(f"\n{'=' * 64}")
        print(f"已分叉: {fork.id} 来源 {fork.parent}（历史 {len(fork.history)} 条、"
              f"计划 {len(fork.plan)} 步原样复制，与主线互不影响）")
        print(f"全局记忆(磁盘持久化): {service.global_memory.items()}")
        print(f"s1 计划终态: {[(p['title'], p['status']) for p in s1.plan]}")
        print(f"s1 审计记录: {len(s1.runs)} 条  trace 事件: {len(service.tracer.events)}")
        print(f"会话文件: {sorted(p.name for p in (Path(tmp) / 'sessions').glob('*.json'))}")


if __name__ == "__main__":
    main()
