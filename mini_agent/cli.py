"""终端前端：消费 AgentService 事件流，流式渲染（与 Web 前端共用同一内核）。

用法（模型 Provider 由 LLM_PROVIDER 环境变量决定，见 mini_agent/llm/factory.py）:
    python -m mini_agent.cli

命令:
    /new [标题]     新建共享全局记忆的会话      /new! [标题]  新建记忆隔离的会话
    /sessions       列出所有会话（含磁盘恢复）  /switch <id>  切换会话
    /model [名称]   查看/切换当前会话的模型 Provider（跨模型接力）
    /fork [n]       分叉当前会话（可选：截到第 n 条消息），平行试另一条路
    /inspect        运行审计：每次 run 的轮次/工具耗时/token/缓存命中/成本
    /history        当前会话对话历史            /plan         当前任务计划
    /trace          当前会话执行日志            /memory       全局与私有记忆
    /usage          当前会话 token 用量         /quit         退出
其余输入均作为用户消息发给 Agent。
"""

from __future__ import annotations

import json
import sys

from .core.events import AgentEvent
from .observability.inspector import format_run
from .core.llm import LLMError
from .core.llm_factory import create_llm
from .service import AgentService, SessionBusyError

# ---- ANSI 样式 -----------------------------------------------------------
DIM, RESET, BOLD = "\x1b[2m", "\x1b[0m", "\x1b[1m"
ORANGE, GREEN, RED, CYAN = "\x1b[38;5;209m", "\x1b[32m", "\x1b[31m", "\x1b[36m"


class CliRenderer:
    """事件流 -> 终端渲染。thinking 灰色斜体，工具调用一行一卡，文本流式输出。"""

    def __init__(self) -> None:
        self._streaming_text = False
        self._streaming_thinking = False
        self._answered = False  # 本次 run 是否已渲染答案（防重复/防丢失）

    def __call__(self, event: AgentEvent) -> None:
        etype, data = event.type, event.data
        if etype == "run_start":
            self._answered = False
        elif etype == "thinking_start":
            sys.stdout.write(f"\n{DIM}∴ ")
            self._streaming_thinking = True
        elif etype == "thinking_delta":
            if self._streaming_thinking:
                sys.stdout.write(data["text"])
                sys.stdout.flush()
        elif etype == "thinking_end":
            self._close_thinking()
            print(f"{DIM}  (思考 {data.get('duration_ms', 0) / 1000:.1f}s){RESET}")
        elif etype == "assistant_delta":
            if not self._streaming_text:
                sys.stdout.write(f"\n{BOLD}{ORANGE}⏺{RESET} ")
                self._streaming_text = True
            sys.stdout.write(data["text"])
            sys.stdout.flush()
            if data.get("text"):
                self._answered = True
        elif etype == "assistant_final":
            # 未逐字流式的 Provider：answer 只在这里到达，补渲染（已流式过则跳过防重复）
            if data.get("content") and not self._answered:
                self._close_all()
                print(f"{BOLD}{ORANGE}⏺{RESET} {data['content']}")
                self._answered = True
        elif etype == "tool_start":
            self._close_all()
            args = json.dumps(data["arguments"], ensure_ascii=False)
            print(f"{CYAN}⚙ {data['tool']}{RESET}{DIM}({args[:120]}){RESET}")
        elif etype == "tool_result":
            first_line = str(data["result"]).split("\n")[0][:110]
            print(f"  {GREEN}✓{RESET} {DIM}{first_line} · {data.get('duration_ms', 0)}ms{RESET}")
        elif etype == "tool_error":
            first_line = str(data["error"]).split("\n")[0][:110]
            print(f"  {RED}✗{RESET} {DIM}{first_line} · {data.get('duration_ms', 0)}ms{RESET}")
        elif etype == "memory_update":
            print(f"{DIM}◆ 记忆写入[{data.get('scope', '')}]: {data.get('key', '')}{RESET}")
        elif etype == "plan_update":
            icons = {"pending": "○", "in_progress": "◐", "done": "●", "skipped": "⊘"}
            print(f"{DIM}┌ 计划{RESET}")
            for i, step in enumerate(data["plan"]):
                icon = icons.get(step["status"], "?")
                style = GREEN if step["status"] == "done" else (BOLD if step["status"] == "in_progress" else DIM)
                print(f"{DIM}│{RESET} {style}{icon} {i + 1}. {step['title']}{RESET}")
            print(f"{DIM}└{RESET}")
        elif etype == "compaction":
            print(f"{DIM}⇣ 上下文已压缩：{data['dropped_messages']} 条历史 -> 摘要{RESET}")
        elif etype == "usage":
            cached = data.get("cache_read_input_tokens", 0)
            print(f"{DIM}  · tokens in={data.get('input_tokens', 0)} "
                  f"out={data.get('output_tokens', 0)} cache_hit={cached}{RESET}")
        elif etype == "error":
            self._close_all()
            print(f"{RED}✗ {data['message']}{RESET}")
        elif etype == "run_end":
            self._close_all()
            # 兜底：本次 run 一个答案都没渲染（含达轮次上限/预算早停等终态消息）
            if not self._answered and data.get("answer"):
                print(f"{BOLD}{ORANGE}⏺{RESET} {data['answer']}")
                self._answered = True

    def _close_thinking(self) -> None:
        if self._streaming_thinking:
            sys.stdout.write(f"{RESET}\n")
            self._streaming_thinking = False

    def _close_all(self) -> None:
        self._close_thinking()
        if self._streaming_text:
            sys.stdout.write("\n")
            self._streaming_text = False


def main() -> None:
    try:
        llm = create_llm()
    except LLMError as exc:
        print(f"LLM 初始化失败: {exc}")
        print("离线体验: python demo.py（脚本化 Mock LLM，无需 API key）")
        sys.exit(1)

    service = AgentService(llm=llm)
    existing = service.list_sessions()
    if existing:
        current = service.get_session(existing[-1]["id"])
        print(f"已从磁盘恢复 {len(existing)} 个会话，当前会话 {current.id}（{current.title}）")
    else:
        current = service.create_session(title="默认会话")
        print(f"mini_agent 已启动，当前会话 {current.id}（{current.title}，共享记忆）")
    print(f"{DIM}数据目录: {service.base_dir}   /quit 退出，/sessions 查看窗口{RESET}")
    renderer = CliRenderer()

    while True:
        try:
            line = input(f"\n{BOLD}[{current.id}] ›{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if not line:
            continue

        if line.startswith("/"):
            parts = line.split(maxsplit=1)
            cmd, arg = parts[0], (parts[1] if len(parts) > 1 else "")
            if cmd == "/quit":
                print("再见！")
                break
            elif cmd in ("/new", "/new!"):
                shared = cmd == "/new"
                current = service.create_session(title=arg, shared_memory=shared)
                print(f"已创建并切换到 {current.id}（{current.title}，"
                      f"{'共享全局记忆' if shared else '记忆隔离'}）")
            elif cmd == "/sessions":
                for info in service.list_sessions():
                    mark = "*" if info["id"] == current.id else " "
                    mem = "共享" if info["shared_memory"] else "隔离"
                    print(f" {mark} {info['id']}  {info['title']}  记忆:{mem}  "
                          f"消息:{info['messages']}  上下文:{info['context_tokens']}tok")
            elif cmd == "/switch":
                try:
                    current = service.get_session(arg)
                    print(f"已切换到 {current.id}（{current.title}）")
                except KeyError as exc:
                    print(str(exc))
            elif cmd == "/model":
                info = service.providers()
                if not arg:
                    label = service.set_provider(current.id, current.provider)
                    print(f"  当前模型: {label}"
                          f"（会话覆盖: {current.provider or '无，使用默认'}）")
                    print(f"  可用 Provider: {', '.join(info['available'])}，"
                          f"用 /model <名称> 切换，/model default 回到默认")
                else:
                    try:
                        label = service.set_provider(current.id, arg)
                        print(f"  已切换，本会话后续由 {label} 接力"
                              f"（历史与记忆原样保留）")
                    except LLMError as exc:
                        print(f"  切换失败: {exc}")
            elif cmd == "/fork":
                try:
                    at = int(arg) if arg else None
                    forked = service.fork_session(current.id, at_message=at)
                    current = forked
                    print(f"已分叉并切换到 {forked.id}（{forked.title}，"
                          f"来源 {forked.parent}）。可用 /model 换模型重走这条路。")
                except (ValueError, KeyError) as exc:
                    print(f"  分叉失败: {exc}")
            elif cmd == "/inspect":
                records = service.runs(current.id)
                if not records:
                    print("  本会话还没有运行记录")
                for r in records:
                    print(f"  [{r['started_at']}] {format_run(r)}")
                    print(f"    {DIM}› {r['input'][:70]}{RESET}")
                    for t in r.get("tools", []):
                        mark = "✓" if t["ok"] else "✗"
                        print(f"    {DIM}⚙ {t['tool']} {mark} {t['duration_ms']}ms{RESET}")
            elif cmd == "/history":
                for msg in current.history:
                    content = msg["content"]
                    if isinstance(content, list):
                        content = " ".join(
                            b.get("text", f"<{b.get('type')}>") for b in content
                            if isinstance(b, dict) and b.get("type") in ("text", "tool_use", "tool_result"))
                    print(f"  {DIM}[{msg['role']}]{RESET} {str(content)[:160]}")
                if not current.history:
                    print("  (空)")
            elif cmd == "/plan":
                if not current.plan:
                    print("  当前没有任务计划")
                for i, step in enumerate(current.plan):
                    print(f"  {i + 1}. [{step['status']}] {step['title']}")
            elif cmd == "/trace":
                print(service.trace(current.id))
            elif cmd == "/memory":
                snapshot = service.memory_snapshot(current.id)
                print("全局记忆:")
                for k, v in snapshot["global"].items() or {"": ""}.items():
                    if k:
                        print(f"  - {k}: {v}")
                if not snapshot["global"]:
                    print("  (空)")
                print(f"会话({current.id})私有记忆:")
                for k, v in snapshot.get("local", {}).items():
                    print(f"  - {k}: {v}")
                if not snapshot.get("local"):
                    print("  (空)")
            elif cmd == "/usage":
                total = current.total_usage
                print(f"  累计: in={total.get('input_tokens', 0)} out={total.get('output_tokens', 0)} "
                      f"cache_read={total.get('cache_read_input_tokens', 0)} "
                      f"cache_write={total.get('cache_creation_input_tokens', 0)}")
                print(f"  当前上下文规模: {current.context_tokens()} tokens")
            else:
                print(f"未知命令 {cmd}")
            continue

        try:
            result = service.send(current.id, line, on_event=renderer)
        except SessionBusyError as exc:
            print(f"{RED}{exc}{RESET}")
            continue
        if current.runs:
            print(f"{DIM}  ↳ {format_run(current.runs[-1])}{RESET}")
        if result.error:
            print(f"{DIM}(异常结束: {result.error}){RESET}")


if __name__ == "__main__":
    main()
