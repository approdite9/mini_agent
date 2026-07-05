"""Execution IDE 可交互演示：规则驱动的 Mock 模型，无需任何 API key。

运行: python ui_demo.py [端口，默认 8899]
打开: http://127.0.0.1:8899

MockLLM 是 LLMClient 的又一个 Provider 实现：根据用户输入的关键词决定调用
哪些工具（计划/计算/天气/记忆/搜索），走真实的 Agent 循环、真实的事件流、
真实的持久化与审计 —— 只有"推理"是规则模拟的。

试试这些输入：
  帮我规划北京出差：预算 (1200+800)*1.1，再看看天气
  记住我的项目代号是蓝鲸七号
  搜索一下 ReAct 是什么
  (1+2+3)*4 等于多少
也可以 📎 附一张图片、顶部分叉会话、切换时间线视图。
"""

from __future__ import annotations

import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mini_agent import AgentService
from mini_agent import LLMClient, LLMResponse, ToolCallRequest
from mini_agent.server import make_server

_CITIES = ["北京", "上海", "广州", "深圳", "杭州"]
_EXPR_RE = re.compile(r"[(\d][\d\s+\-*/().%]*[\d)]")
_MEM_RE = re.compile(r"记住[，,:：]?\s*(.+?)(?:是|=|：|:)\s*(.+)", re.S)

Action = Tuple[str, Dict[str, Any], str]  # (tool, args, thinking)


class MockLLM(LLMClient):
    """规则驱动的演示模型：关键词 -> 工具序列 -> 汇总回答。"""

    pricing = {"input": 5.0, "output": 25.0, "cache_read_multiplier": 0.1,
               "cache_write_multiplier": 1.25, "currency": "USD"}

    def describe(self) -> str:
        return "mock/demo-model"

    # ------------------------------------------------------------------
    def complete(self, *, system, messages, tools=None, on_delta=None) -> LLMResponse:
        text, images, results_after = self._parse_state(messages)
        actions = self._build_actions(text)
        stage = self._count_stage(messages)

        if stage < len(actions):
            tool, args, thinking = actions[stage]
            return self._respond(thinking=thinking, tool=(tool, args),
                                 on_delta=on_delta, turn=stage)
        answer = self._final_answer(text, images, results_after)
        return self._respond(thinking="信息已齐，汇总给用户。", text=answer,
                             on_delta=on_delta, turn=stage)

    # ---- 状态解析 -------------------------------------------------------
    @staticmethod
    def _parse_state(messages) -> Tuple[str, int, List[str]]:
        """返回 (最近一条真实用户输入, 附图数, 此后累计的工具结果)。"""
        text, images, idx = "", 0, -1
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if msg["role"] != "user":
                continue
            content = msg["content"]
            if isinstance(content, str):
                text, idx = content, i
                break
            texts = [b.get("text", "") for b in content if b.get("type") == "text"]
            if any(b.get("type") == "tool_result" for b in content):
                continue
            text = " ".join(texts)
            images = sum(1 for b in content if b.get("type") == "image")
            idx = i
            break
        results: List[str] = []
        for msg in messages[idx + 1:]:
            if msg["role"] != "user" or isinstance(msg["content"], str):
                continue
            for block in msg["content"]:
                if block.get("type") == "tool_result":
                    raw = block.get("content", "")
                    if isinstance(raw, list):
                        raw = " ".join(str(b.get("text", "")) for b in raw)
                    results.append(str(raw))
        return text, images, results

    @staticmethod
    def _count_stage(messages) -> int:
        """自最近真实用户输入以来，已发出的工具调用轮数。"""
        stage = 0
        for msg in reversed(messages):
            content = msg.get("content")
            if msg["role"] == "user" and (
                isinstance(content, str)
                or not any(b.get("type") == "tool_result" for b in content)
            ):
                break
            if msg["role"] == "assistant" and isinstance(content, list) and any(
                    b.get("type") == "tool_use" for b in content):
                stage += 1
        return stage

    # ---- 规则：输入 -> 工具序列 ------------------------------------------
    def _build_actions(self, text: str) -> List[Action]:
        main: List[Tuple[str, Tuple[str, Dict[str, Any]]]] = []

        for expr in _EXPR_RE.findall(text):
            expr = expr.strip()
            if any(op in expr for op in "+-*/") and len(expr) >= 3:
                main.append((f"计算 {expr}", ("calculator", {"expression": expr})))
        if "天气" in text:
            city = next((c for c in _CITIES if c in text), None)
            if city is None:
                # 演示错误自纠：先查一个不存在的城市，再用北京重试
                main.append(("查询天气", ("weather", {"city": "北平"})))
                main.append(("修正城市重查", ("weather", {"city": "北京"})))
            else:
                main.append((f"查询{city}天气", ("weather", {"city": city})))
        mem = _MEM_RE.search(text)
        if mem:
            key, value = mem.group(1).strip()[:20], mem.group(2).strip()[:60]
            main.append((f"写入记忆 {key}", ("memory", {"action": "save",
                                                        "key": key, "value": value})))
        elif "记住" in text:
            main.append(("写入记忆", ("memory", {"action": "save", "key": "备忘",
                                                 "value": text[:60]})))
        if "搜索" in text or "什么是" in text or "是什么" in text:
            query = re.sub(r"搜索一?下?|什么是|是什么|[？?，。 ]", " ", text).strip() or text
            main.append((f"搜索 {query[:12]}", ("search", {"query": query[:30]})))
        if "文档" in text:
            main.append(("查看文档列表", ("read_docs", {})))

        wants_plan = any(k in text for k in ("规划", "计划", "安排", "帮我")) and len(main) >= 2
        actions: List[Action] = []
        if wants_plan:
            steps = [label for label, _ in main]
            actions.append(("plan", {"action": "set", "steps": steps},
                            "这是个多步任务，先用 plan 工具拆解。"))
            for i, (label, (tool, args)) in enumerate(main):
                actions.append((tool, args, f"执行第 {i + 1} 步：{label}。"))
                actions.append(("plan", {"action": "update", "step": i + 1,
                                         "status": "done"}, ""))
        else:
            for label, (tool, args) in main:
                actions.append((tool, args, f"需要{label}，调用工具。"))
        return actions

    def _final_answer(self, text: str, images: int, results: List[str]) -> str:
        parts = []
        if images:
            parts.append(f"收到你发来的 {images} 张图片（演示模型不做真实识图，"
                         f"真实模型下会走多模态推理）。")
        useful = [r for r in results if not r.startswith("计划已")]
        if useful:
            parts.append("已完成：" + "；".join(u.split("\n")[0] for u in useful[:4]) + "。")
        if not parts:
            parts.append(f"收到：「{text[:40]}」。我是规则驱动的演示模型，"
                         f"试试让我算数、查天气、做规划或记住某件事。")
        return " ".join(parts)

    # ---- 流式输出模拟 -----------------------------------------------------
    def _respond(self, thinking: str = "", text: str = "",
                 tool: Optional[Tuple[str, Dict[str, Any]]] = None,
                 on_delta=None, turn: int = 0) -> LLMResponse:
        def stream(kind: str, content: str):
            if not on_delta or not content:
                return
            for i in range(0, len(content), 6):
                on_delta(kind, {"text": content[i:i + 6]})
                time.sleep(0.02)  # 让前端呈现真实的流式节奏

        stream("thinking_delta", thinking)
        stream("text_delta", text)

        raw: List[Dict[str, Any]] = []
        if thinking:
            raw.append({"type": "thinking", "thinking": thinking, "signature": "mock"})
        if text:
            raw.append({"type": "text", "text": text})
        tool_calls: List[ToolCallRequest] = []
        if tool:
            call_id = f"toolu_mock_{time.time_ns()}"
            raw.append({"type": "tool_use", "id": call_id,
                        "name": tool[0], "input": tool[1]})
            tool_calls.append(ToolCallRequest(id=call_id, name=tool[0],
                                              arguments=tool[1]))
        return LLMResponse(
            stop_reason="tool_use" if tool else "end_turn",
            text=text, thinking=thinking,
            tool_calls=tool_calls, raw_content=raw,
            usage={"input_tokens": 180 + 40 * turn, "output_tokens": 30 + len(text) // 3,
                   "cache_read_input_tokens": 900 * min(turn, 3),
                   "cache_creation_input_tokens": 120 if turn == 0 else 0},
        )


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8899
    home = Path(tempfile.mkdtemp(prefix="mini_agent_ui_demo_"))
    service = AgentService(llm=MockLLM(), base_dir=home)
    server = make_server(service, port=port)
    print(f"Execution IDE 演示已启动: http://127.0.0.1:{port}（无需 API key）")
    print(f"数据目录: {home}")
    print(__doc__.split("试试这些输入：")[1])
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")


if __name__ == "__main__":
    main()
