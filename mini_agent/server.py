"""Web 前端的后端：纯标准库 HTTP 服务 + SSE 事件流。

与 CLI 共用同一个 AgentService —— 事件流架构下"前后端可切换"的另一半。

用法（模型 Provider 由 LLM_PROVIDER 环境变量决定，见 mini_agent/llm/factory.py）:
    python -m mini_agent.server [端口，默认 8642]
    打开 http://127.0.0.1:8642

API:
    GET  /                          Web UI（代码终端风格单页应用）
    GET  /api/sessions              会话列表
    POST /api/sessions              新建会话 {"title": "...", "shared_memory": true}
    DELETE /api/sessions/<id>       删除会话（运行中返回 409）
    PATCH  /api/sessions/<id>       重命名会话 {"title": "..."}
    GET  /api/sessions/<id>/history 会话历史（渲染友好格式）
    GET  /api/sessions/<id>/runs    运行审计记录（Run Inspector）
    GET  /api/sessions/<id>/timeline 结构化时间线（trace 事件 + 真实时间戳）
    GET  /api/sessions/<id>/trace   执行日志（文本）
    GET  /api/providers             可用模型 Provider 列表
    POST /api/sessions/<id>/provider 切换会话模型 {"provider": "..."}（跨模型接力）
    POST /api/sessions/<id>/fork    分叉会话 {"at_message": n?, "provider": "..."?}
    GET  /api/memory?session=<id>   记忆快照
    POST /api/sessions/<id>/messages
         {"text": "...", "attachments": [{type:"image", media_type, data(base64)}]?}
         -> SSE 流（data: {type, data} 逐事件推送，结束前附带 run_stats 审计事件）

并发模型：ThreadingHTTPServer 每请求一线程；同一会话由 Session.lock 串行化
（重复发送返回 409），不同会话可并行运行。
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

from .core.events import AgentEvent
from .core.llm import LLMError
from .core.llm_factory import create_llm
from .service import AgentService, SessionBusyError

# LLMError 同时用于 provider 切换/分叉时的参数校验错误（返回 400）

WEB_DIR = Path(__file__).resolve().parent / "web"


def _history_for_render(history) -> list:
    """把原生消息转成前端易渲染的条目序列。"""
    items = []
    for msg in history:
        role, content = msg["role"], msg["content"]
        if isinstance(content, str):
            items.append({"kind": "user" if role == "user" else "text", "content": content})
            continue
        for block in content:
            btype = block.get("type")
            if role == "user" and btype == "tool_result":
                items.append({"kind": "tool_result",
                              "id": block.get("tool_use_id", ""),
                              "content": _tool_result_text(block),
                              "ok": not block.get("is_error", False)})
            elif btype == "text":
                items.append({"kind": "user" if role == "user" else "text",
                              "content": block.get("text", "")})
            elif btype == "image":
                source = block.get("source", {})
                items.append({"kind": "image",
                              "media_type": source.get("media_type", "image/png"),
                              "data": source.get("data", "")})
            elif btype == "thinking" and block.get("thinking"):
                items.append({"kind": "thinking", "content": block["thinking"]})
            elif btype == "tool_use":
                items.append({"kind": "tool_call", "id": block.get("id", ""),
                              "tool": block.get("name", ""),
                              "arguments": block.get("input", {})})
    return items


def _tool_result_text(block) -> str:
    content = block.get("content", "")
    if isinstance(content, list):
        return " ".join(str(b.get("text", "")) for b in content if isinstance(b, dict))
    return str(content)


class _Handler(BaseHTTPRequestHandler):
    service: AgentService  # 由 make_server 注入到子类属性

    # ---- 基础设施 -------------------------------------------------------
    def log_message(self, fmt, *args):  # 静默默认访问日志，trace 已足够
        pass

    def _json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    MAX_BODY_BYTES = 8 * 1024 * 1024  # 请求体上限，防超大 Content-Length 撑爆内存

    def _read_body(self) -> Dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", 0))
        except (TypeError, ValueError):
            return {}
        if length <= 0:
            return {}
        if length > self.MAX_BODY_BYTES:
            # 边读边丢弃，内存恒定（不 read(length) 一次性分配），排空后由上层
            # 返回干净的 413，而不是不读 body 导致连接被 RST
            remaining = min(length, 64 * 1024 * 1024)  # 排空上限，超巨型声明则任其 reset
            while remaining > 0:
                chunk = self.rfile.read(min(65536, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
            return {"__too_large__": True}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}

    def _static(self, filename: str, content_type: str) -> None:
        path = WEB_DIR / filename
        if not path.is_file():
            self._json(404, {"error": f"缺少静态文件 {filename}"})
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ---- 路由 -----------------------------------------------------------
    def do_GET(self) -> None:
        path, _, query = self.path.partition("?")
        try:
            if path in ("/", "/index.html"):
                self._static("index.html", "text/html; charset=utf-8")
            elif path == "/api/sessions":
                self._json(200, self.service.list_sessions())
            elif path.startswith("/api/sessions/") and path.endswith("/history"):
                sid = path.split("/")[3]
                session = self.service.get_session(sid)
                self._json(200, {
                    "id": session.id, "title": session.title,
                    "shared_memory": session.use_global_memory,
                    "plan": session.plan,
                    "items": _history_for_render(session.history),
                    "total_usage": session.total_usage,
                    "context_tokens": session.context_tokens(),
                    "provider": session.provider,
                    "parent": session.parent,
                })
            elif path.startswith("/api/sessions/") and path.endswith("/runs"):
                sid = path.split("/")[3]
                self._json(200, {"runs": self.service.runs(sid)})
            elif path.startswith("/api/sessions/") and path.endswith("/timeline"):
                sid = path.split("/")[3]
                self._json(200, {"events": self.service.timeline(sid)})
            elif path.startswith("/api/sessions/") and path.endswith("/run_ids"):
                sid = path.split("/")[3]
                self._json(200, {"run_ids": self.service.list_runs(sid)})
            elif path.startswith("/api/sessions/") and path.endswith("/metrics"):
                sid = path.split("/")[3]
                self._json(200, self.service.metrics(sid))
            elif path.startswith("/api/runs/") and path.endswith("/events"):
                rid = path.split("/")[3]
                self._json(200, {
                    "run_id": rid,
                    "events": self.service.run_events(rid),
                    "transitions": self.service.run_transitions(rid),
                })
            elif path == "/api/providers":
                self._json(200, self.service.providers())
            elif path.startswith("/api/sessions/") and path.endswith("/trace"):
                sid = path.split("/")[3]
                self._json(200, {"trace": self.service.trace(sid)})
            elif path == "/api/memory":
                sid = None
                for pair in query.split("&"):
                    if pair.startswith("session="):
                        sid = pair.split("=", 1)[1] or None
                self._json(200, self.service.memory_snapshot(sid))
            else:
                self._json(404, {"error": "not found"})
        except KeyError as exc:
            self._json(404, {"error": str(exc)})
        except Exception as exc:  # 服务端兜底，避免线程静默崩溃
            self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

    def do_POST(self) -> None:
        path = self.path.partition("?")[0]
        try:
            if path == "/api/sessions":
                body = self._read_body()
                session = self.service.create_session(
                    title=str(body.get("title", "")),
                    shared_memory=bool(body.get("shared_memory", True)))
                self._json(201, {"id": session.id, "title": session.title,
                                 "shared_memory": session.use_global_memory})
            elif path.startswith("/api/sessions/") and path.endswith("/messages"):
                self._handle_message(path.split("/")[3])
            elif path.startswith("/api/sessions/") and path.endswith("/provider"):
                body = self._read_body()
                label = self.service.set_provider(
                    path.split("/")[3], body.get("provider"))
                self._json(200, {"model": label})
            elif path.startswith("/api/sessions/") and path.endswith("/fork"):
                body = self._read_body()
                at = body.get("at_message")
                forked = self.service.fork_session(
                    path.split("/")[3],
                    at_message=int(at) if at is not None else None,
                    provider=body.get("provider") or None)
                self._json(201, {"id": forked.id, "title": forked.title,
                                 "parent": forked.parent})
            else:
                self._json(404, {"error": "not found"})
        except KeyError as exc:
            self._json(404, {"error": str(exc)})
        except (ValueError, LLMError) as exc:
            self._json(400, {"error": str(exc)})
        except Exception as exc:
            self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

    def do_DELETE(self) -> None:
        path = self.path.partition("?")[0]
        try:
            parts = path.split("/")
            if path.startswith("/api/sessions/") and len(parts) == 4 and parts[3]:
                self.service.delete_session(parts[3])
                self._json(200, {"ok": True, "id": parts[3]})
            else:
                self._json(404, {"error": "not found"})
        except KeyError as exc:
            self._json(404, {"error": str(exc)})
        except SessionBusyError as exc:
            self._json(409, {"error": str(exc)})
        except Exception as exc:
            self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

    def do_PATCH(self) -> None:
        path = self.path.partition("?")[0]
        try:
            parts = path.split("/")
            if path.startswith("/api/sessions/") and len(parts) == 4 and parts[3]:
                body = self._read_body()
                session = self.service.rename_session(
                    parts[3], str(body.get("title", "")))
                self._json(200, {"id": session.id, "title": session.title})
            else:
                self._json(404, {"error": "not found"})
        except KeyError as exc:
            self._json(404, {"error": str(exc)})
        except ValueError as exc:
            self._json(400, {"error": str(exc)})
        except Exception as exc:
            self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

    # ---- SSE 消息流 -------------------------------------------------------
    MAX_ATTACHMENT_BYTES = 4 * 1024 * 1024  # base64 后约 4MB 上限

    def _handle_message(self, session_id: str) -> None:
        body = self._read_body()
        if body.get("__too_large__"):
            self._json(413, {"error": "请求体过大（上限 8MB）"})
            return
        text = str(body.get("text", "")).strip()
        attachments = body.get("attachments") or []
        if not text and not attachments:
            self._json(400, {"error": "text 不能为空"})
            return
        # 多模态接口：校验并转为框架 canonical image block
        canonical = []
        total = 0
        for att in attachments:
            data = str(att.get("data", ""))
            total += len(data)
            if att.get("type") != "image" or not data:
                self._json(400, {"error": "attachment 必须是 {type:'image', media_type, data}"})
                return
            if total > self.MAX_ATTACHMENT_BYTES:
                self._json(400, {"error": "附件过大（上限 4MB）"})
                return
            canonical.append({
                "type": "image",
                "source": {"type": "base64",
                           "media_type": att.get("media_type", "image/png"),
                           "data": data},
            })

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        def push(event: AgentEvent) -> None:
            try:
                payload = json.dumps(event.to_dict(), ensure_ascii=False)
                self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass  # 客户端断开：Agent 继续跑完并落盘，历史不丢

        try:
            self.service.send(session_id, text, on_event=push,
                              attachments=canonical or None)
            records = self.service.runs(session_id)
            if records:  # 运行审计：结束前推送本次 run 的结构化记录
                push(AgentEvent(type="run_stats", session_id=session_id,
                                data=records[-1]))
        except SessionBusyError as exc:
            push(AgentEvent(type="error", session_id=session_id,
                            data={"message": str(exc), "busy": True}))
        except KeyError as exc:
            push(AgentEvent(type="error", session_id=session_id,
                            data={"message": str(exc)}))


def make_server(service: AgentService, host: str = "127.0.0.1",
                port: int = 8642) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (_Handler,), {"service": service})
    return ThreadingHTTPServer((host, port), handler)


def main() -> None:
    try:
        llm = create_llm()
    except LLMError as exc:
        print(f"LLM 初始化失败: {exc}")
        sys.exit(1)
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8642
    service = AgentService(llm=llm)
    server = make_server(service, port=port)
    print(f"mini_agent Web 前端已启动: http://127.0.0.1:{port}")
    print(f"数据目录: {service.base_dir}（与 CLI 共享，会话互通）")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")


if __name__ == "__main__":
    main()
