"""评测运行入口：
python -m eval.runner [--config F] [--offline] [--json OUT] [--workdir DIR] [--repeat K]

模式：
- 真实 API（默认，当 MINI_AGENT_LIVE=1 且配置了对应 key）：用 create_llm() 驱动完整
  Controller 在真实执行环境上跑每个任务，状态级判分。
- 离线自检（无 key 或 --offline）：用每个任务自带的 scripted 轨迹驱动，校验判分器与
  环境本身正确（不产生费用）。CI 用这个，等价于 test_live_llm 的门控方式。

留痕：默认在临时目录运行、跑完即清理；给 --workdir DIR 时，每次评测在
DIR/<时间戳>/ 下保留完整评测集 —— 逐任务的会话历史、可回放事件日志、记忆终态，
外加 manifest.json（goal/判分/产物索引）与 report.json（汇总报告）。

配置（JSON，避免引入 YAML 依赖）：{"provider", "model", "families"}。
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

from eval.core.report import dataset_manifest, print_console, summarize
from eval.core.runner import EvalRunner
from eval.tasks import select
from mini_agent import ScriptedLLM, create_llm

_DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "minimal.json"


def _load_config(path: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"provider": None, "model": None, "families": []}


def _live_ready(provider: str) -> bool:
    if os.environ.get("MINI_AGENT_LIVE") != "1":
        return False
    key = {"qwen": "DASHSCOPE_API_KEY"}.get((provider or "").lower(), "ANTHROPIC_API_KEY")
    if (provider or "").lower() not in ("qwen",) and not os.environ.get("LLM_PROVIDER"):
        # 默认 anthropic
        return bool(os.environ.get("ANTHROPIC_API_KEY"))
    return bool(os.environ.get(key))


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    config_path = _DEFAULT_CONFIG
    force_offline = "--offline" in argv
    json_out = None
    workdir = None
    repeat_arg = None
    i = 0
    while i < len(argv):
        if argv[i] == "--config" and i + 1 < len(argv):
            config_path = Path(argv[i + 1]); i += 2
        elif argv[i] == "--json" and i + 1 < len(argv):
            json_out = Path(argv[i + 1]); i += 2
        elif argv[i] == "--workdir" and i + 1 < len(argv):
            workdir = Path(argv[i + 1]); i += 2
        elif argv[i] == "--repeat" and i + 1 < len(argv):
            repeat_arg = int(argv[i + 1]); i += 2
        else:
            i += 1

    cfg = _load_config(config_path)
    provider = cfg.get("provider")
    model = cfg.get("model")
    repeat = max(1, repeat_arg if repeat_arg is not None else int(cfg.get("repeat") or 1))
    tasks = select(cfg.get("families") or None)

    live = (not force_offline) and _live_ready(provider)
    rep_txt = f" × {repeat}次采样" if repeat > 1 else ""
    if live:
        print(f"[真实 API 模式] provider={provider or os.environ.get('LLM_PROVIDER') or 'anthropic'} "
              f"model={model or '默认'}  ({len(tasks)} 任务{rep_txt})")
        llm_factory = lambda task: create_llm(provider=provider, model=model)
    else:
        why = "强制 --offline" if force_offline else "未检测到 MINI_AGENT_LIVE=1 + API key"
        print(f"[离线自检模式：{why}] 用各任务的 scripted 轨迹校验判分器/环境  ({len(tasks)} 任务)")
        llm_factory = lambda task: ScriptedLLM(task.scripted)

    if workdir:
        # 留痕模式：每次评测独占 DIR/<时间戳>/，避免复用目录时会话/日志互相污染
        root = workdir / time.strftime("%Y%m%d-%H%M%S")
        n = 1
        while root.exists():
            root = workdir / (time.strftime("%Y%m%d-%H%M%S") + f"-{n}"); n += 1
        root.mkdir(parents=True)
        print(f"[留痕] 评测集目录: {root}")
        results = EvalRunner(llm_factory).run_all(tasks, base_dir=root, repeat=repeat)
    else:
        with tempfile.TemporaryDirectory(prefix="mini_agent_eval_") as tmp:
            results = EvalRunner(llm_factory).run_all(tasks, base_dir=Path(tmp),
                                                      repeat=repeat)

    print_console(results)
    summary = summarize(results)
    if workdir:
        manifest = dataset_manifest(tasks, results, root, {
            "mode": "live" if live else "offline",
            "provider": (provider or os.environ.get("LLM_PROVIDER") or "anthropic")
                        if live else "scripted",
            "model": model,
            "repeat": repeat,
            "config": str(config_path),
        })
        (root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        (root / "report.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"评测集已保存: {root}（manifest.json + report.json + 每任务 sessions/runs/memory）")
    if json_out:
        json_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        print(f"JSON 报告已写入: {json_out}")
    # 退出码：有失败则非 0（便于 CI）
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
