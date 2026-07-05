"""确定性回放 CLI —— 从持久化事件日志重建并复现一次 run。

    python -m mini_agent.replay <run_id> [--home DIR] [--steps]

- 默认：打印重建的状态迁移时间线（每步的 from→to 与 reason）+ 结构化指标。
- --steps：逐事件单步展开（deterministic debug 的文本版）。

任意 run 只要有 <home>/runs/<run_id>.jsonl 即可完全复现执行过程，无需真实 API。
"""

from __future__ import annotations

import sys
from pathlib import Path

from .observability.metrics import compute_run_metrics
from .runtime.replay import Replayer
from .service import default_home

DIM, RESET, ORANGE, GREEN, RED, CYAN = (
    "\x1b[2m", "\x1b[0m", "\x1b[38;5;209m", "\x1b[32m", "\x1b[31m", "\x1b[36m")


def _fmt_state(s: str) -> str:
    if s in ("completed",):
        return f"{GREEN}{s}{RESET}"
    if s in ("failed", "stopped_budget", "stopped_max_turns"):
        return f"{RED}{s}{RESET}"
    return f"{CYAN}{s}{RESET}"


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    run_id = argv[0]
    home = default_home()
    steps = False
    i = 1
    while i < len(argv):
        if argv[i] == "--home" and i + 1 < len(argv):
            home = Path(argv[i + 1]); i += 2
        elif argv[i] == "--steps":
            steps = True; i += 1
        else:
            i += 1

    runs_dir = Path(home) / "runs"
    log_path = runs_dir / f"{run_id}.jsonl"
    if not log_path.is_file():
        print(f"找不到事件日志: {log_path}")
        return 1

    replayer = Replayer.from_run(run_id, runs_dir)
    print(f"{ORANGE}◆ 回放 run {run_id}{RESET}  {DIM}({len(replayer.events)} 事件){RESET}\n")

    if steps:
        for event, state in replayer.iter_states():
            tag = event.type
            extra = event.reason or ""
            print(f"{DIM}[{state.current.value:<16}]{RESET} {tag} {DIM}{extra}{RESET}")
    else:
        print(f"{DIM}状态迁移时间线（为什么发生）:{RESET}")
        for t in replayer.transitions():
            print(f"  {DIM}{t['from']:>16}{RESET} → {_fmt_state(t['to']):<24} "
                  f"{DIM}{t['reason']}{RESET}")

    final = replayer.reconstruct_state()
    print(f"\n终态: {_fmt_state(final.current.value)}")

    m = compute_run_metrics(replayer.events)
    c = m["counters"]
    lat = m["tool_latency_ms"]
    print(f"{DIM}指标: 轮次{c['turns']} 迁移{c['transitions']} "
          f"工具{c['tool_ok']}✓{c['tool_error']}✗ 重试{c['tool_retry']} "
          f"压缩{c['compactions']} 早停{c['early_stops']}{RESET}")
    if lat["count"]:
        print(f"{DIM}工具延迟: avg {lat['avg']}ms  p95 {lat['p95']}ms  max {lat['max']}ms{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
