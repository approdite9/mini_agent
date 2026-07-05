"""mini_agent.observability —— 系统级可观测（持久化，非仅 UI）。

- tracer   : Tracer / TraceEvent（进程内 span；持久化的完整轨迹由 EventLog 承担）
- metrics  : compute_run_metrics / aggregate_metrics（从事件日志复算的结构化指标）
- inspector: RunCollector / 审计记录（面向人的摘要，按需从 .inspector 导入以避免循环）
"""

from .tracer import TraceEvent, Tracer
from .metrics import aggregate_metrics, compute_run_metrics

__all__ = ["TraceEvent", "Tracer", "compute_run_metrics", "aggregate_metrics"]
