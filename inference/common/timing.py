"""Wall-clock timing helpers for LOCA-bench tasks.

Task-level timings are recorded by the runners in ``eval.json`` /
``results.json`` and aggregated into ``task_timings.json`` at the output root.
These helpers are shared by the runner (to aggregate at the end of a run) and
by ``loca timings`` (to re-analyse an existing output directory).
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

_SIZE_RE = re.compile(r"_(?P<size>\d+)k$", re.IGNORECASE)
_STATE_RE = re.compile(r"^state(?P<id>\d+)$")
_RUN_RE = re.compile(r"^run_(?P<id>\d+)$")


def utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def parse_size_bucket(config_name: Optional[str]) -> str:
    """Map a task/config name to its context-size bucket, e.g. ``8k``.

    LOCA configs name tasks like ``CanvasListTestS2LEnv_8k``. Anything that
    does not carry a trailing ``_<n>k`` suffix is grouped under ``unknown``.
    """
    if not config_name:
        return "unknown"
    match = _SIZE_RE.search(config_name)
    return f"{int(match.group('size'))}k" if match else "unknown"


def _percentile(sorted_values: Sequence[float], fraction: float) -> Optional[float]:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return round(float(sorted_values[0]), 3)
    position = (len(sorted_values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    if lower == upper:
        value = sorted_values[lower]
    else:
        value = sorted_values[lower] + (
            sorted_values[upper] - sorted_values[lower]
        ) * (position - lower)
    return round(float(value), 3)


def summarize_seconds(values: Iterable[Optional[float]]) -> Dict[str, Any]:
    """Summarize a collection of wall-clock durations (seconds)."""
    seconds = sorted(float(v) for v in values if v is not None)
    if not seconds:
        return {
            "count": 0,
            "total_seconds": None,
            "avg_seconds": None,
            "p50_seconds": None,
            "p90_seconds": None,
            "min_seconds": None,
            "max_seconds": None,
        }

    total = sum(seconds)
    return {
        "count": len(seconds),
        "total_seconds": round(total, 3),
        "avg_seconds": round(total / len(seconds), 3),
        "p50_seconds": _percentile(seconds, 0.50),
        "p90_seconds": _percentile(seconds, 0.90),
        "min_seconds": round(seconds[0], 3),
        "max_seconds": round(seconds[-1], 3),
    }


def summarize_tasks(tasks: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Aggregate task timings by context-size bucket (plus an ``all`` bucket)."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for task in tasks:
        bucket = task.get("size") or parse_size_bucket(task.get("config_name"))
        grouped.setdefault(str(bucket), []).append(task)

    def _stats(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        stats = summarize_seconds(item.get("wall_clock_seconds") for item in items)
        stats["tasks"] = len(items)
        stats["success"] = sum(
            1 for item in items if item.get("status") == "success"
        )
        stats["error"] = len(items) - stats["success"]
        return stats

    result = {bucket: _stats(items) for bucket, items in grouped.items()}
    result["all"] = _stats(list(tasks))
    return _sort_buckets(result)


def _sort_buckets(buckets: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    def key(name: str) -> tuple:
        if name == "all":
            return (2, 0)
        if name == "unknown":
            return (1, 0)
        try:
            return (0, int(name.rstrip("kK")))
        except ValueError:
            return (1, 0)

    return {name: buckets[name] for name in sorted(buckets, key=key)}


def format_duration(seconds: Optional[float]) -> str:
    """Render seconds as ``1h 02m 03s`` / ``12m 03s`` / ``42.3s``."""
    if seconds is None:
        return "n/a"
    seconds = float(seconds)
    if seconds >= 3600:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(round(seconds % 60))
        if secs == 60:
            secs = 0
            minutes += 1
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if seconds >= 60:
        minutes = int(seconds // 60)
        secs = int(round(seconds % 60))
        if secs == 60:
            secs = 0
            minutes += 1
        return f"{minutes}m{secs:02d}s"
    return f"{seconds:.1f}s"


def collect_task_timings(output_dir: str | Path) -> Dict[str, Any]:
    """Load task timings from an output directory.

    Prefers ``task_timings.json`` written at the end of a run. Falls back to
    scanning ``tasks/<name>/state<N>/eval.json`` (and trajectory metrics) so
    existing/interrupted runs can still be analysed.
    """
    output_path = Path(output_dir)
    timing_file = output_path / "task_timings.json"
    if timing_file.is_file():
        with open(timing_file) as f:
            payload = json.load(f)
        if "buckets" not in payload:
            payload["buckets"] = summarize_tasks(payload.get("tasks", []))
        return payload

    tasks: List[Dict[str, Any]] = []
    tasks_dir = output_path / "tasks"
    scan_root = tasks_dir if tasks_dir.is_dir() else output_path

    for eval_file in sorted(scan_root.glob("*/state*/eval.json")):
        task_dir = eval_file.parent
        config_name = task_dir.parent.name
        state_match = _STATE_RE.match(task_dir.name)
        run_id = int(state_match.group("id")) if state_match else 0
        tasks.append(_read_task_timing(eval_file, config_name, run_id, None))

    # Older layout: config_<id>/run_<id>/eval.json
    for eval_file in sorted(scan_root.glob("config_*/run_*/eval.json")):
        task_dir = eval_file.parent
        config_dir = task_dir.parent.name
        run_match = _RUN_RE.match(task_dir.name)
        run_id = int(run_match.group("id")) if run_match else 0
        tasks.append(_read_task_timing(eval_file, config_dir, run_id, None))

    run_elapsed = None
    results_file = output_path / "results.json"
    if results_file.is_file():
        try:
            with open(results_file) as f:
                run_elapsed = json.load(f).get("metadata", {}).get("elapsed_seconds")
        except (OSError, json.JSONDecodeError):
            pass

    return {
        "generated_at": utc_now_iso(),
        "output_dir": str(output_path),
        "run_elapsed_seconds": run_elapsed,
        "task_count": len(tasks),
        "buckets": summarize_tasks(tasks),
        "tasks": tasks,
    }


def _read_task_timing(
    eval_file: Path,
    config_name: str,
    run_id: int,
    task_id: Optional[int],
) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    try:
        with open(eval_file) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        pass

    wall_clock = data.get("wall_clock_seconds")
    if wall_clock is None:
        traj_file = eval_file.parent / "trajectory.json"
        try:
            with open(traj_file) as f:
                traj = json.load(f)
            metrics = traj.get("metrics", {})
            wall_clock = metrics.get("wall_clock_seconds")
        except (OSError, json.JSONDecodeError):
            pass

    return {
        "task_id": task_id,
        "config_name": config_name,
        "size": parse_size_bucket(config_name),
        "run_id": run_id,
        "status": data.get("status"),
        "accuracy": data.get("accuracy"),
        "steps": data.get("steps"),
        "wall_clock_seconds": wall_clock,
        "start_time": data.get("start_time"),
        "end_time": data.get("end_time"),
        "error": data.get("feedback") if data.get("status") == "error" else None,
    }
