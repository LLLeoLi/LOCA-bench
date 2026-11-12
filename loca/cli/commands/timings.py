# Copyright 2025 LOCA-bench Contributors. All Rights Reserved.
#
# Licensed under the MIT License.

"""``loca timings``: summarize per-task wall-clock time by context size."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

from inference.common.timing import (
    collect_task_timings,
    format_duration,
    summarize_tasks,
    utc_now_iso,
)
from loca.cli.utils.config_resolver import PROJECT_ROOT

console = Console()


def _find_latest_output() -> Path:
    outputs = PROJECT_ROOT / "outputs"
    candidates = list(outputs.rglob("task_timings.json")) if outputs.is_dir() else []
    if not candidates and outputs.is_dir():
        candidates = list(outputs.rglob("results.json"))

    # `loca run -o <name>` writes directly under the project root rather than
    # under outputs/. Fall back to top-level run directories in that case.
    if not candidates:
        candidates = list(PROJECT_ROOT.glob("*/task_timings.json"))
        if not candidates:
            candidates = list(PROJECT_ROOT.glob("*/results.json"))

    if not candidates:
        raise FileNotFoundError(
            "No task_timings.json or results.json found under "
            f"{outputs} or {PROJECT_ROOT}. Pass --input explicitly."
        )
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest.parent


def timings_command(
    input_dir: Annotated[
        Optional[str],
        typer.Option(
            "--input",
            "-i",
            help=(
                "Output directory to analyse (the directory containing "
                "task_timings.json / tasks/). Defaults to the newest under outputs/."
            ),
        ),
    ] = None,
    output: Annotated[
        Optional[str],
        typer.Option(
            "--output",
            "-o",
            help="Where to write timing_summary.json (defaults to the input directory).",
        ),
    ] = None,
    per_task: Annotated[
        bool,
        typer.Option(
            "--per-task/--no-per-task",
            help="Also print one row per task.",
        ),
    ] = False,
) -> None:
    """Summarize per-task wall-clock time grouped by 8k/16k/32k/64k.

    Examples:
        loca timings
        loca timings -i outputs/inf_react_..._20261112_051837
        loca timings -i outputs/run --per-task
    """
    try:
        target = Path(input_dir).resolve() if input_dir else _find_latest_output()
    except FileNotFoundError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)

    if not target.is_dir():
        console.print(f"[red]Error:[/red] Not a directory: {target}")
        raise typer.Exit(1)

    payload = collect_task_timings(target)
    tasks = payload.get("tasks") or []
    buckets = payload.get("buckets") or summarize_tasks(tasks)

    if not tasks:
        console.print(
            f"[yellow]No task timings found in {target}.[/yellow]\n"
            "Run with the updated runner (eval.json/trajectory.json now store "
            "wall_clock_seconds), or pass the correct output directory."
        )
        raise typer.Exit(1)

    header = Table(title=f"LOCA wall-clock timings — {target}", header_style="bold", padding=(0, 1))
    for column in ("Size", "N", "Total", "Avg", "p50", "p90", "Min", "Max"):
        header.add_column(column, justify="right" if column != "Size" else "left", no_wrap=True)

    for name, stats in buckets.items():
        if name == "all":
            continue
        header.add_row(
            name,
            str(stats.get("tasks", 0)),
            format_duration(stats.get("total_seconds")),
            format_duration(stats.get("avg_seconds")),
            format_duration(stats.get("p50_seconds")),
            format_duration(stats.get("p90_seconds")),
            format_duration(stats.get("min_seconds")),
            format_duration(stats.get("max_seconds")),
        )
    overall = buckets.get("all", {})
    header.add_section()
    header.add_row(
        "all",
        str(overall.get("tasks", 0)),
        format_duration(overall.get("total_seconds")),
        format_duration(overall.get("avg_seconds")),
        format_duration(overall.get("p50_seconds")),
        format_duration(overall.get("p90_seconds")),
        format_duration(overall.get("min_seconds")),
        format_duration(overall.get("max_seconds")),
    )
    console.print(header)

    run_elapsed = payload.get("run_elapsed_seconds")
    if run_elapsed is not None:
        console.print(
            f"Run wall clock: [bold]{format_duration(run_elapsed)}[/bold] "
            f"({float(run_elapsed):.1f}s), tasks: {len(tasks)}"
        )

    if per_task:
        detail = Table(title="Per-task timings", header_style="bold")
        for column in ("Config", "State", "Status", "Acc", "Steps", "Wall clock"):
            detail.add_column(column, justify="right" if column not in ("Config", "Status") else "left", no_wrap=True)
        for task in sorted(
            tasks,
            key=lambda t: (
                t.get("size") or "",
                t.get("config_name") or "",
                t.get("run_id") or 0,
            ),
        ):
            detail.add_row(
                str(task.get("config_name") or ""),
                str(task.get("run_id") if task.get("run_id") is not None else ""),
                str(task.get("status") or ""),
                f"{float(task['accuracy']):.2f}" if task.get("accuracy") is not None else "",
                str(task.get("steps") if task.get("steps") is not None else ""),
                format_duration(task.get("wall_clock_seconds")),
            )
        console.print(detail)

    summary = {
        "generated_at": utc_now_iso(),
        "output_dir": str(target),
        "run_elapsed_seconds": run_elapsed,
        "task_count": len(tasks),
        "buckets": buckets,
        "tasks": tasks,
    }
    out_path = Path(output).resolve() if output else target / "timing_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    console.print(f"Timing summary written to [bold]{out_path}[/bold]")
