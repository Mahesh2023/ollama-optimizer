"""
CLI entry point for ollama-optimizer.

Uses Click for command handling and Rich for terminal output.
All module calls use the real dataclass-based APIs from the package.
"""

import click
from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
import logging
import sys
import json
import os
import dataclasses
from statistics import mean

from ollama_optimizer.system_profiler import detect_system, estimate_model_capacity
from ollama_optimizer.ollama_client import OllamaClient, OllamaModel
from ollama_optimizer.quantization import QuantizationEngine, GGUF_QUANT_LEVELS, GGUFQuantLevel
from ollama_optimizer.optimizer import (
    ModelOptimizer, ModelAnalysis, OptimizationPlan, OptimizationResult,
    EnvironmentConfig,
)
from ollama_optimizer.benchmark import BenchmarkRunner, BenchmarkResult, BenchmarkComparison
from ollama_optimizer.reporter import Reporter

console = Console()
logger = logging.getLogger(__name__)

BANNER = r"""[bold cyan]
   ___  _ _                        ___        _   _       _
  / _ \| | | __ _ _ __ ___   __ _ / _ \ _ __ | |_(_)_ __ (_)_______ _ __
 | | | | | |/ _` | '_ ` _ \ / _` | | | | '_ \| __| | '_ \| |_  / _ \ '__|
 | |_| | | | (_| | | | | | | (_| | |_| | |_) | |_| | | | | |/ /  __/ |
  \___/|_|_|\__,_|_| |_| |_|\__,_|\___/| .__/ \__|_|_| |_|_/___\___|_|
                                        |_|
[/bold cyan][dim]Quantization-powered model optimization for Ollama  v1.0.0[/dim]
[dim]Based on ngrok's "Quantization from the Ground Up" blog[/dim]
"""

OLLAMA_NOT_RUNNING_MSG = (
    "[bold red]Error:[/bold red] Ollama is not running or not reachable.\n\n"
    "Start it with:\n"
    "  [green]Linux/Mac:[/green]  ollama serve\n"
    "  [green]Windows:[/green]   Start Ollama from the Start menu\n"
    "  [green]Docker:[/green]    docker run -d -p 11434:11434 ollama/ollama\n\n"
    "Install Ollama: [link=https://ollama.ai/download]https://ollama.ai/download[/link]"
)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _check_ollama(client: OllamaClient) -> bool:
    """Return True if Ollama is reachable, else print error and return False."""
    with console.status("[bold yellow]Checking Ollama connection...[/bold yellow]"):
        if not client.is_running():
            console.print(OLLAMA_NOT_RUNNING_MSG)
            return False
    console.print("[green]\u2713[/green] Ollama is running\n")
    return True


def _fmt_size(gb: float) -> str:
    if gb >= 1:
        return f"{gb:.1f} GB"
    return f"{gb * 1024:.0f} MB"


def _score_style(score: float) -> str:
    if score >= 0.85:
        return "green"
    if score >= 0.65:
        return "yellow"
    return "red"


def _display_system(profile) -> None:
    """Print system hardware info from a SystemProfile dataclass."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Property", style="bold cyan", min_width=18)
    table.add_column("Value")

    table.add_row("OS", f"{profile.os_type} {profile.os_version}")
    table.add_row("CPU", f"{profile.cpu_name}")
    table.add_row("Cores", f"{profile.cpu_cores_physical} physical / {profile.cpu_cores_logical} logical")
    table.add_row("Total RAM", f"{profile.ram_total_gb:.1f} GB")
    table.add_row("Available RAM", f"{profile.ram_available_gb:.1f} GB")

    if profile.gpu_detected:
        table.add_row("GPU", f"{profile.gpu_name} ({profile.gpu_vram_gb:.1f} GB VRAM)")
        cc = getattr(profile, 'gpu_compute_capability', 0.0)
        if cc > 0:
            table.add_row("Compute Cap.", f"{cc:.1f}")
        fa = getattr(profile, 'gpu_supports_flash_attn', False)
        table.add_row("Flash Attention", "[green]Supported[/green]" if fa else "[yellow]Not supported[/yellow]")
    else:
        table.add_row("GPU", "[dim]Not detected (CPU-only mode)[/dim]")

    numa = getattr(profile, 'numa_available', False)
    if numa:
        nodes = getattr(profile, 'numa_node_count', 1)
        table.add_row("NUMA", f"{nodes} nodes")

    table.add_row("Disk Free", f"{profile.disk_free_gb:.1f} GB")
    console.print(Panel(table, title="[bold]System Hardware[/bold]", border_style="blue"))
    console.print()


def _display_models(analyses: list) -> None:
    """Print model analysis table from list of ModelAnalysis dataclasses."""
    table = Table(title="Installed Models", show_lines=True)
    table.add_column("Model", style="bold cyan", min_width=20)
    table.add_column("Family", justify="center")
    table.add_column("Params", justify="right")
    table.add_column("Quant", justify="center")
    table.add_column("Size", justify="right", style="yellow")
    table.add_column("Quality", justify="center")
    table.add_column("Speed", justify="center")
    table.add_column("RAM", justify="center")
    table.add_column("VRAM", justify="center")

    for a in analyses:
        qs = _score_style(a.current_quality_score)
        table.add_row(
            a.model,
            a.family or "?",
            a.parameter_count or "?",
            a.current_quant or "?",
            _fmt_size(a.current_size_gb),
            f"[{qs}]{a.current_quality_score:.0%}[/{qs}]",
            f"{a.current_speed_multiplier:.1f}x",
            "[green]\u2714[/green]" if a.fits_in_ram else "[red]\u2718[/red]",
            "[green]\u2714[/green]" if a.fits_in_vram else "[dim]-[/dim]",
        )

    console.print(table)
    console.print()


def _display_plans(plans: list) -> None:
    """Print optimization plans from list of OptimizationPlan dataclasses."""
    table = Table(title="Optimization Plans", show_lines=True)
    table.add_column("Model", style="bold cyan", min_width=20)
    table.add_column("Current", justify="center")
    table.add_column("", justify="center")
    table.add_column("Target", justify="center", style="green")
    table.add_column("Action", min_width=18)
    table.add_column("Size", justify="right")
    table.add_column("Quality", justify="center")
    table.add_column("Speed", justify="center")

    for p in plans:
        if p.size_reduction_pct > 0:
            size_str = f"{_fmt_size(p.optimized_size_gb)} [green](-{p.size_reduction_pct:.0f}%)[/green]"
        elif p.size_reduction_pct < 0:
            size_str = f"{_fmt_size(p.optimized_size_gb)} [yellow](+{abs(p.size_reduction_pct):.0f}%)[/yellow]"
        else:
            size_str = _fmt_size(p.optimized_size_gb)

        qd = p.quality_delta
        q_str = f"[green]+{qd:.0%}[/green]" if qd > 0 else f"[yellow]{qd:.0%}[/yellow]" if qd < 0 else "="

        sp = p.speed_improvement
        s_str = f"[green]{sp:.1f}x[/green]" if sp > 1 else f"[yellow]{sp:.1f}x[/yellow]" if sp < 1 else "1.0x"

        table.add_row(
            p.model_name,
            p.current_quant or "?",
            "\u2192",
            p.recommended_quant or "?",
            p.action.replace("_", " ").title(),
            size_str,
            q_str,
            s_str,
        )

    console.print(table)

    for p in plans:
        if p.reason:
            console.print(f"  [dim]{p.model_name}: {p.reason}[/dim]")
    console.print()


def _display_bench_results(results: list, title: str = "Benchmark Results") -> None:
    """Print a list of BenchmarkResult dataclasses."""
    table = Table(title=title, show_lines=True)
    table.add_column("Model", style="bold cyan", min_width=18)
    table.add_column("Tokens/sec", justify="right", style="green")
    table.add_column("TTFT (ms)", justify="right")
    table.add_column("Prompt tok/s", justify="right")
    table.add_column("Total Time", justify="right")
    table.add_column("Memory (MB)", justify="right")

    for r in results:
        table.add_row(
            r.model_name,
            f"{r.tokens_per_second:.1f}",
            f"{r.time_to_first_token_ms:.0f}",
            f"{r.prompt_eval_rate:.1f}",
            f"{r.total_time_seconds:.2f}s",
            f"{r.memory_usage_mb:.0f}",
        )

    console.print(table)
    console.print()


def _display_comparison(comp: BenchmarkComparison) -> None:
    """Print a BenchmarkComparison showing before/after."""
    table = Table(
        title=f"Comparison: {comp.original_model} vs {comp.optimized_model}",
        show_lines=True,
    )
    table.add_column("Metric", style="bold")
    table.add_column("Original", justify="right")
    table.add_column("Optimized", justify="right")
    table.add_column("Change", justify="right")

    orig = comp.original_result
    opt = comp.optimized_result

    def _delta(old_val, new_val, higher_is_better=True):
        if old_val == 0:
            return "[dim]N/A[/dim]"
        pct = ((new_val - old_val) / old_val) * 100
        good = (pct > 0) == higher_is_better
        style = "green" if good else "red"
        return f"[{style}]{pct:+.1f}%[/{style}]"

    table.add_row(
        "Tokens/sec", f"{orig.tokens_per_second:.1f}", f"{opt.tokens_per_second:.1f}",
        _delta(orig.tokens_per_second, opt.tokens_per_second),
    )
    table.add_row(
        "Time to First Token", f"{orig.time_to_first_token_ms:.0f} ms", f"{opt.time_to_first_token_ms:.0f} ms",
        _delta(orig.time_to_first_token_ms, opt.time_to_first_token_ms, higher_is_better=False),
    )
    table.add_row(
        "Prompt Processing", f"{orig.prompt_eval_rate:.1f} tok/s", f"{opt.prompt_eval_rate:.1f} tok/s",
        _delta(orig.prompt_eval_rate, opt.prompt_eval_rate),
    )
    table.add_row(
        "Memory", f"{orig.memory_usage_mb:.0f} MB", f"{opt.memory_usage_mb:.0f} MB",
        _delta(orig.memory_usage_mb, opt.memory_usage_mb, higher_is_better=False),
    )

    console.print(table)

    summary_parts = []
    if comp.speedup_ratio > 1:
        summary_parts.append(f"[green]{comp.speedup_ratio:.2f}x faster[/green]")
    elif comp.speedup_ratio < 1:
        summary_parts.append(f"[yellow]{comp.speedup_ratio:.2f}x slower[/yellow]")
    if comp.memory_reduction_pct > 0:
        summary_parts.append(f"[green]{comp.memory_reduction_pct:.0f}% less memory[/green]")
    if comp.quality_note:
        summary_parts.append(f"[dim]{comp.quality_note}[/dim]")

    if summary_parts:
        console.print(Panel(" | ".join(summary_parts), title="Summary", border_style="cyan"))
    console.print()


def _aggregate_bench(results: list) -> BenchmarkResult:
    """Compute an aggregate BenchmarkResult from a suite of results."""
    if not results:
        return None
    return BenchmarkResult(
        model_name=results[0].model_name,
        timestamp=results[0].timestamp,
        prompt_tokens=sum(r.prompt_tokens for r in results),
        completion_tokens=sum(r.completion_tokens for r in results),
        total_tokens=sum(r.total_tokens for r in results),
        time_to_first_token_ms=mean(r.time_to_first_token_ms for r in results),
        tokens_per_second=mean(r.tokens_per_second for r in results) if results else 0,
        total_time_seconds=sum(r.total_time_seconds for r in results),
        prompt_eval_rate=mean(r.prompt_eval_rate for r in results) if results else 0,
        eval_rate=mean(r.eval_rate for r in results) if results else 0,
        memory_usage_mb=max((r.memory_usage_mb for r in results), default=0),
        load_time_seconds=results[0].load_time_seconds,
        prompt_used="(aggregated)",
        raw_response={},
    )


def _display_env_config(env_config: EnvironmentConfig) -> None:
    """Print environment variable recommendations."""
    env_dict = env_config.to_env_dict()
    if not env_dict:
        console.print("[dim]Default Ollama settings are optimal for your system.[/dim]\n")
        return

    table = Table(title="Server Environment Recommendations", show_lines=True)
    table.add_column("Variable", style="bold cyan", min_width=28)
    table.add_column("Value", style="bold green", min_width=8)
    table.add_column("Why", min_width=40)

    reasons = env_config.reasons or {}
    for var, val in env_dict.items():
        table.add_row(var, str(val), reasons.get(var, ""))

    console.print(table)
    console.print()

    console.print(Panel(
        env_config.to_shell_exports(),
        title="[bold]Shell Exports[/bold] (add to ~/.bashrc or ~/.zshrc)",
        border_style="blue",
    ))
    console.print()


# ──────────────────────────────────────────────────────────────────────
# Click group
# ──────────────────────────────────────────────────────────────────────

@click.group()
@click.version_option(version="1.0.0", prog_name="ollama-optimizer")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def main(verbose):
    """Ollama Optimizer - Quantization-powered model optimization tool.

    Automatically optimize your Ollama models for maximum performance
    on your hardware using quantization techniques.

    Based on concepts from ngrok's "Quantization from the Ground Up" blog.
    """
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(name)s - %(levelname)s - %(message)s")


# ──────────────────────────────────────────────────────────────────────
# scan
# ──────────────────────────────────────────────────────────────────────

@main.command()
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
def scan(json_output):
    """Scan system hardware and list all installed Ollama models with analysis."""
    try:
        console.print(BANNER)
        client = OllamaClient()
        if not _check_ollama(client):
            sys.exit(1)

        with console.status("[bold yellow]Detecting hardware...[/bold yellow]"):
            profile = detect_system()

        if not json_output:
            _display_system(profile)

        with console.status("[bold yellow]Fetching installed models...[/bold yellow]"):
            models = client.list_models()

        if not models:
            console.print("[yellow]No models installed.[/yellow] Install one with:")
            console.print("  [green]ollama pull llama3.2[/green]")
            sys.exit(0)

        console.print(f"[green]\u2713[/green] Found {len(models)} model(s)\n")

        optimizer = ModelOptimizer(client=client, system=profile)
        analyses = [optimizer.analyze_model(m) for m in models]

        if json_output:
            output = {
                "system": dataclasses.asdict(profile),
                "models": [dataclasses.asdict(m) for m in models],
                "analyses": [dataclasses.asdict(a) for a in analyses],
            }
            console.print_json(json.dumps(output, default=str))
        else:
            _display_models(analyses)

            plans = [optimizer.create_optimization_plan(a) for a in analyses]
            actionable = [p for p in plans if p.action != "already_optimal"]
            if actionable:
                console.print(
                    Panel(
                        "\n".join(
                            f"  [yellow]\u2022[/yellow] [bold]{p.model_name}[/bold]: "
                            f"{p.current_quant} \u2192 {p.recommended_quant} ({p.reason})"
                            for p in actionable
                        )
                        + "\n\n[dim]Run [bold]ollama-optimizer optimize[/bold] to apply.[/dim]",
                        title=f"[bold]Recommendations ({len(actionable)})[/bold]",
                        border_style="yellow",
                    )
                )
            else:
                console.print(
                    Panel(
                        "[bold green]All models are optimally configured for your hardware.[/bold green]",
                        title="Recommendations",
                        border_style="green",
                    )
                )

            # Environment variable recommendations
            env_config = optimizer.recommend_environment()
            if env_config.to_env_dict():
                console.print("\n[bold]Server-Level Optimizations[/bold]\n")
                _display_env_config(env_config)

    except KeyboardInterrupt:
        console.print("\n[dim]Cancelled.[/dim]")
        sys.exit(130)
    except Exception as exc:
        logger.debug("scan failed", exc_info=True)
        console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────
# optimize
# ──────────────────────────────────────────────────────────────────────

@main.command()
@click.option("--model", "-m", default=None, help="Specific model to optimize (default: all)")
@click.option(
    "--priority", "-p",
    type=click.Choice(["quality", "speed", "balanced", "minimum"]),
    default="balanced",
    help="Optimization priority",
)
@click.option("--dry-run", "-d", is_flag=True, help="Show plans without applying")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompts")
@click.option("--benchmark/--no-benchmark", "run_bench", default=True, help="Run benchmarks before and after")
def optimize(model, priority, dry_run, yes, run_bench):
    """Optimize installed Ollama models for your hardware."""
    try:
        console.print(BANNER)
        client = OllamaClient()
        if not _check_ollama(client):
            sys.exit(1)

        with console.status("[bold yellow]Detecting hardware...[/bold yellow]"):
            profile = detect_system()
        _display_system(profile)

        with console.status("[bold yellow]Fetching models...[/bold yellow]"):
            models = client.list_models()
        if not models:
            console.print("[yellow]No models installed.[/yellow]")
            sys.exit(0)

        if model:
            models = [m for m in models if m.name.startswith(model) or m.full_name.startswith(model)]
            if not models:
                console.print(f"[bold red]Error:[/bold red] Model '{model}' not found.")
                sys.exit(1)

        # Filter out embedding-only models (they don't support generate)
        embed_models = [m for m in models if m.is_embedding_model]
        models = [m for m in models if not m.is_embedding_model]
        if embed_models:
            names = ", ".join(m.full_name for m in embed_models)
            console.print(f"[dim]Skipping embedding model(s): {names}[/dim]")
        if not models:
            console.print("[yellow]No generative models to optimize.[/yellow]")
            sys.exit(0)

        console.print(f"[green]\u2713[/green] {len(models)} model(s) to optimize\n")

        optimizer = ModelOptimizer(client=client, system=profile, priority=priority)

        analyses = [optimizer.analyze_model(m) for m in models]
        plans = [optimizer.create_optimization_plan(a) for a in analyses]
        actionable = [p for p in plans if p.action != "already_optimal"]

        if not actionable:
            console.print("[bold green]All models are already optimal![/bold green]")
            sys.exit(0)

        _display_plans(actionable)

        if dry_run:
            console.print("[dim]Dry run complete. No changes applied.[/dim]")
            sys.exit(0)

        if not yes:
            if not Confirm.ask(f"Apply optimizations to {len(actionable)} model(s)?"):
                console.print("[dim]Cancelled.[/dim]")
                sys.exit(0)

        # ── Pre-optimization benchmarks ─────────────────────────────
        pre_bench = {}
        # Use a generous timeout — large models (70B+) can take minutes to load
        runner = BenchmarkRunner(client, timeout=600)
        if run_bench:
            console.print("\n[bold]Pre-optimization benchmarks[/bold]\n")
            for p in actionable:
                with console.status(f"[yellow]Benchmarking {p.model_name}...[/yellow]"):
                    suite = runner.run_benchmark_suite(p.model_name, runs_per_prompt=1)
                    agg = _aggregate_bench(suite)
                    if agg:
                        pre_bench[p.model_name] = agg
                        console.print(f"  [green]\u2713[/green] {p.model_name}: {agg.tokens_per_second:.1f} tok/s")
            if pre_bench:
                _display_bench_results(list(pre_bench.values()), "Pre-Optimization")

        # ── Apply ───────────────────────────────────────────────────
        console.print("[bold]Applying optimizations...[/bold]\n")
        results = []
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(),
                       TaskProgressColumn(), console=console) as prog:
            task = prog.add_task("Optimizing", total=len(actionable))
            for p in actionable:
                prog.update(task, description=f"Optimizing [cyan]{p.model_name}[/cyan]")
                result = optimizer.apply_optimization(p)
                results.append(result)
                if result.success:
                    console.print(f"  [green]\u2713[/green] {p.model_name} \u2192 {result.new_model_tag}")
                else:
                    console.print(f"  [red]\u2718[/red] {p.model_name}: {result.error}")
                prog.advance(task)

        console.print()

        # ── Post-optimization benchmarks ────────────────────────────
        post_bench = {}
        if run_bench:
            successful = [r for r in results if r.success]
            if successful:
                console.print("[bold]Post-optimization benchmarks[/bold]\n")
                for r in successful:
                    with console.status(f"[yellow]Benchmarking {r.new_model_tag}...[/yellow]"):
                        suite = runner.run_benchmark_suite(r.new_model_tag, runs_per_prompt=1)
                        agg = _aggregate_bench(suite)
                        if agg:
                            post_bench[r.plan.model_name] = agg
                            console.print(f"  [green]\u2713[/green] {r.new_model_tag}: {agg.tokens_per_second:.1f} tok/s")
                if post_bench:
                    _display_bench_results(list(post_bench.values()), "Post-Optimization")

        # ── Comparison ──────────────────────────────────────────────
        if pre_bench and post_bench:
            console.print("[bold]Performance Comparison[/bold]\n")
            comp_table = Table(title="Before vs After", show_lines=True)
            comp_table.add_column("Model", style="bold cyan")
            comp_table.add_column("Before (tok/s)", justify="right")
            comp_table.add_column("After (tok/s)", justify="right")
            comp_table.add_column("Speedup", justify="right")
            comp_table.add_column("Memory Before", justify="right")
            comp_table.add_column("Memory After", justify="right")

            for name in pre_bench:
                if name in post_bench:
                    pre = pre_bench[name]
                    post = post_bench[name]
                    ratio = post.tokens_per_second / pre.tokens_per_second if pre.tokens_per_second > 0 else 0
                    r_style = "green" if ratio > 1 else "red"
                    comp_table.add_row(
                        name,
                        f"{pre.tokens_per_second:.1f}",
                        f"{post.tokens_per_second:.1f}",
                        f"[{r_style}]{ratio:.2f}x[/{r_style}]",
                        f"{pre.memory_usage_mb:.0f} MB",
                        f"{post.memory_usage_mb:.0f} MB",
                    )
            console.print(comp_table)
            console.print()

        # ── Summary ─────────────────────────────────────────────────
        ok = sum(1 for r in results if r.success)
        fail = sum(1 for r in results if not r.success)
        lines = [f"[green]\u2713[/green] {ok} model(s) optimized", f"[dim]Priority: {priority}[/dim]"]
        if fail:
            lines.append(f"[red]\u2718 {fail} failed[/red]")
        console.print(Panel("\n".join(lines), title="[bold]Done[/bold]", border_style="green"))

        # ── Environment recommendations ─────────────────────────────
        env_config = optimizer.recommend_environment()
        if env_config.to_env_dict():
            console.print("\n[bold]Additional: Server-Level Optimizations[/bold]")
            console.print("[dim]These environment variables can provide further speedups.[/dim]\n")
            _display_env_config(env_config)

    except KeyboardInterrupt:
        console.print("\n[dim]Cancelled.[/dim]")
        sys.exit(130)
    except Exception as exc:
        logger.debug("optimize failed", exc_info=True)
        console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────
# benchmark
# ──────────────────────────────────────────────────────────────────────

@main.command("benchmark")
@click.option("--model", "-m", default=None, help="Model to benchmark (default: all)")
@click.option("--compare", "-c", default=None, help="Second model for comparison")
@click.option("--runs", "-r", default=2, help="Runs per prompt")
@click.option("--save", "-s", is_flag=True, help="Save results to file")
def benchmark_cmd(model, compare, runs, save):
    """Run performance benchmarks on Ollama models."""
    try:
        console.print(BANNER)
        client = OllamaClient()
        if not _check_ollama(client):
            sys.exit(1)

        runner = BenchmarkRunner(client, timeout=600)

        if compare:
            if not model:
                console.print("[bold red]Error:[/bold red] --compare requires --model.")
                sys.exit(1)
            console.print(f"[bold]Comparing [cyan]{model}[/cyan] vs [cyan]{compare}[/cyan][/bold]\n")

            comp = runner.compare_models(model, compare)
            _display_comparison(comp)

        elif model:
            console.print(f"[bold]Benchmarking [cyan]{model}[/cyan] ({runs} runs per prompt)[/bold]\n")
            with console.status(f"[yellow]Running benchmarks...[/yellow]"):
                suite = runner.run_benchmark_suite(model, runs_per_prompt=runs)
            _display_bench_results(suite)
            agg = _aggregate_bench(suite)
            if agg:
                console.print(
                    Panel(
                        f"[bold]Average:[/bold] {agg.tokens_per_second:.1f} tok/s | "
                        f"TTFT: {agg.time_to_first_token_ms:.0f}ms | "
                        f"Memory: {agg.memory_usage_mb:.0f} MB",
                        border_style="cyan",
                    )
                )

        else:
            with console.status("[yellow]Fetching models...[/yellow]"):
                models = client.list_models()
            if not models:
                console.print("[yellow]No models installed.[/yellow]")
                sys.exit(0)

            # Filter out embedding-only models (they don't support generate)
            embed_models = [m for m in models if m.is_embedding_model]
            models = [m for m in models if not m.is_embedding_model]
            if embed_models:
                names = ", ".join(m.full_name for m in embed_models)
                console.print(f"[dim]Skipping embedding model(s): {names}[/dim]")
            if not models:
                console.print("[yellow]No generative models to benchmark.[/yellow]")
                sys.exit(0)

            console.print(f"[bold]Benchmarking {len(models)} model(s)[/bold]\n")
            all_results = []
            for m in models:
                name = m.full_name
                with console.status(f"[yellow]Benchmarking {name}...[/yellow]"):
                    suite = runner.run_benchmark_suite(name, runs_per_prompt=runs)
                agg = _aggregate_bench(suite)
                if agg:
                    all_results.append(agg)
                    console.print(f"  [green]\u2713[/green] {name}: {agg.tokens_per_second:.1f} tok/s")
            console.print()
            if all_results:
                _display_bench_results(all_results, "All Models")

        if save:
            console.print("[dim]Saving results...[/dim]")
            suite_obj = runner.run_full_benchmark([])
            path = runner.save_results(suite_obj)
            console.print(f"[green]\u2713[/green] Saved to {path}")

    except KeyboardInterrupt:
        console.print("\n[dim]Cancelled.[/dim]")
        sys.exit(130)
    except Exception as exc:
        logger.debug("benchmark failed", exc_info=True)
        console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────
# report
# ──────────────────────────────────────────────────────────────────────

@main.command()
@click.option("--file", "-f", "filepath", default=None, help="Load results from JSON file")
def report(filepath):
    """Show a saved benchmark report."""
    try:
        console.print(BANNER)
        runner = BenchmarkRunner()

        if filepath:
            if not os.path.exists(filepath):
                console.print(f"[bold red]Error:[/bold red] File not found: {filepath}")
                sys.exit(1)
            suite = runner.load_results(filepath)
        else:
            default_dir = os.path.expanduser("~/.ollama-optimizer/benchmarks")
            if not os.path.isdir(default_dir):
                console.print("[yellow]No saved benchmarks found.[/yellow]")
                console.print("[dim]Run [bold]ollama-optimizer benchmark --save[/bold] first.[/dim]")
                sys.exit(0)
            files = sorted(
                [os.path.join(default_dir, f) for f in os.listdir(default_dir) if f.endswith(".json")],
                key=os.path.getmtime,
                reverse=True,
            )
            if not files:
                console.print("[yellow]No saved benchmarks found.[/yellow]")
                sys.exit(0)
            suite = runner.load_results(files[0])
            console.print(f"[dim]Loading: {files[0]}[/dim]\n")

        if suite.results:
            _display_bench_results(suite.results, "Benchmark Results")
        for comp in suite.comparisons:
            _display_comparison(comp)

        console.print(f"[dim]Total duration: {suite.total_duration_seconds:.1f}s | {suite.timestamp}[/dim]")

    except KeyboardInterrupt:
        console.print("\n[dim]Cancelled.[/dim]")
        sys.exit(130)
    except Exception as exc:
        logger.debug("report failed", exc_info=True)
        console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────
# explain
# ──────────────────────────────────────────────────────────────────────

@main.command()
def explain():
    """Learn about quantization techniques and GGUF formats."""
    try:
        console.print(BANNER)

        console.print(
            Panel(
                "[bold]What is Quantization?[/bold]\n\n"
                "LLM parameters are stored as floating-point numbers. A 70B model at FP16\n"
                "needs ~140 GB RAM. Quantization compresses these numbers to lower precision\n"
                "(INT8, INT4) trading a small accuracy loss for massive memory + speed gains.\n\n"
                "[bold]Key techniques (from the ngrok blog):[/bold]\n\n"
                "[cyan]Absmax (Symmetric):[/cyan] scale = max(|x|) / 127\n"
                "  Simple, maps values symmetrically around zero. Used in Q8_0.\n\n"
                "[cyan]Zero-Point (Asymmetric):[/cyan] adds offset for skewed distributions\n"
                "  Better range utilization. Used in Q4_1.\n\n"
                "[cyan]Per-Group:[/cyan] separate scale per group of 32-128 weights\n"
                "  What K-quants (Q4_K_M, Q5_K_M, etc.) use internally.\n"
                "  Each group has its own scale factor for higher accuracy.\n\n"
                "[cyan]Perplexity:[/cyan] standard metric for quality loss measurement\n"
                "  Lower = better. INT8 adds <0.1 perplexity; INT4 with GPTQ adds ~0.1-0.5",
                title="[bold cyan]Quantization from the Ground Up[/bold cyan]",
                subtitle="[link=https://ngrok.com/blog/quantization]ngrok.com/blog/quantization[/link]",
                border_style="cyan",
            )
        )

        # GGUF levels table
        console.print("\n[bold]GGUF Quantization Levels[/bold]\n")
        table = Table(show_lines=True)
        table.add_column("Level", style="bold cyan", min_width=8)
        table.add_column("Bits/W", justify="right")
        table.add_column("Quality", justify="center")
        table.add_column("Speed", justify="center")
        table.add_column("Memory", justify="center")
        table.add_column("Description", min_width=30)
        table.add_column("Best For")

        for name, q in GGUF_QUANT_LEVELS.items():
            qs = _score_style(q.quality_score)
            table.add_row(
                name,
                f"{q.bits_per_weight:.2f}",
                f"[{qs}]{q.quality_score:.0%}[/{qs}]",
                f"{q.speed_multiplier:.1f}x",
                f"{q.memory_multiplier:.0%}",
                q.description,
                q.recommended_for,
            )

        console.print(table)

        # Live quantization demo
        console.print("\n[bold]Live Quantization Demo[/bold]\n")
        console.print("[dim]Running quantization on synthetic LLM-like weight distribution...[/dim]\n")

        engine = QuantizationEngine()
        demo = engine.demonstrate_quantization()

        demo_table = Table(title="Error Comparison Across Methods", show_lines=True)
        demo_table.add_column("Method", style="bold cyan")
        demo_table.add_column("Bits", justify="center")
        demo_table.add_column("MSE", justify="right")
        demo_table.add_column("Max Error", justify="right")
        demo_table.add_column("SNR (dB)", justify="right")

        for method_name, data in demo.items():
            if isinstance(data, dict) and "error" in data:
                err = data["error"]
                demo_table.add_row(
                    method_name,
                    str(data.get("bits", "?")),
                    f"{err.get('mse', 0):.6f}",
                    f"{err.get('max_error', 0):.6f}",
                    f"{err.get('snr_db', 0):.1f}",
                )

        console.print(demo_table)
        console.print()

    except KeyboardInterrupt:
        console.print("\n[dim]Cancelled.[/dim]")
        sys.exit(130)
    except Exception as exc:
        logger.debug("explain failed", exc_info=True)
        console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────
# status
# ──────────────────────────────────────────────────────────────────────

@main.command()
def status():
    """Check Ollama status and list running/installed models."""
    try:
        console.print(BANNER)
        client = OllamaClient()
        if not _check_ollama(client):
            sys.exit(1)

        running = client.get_running_models()
        if running:
            table = Table(title="Running Models")
            table.add_column("Model", style="bold cyan")
            table.add_column("Size", justify="right")
            table.add_column("Processor", justify="center")
            table.add_column("Until", justify="right", style="dim")
            for rm in running:
                table.add_row(
                    str(rm.get("name", "?")),
                    str(rm.get("size", "?")),
                    str(rm.get("size_vram", "?")),
                    str(rm.get("expires_at", "?")),
                )
            console.print(table)
        else:
            console.print("[dim]No models currently loaded in memory.[/dim]")

        console.print()
        models = client.list_models()
        console.print(f"[bold]Installed models: {len(models)}[/bold]")
        for m in models:
            console.print(
                f"  [cyan]\u2022[/cyan] {m.full_name} "
                f"[dim]({_fmt_size(m.size_gb)} | {m.quantization_level or '?'} | {m.parameter_size})[/dim]"
            )

    except KeyboardInterrupt:
        console.print("\n[dim]Cancelled.[/dim]")
        sys.exit(130)
    except Exception as exc:
        logger.debug("status failed", exc_info=True)
        console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────
# env-config
# ──────────────────────────────────────────────────────────────────────

@main.command("env-config")
@click.option(
    "--format", "-f", "output_format",
    type=click.Choice(["shell", "systemd", "launchd", "json"]),
    default="shell",
    help="Output format for environment configuration",
)
@click.option("--save", "-s", is_flag=True, help="Save config to file")
def env_config_cmd(output_format, save):
    """Generate optimized Ollama server environment configuration.

    Analyzes your hardware and recommends server-level environment variables
    for maximum performance. These settings often provide the biggest speedups
    (Flash Attention alone gives 1.5-3x improvement).
    """
    try:
        console.print(BANNER)

        with console.status("[bold yellow]Detecting hardware...[/bold yellow]"):
            profile = detect_system()
        _display_system(profile)

        optimizer = ModelOptimizer(system=profile)
        env_config = optimizer.recommend_environment()

        env_dict = env_config.to_env_dict()
        if not env_dict:
            console.print(
                Panel(
                    "[bold green]Default Ollama settings are already optimal "
                    "for your hardware.[/bold green]",
                    title="Environment Config",
                    border_style="green",
                )
            )
            sys.exit(0)

        if output_format == "json":
            output = {
                "environment": env_dict,
                "reasons": env_config.reasons,
            }
            console.print_json(json.dumps(output, default=str))
        elif output_format == "systemd":
            config_text = env_config.to_systemd_override()
            console.print(Panel(
                config_text,
                title="[bold]Systemd Override Config[/bold]",
                subtitle="Save to /etc/systemd/system/ollama.service.d/override.conf",
                border_style="yellow",
            ))
            if save:
                out_path = os.path.expanduser("~/.ollama-optimizer/ollama-override.conf")
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                with open(out_path, "w") as f:
                    f.write(config_text + "\n")
                console.print(f"\n[green]\u2713[/green] Saved to {out_path}")
                console.print("[dim]Copy to /etc/systemd/system/ollama.service.d/override.conf[/dim]")
                console.print("[dim]Then: sudo systemctl daemon-reload && sudo systemctl restart ollama[/dim]")
        elif output_format == "launchd":
            config_text = env_config.to_launchd_plist_fragment()
            console.print(Panel(
                config_text,
                title="[bold]macOS launchd Config[/bold]",
                subtitle="Add to Ollama's plist",
                border_style="yellow",
            ))
            if save:
                out_path = os.path.expanduser("~/.ollama-optimizer/ollama-launchd.plist")
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                with open(out_path, "w") as f:
                    f.write(config_text + "\n")
                console.print(f"\n[green]\u2713[/green] Saved to {out_path}")
        else:
            # Default: shell exports
            _display_env_config(env_config)
            if save:
                out_path = os.path.expanduser("~/.ollama-optimizer/ollama-env.sh")
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                with open(out_path, "w") as f:
                    f.write(env_config.to_shell_exports() + "\n")
                console.print(f"[green]\u2713[/green] Saved to {out_path}")
                console.print("[dim]Source it: source ~/.ollama-optimizer/ollama-env.sh[/dim]")

    except KeyboardInterrupt:
        console.print("\n[dim]Cancelled.[/dim]")
        sys.exit(130)
    except Exception as exc:
        logger.debug("env-config failed", exc_info=True)
        console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
