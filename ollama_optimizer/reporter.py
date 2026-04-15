"""
Reporter module for ollama-optimizer.

Generates beautiful terminal reports using the `rich` library, showing
before/after comparisons for Ollama model optimization. Covers system
profiling, model analysis, optimization plans, benchmarks, and
quantization education.
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich.columns import Columns
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)
from rich.tree import Tree
from rich.rule import Rule
from rich import box

import dataclasses
import typing
import math
import json
import os
import pathlib


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(value, default: float = 0.0) -> float:
    """Safely convert a value to float, returning *default* on failure."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _fmt_size(gb: float) -> str:
    """Format a size in GB for display."""
    if gb is None:
        return "N/A"
    if gb < 1:
        return f"{gb * 1024:.0f} MB"
    return f"{gb:.2f} GB"


def _delta_color(value: float, higher_is_better: bool = True) -> str:
    """Return a rich color name based on whether the delta is good or bad."""
    if value is None:
        return "white"
    if higher_is_better:
        if value > 0:
            return "green"
        elif value < 0:
            return "red"
        return "yellow"
    else:
        if value < 0:
            return "green"
        elif value > 0:
            return "red"
        return "yellow"


def _arrow(value: float, higher_is_better: bool = True) -> str:
    """Return a colored arrow string showing improvement/regression."""
    if value is None or value == 0:
        return "[yellow]\u2194[/yellow]"  # left-right arrow
    if (higher_is_better and value > 0) or (not higher_is_better and value < 0):
        return "[green]\u2191[/green]"  # up arrow
    return "[red]\u2193[/red]"  # down arrow


def _pct_str(value: float | None, signed: bool = True) -> str:
    """Format a percentage value with optional sign."""
    if value is None:
        return "N/A"
    sign = "+" if signed and value > 0 else ""
    return f"{sign}{value:.1f}%"


def _score_color(score: float | None, thresholds: tuple[float, float] = (0.5, 0.75)) -> str:
    """Return a color based on score thresholds (low, high)."""
    if score is None:
        return "white"
    low, high = thresholds
    if score >= high:
        return "green"
    elif score >= low:
        return "yellow"
    return "red"


def _bool_icon(val: bool | None) -> str:
    """Return a colored check/cross for booleans."""
    if val is None:
        return "[dim]?[/dim]"
    return "[green]\u2714[/green]" if val else "[red]\u2718[/red]"


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------

class Reporter:
    """Generates rich terminal output for every stage of the optimization
    pipeline: system profiling, model analysis, optimization planning,
    benchmarking, and comparison."""

    def __init__(self):
        self.console = Console()

    # ------------------------------------------------------------------ 1
    def print_banner(self):
        """Print a styled ASCII art banner for ollama-optimizer."""
        ascii_art = (
            "   ___  _ _                        ___        _   _           _              \n"
            "  / _ \\| | |__ _ _ __  __ _ ___   / _ \\ _ __ | |_(_)_ __ ___ (_)_______ _ __ \n"
            " | | | | | / _` | '_ \\/  _` |___| | | | '_ \\| __| | '_ ` _ \\| |_  / _ \\ '__|\n"
            " | |_| | | | (_| | | | | (_| |___| |_| | |_) | |_| | | | | | | |/ /  __/ |   \n"
            "  \\___/|_|_|\\__,_|_| |_|\\__,_|    \\___/| .__/ \\__|_|_| |_| |_|_/___\\___|_|   \n"
            "                                        |_|                                    "
        )
        banner_text = Text(ascii_art, style="bold cyan")
        panel = Panel(
            banner_text,
            title="[bold white]v1.0.0[/bold white]",
            subtitle="[italic]Quantization-Powered Model Optimization[/italic]",
            border_style="bright_blue",
            box=box.DOUBLE,
            padding=(1, 2),
        )
        self.console.print(panel)

    # ------------------------------------------------------------------ 2
    def print_system_info(self, profile):
        """Print a rich table with system hardware information.

        Parameters
        ----------
        profile : SystemProfile
            Dataclass with fields: os_type, cpu_name, cpu_cores, cpu_threads,
            total_ram_gb, available_ram_gb, gpu_name, gpu_vram_gb, disk_free_gb.
        """
        if profile is None:
            self.console.print("[dim]No system profile available.[/dim]")
            return

        table = Table(
            title="\U0001f5a5  System Hardware Profile",
            box=box.ROUNDED,
            title_style="bold bright_white",
            border_style="bright_blue",
            show_lines=True,
            padding=(0, 1),
        )
        table.add_column("Component", style="bold cyan", min_width=14)
        table.add_column("Detail", style="white", min_width=30)
        table.add_column("Status", justify="center", min_width=10)

        # OS
        table.add_row(
            "\U0001f4bb OS",
            str(getattr(profile, "os_type", "Unknown")),
            "",
        )

        # CPU
        cpu_name = getattr(profile, "cpu_name", "Unknown")
        cpu_cores = getattr(profile, "cpu_cores", "?")
        cpu_threads = getattr(profile, "cpu_threads", "?")
        table.add_row(
            "\u2699\ufe0f  CPU",
            f"{cpu_name}",
            f"{cpu_cores}C / {cpu_threads}T",
        )

        # RAM
        total_ram = _safe_float(getattr(profile, "total_ram_gb", None))
        avail_ram = _safe_float(getattr(profile, "available_ram_gb", None))
        if total_ram > 32:
            ram_color = "green"
        elif total_ram > 16:
            ram_color = "yellow"
        else:
            ram_color = "red"
        ram_detail = f"{total_ram:.1f} GB total / {avail_ram:.1f} GB available"
        table.add_row(
            "\U0001f9e0 RAM",
            f"[{ram_color}]{ram_detail}[/{ram_color}]",
            "[{c}]{icon}[/{c}]".format(c=ram_color, icon="\u2714" if total_ram > 16 else "\u26a0"),
        )

        # GPU
        gpu_name = getattr(profile, "gpu_name", None)
        gpu_vram = _safe_float(getattr(profile, "gpu_vram_gb", None))
        if gpu_name:
            gpu_detail = f"{gpu_name} ({gpu_vram:.1f} GB VRAM)" if gpu_vram else str(gpu_name)
            table.add_row("\U0001f3ae GPU", gpu_detail, "[green]\u2714[/green]")
        else:
            table.add_row("\U0001f3ae GPU", "[dim]Not detected[/dim]", "[yellow]CPU-only[/yellow]")

        # Flash Attention support
        flash_attn = getattr(profile, "gpu_supports_flash_attn", None)
        compute_cap = _safe_float(getattr(profile, "gpu_compute_capability", None))
        if flash_attn is not None:
            if flash_attn:
                fa_detail = f"[green]Supported[/green] (compute capability {compute_cap:.1f})"
            elif compute_cap > 0:
                fa_detail = f"[yellow]Not supported[/yellow] (compute capability {compute_cap:.1f} < 7.0)"
            else:
                fa_detail = "[dim]Unknown (no NVIDIA GPU detected)[/dim]"
            table.add_row("\u26a1 Flash Attn", fa_detail, _bool_icon(flash_attn))
        
        # NUMA
        numa_avail = getattr(profile, "numa_available", None)
        numa_nodes = getattr(profile, "numa_node_count", 1)
        if numa_avail is not None and numa_avail:
            table.add_row(
                "\U0001f9f1 NUMA",
                f"[cyan]{numa_nodes} nodes[/cyan]",
                "[green]\u2714[/green]",
            )

        # Disk
        disk_free = _safe_float(getattr(profile, "disk_free_gb", None))
        disk_color = "green" if disk_free > 50 else ("yellow" if disk_free > 20 else "red")
        table.add_row(
            "\U0001f4be Disk",
            f"[{disk_color}]{disk_free:.1f} GB free[/{disk_color}]",
            "[{c}]{icon}[/{c}]".format(c=disk_color, icon="\u2714" if disk_free > 20 else "\u26a0"),
        )

        self.console.print()
        self.console.print(table)
        self.console.print()

    # ------------------------------------------------------------------ 3
    def print_model_analysis(self, analyses: list):
        """Print a table showing all installed models and their current status.

        Parameters
        ----------
        analyses : list[ModelAnalysis]
        """
        if not analyses:
            self.console.print("[dim]No models to analyse.[/dim]")
            return

        table = Table(
            title="\U0001f50d Installed Model Analysis",
            box=box.ROUNDED,
            title_style="bold bright_white",
            border_style="bright_blue",
            show_lines=True,
            padding=(0, 1),
        )
        table.add_column("Model Name", style="bold cyan", max_width=30)
        table.add_column("Family", style="magenta")
        table.add_column("Parameters", justify="right")
        table.add_column("Quantization", justify="center")
        table.add_column("Size (GB)", justify="right")
        table.add_column("Quality", justify="center")
        table.add_column("Speed", justify="center")
        table.add_column("RAM", justify="center")
        table.add_column("VRAM", justify="center")

        for a in analyses:
            model = getattr(a, "model", "?")
            family = getattr(a, "family", "?") or "?"
            param_b = getattr(a, "parameter_billions", None)
            param_count = getattr(a, "parameter_count", None)
            params_str = f"{param_b:.1f}B" if param_b else (f"{param_count:,}" if param_count else "?")
            quant = getattr(a, "current_quant", "?") or "?"
            size_gb = _safe_float(getattr(a, "current_size_gb", None))
            quality = _safe_float(getattr(a, "current_quality_score", None))
            speed = _safe_float(getattr(a, "current_speed_multiplier", None))
            fits_ram = getattr(a, "fits_in_ram", None)
            fits_vram = getattr(a, "fits_in_vram", None)

            q_color = _score_color(quality, (0.5, 0.75))
            s_color = _score_color(speed, (0.8, 1.2))

            table.add_row(
                model,
                family,
                params_str,
                quant,
                f"{size_gb:.2f}",
                f"[{q_color}]{quality:.2f}[/{q_color}]",
                f"[{s_color}]{speed:.2f}x[/{s_color}]",
                _bool_icon(fits_ram),
                _bool_icon(fits_vram),
            )

        self.console.print()
        self.console.print(table)
        self.console.print()

    # ------------------------------------------------------------------ 4
    def print_optimization_plans(self, plans: list):
        """Print detailed optimization plans for each model.

        Parameters
        ----------
        plans : list[OptimizationPlan]
        """
        if not plans:
            self.console.print("[dim]No optimization plans to display.[/dim]")
            return

        self.console.print()
        self.console.print(
            Rule(
                "\U0001f680 Optimization Plans",
                style="bright_blue",
            )
        )
        self.console.print()

        for plan in plans:
            model_name = getattr(plan, "model_name", "Unknown")
            current_q = getattr(plan, "current_quant", "?")
            rec_q = getattr(plan, "recommended_quant", "?")
            cur_size = _safe_float(getattr(plan, "current_size_gb", None))
            opt_size = _safe_float(getattr(plan, "optimized_size_gb", None))
            size_red = _safe_float(getattr(plan, "size_reduction_pct", None))
            quality_d = _safe_float(getattr(plan, "quality_delta", None))
            speed_imp = _safe_float(getattr(plan, "speed_improvement", None))
            gpu_layers = getattr(plan, "gpu_layers", None)
            num_threads = getattr(plan, "num_threads", None)
            ctx_size = getattr(plan, "context_size", None)
            batch_size = getattr(plan, "batch_size", None)
            rec_opts = getattr(plan, "recommended_options", None) or {}
            action = getattr(plan, "action", "?")
            reason = getattr(plan, "reason", "")

            # Build a tree view for the plan
            tree = Tree(
                f"[bold bright_white]\U0001f4e6 {model_name}[/bold bright_white]",
                guide_style="bright_blue",
            )

            # Quantization change
            quant_branch = tree.add("[bold]Quantization[/bold]")
            if current_q != rec_q:
                quant_branch.add(
                    f"[yellow]{current_q}[/yellow] \u2192 [green]{rec_q}[/green]"
                )
            else:
                quant_branch.add(f"[green]{current_q}[/green] (no change)")

            # Size
            size_branch = tree.add("[bold]Size[/bold]")
            size_color = "green" if size_red > 0 else "yellow"
            size_branch.add(
                f"{_fmt_size(cur_size)} \u2192 [{size_color}]{_fmt_size(opt_size)}[/{size_color}]  "
                f"([{size_color}]{_pct_str(-size_red, signed=True)} reduction[/{size_color}])"
            )

            # Performance
            perf_branch = tree.add("[bold]Performance[/bold]")
            spd_color = _delta_color(speed_imp, higher_is_better=True)
            perf_branch.add(
                f"Speed: [{spd_color}]{_pct_str(speed_imp)}[/{spd_color}] "
                f"{_arrow(speed_imp)}"
            )
            qd_color = _delta_color(quality_d, higher_is_better=True)
            perf_branch.add(
                f"Quality: [{qd_color}]{_pct_str(quality_d)}[/{qd_color}] "
                f"{_arrow(quality_d)}"
            )

            # Runtime parameters
            params_branch = tree.add("[bold]Runtime Parameters[/bold]")
            if gpu_layers is not None:
                params_branch.add(f"GPU layers: [cyan]{gpu_layers}[/cyan]")
            if num_threads is not None:
                params_branch.add(f"Threads: [cyan]{num_threads}[/cyan]")
            if ctx_size is not None:
                params_branch.add(f"Context size: [cyan]{ctx_size}[/cyan]")
            if batch_size is not None:
                params_branch.add(f"Batch size: [cyan]{batch_size}[/cyan]")
            if rec_opts:
                for k, v in rec_opts.items():
                    params_branch.add(f"{k}: [cyan]{v}[/cyan]")

            # Action & reason
            action_branch = tree.add("[bold]Action[/bold]")
            action_branch.add(f"[bold bright_yellow]{action}[/bold bright_yellow]")
            if reason:
                action_branch.add(f"[dim italic]{reason}[/dim italic]")

            panel = Panel(
                tree,
                border_style="bright_blue",
                box=box.ROUNDED,
                padding=(0, 1),
            )
            self.console.print(panel)

    # ------------------------------------------------------------------ 4b
    def print_env_config(self, env_config):
        """Print recommended Ollama environment variable configuration.
        
        Parameters
        ----------
        env_config : EnvironmentConfig
            Dataclass with fields: flash_attention, kv_cache_type, keep_alive,
            num_parallel, max_loaded_models, sched_spread, reasons, and methods
            to_env_dict(), to_shell_exports(), to_systemd_override().
        """
        if env_config is None:
            self.console.print("[dim]No environment configuration available.[/dim]")
            return
        
        self.console.print()
        self.console.print(
            Rule(
                "\u2699\ufe0f  Server Environment Recommendations",
                style="bright_green",
            )
        )
        self.console.print()
        
        env_dict = env_config.to_env_dict()
        reasons = getattr(env_config, 'reasons', {}) or {}
        
        if not env_dict:
            self.console.print(
                Panel(
                    "[green]Default Ollama settings are already optimal for your system.[/green]",
                    border_style="green",
                )
            )
            return
        
        # Impact table
        table = Table(
            title="Recommended Environment Variables",
            box=box.ROUNDED,
            title_style="bold bright_white",
            border_style="bright_green",
            show_lines=True,
            padding=(0, 1),
        )
        table.add_column("Variable", style="bold cyan", min_width=28)
        table.add_column("Value", style="bold green", min_width=10)
        table.add_column("Impact", style="white", min_width=40)
        
        # Ordered by estimated impact
        impact_order = [
            "OLLAMA_FLASH_ATTENTION",
            "OLLAMA_KV_CACHE_TYPE",
            "OLLAMA_KEEP_ALIVE",
            "OLLAMA_NUM_PARALLEL",
            "OLLAMA_MAX_LOADED_MODELS",
            "OLLAMA_SCHED_SPREAD",
            "OLLAMA_RUNNERS_DIR",
        ]
        
        for var_name in impact_order:
            if var_name in env_dict:
                value = env_dict[var_name]
                reason = reasons.get(var_name, "")
                table.add_row(var_name, str(value), reason)
        
        # Any remaining vars not in the ordered list
        for var_name, value in env_dict.items():
            if var_name not in impact_order:
                reason = reasons.get(var_name, "")
                table.add_row(var_name, str(value), reason)
        
        # Show NUMA info note if present (not an env var, just advisory)
        numa_info = reasons.get("_NUMA_INFO", "")
        if numa_info:
            table.add_row(
                "[dim]NUMA (no env var)[/dim]",
                "[dim]N/A[/dim]",
                f"[dim]{numa_info}[/dim]",
            )
        
        self.console.print(table)
        self.console.print()
        
        # Shell export block
        shell_exports = env_config.to_shell_exports()
        self.console.print(
            Panel(
                shell_exports,
                title="[bold]Shell Exports[/bold] (add to ~/.bashrc or ~/.zshrc)",
                border_style="bright_blue",
                padding=(1, 2),
            )
        )
        
        # Systemd config
        systemd_config = env_config.to_systemd_override()
        self.console.print(
            Panel(
                systemd_config,
                title="[bold]Systemd Override[/bold] (Linux service)",
                border_style="yellow",
                padding=(1, 2),
            )
        )
        self.console.print()

    # ------------------------------------------------------------------ 5
    def print_benchmark_results(self, results: list):
        """Print benchmark results for a set of model runs.

        Parameters
        ----------
        results : list[BenchmarkResult]
        """
        if not results:
            self.console.print("[dim]No benchmark results to display.[/dim]")
            return

        table = Table(
            title="\u23f1  Benchmark Results",
            box=box.ROUNDED,
            title_style="bold bright_white",
            border_style="bright_blue",
            show_lines=True,
            padding=(0, 1),
        )
        table.add_column("Prompt", style="italic", max_width=40, overflow="ellipsis")
        table.add_column("Tokens/sec", justify="right", style="bold")
        table.add_column("TTFT (ms)", justify="right")
        table.add_column("Total Time (s)", justify="right")
        table.add_column("Memory (MB)", justify="right")

        for r in results:
            prompt = getattr(r, "prompt_used", "") or ""
            tps = _safe_float(getattr(r, "tokens_per_second", None))
            ttft = _safe_float(getattr(r, "time_to_first_token_ms", None))
            total = _safe_float(getattr(r, "total_time_seconds", None))
            mem = _safe_float(getattr(r, "memory_usage_mb", None))

            tps_color = "green" if tps > 30 else ("yellow" if tps > 10 else "red")
            ttft_color = "green" if ttft < 500 else ("yellow" if ttft < 2000 else "red")

            display_prompt = (prompt[:37] + "...") if len(prompt) > 40 else prompt

            table.add_row(
                display_prompt,
                f"[{tps_color}]{tps:.1f}[/{tps_color}]",
                f"[{ttft_color}]{ttft:.0f}[/{ttft_color}]",
                f"{total:.2f}",
                f"{mem:.0f}",
            )

        self.console.print()
        self.console.print(table)
        self.console.print()

    # ------------------------------------------------------------------ 6
    def print_benchmark_comparison(self, comparison):
        """Print side-by-side comparison of original vs optimized model.

        Parameters
        ----------
        comparison : BenchmarkComparison
        """
        if comparison is None:
            self.console.print("[dim]No comparison data available.[/dim]")
            return

        model_name = getattr(comparison, "model_name", "Unknown")
        original = getattr(comparison, "original_result", None)
        optimized = getattr(comparison, "optimized_result", None)
        speedup = _safe_float(getattr(comparison, "speedup_ratio", None))
        mem_red = _safe_float(getattr(comparison, "memory_reduction_pct", None))
        ttft_imp = _safe_float(getattr(comparison, "ttft_improvement_pct", None))
        thr_imp = _safe_float(getattr(comparison, "throughput_improvement_pct", None))
        quality_note = getattr(comparison, "quality_note", "") or ""
        orig_model_name = getattr(comparison, "original_model", "original")
        opt_model_name = getattr(comparison, "optimized_model", "optimized")

        self.console.print()
        self.console.print(
            Rule(
                f"\U0001f4ca  {model_name} \u2014 Before vs After",
                style="bright_blue",
            )
        )

        # Helper to extract stats from a BenchmarkResult
        def _stats(result):
            if result is None:
                return {}
            return {
                "tokens_per_second": _safe_float(getattr(result, "tokens_per_second", None)),
                "ttft_ms": _safe_float(getattr(result, "time_to_first_token_ms", None)),
                "total_time": _safe_float(getattr(result, "total_time_seconds", None)),
                "memory_mb": _safe_float(getattr(result, "memory_usage_mb", None)),
                "load_time": _safe_float(getattr(result, "load_time_seconds", None)),
                "prompt_eval_rate": _safe_float(getattr(result, "prompt_eval_rate", None)),
                "eval_rate": _safe_float(getattr(result, "eval_rate", None)),
            }

        orig_s = _stats(original)
        opt_s = _stats(optimized)

        # Comparison table
        table = Table(
            box=box.ROUNDED,
            border_style="bright_blue",
            show_lines=True,
            padding=(0, 1),
        )
        table.add_column("Metric", style="bold cyan", min_width=20)
        table.add_column(f"Original\n({orig_model_name})", justify="right", style="white", min_width=14)
        table.add_column("Delta", justify="center", min_width=16)
        table.add_column(f"Optimized\n({opt_model_name})", justify="right", style="white", min_width=14)

        metrics = [
            ("Tokens/sec", "tokens_per_second", True, "{:.1f}", ""),
            ("TTFT (ms)", "ttft_ms", False, "{:.0f}", " ms"),
            ("Total time (s)", "total_time", False, "{:.2f}", " s"),
            ("Memory (MB)", "memory_mb", False, "{:.0f}", " MB"),
            ("Load time (s)", "load_time", False, "{:.2f}", " s"),
            ("Prompt eval (t/s)", "prompt_eval_rate", True, "{:.1f}", ""),
            ("Eval rate (t/s)", "eval_rate", True, "{:.1f}", ""),
        ]

        for label, key, higher_better, fmt, suffix in metrics:
            o_val = orig_s.get(key, 0.0)
            n_val = opt_s.get(key, 0.0)
            if o_val == 0 and n_val == 0:
                continue  # skip metrics not present in either

            diff = n_val - o_val
            if o_val != 0:
                diff_pct = (diff / abs(o_val)) * 100
            else:
                diff_pct = 0.0

            d_color = _delta_color(diff, higher_is_better=higher_better)
            arrow = _arrow(diff, higher_is_better=higher_better)

            table.add_row(
                label,
                fmt.format(o_val) + suffix,
                f"[{d_color}]{_pct_str(diff_pct)}[/{d_color}] {arrow}",
                f"[{d_color}]{fmt.format(n_val)}{suffix}[/{d_color}]",
            )

        self.console.print()
        self.console.print(table)

        # Summary strip
        summary_parts = []
        sp_color = "green" if speedup >= 1.0 else "red"
        summary_parts.append(
            f"[{sp_color}]\u26a1 Speedup: {speedup:.2f}x[/{sp_color}]"
        )
        mr_color = "green" if mem_red > 0 else ("yellow" if mem_red == 0 else "red")
        summary_parts.append(
            f"[{mr_color}]\U0001f4be Memory: {_pct_str(-mem_red)} saved[/{mr_color}]"
        )
        tt_color = "green" if ttft_imp > 0 else ("yellow" if ttft_imp == 0 else "red")
        summary_parts.append(
            f"[{tt_color}]\u23f1  TTFT: {_pct_str(ttft_imp)} faster[/{tt_color}]"
        )
        th_color = "green" if thr_imp > 0 else ("yellow" if thr_imp == 0 else "red")
        summary_parts.append(
            f"[{th_color}]\U0001f4c8 Throughput: {_pct_str(thr_imp)}[/{th_color}]"
        )

        summary_text = "  \u2502  ".join(summary_parts)
        if quality_note:
            summary_text += f"\n[dim italic]Quality note: {quality_note}[/dim italic]"

        self.console.print(
            Panel(
                summary_text,
                title="[bold]Summary[/bold]",
                border_style="bright_green",
                box=box.ROUNDED,
                padding=(0, 1),
            )
        )
        self.console.print()

    # ------------------------------------------------------------------ 7
    def print_full_report(self, suite, system_profile=None):
        """Print the complete benchmark report with all comparisons.

        Parameters
        ----------
        suite : BenchmarkSuite
            Dataclass with fields: results, comparisons, system_info,
            timestamp, total_duration_seconds.
        system_profile : SystemProfile, optional
        """
        if suite is None:
            self.console.print("[dim]No benchmark suite data available.[/dim]")
            return

        # Banner
        self.print_banner()

        # System info
        profile = system_profile or getattr(suite, "system_info", None)
        if profile is not None:
            self.print_system_info(profile)

        # Timestamp & duration
        timestamp = getattr(suite, "timestamp", None)
        duration = _safe_float(getattr(suite, "total_duration_seconds", None))
        if timestamp or duration:
            meta_parts: list[str] = []
            if timestamp:
                meta_parts.append(f"\U0001f4c5 {timestamp}")
            if duration:
                mins, secs = divmod(duration, 60)
                meta_parts.append(f"\u23f1  Duration: {int(mins)}m {secs:.1f}s")
            self.console.print(
                Panel(
                    "   ".join(meta_parts),
                    border_style="dim",
                    box=box.ROUNDED,
                )
            )

        # Individual results
        results = getattr(suite, "results", None) or []
        if results:
            self.console.print(
                Rule("\U0001f9ea Individual Benchmark Results", style="bright_blue")
            )
            self.print_benchmark_results(results)

        # Comparisons
        comparisons = getattr(suite, "comparisons", None) or []
        if comparisons:
            self.console.print(
                Rule(
                    "\U0001f504 Before / After Comparisons",
                    style="bright_blue",
                )
            )
            for comp in comparisons:
                self.print_benchmark_comparison(comp)

        # Aggregate summary
        if comparisons:
            self._print_aggregate_summary(comparisons, duration)

    def _print_aggregate_summary(self, comparisons: list, total_duration: float | None = None):
        """Internal helper to print an aggregate summary across all comparisons."""
        speedups: list[float] = []
        mem_savings: list[float] = []
        thr_improvements: list[float] = []

        for c in comparisons:
            sp = _safe_float(getattr(c, "speedup_ratio", None))
            mr = _safe_float(getattr(c, "memory_reduction_pct", None))
            ti = _safe_float(getattr(c, "throughput_improvement_pct", None))
            if sp:
                speedups.append(sp)
            if mr:
                mem_savings.append(mr)
            if ti:
                thr_improvements.append(ti)

        avg_speedup = sum(speedups) / len(speedups) if speedups else 0
        avg_mem = sum(mem_savings) / len(mem_savings) if mem_savings else 0
        avg_thr = sum(thr_improvements) / len(thr_improvements) if thr_improvements else 0

        lines: list[str] = [
            f"[bold]Models compared:[/bold]     {len(comparisons)}",
            f"[bold]Avg speedup:[/bold]         [green]{avg_speedup:.2f}x[/green]",
            f"[bold]Avg memory saved:[/bold]    [green]{avg_mem:.1f}%[/green]",
            f"[bold]Avg throughput gain:[/bold] [green]{_pct_str(avg_thr)}[/green]",
        ]
        if total_duration:
            mins, secs = divmod(total_duration, 60)
            lines.append(
                f"[bold]Total duration:[/bold]     {int(mins)}m {secs:.1f}s"
            )

        self.console.print()
        self.console.print(
            Panel(
                "\n".join(lines),
                title="[bold bright_white]\U0001f3c6 Overall Summary[/bold bright_white]",
                border_style="bright_green",
                box=box.DOUBLE,
                padding=(1, 2),
            )
        )
        self.console.print()

    # ------------------------------------------------------------------ 8
    def print_quantization_explainer(self):
        """Print an educational table showing all GGUF quantization levels.

        Reference: ngrok blog \u201cQuantization from the Ground Up\u201d concepts
        including absmax, zero-point, per-tensor/channel/group granularity.
        """
        self.console.print()
        self.console.print(
            Rule(
                "\U0001f4da GGUF Quantization Levels Explained",
                style="bright_blue",
            )
        )
        self.console.print()

        table = Table(
            box=box.ROUNDED,
            title_style="bold bright_white",
            border_style="bright_blue",
            show_lines=True,
            padding=(0, 1),
        )
        table.add_column("Name", style="bold cyan", min_width=8)
        table.add_column("Bits/Weight", justify="center", min_width=10)
        table.add_column("Quality", justify="center", min_width=10)
        table.add_column("Speed", justify="center", min_width=10)
        table.add_column("Memory", justify="center", min_width=10)
        table.add_column("Best For", style="italic", min_width=22)
        table.add_column("Method", style="dim", min_width=30)

        levels = [
            (
                "F16",
                "16",
                "[bold green]\u2588\u2588\u2588\u2588\u2588[/bold green]",
                "[red]\u2588\u2588[/red]",
                "[red]\u2588\u2588\u2588\u2588\u2588[/red]",
                "Maximum accuracy tasks",
                "Full half-precision, no quantization",
            ),
            (
                "Q8_0",
                "8.0",
                "[green]\u2588\u2588\u2588\u2588\u2589[/green]",
                "[yellow]\u2588\u2588\u2588[/yellow]",
                "[yellow]\u2588\u2588\u2588\u2588[/yellow]",
                "Near-lossless, large VRAM",
                "Absmax (symmetric) per-tensor",
            ),
            (
                "Q6_K",
                "6.6",
                "[green]\u2588\u2588\u2588\u2588[/green]",
                "[yellow]\u2588\u2588\u2588\u258c[/yellow]",
                "[yellow]\u2588\u2588\u2588\u258c[/yellow]",
                "High-quality local inference",
                "Super-blocks, 6-bit with K-quants",
            ),
            (
                "Q5_K_M",
                "5.7",
                "[green]\u2588\u2588\u2588\u258c[/green]",
                "[green]\u2588\u2588\u2588\u2588[/green]",
                "[green]\u2588\u2588\u2588[/green]",
                "Recommended default for most",
                "Mixed precision K-quant, medium",
            ),
            (
                "Q5_K_S",
                "5.5",
                "[green]\u2588\u2588\u2588\u258c[/green]",
                "[green]\u2588\u2588\u2588\u2588[/green]",
                "[green]\u2588\u2588\u2588[/green]",
                "Good balance, slightly smaller",
                "Mixed precision K-quant, small",
            ),
            (
                "Q5_0",
                "5.0",
                "[green]\u2588\u2588\u2588[/green]",
                "[green]\u2588\u2588\u2588\u2588[/green]",
                "[green]\u2588\u2588\u2588[/green]",
                "Legacy, prefer K-quants",
                "Absmax symmetric, 5-bit uniform",
            ),
            (
                "Q4_K_M",
                "4.8",
                "[yellow]\u2588\u2588\u2588[/yellow]",
                "[green]\u2588\u2588\u2588\u2588[/green]",
                "[green]\u2588\u2588\u258c[/green]",
                "Great speed/quality trade-off",
                "Mixed 4/5-bit K-quant, medium",
            ),
            (
                "Q4_K_S",
                "4.5",
                "[yellow]\u2588\u2588\u258c[/yellow]",
                "[green]\u2588\u2588\u2588\u2588\u258c[/green]",
                "[green]\u2588\u2588\u258c[/green]",
                "Smaller 4-bit, good quality",
                "Mixed 4/5-bit K-quant, small",
            ),
            (
                "Q4_0",
                "4.0",
                "[yellow]\u2588\u2588\u258c[/yellow]",
                "[green]\u2588\u2588\u2588\u2588\u258c[/green]",
                "[green]\u2588\u2588[/green]",
                "Legacy, prefer K-quants",
                "Absmax symmetric, 4-bit uniform",
            ),
            (
                "Q3_K_M",
                "3.9",
                "[yellow]\u2588\u2588[/yellow]",
                "[green]\u2588\u2588\u2588\u2588\u258c[/green]",
                "[green]\u2588\u2588[/green]",
                "Constrained RAM, ok quality",
                "Mixed 3/4-bit K-quant, medium",
            ),
            (
                "Q3_K_S",
                "3.5",
                "[red]\u2588\u258c[/red]",
                "[green]\u2588\u2588\u2588\u2588\u2588[/green]",
                "[green]\u2588\u258c[/green]",
                "Very low RAM, speed priority",
                "Mixed 3/4-bit K-quant, small",
            ),
            (
                "Q2_K",
                "2.6",
                "[red]\u2588[/red]",
                "[green]\u2588\u2588\u2588\u2588\u2588[/green]",
                "[green]\u2588[/green]",
                "Extreme compression, testing",
                "Zero-point asymmetric, per-group 2-bit",
            ),
            (
                "IQ2_XXS",
                "2.1",
                "[red]\u258c[/red]",
                "[green]\u2588\u2588\u2588\u2588\u2588[/green]",
                "[green]\u258c[/green]",
                "Research / extreme edge deploy",
                "Importance-weighted 2-bit quantization",
            ),
        ]

        for row in levels:
            table.add_row(*row)

        self.console.print(table)

        # Legend
        self.console.print()
        legend = (
            "[bold]Quantization Methods:[/bold]\n"
            "  [cyan]Absmax (Symmetric):[/cyan]  scale = max(|x|) / (2^(b-1) - 1).  "
            "Simple, fast, slight asymmetry waste.\n"
            "  [cyan]Zero-point (Asymmetric):[/cyan]  uses offset to handle skewed "
            "distributions \u2014 better range utilisation.\n"
            "  [cyan]K-quants:[/cyan]  GGML super-block mixed-precision; assigns more "
            "bits to sensitive layers (attention).\n"
            "  [cyan]Importance-weighted (IQ):[/cyan]  allocates bits proportional to "
            "weight importance across groups.\n\n"
            "[dim]Based on concepts from the ngrok blog "
            "\u201cQuantization from the Ground Up\u201d.[/dim]"
        )
        self.console.print(
            Panel(
                legend,
                border_style="dim",
                box=box.ROUNDED,
                padding=(0, 1),
            )
        )
        self.console.print()

    # ------------------------------------------------------------------ 9
    def get_progress_bar(self) -> Progress:
        """Return a configured rich Progress bar for long operations."""
        return Progress(
            SpinnerColumn(spinner_name="dots", style="bright_blue"),
            TextColumn("[bold blue]{task.description}[/bold blue]"),
            BarColumn(bar_width=40, style="bright_blue", complete_style="green", finished_style="bold green"),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=self.console,
            transient=False,
        )

    # ------------------------------------------------------------------ 10
    def print_summary_stats(self, plans: list, comparisons: list = None):
        """Print a final summary panel with aggregate statistics.

        Parameters
        ----------
        plans : list[OptimizationPlan]
        comparisons : list[BenchmarkComparison], optional
        """
        plans = plans or []
        comparisons = comparisons or []

        total_models = len(plans)
        models_optimized = sum(
            1
            for p in plans
            if getattr(p, "current_quant", None) != getattr(p, "recommended_quant", None)
        )
        total_mem_saved = sum(
            _safe_float(getattr(p, "current_size_gb", 0))
            - _safe_float(getattr(p, "optimized_size_gb", 0))
            for p in plans
        )

        avg_speedup = 0.0
        avg_quality_delta = 0.0
        if comparisons:
            speedups = [
                _safe_float(getattr(c, "speedup_ratio", 0)) for c in comparisons
            ]
            avg_speedup = sum(speedups) / len(speedups) if speedups else 0
        elif plans:
            speed_imps = [_safe_float(getattr(p, "speed_improvement", 0)) for p in plans]
            avg_speedup_pct = sum(speed_imps) / len(speed_imps) if speed_imps else 0
            avg_speedup = 1.0 + avg_speedup_pct / 100.0

        if plans:
            quality_deltas = [_safe_float(getattr(p, "quality_delta", 0)) for p in plans]
            avg_quality_delta = sum(quality_deltas) / len(quality_deltas) if quality_deltas else 0

        # Build grid table
        grid = Table(
            box=box.ROUNDED,
            border_style="bright_blue",
            show_header=False,
            show_lines=True,
            padding=(0, 2),
        )
        grid.add_column("Label", style="bold white", min_width=22)
        grid.add_column("Value", style="bold", justify="right", min_width=14)

        grid.add_row(
            "\U0001f50d Models analysed",
            f"[bright_white]{total_models}[/bright_white]",
        )
        grid.add_row(
            "\U0001f527 Models optimized",
            f"[cyan]{models_optimized}[/cyan]",
        )
        mem_color = "green" if total_mem_saved > 0 else "yellow"
        grid.add_row(
            "\U0001f4be Total memory saved",
            f"[{mem_color}]{_fmt_size(total_mem_saved)}[/{mem_color}]",
        )
        sp_color = "green" if avg_speedup >= 1.0 else "red"
        grid.add_row(
            "\u26a1 Avg speedup",
            f"[{sp_color}]{avg_speedup:.2f}x[/{sp_color}]",
        )
        qd_color = _delta_color(avg_quality_delta, higher_is_better=True)
        grid.add_row(
            "\U0001f3af Avg quality delta",
            f"[{qd_color}]{_pct_str(avg_quality_delta)}[/{qd_color}]",
        )

        self.console.print()
        self.console.print(
            Panel(
                grid,
                title="[bold bright_white]\U0001f4cb Final Summary[/bold bright_white]",
                subtitle="[dim]ollama-optimizer v1.0.0[/dim]",
                border_style="bright_green",
                box=box.DOUBLE,
                padding=(1, 2),
            )
        )
        self.console.print()
