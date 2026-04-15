"""
Main optimization engine for ollama-optimizer.

Analyzes installed Ollama models and applies optimal quantization and runtime
settings based on detected hardware capabilities and user priorities. Ties
together the system profiler, Ollama client, and quantization knowledge to
produce actionable optimization plans.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ollama_optimizer.system_profiler import (
    SystemProfile,
    detect_system,
    recommend_gpu_layers,
)
from ollama_optimizer.ollama_client import OllamaClient, OllamaModel
from ollama_optimizer.quantization import (
    QuantizationEngine,
    GGUF_QUANT_LEVELS,
    GGUFQuantLevel,
    recommend_quantization,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Priority presets
# ---------------------------------------------------------------------------
PRIORITY_PRESETS = {
    "quality": {"quality_weight": 0.8, "speed_weight": 0.1, "size_weight": 0.1},
    "speed": {"quality_weight": 0.1, "speed_weight": 0.8, "size_weight": 0.1},
    "balanced": {"quality_weight": 0.4, "speed_weight": 0.3, "size_weight": 0.3},
    "minimum": {"quality_weight": 0.05, "speed_weight": 0.15, "size_weight": 0.8},
}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ModelAnalysis:
    """Result of analyzing a single installed Ollama model."""

    model: str
    current_quant: str
    current_size_gb: float
    parameter_count: str  # e.g. "7B"
    parameter_billions: float
    family: str
    fits_in_ram: bool
    fits_in_vram: bool
    current_quality_score: float
    current_speed_multiplier: float


@dataclass
class OptimizationPlan:
    """A concrete plan describing how to optimize a single model."""

    model_name: str
    current_quant: str
    recommended_quant: str
    current_size_gb: float
    optimized_size_gb: float
    size_reduction_pct: float
    quality_delta: float
    speed_improvement: float
    gpu_layers: int
    num_threads: int
    context_size: int
    batch_size: int
    recommended_options: Dict[str, object] = field(default_factory=dict)
    action: str = "already_optimal"
    reason: str = ""


@dataclass
class OptimizationResult:
    """Outcome of applying an OptimizationPlan."""

    model_name: str
    plan: OptimizationPlan
    success: bool
    old_model_tag: str
    new_model_tag: str
    modelfile_content: str
    error: str = ""


@dataclass
class EnvironmentConfig:
    """Recommended Ollama server environment variables.
    
    These are server-level settings that affect ALL models and often provide
    the biggest performance gains (Flash Attention alone gives 1.5-3x speedup).
    """
    
    flash_attention: bool = False
    kv_cache_type: str = "f16"          # "f16", "q8_0", or "q4_0"
    keep_alive: str = "5m"              # "-1" for permanent, "5m" default
    num_parallel: int = 1
    max_loaded_models: int = 1
    sched_spread: bool = False          # spread layers across all GPUs
    runners_dir: str = ""               # custom runners directory
    
    # Explanations for each recommendation
    reasons: Dict[str, str] = field(default_factory=dict)
    
    def to_env_dict(self) -> Dict[str, str]:
        """Convert to a dict of env var name -> value for shell export."""
        env = {}
        if self.flash_attention:
            env["OLLAMA_FLASH_ATTENTION"] = "1"
        if self.kv_cache_type != "f16":
            # Quantized KV cache requires Flash Attention in Ollama
            if self.flash_attention:
                env["OLLAMA_KV_CACHE_TYPE"] = self.kv_cache_type
        if self.keep_alive != "5m":
            env["OLLAMA_KEEP_ALIVE"] = self.keep_alive
        if self.num_parallel > 1:
            env["OLLAMA_NUM_PARALLEL"] = str(self.num_parallel)
        if self.max_loaded_models > 1:
            env["OLLAMA_MAX_LOADED_MODELS"] = str(self.max_loaded_models)
        if self.sched_spread:
            env["OLLAMA_SCHED_SPREAD"] = "1"
        if self.runners_dir:
            env["OLLAMA_RUNNERS_DIR"] = self.runners_dir
        return env
    
    def to_shell_exports(self) -> str:
        """Generate shell export statements."""
        lines = ["# Ollama environment optimization by ollama-optimizer"]
        for key, value in self.to_env_dict().items():
            lines.append(f'export {key}="{value}"')
        return "\n".join(lines)
    
    def to_systemd_override(self) -> str:
        """Generate systemd override configuration."""
        lines = [
            "# Ollama service optimization by ollama-optimizer",
            "# Save to: /etc/systemd/system/ollama.service.d/override.conf",
            "# Then run: sudo systemctl daemon-reload && sudo systemctl restart ollama",
            "",
            "[Service]",
        ]
        for key, value in self.to_env_dict().items():
            lines.append(f'Environment="{key}={value}"')
        return "\n".join(lines)
    
    def to_launchd_plist_fragment(self) -> str:
        """Generate macOS launchd plist EnvironmentVariables fragment."""
        lines = [
            "<!-- Ollama environment optimization by ollama-optimizer -->",
            "<!-- Add inside the main <dict> of the Ollama plist -->",
            "<key>EnvironmentVariables</key>",
            "<dict>",
        ]
        for key, value in self.to_env_dict().items():
            lines.append(f"    <key>{key}</key>")
            lines.append(f"    <string>{value}</string>")
        lines.append("</dict>")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_parameter_billions(param_str: str) -> float:
    """Parse ``'7B'``, ``'70B'``, ``'1.5B'`` -> float billions."""
    if not param_str:
        return 0.0
    cleaned = param_str.strip().upper()
    if cleaned.endswith("B"):
        cleaned = cleaned[:-1]
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        logger.warning("Unable to parse parameter count from %r", param_str)
        return 0.0


def _estimate_model_size_gb(parameter_billions: float, bits: float) -> float:
    """Estimate model size: params * (bits/8) * 1.1 overhead, in GB."""
    if parameter_billions <= 0 or bits <= 0:
        return 0.0
    size_bytes = parameter_billions * 1e9 * (bits / 8.0) * 1.1
    return size_bytes / (1024 ** 3)


def _quant_tag_to_key(quant_str: str) -> str:
    """Normalise quant tag to upper-case key for GGUF_QUANT_LEVELS."""
    if not quant_str:
        return ""
    return quant_str.strip().upper()


def _get_quant_level(quant_key: str) -> Optional[GGUFQuantLevel]:
    """Look up a GGUFQuantLevel by canonical key."""
    return GGUF_QUANT_LEVELS.get(_quant_tag_to_key(quant_key))


# ---------------------------------------------------------------------------
# Main optimizer class
# ---------------------------------------------------------------------------


class ModelOptimizer:
    """Core optimization engine.

    Parameters
    ----------
    client:
        An OllamaClient. Created automatically if None.
    system:
        A SystemProfile. Detected automatically if None.
    priority:
        One of "quality", "speed", "balanced", or "minimum".
    """

    def __init__(
        self,
        client: OllamaClient = None,
        system: SystemProfile = None,
        priority: str = "balanced",
    ) -> None:
        self.client = client or OllamaClient()
        self.system = system or detect_system()
        self.quant_engine = QuantizationEngine()
        self.priority = priority if priority in PRIORITY_PRESETS else "balanced"
        logger.info(
            "ModelOptimizer initialised – priority=%s, RAM=%.1f GB, VRAM=%.1f GB, "
            "CPU cores=%d",
            self.priority,
            self.system.ram_total_gb,
            self.system.gpu_vram_gb,
            self.system.cpu_cores_physical,
        )

    # ------------------------------------------------------------------
    # 1. analyze_model
    # ------------------------------------------------------------------

    def analyze_model(self, model: OllamaModel) -> ModelAnalysis:
        """Analyse a single installed model."""
        logger.debug("Analysing model: %s", model.full_name)

        current_quant = model.quantization_level or ""
        current_size_gb = model.size_bytes / (1024 ** 3) if model.size_bytes else 0.0
        parameter_count = model.parameter_size or ""
        family = model.family or ""

        parameter_billions = _parse_parameter_billions(parameter_count)

        quant_level = _get_quant_level(current_quant)
        if quant_level is not None:
            quality_score = quant_level.quality_score
            speed_multiplier = quant_level.speed_multiplier
        else:
            quality_score = 0.5
            speed_multiplier = 1.0

        fits_in_ram = current_size_gb <= self.system.ram_available_gb * 0.85
        fits_in_vram = (
            self.system.gpu_detected
            and current_size_gb <= self.system.gpu_vram_gb * 0.90
        )

        return ModelAnalysis(
            model=model.full_name,
            current_quant=current_quant,
            current_size_gb=round(current_size_gb, 2),
            parameter_count=parameter_count,
            parameter_billions=parameter_billions,
            family=family,
            fits_in_ram=fits_in_ram,
            fits_in_vram=fits_in_vram,
            current_quality_score=quality_score,
            current_speed_multiplier=speed_multiplier,
        )

    # ------------------------------------------------------------------
    # 2. analyze_all_models
    # ------------------------------------------------------------------

    def analyze_all_models(self) -> List[ModelAnalysis]:
        """Analyse every model installed in the local Ollama instance."""
        try:
            models = self.client.list_models()
        except Exception:
            logger.exception("Failed to list installed Ollama models")
            return []

        analyses: List[ModelAnalysis] = []
        for model in models:
            try:
                analyses.append(self.analyze_model(model))
            except Exception:
                logger.exception("Error analysing model %s", model.name)
        return analyses

    # ------------------------------------------------------------------
    # 3. create_optimization_plan
    # ------------------------------------------------------------------

    def create_optimization_plan(self, analysis: ModelAnalysis) -> OptimizationPlan:
        """Create an optimization plan for a single model."""
        logger.debug(
            "Creating plan for %s (current=%s, %.1f GB)",
            analysis.model, analysis.current_quant, analysis.current_size_gb,
        )

        # -- Quantization recommendation (module-level function) ----------
        recommendations = recommend_quantization(
            model_params_billions=analysis.parameter_billions,
            available_ram_gb=self.system.ram_available_gb,
            available_vram_gb=self.system.gpu_vram_gb,
            priority=self.priority,
        )

        if recommendations:
            recommended = recommendations[0]
            recommended_quant = recommended.name
        else:
            recommended_quant = analysis.current_quant
            logger.warning("No quant recommendation for %s", analysis.model)

        current_ql = _get_quant_level(analysis.current_quant)
        recommended_ql = _get_quant_level(recommended_quant)

        current_quality = analysis.current_quality_score
        current_speed = analysis.current_speed_multiplier
        current_bits = current_ql.bits_per_weight if current_ql else 16.0
        recommended_bits = recommended_ql.bits_per_weight if recommended_ql else current_bits

        new_quality = recommended_ql.quality_score if recommended_ql else current_quality
        new_speed = recommended_ql.speed_multiplier if recommended_ql else current_speed

        if analysis.parameter_billions > 0 and recommended_bits > 0:
            optimized_size_gb = _estimate_model_size_gb(analysis.parameter_billions, recommended_bits)
        else:
            optimized_size_gb = analysis.current_size_gb

        optimized_size_gb = round(optimized_size_gb, 2)

        if analysis.current_size_gb > 0:
            size_reduction_pct = round((1.0 - optimized_size_gb / analysis.current_size_gb) * 100.0, 1)
        else:
            size_reduction_pct = 0.0

        quality_delta = round(new_quality - current_quality, 4)
        speed_improvement = round(new_speed / current_speed, 2) if current_speed > 0 else 1.0

        # -- Determine action -------------------------------------------
        current_key = _quant_tag_to_key(analysis.current_quant)
        recommended_key = _quant_tag_to_key(recommended_quant)

        if current_key == recommended_key:
            action = "tune_runtime"
            reason = (
                f"Already at {recommended_quant}. "
                f"Runtime parameters will be tuned for this system."
            )
        elif recommended_bits < current_bits:
            action = "requantize_down"
            reason = (
                f"{analysis.current_quant} -> {recommended_quant}: "
                f"saves ~{abs(size_reduction_pct):.0f}% memory, "
                f"{speed_improvement:.1f}x speed, "
                f"quality delta {quality_delta:+.3f}"
            )
        else:
            action = "requantize_up"
            reason = (
                f"System can handle {recommended_quant}: "
                f"quality +{quality_delta:.3f}, "
                f"~{abs(size_reduction_pct):.0f}% more memory"
            )

        # -- Runtime tuning (always applied) -----------------------------
        num_threads = max(1, self.system.cpu_cores_physical)

        gpu_layers = recommend_gpu_layers(self.system, optimized_size_gb)

        effective_ram = self.system.ram_available_gb - optimized_size_gb
        if effective_ram >= 8.0:
            context_size = 8192
        elif effective_ram >= 4.0:
            context_size = 4096
        else:
            context_size = 2048

        batch_size = 512 if effective_ram >= 2.0 else 256

        num_gpu = 1 if self.system.gpu_detected and self.system.gpu_vram_gb > 2.0 else 0

        use_mmap = True
        use_mlock = effective_ram >= 4.0 and optimized_size_gb <= self.system.ram_available_gb * 0.75
        # Note: f16_kv is deprecated in Ollama. KV cache type is now controlled
        # by the OLLAMA_KV_CACHE_TYPE environment variable (server-level setting).
        # We still include it for backward compatibility with older Ollama versions.
        f16_kv = self.system.gpu_detected and self.system.gpu_vram_gb >= 4.0

        recommended_options: Dict[str, object] = {
            "num_thread": num_threads,
            "num_gpu": num_gpu,
            "num_ctx": context_size,
            "num_batch": batch_size,
            "num_keep": -1,
            "f16_kv": f16_kv,
            # Note: prefer OLLAMA_KV_CACHE_TYPE env var over f16_kv param
            "use_mmap": use_mmap,
            "use_mlock": use_mlock,
        }

        return OptimizationPlan(
            model_name=analysis.model,
            current_quant=analysis.current_quant,
            recommended_quant=recommended_quant,
            current_size_gb=analysis.current_size_gb,
            optimized_size_gb=optimized_size_gb,
            size_reduction_pct=size_reduction_pct,
            quality_delta=quality_delta,
            speed_improvement=speed_improvement,
            gpu_layers=gpu_layers,
            num_threads=num_threads,
            context_size=context_size,
            batch_size=batch_size,
            recommended_options=recommended_options,
            action=action,
            reason=reason,
        )

    # ------------------------------------------------------------------
    # 4. create_all_plans
    # ------------------------------------------------------------------

    def create_all_plans(self) -> List[OptimizationPlan]:
        """Create optimization plans for every installed model."""
        analyses = self.analyze_all_models()
        plans: List[OptimizationPlan] = []
        for analysis in analyses:
            try:
                plans.append(self.create_optimization_plan(analysis))
            except Exception:
                logger.exception("Failed to create plan for %s", analysis.model)
        return plans

    # ------------------------------------------------------------------
    # 5. recommend_environment
    # ------------------------------------------------------------------

    def recommend_environment(self) -> EnvironmentConfig:
        """Recommend Ollama server-level environment variables.
        
        These settings often provide the biggest performance gains:
        - Flash Attention: 1.5-3x speedup (requires GPU compute >= 7.0)
        - KV cache quantization: frees VRAM for more GPU layers
          (requires Flash Attention to be enabled)
        - Keep-alive: eliminates model reload latency
        - Sched spread: distributes layers across all GPUs
        """
        config = EnvironmentConfig()
        reasons = {}
        
        # Flash Attention
        if hasattr(self.system, 'gpu_supports_flash_attn') and self.system.gpu_supports_flash_attn:
            config.flash_attention = True
            cc = getattr(self.system, 'gpu_compute_capability', 0.0)
            reasons["OLLAMA_FLASH_ATTENTION"] = (
                f"GPU supports Flash Attention (compute capability {cc:.1f}). "
                "Expected 1.5-3x speedup for long contexts. "
                "This is OFF by default in Ollama."
            )
        elif self.system.gpu_detected:
            cc = getattr(self.system, 'gpu_compute_capability', 0.0)
            if cc > 0 and cc < 7.0:
                reasons["OLLAMA_FLASH_ATTENTION"] = (
                    f"GPU compute capability {cc:.1f} < 7.0. "
                    "Flash Attention requires Volta (7.0) or newer."
                )
            else:
                reasons["OLLAMA_FLASH_ATTENTION"] = (
                    "Could not determine GPU compute capability. "
                    "Try setting OLLAMA_FLASH_ATTENTION=1 manually if you have "
                    "a Volta+ (V100, RTX 20xx+) GPU."
                )
        
        # KV Cache Type (requires Flash Attention for quantized types)
        if self.system.gpu_detected and self.system.gpu_vram_gb > 0:
            if config.flash_attention:
                if self.system.gpu_vram_gb < 8:
                    config.kv_cache_type = "q4_0"
                    reasons["OLLAMA_KV_CACHE_TYPE"] = (
                        f"Limited VRAM ({self.system.gpu_vram_gb:.0f} GB). "
                        "q4_0 KV cache saves ~75% KV memory, freeing VRAM "
                        "for more GPU layers. (Requires Flash Attention.)"
                    )
                elif self.system.gpu_vram_gb < 16:
                    config.kv_cache_type = "q8_0"
                    reasons["OLLAMA_KV_CACHE_TYPE"] = (
                        f"Moderate VRAM ({self.system.gpu_vram_gb:.0f} GB). "
                        "q8_0 KV cache saves ~50% KV memory with minimal "
                        "quality impact. (Requires Flash Attention.)"
                    )
                else:
                    reasons["OLLAMA_KV_CACHE_TYPE"] = (
                        f"Ample VRAM ({self.system.gpu_vram_gb:.0f} GB). "
                        "f16 KV cache recommended for maximum quality."
                    )
            else:
                reasons["OLLAMA_KV_CACHE_TYPE"] = (
                    "Quantized KV cache (q8_0/q4_0) requires Flash Attention "
                    "to be enabled. Enable OLLAMA_FLASH_ATTENTION first."
                )
        
        # Keep-alive
        if self.system.ram_available_gb >= 16 or (
            self.system.gpu_detected and self.system.gpu_vram_gb >= 8
        ):
            config.keep_alive = "-1"
            reasons["OLLAMA_KEEP_ALIVE"] = (
                "Sufficient memory for persistent model loading. "
                "Setting to -1 prevents model unloading, eliminating "
                "5-30 second cold start delays."
            )
        elif self.system.ram_available_gb >= 8:
            config.keep_alive = "30m"
            reasons["OLLAMA_KEEP_ALIVE"] = (
                "Moderate memory available. Extended keep-alive (30m) "
                "reduces reload frequency while allowing memory reclaim."
            )
        
        # Parallel requests
        if self.system.ram_available_gb >= 64:
            config.num_parallel = 4
            config.max_loaded_models = 3
            reasons["OLLAMA_NUM_PARALLEL"] = (
                f"High memory system ({self.system.ram_available_gb:.0f} GB RAM). "
                "Can handle 4 parallel requests and 3 loaded models."
            )
        elif self.system.ram_available_gb >= 32 and self.system.gpu_detected:
            config.num_parallel = 4
            config.max_loaded_models = 2
            reasons["OLLAMA_NUM_PARALLEL"] = (
                f"High memory system ({self.system.ram_available_gb:.0f} GB RAM, "
                f"{self.system.gpu_vram_gb:.0f} GB VRAM). "
                "Can handle 4 parallel requests."
            )
        elif self.system.ram_available_gb >= 16:
            config.num_parallel = 2
            reasons["OLLAMA_NUM_PARALLEL"] = (
                f"Moderate memory ({self.system.ram_available_gb:.0f} GB RAM). "
                "Can handle 2 parallel requests."
            )
        
        # Sched Spread (multi-GPU)
        gpu_count = getattr(self.system, 'gpu_count', 1) or 1
        if gpu_count > 1:
            config.sched_spread = True
            reasons["OLLAMA_SCHED_SPREAD"] = (
                f"Multiple GPUs detected ({gpu_count}). Spreading model "
                "layers across all GPUs maximizes total VRAM utilisation."
            )
        
        # NUMA (informational only - Ollama does not currently expose
        # a NUMA env var; see llm/server.go TODO comment)
        numa_avail = getattr(self.system, 'numa_available', False)
        numa_nodes = getattr(self.system, 'numa_node_count', 1)
        if numa_avail and numa_nodes > 1:
            reasons["_NUMA_INFO"] = (
                f"NUMA topology detected ({numa_nodes} nodes). Ollama does "
                "not currently expose a NUMA env var. For manual tuning, "
                "consider running Ollama under numactl --interleave=all."
            )
        
        config.reasons = reasons
        return config

    # ------------------------------------------------------------------
    # 6. generate_modelfile
    # ------------------------------------------------------------------

    def generate_modelfile(self, plan: OptimizationPlan) -> str:
        """Generate an Ollama Modelfile that applies the optimization.

        For requantization, FROM points to the new quant variant tag.
        For runtime tuning only, FROM references the current model.
        """
        lines: List[str] = []

        if plan.action in ("requantize_down", "requantize_up"):
            # Construct Ollama model tag: base:paramsize-quantlevel
            # e.g. "llama3.2:3b-q4_K_M" from model "llama3.2:3b" + quant "Q4_K_M"
            base_name = plan.model_name.split(":")[0]
            # Try to extract param size from the current tag (e.g., "3b" from "llama3.2:3b")
            current_tag = plan.model_name.split(":")[-1] if ":" in plan.model_name else ""
            param_part = ""
            for part in current_tag.replace("-", " ").split():
                if part and part[0].isdigit() and part[-1].lower() == "b":
                    param_part = part.lower()
                    break

            quant_tag = plan.recommended_quant.lower()
            if param_part:
                from_ref = f"{base_name}:{param_part}-{quant_tag}"
            else:
                from_ref = f"{base_name}:{quant_tag}"
        else:
            from_ref = plan.model_name

        lines.append(f"FROM {from_ref}")
        lines.append("")
        lines.append("# Runtime parameters tuned by ollama-optimizer")
        for key, value in plan.recommended_options.items():
            if isinstance(value, bool):
                param_value = "true" if value else "false"
            else:
                param_value = str(value)
            lines.append(f"PARAMETER {key} {param_value}")

        lines.append("")
        lines.append(
            'SYSTEM """You are a helpful assistant. '
            "This model has been optimized by ollama-optimizer "
            f'(priority={plan.recommended_quant})."""'
        )

        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # 7. apply_optimization
    # ------------------------------------------------------------------

    def apply_optimization(self, plan: OptimizationPlan) -> OptimizationResult:
        """Apply an optimization plan.

        1. Generate a Modelfile from the plan.
        2. If requantizing, pull the target quant variant.
        3. Use ``ollama create`` to register the optimized model.
        4. Naming: ``<original_name>-optimized``.
        """
        old_tag = plan.model_name
        base_name = plan.model_name.split(":")[0]
        new_tag = f"{base_name}-optimized"
        modelfile_content = self.generate_modelfile(plan)

        logger.info("Applying: %s -> %s (action=%s)", old_tag, new_tag, plan.action)

        try:
            # Pull the new quant variant if requantizing
            if plan.action in ("requantize_down", "requantize_up"):
                # Extract the FROM reference from the modelfile
                from_line = modelfile_content.split("\n")[0]
                pull_target = from_line.replace("FROM ", "").strip()
                logger.info("Pulling quantization variant: %s", pull_target)
                try:
                    for status in self.client.pull_model(pull_target):
                        if status.get("status") == "success":
                            logger.info("Pull complete: %s", pull_target)
                        elif "error" in status:
                            logger.warning("Pull warning: %s", status.get("error"))
                except Exception:
                    logger.exception(
                        "Failed to pull %s – attempting create anyway", pull_target
                    )

            # Create the optimized model
            logger.info("Creating optimized model: %s", new_tag)
            for status in self.client.create_model(name=new_tag, modelfile=modelfile_content):
                if status.get("status") == "success":
                    logger.info("Model created: %s", new_tag)
                elif "error" in status:
                    raise RuntimeError(f"ollama create failed: {status.get('error')}")

            return OptimizationResult(
                model_name=plan.model_name,
                plan=plan,
                success=True,
                old_model_tag=old_tag,
                new_model_tag=new_tag,
                modelfile_content=modelfile_content,
            )

        except Exception as exc:
            error_msg = f"Optimisation failed for {plan.model_name}: {exc}"
            logger.exception(error_msg)
            return OptimizationResult(
                model_name=plan.model_name,
                plan=plan,
                success=False,
                old_model_tag=old_tag,
                new_model_tag=new_tag,
                modelfile_content=modelfile_content,
                error=error_msg,
            )

    # ------------------------------------------------------------------
    # 8. apply_all_optimizations
    # ------------------------------------------------------------------

    def apply_all_optimizations(
        self, plans: Optional[List[OptimizationPlan]] = None
    ) -> List[OptimizationResult]:
        """Apply optimization plans for all models."""
        if plans is None:
            plans = self.create_all_plans()

        results: List[OptimizationResult] = []
        for plan in plans:
            result = self.apply_optimization(plan)
            results.append(result)
            if result.success:
                logger.info("OK %s -> %s", result.old_model_tag, result.new_model_tag)
            else:
                logger.error("FAIL %s: %s", result.old_model_tag, result.error)
        return results
