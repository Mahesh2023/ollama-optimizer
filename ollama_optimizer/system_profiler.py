"""
System hardware profiler for Ollama model optimization.

Detects CPU, RAM, GPU, and disk capabilities across Windows, Linux, and macOS
to determine optimal quantization settings and GPU offloading strategies for
running large language models via Ollama.
"""

from __future__ import annotations

import json
import logging
import math
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import psutil

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Bits per parameter for each GGML quantization type.
QUANTIZATION_BITS: Dict[str, float] = {
    "IQ1_S":  1.56,
    "IQ1_M":  1.75,
    "IQ2_XXS": 2.06,
    "IQ2_XS": 2.31,
    "IQ2_S":  2.56,
    "Q2_K":   2.63,
    "IQ3_XXS": 3.06,
    "IQ3_XS": 3.30,
    "IQ3_S":  3.44,
    "Q3_K_M": 3.44,
    "IQ4_XS": 4.25,
    "Q4_0":   4.50,
    "IQ4_NL": 4.50,
    "Q4_K_M": 4.83,
    "Q5_0":   5.50,
    "Q5_K_M": 5.69,
    "Q6_K":   6.56,
    "Q8_0":   8.50,
    "F16":   16.00,
}

# Overhead multiplier applied to raw model weight size to account for KV-cache,
# computation buffers, and OS/runtime memory that must remain free.
_MEMORY_OVERHEAD_FACTOR: float = 1.20  # 20 % headroom
# Fraction of total VRAM assumed usable (rest reserved by driver / desktop).
_USABLE_VRAM_FRACTION: float = 0.90
# Minimum amount of system RAM (bytes) to keep free for the OS.
_MIN_FREE_RAM_BYTES: int = 2 * 1024 ** 3  # 2 GiB


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SystemProfile:
    """Snapshot of the host machine's hardware capabilities."""

    # -- CPU --
    cpu_name: str = "Unknown"
    cpu_cores_physical: int = 0
    cpu_cores_logical: int = 0
    cpu_freq_mhz: float = 0.0

    # -- Memory --
    ram_total_bytes: int = 0
    ram_available_bytes: int = 0

    # -- GPU --
    gpu_name: str = "Unknown"
    gpu_vram_bytes: int = 0
    gpu_detected: bool = False

    # -- OS / Disk --
    os_type: str = "Unknown"           # "Windows", "Linux", or "Darwin"
    os_version: str = "Unknown"
    disk_free_bytes: int = 0

    # -- GPU extended --
    gpu_compute_capability: float = 0.0   # e.g. 7.5 for Turing
    gpu_supports_flash_attn: bool = False

    # -- NUMA --
    numa_available: bool = False
    numa_node_count: int = 1

    # -- Derived helpers (populated by detect_system) --
    ram_total_gb: float = field(init=False, default=0.0)
    ram_available_gb: float = field(init=False, default=0.0)
    gpu_vram_gb: float = field(init=False, default=0.0)
    disk_free_gb: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self.ram_total_gb = round(self.ram_total_bytes / (1024 ** 3), 2)
        self.ram_available_gb = round(self.ram_available_bytes / (1024 ** 3), 2)
        self.gpu_vram_gb = round(self.gpu_vram_bytes / (1024 ** 3), 2)
        self.disk_free_gb = round(self.disk_free_bytes / (1024 ** 3), 2)


# ---------------------------------------------------------------------------
# Internal helpers – GPU detection per platform
# ---------------------------------------------------------------------------

def _run_command(cmd: List[str], timeout: int = 10) -> Optional[str]:
    """Run *cmd* and return stripped stdout, or ``None`` on any failure."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        logger.debug(
            "Command %s exited with code %d: %s",
            cmd, result.returncode, result.stderr.strip(),
        )
    except FileNotFoundError:
        logger.debug("Command not found: %s", cmd[0])
    except subprocess.TimeoutExpired:
        logger.warning("Command timed out: %s", cmd)
    except Exception:
        logger.debug("Unexpected error running %s", cmd, exc_info=True)
    return None


def _detect_gpu_nvidia() -> Tuple[str, int]:
    """Attempt to detect an NVIDIA GPU via ``nvidia-smi``.

    Returns:
        (gpu_name, vram_bytes) or ("Unknown", 0) on failure.
    """
    # nvidia-smi returns memory in MiB by default.
    output = _run_command([
        "nvidia-smi",
        "--query-gpu=name,memory.total",
        "--format=csv,noheader,nounits",
    ])
    if output:
        try:
            # May contain multiple GPUs – take the first one.
            first_line = output.splitlines()[0]
            parts = [p.strip() for p in first_line.split(",")]
            name = parts[0]
            vram_mib = float(parts[1])
            vram_bytes = int(vram_mib * 1024 * 1024)
            logger.info("Detected NVIDIA GPU: %s (%d MiB VRAM)", name, int(vram_mib))
            return name, vram_bytes
        except (IndexError, ValueError) as exc:
            logger.debug("Failed to parse nvidia-smi output: %s", exc)
    return "Unknown", 0


def _detect_gpu_compute_capability() -> float:
    """Detect NVIDIA GPU compute capability via nvidia-smi.

    Compute capability >= 7.0 (Volta) is needed for Flash Attention.
    Returns 0.0 if detection fails or no NVIDIA GPU.
    """
    output = _run_command([
        "nvidia-smi",
        "--query-gpu=compute_cap",
        "--format=csv,noheader,nounits",
    ])
    if output:
        try:
            first_line = output.splitlines()[0].strip()
            return float(first_line)
        except (IndexError, ValueError) as exc:
            logger.debug("Failed to parse compute capability: %s", exc)

    # Fallback: try to infer from GPU name
    return 0.0


def _infer_compute_capability(gpu_name: str) -> float:
    """Infer compute capability from GPU name as a fallback.

    This covers common GPU families when nvidia-smi --query-gpu=compute_cap
    is not available (older drivers).  Values sourced from the NVIDIA CUDA
    GPUs page (https://developer.nvidia.com/cuda-gpus).
    """
    if not gpu_name or gpu_name == "Unknown":
        return 0.0

    name_upper = gpu_name.upper()

    # Blackwell (B100, B200, GB200) = 10.0
    if any(x in name_upper for x in ["B100", "B200", "GB200", "BLACKWELL"]):
        return 10.0
    # Hopper (H100, H200, H800, GH200) = 9.0
    if any(x in name_upper for x in ["H100", "H200", "H800", "GH200", "HOPPER"]):
        return 9.0
    # Ada Lovelace (RTX 40xx, L4, L40, RTX 6000 Ada) = 8.9
    if any(x in name_upper for x in ["RTX 40", "L4", "L40", "ADA",
                                      "RTX 6000"]):
        return 8.9
    # Ampere data-center: A100 / A30 are GA100 (SM 8.0)
    if any(x in name_upper for x in ["A100", "A800", "A30"]):
        return 8.0
    # Ampere consumer/pro: RTX 30xx, A10, A40, A6000, A5000, A4000 are GA10x (SM 8.6)
    if any(x in name_upper for x in ["RTX 30", "A10", "A40", "A6000",
                                      "A5000", "A4000"]):
        return 8.6
    if "AMPERE" in name_upper:
        return 8.6
    # Turing (RTX 20xx, T4, Quadro RTX) = 7.5
    if any(x in name_upper for x in ["RTX 20", "T4", "TURING", "QUADRO RTX"]):
        return 7.5
    # Volta (V100, Titan V) = 7.0
    if any(x in name_upper for x in ["V100", "VOLTA", "TITAN V"]):
        return 7.0
    # Pascal data-center: P100 is GP100 (SM 6.0)
    if "P100" in name_upper:
        return 6.0
    # Pascal consumer/pro: GTX 10xx, P40, Quadro P = GP10x (SM 6.1)
    if any(x in name_upper for x in ["GTX 10", "P40", "PASCAL"]):
        return 6.1
    # Maxwell (GTX 9xx) = 5.2
    if any(x in name_upper for x in ["GTX 9", "MAXWELL"]):
        return 5.2
    # Apple Silicon / AMD - no CUDA compute capability
    if any(x in name_upper for x in ["APPLE", "RADEON", "AMD"]):
        return 0.0

    return 0.0


def _detect_numa() -> Tuple[bool, int]:
    """Detect NUMA topology on Linux.

    Returns (numa_available, node_count).
    On non-Linux or if detection fails, returns (False, 1).
    """
    system = platform.system()
    if system != "Linux":
        return False, 1

    # Check /sys/devices/system/node/
    try:
        node_dirs = [
            d for d in os.listdir("/sys/devices/system/node/")
            if d.startswith("node")
        ]
        node_count = len(node_dirs)
        if node_count > 1:
            logger.info("NUMA detected: %d nodes", node_count)
            return True, node_count
        return False, 1
    except (OSError, FileNotFoundError):
        pass

    # Fallback: try numactl
    output = _run_command(["numactl", "--hardware"])
    if output:
        for line in output.splitlines():
            if line.startswith("available:"):
                try:
                    parts = line.split()
                    count = int(parts[1])
                    if count > 1:
                        logger.info("NUMA detected via numactl: %d nodes", count)
                        return True, count
                except (IndexError, ValueError):
                    pass

    return False, 1


def _detect_gpu_amd_linux() -> Tuple[str, int]:
    """Attempt to detect an AMD GPU on Linux via ``rocm-smi``."""
    output = _run_command(["rocm-smi", "--showproductname"])
    name = "Unknown"
    if output:
        for line in output.splitlines():
            if "GPU" in line or "Card" in line:
                name = line.split(":")[-1].strip() if ":" in line else line.strip()
                break

    vram_bytes = 0
    mem_output = _run_command(["rocm-smi", "--showmeminfo", "vram"])
    if mem_output:
        for line in mem_output.splitlines():
            lower = line.lower()
            if "total" in lower:
                try:
                    # rocm-smi typically reports bytes or has a clear numeric field.
                    parts = line.split()
                    for part in reversed(parts):
                        part_clean = part.replace(",", "")
                        if part_clean.isdigit():
                            vram_bytes = int(part_clean)
                            break
                except (ValueError, IndexError):
                    pass

    if name != "Unknown" or vram_bytes > 0:
        logger.info("Detected AMD GPU: %s (%d bytes VRAM)", name, vram_bytes)
    return name, vram_bytes


def _detect_gpu_macos() -> Tuple[str, int]:
    """Detect GPU info on macOS via ``system_profiler``."""
    output = _run_command(["system_profiler", "SPDisplaysDataType", "-json"])
    if not output:
        return "Unknown", 0

    try:
        data = json.loads(output)
        displays = data.get("SPDisplaysDataType", [])
        if not displays:
            return "Unknown", 0

        gpu_info = displays[0]
        name = gpu_info.get("sppci_model", "Unknown")
        # VRAM value can be a string like "8 GB" or "8192 MB".
        vram_str: str = gpu_info.get("spdisplays_vram", gpu_info.get("sppci_vram", "0"))
        vram_bytes = _parse_vram_string(vram_str)

        # Apple Silicon unified memory – if VRAM looks like 0 or very small,
        # treat total RAM as shared GPU memory (conservative: half).
        if vram_bytes == 0 and "apple" in name.lower():
            vram_bytes = psutil.virtual_memory().total // 2
            logger.info(
                "Apple Silicon detected (%s); estimating shared VRAM as %d bytes",
                name, vram_bytes,
            )

        logger.info("Detected macOS GPU: %s (%d bytes VRAM)", name, vram_bytes)
        return name, vram_bytes
    except (json.JSONDecodeError, KeyError, IndexError) as exc:
        logger.debug("Failed to parse system_profiler output: %s", exc)
    return "Unknown", 0


def _detect_gpu_windows_wmic() -> Tuple[str, int]:
    """Fallback GPU detection on Windows using ``wmic``."""
    output = _run_command([
        "wmic", "path", "win32_videocontroller", "get",
        "Name,AdapterRAM", "/format:csv",
    ])
    if not output:
        return "Unknown", 0

    try:
        lines = [l.strip() for l in output.splitlines() if l.strip()]
        # The CSV header is the first non-empty line.
        if len(lines) < 2:
            return "Unknown", 0
        header = [h.strip().lower() for h in lines[0].split(",")]
        values = [v.strip() for v in lines[1].split(",")]
        name_idx = header.index("name") if "name" in header else -1
        ram_idx = header.index("adapterram") if "adapterram" in header else -1
        name = values[name_idx] if name_idx >= 0 else "Unknown"
        vram_bytes = int(values[ram_idx]) if ram_idx >= 0 and values[ram_idx].isdigit() else 0
        logger.info("Detected GPU via WMIC: %s (%d bytes VRAM)", name, vram_bytes)
        return name, vram_bytes
    except (ValueError, IndexError) as exc:
        logger.debug("Failed to parse WMIC output: %s", exc)
    return "Unknown", 0


def _parse_vram_string(vram_str: str) -> int:
    """Best-effort conversion of a human-readable VRAM string to bytes.

    Handles formats like ``"8 GB"``, ``"8192 MB"``, ``"8589934592"``, etc.
    """
    if not vram_str:
        return 0
    vram_str = vram_str.strip()

    # Pure numeric (already bytes).
    if vram_str.isdigit():
        return int(vram_str)

    upper = vram_str.upper()
    try:
        if "TB" in upper:
            return int(float(upper.replace("TB", "").strip()) * 1024 ** 4)
        if "GB" in upper:
            return int(float(upper.replace("GB", "").strip()) * 1024 ** 3)
        if "MB" in upper:
            return int(float(upper.replace("MB", "").strip()) * 1024 ** 2)
        if "KB" in upper:
            return int(float(upper.replace("KB", "").strip()) * 1024)
    except ValueError:
        pass
    return 0


def _detect_gpu() -> Tuple[str, int]:
    """Cross-platform GPU detection.

    Tries NVIDIA first (works on all OSes with the driver installed), then
    falls back to OS-specific methods.

    Returns:
        (gpu_name, vram_bytes)
    """
    # 1. Try GPUtil if available (pip install gputil).
    try:
        import GPUtil  # type: ignore[import-untyped]
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            name = gpu.name
            vram_bytes = int(gpu.memoryTotal * 1024 * 1024)  # memoryTotal is MiB
            logger.info("Detected GPU via GPUtil: %s (%d bytes VRAM)", name, vram_bytes)
            return name, vram_bytes
    except Exception:
        logger.debug("GPUtil not available or failed; falling back to CLI detection.")

    # 2. NVIDIA CLI (cross-platform).
    name, vram = _detect_gpu_nvidia()
    if name != "Unknown":
        return name, vram

    # 3. OS-specific fallbacks.
    system = platform.system()
    if system == "Linux":
        name, vram = _detect_gpu_amd_linux()
        if name != "Unknown" or vram > 0:
            return name, vram
    elif system == "Darwin":
        return _detect_gpu_macos()
    elif system == "Windows":
        return _detect_gpu_windows_wmic()

    logger.info("No GPU detected; CPU-only mode will be used.")
    return "Unknown", 0


def _detect_cpu_name() -> str:
    """Return a human-friendly CPU model string."""
    system = platform.system()
    try:
        if system == "Linux":
            with open("/proc/cpuinfo", "r") as fh:
                for line in fh:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        elif system == "Darwin":
            out = _run_command(["sysctl", "-n", "machdep.cpu.brand_string"])
            if out:
                return out
        elif system == "Windows":
            out = _run_command(["wmic", "cpu", "get", "Name", "/format:list"])
            if out:
                for line in out.splitlines():
                    if line.startswith("Name="):
                        return line.split("=", 1)[1].strip()
    except Exception:
        logger.debug("CPU name detection failed", exc_info=True)

    # Ultimate fallback: platform.processor() often gives *something*.
    return platform.processor() or "Unknown"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_system() -> SystemProfile:
    """Probe the current system and return a populated :class:`SystemProfile`.

    This function never raises; every detection step is wrapped in
    try/except so partial results are still returned.
    """
    logger.info("Starting system hardware detection ...")

    # -- CPU --
    cpu_name = _detect_cpu_name()
    try:
        cpu_cores_physical = psutil.cpu_count(logical=False) or 0
    except Exception:
        cpu_cores_physical = 0
    try:
        cpu_cores_logical = psutil.cpu_count(logical=True) or 0
    except Exception:
        cpu_cores_logical = 0
    try:
        freq = psutil.cpu_freq()
        cpu_freq_mhz = freq.current if freq else 0.0
    except Exception:
        cpu_freq_mhz = 0.0

    # -- Memory --
    try:
        vmem = psutil.virtual_memory()
        ram_total = vmem.total
        ram_available = vmem.available
    except Exception:
        ram_total = 0
        ram_available = 0

    # -- GPU --
    gpu_name, gpu_vram = _detect_gpu()
    gpu_detected = gpu_name != "Unknown" and gpu_vram > 0

    # -- GPU compute capability --
    compute_cap = _detect_gpu_compute_capability()
    if compute_cap == 0.0 and gpu_detected:
        compute_cap = _infer_compute_capability(gpu_name)
    supports_flash_attn = compute_cap >= 7.0

    # -- NUMA --
    numa_available, numa_node_count = _detect_numa()

    # -- OS --
    os_type = platform.system() or "Unknown"      # "Windows", "Linux", "Darwin"
    os_version = platform.version() or "Unknown"

    # -- Disk (free space on the partition where Ollama stores models) --
    try:
        ollama_home = os.environ.get("OLLAMA_MODELS", os.path.expanduser("~"))
        disk_usage = shutil.disk_usage(ollama_home)
        disk_free = disk_usage.free
    except Exception:
        disk_free = 0

    profile = SystemProfile(
        cpu_name=cpu_name,
        cpu_cores_physical=cpu_cores_physical,
        cpu_cores_logical=cpu_cores_logical,
        cpu_freq_mhz=cpu_freq_mhz,
        ram_total_bytes=ram_total,
        ram_available_bytes=ram_available,
        gpu_name=gpu_name,
        gpu_vram_bytes=gpu_vram,
        gpu_detected=gpu_detected,
        os_type=os_type,
        os_version=os_version,
        disk_free_bytes=disk_free,
        gpu_compute_capability=compute_cap,
        gpu_supports_flash_attn=supports_flash_attn,
        numa_available=numa_available,
        numa_node_count=numa_node_count,
    )

    logger.info(
        "System profile: CPU=%s (%d cores), RAM=%.1f GiB total / %.1f GiB avail, "
        "GPU=%s (%.1f GiB VRAM, CC=%.1f, FlashAttn=%s), OS=%s, NUMA=%s(%d), Disk free=%.1f GiB",
        profile.cpu_name,
        profile.cpu_cores_physical,
        profile.ram_total_gb,
        profile.ram_available_gb,
        profile.gpu_name,
        profile.gpu_vram_gb,
        profile.gpu_compute_capability,
        "Yes" if profile.gpu_supports_flash_attn else "No",
        profile.os_type,
        "Yes" if profile.numa_available else "No",
        profile.numa_node_count,
        profile.disk_free_gb,
    )
    return profile


def estimate_model_capacity(profile: SystemProfile) -> Dict[str, Dict[str, Any]]:
    """Estimate what model sizes can run at each quantization level.

    Uses **available** RAM (plus usable VRAM, if a GPU is detected) to
    determine the maximum number of parameters that can fit in memory at each
    quantization level, then maps that to common model-size buckets.

    Returns a dict keyed by quantization name, each containing::

        {
            "bits_per_param": float,
            "max_params_billions": float,   # max B-params that fit
            "max_model_size_gb": float,     # weight-file size in GiB
            "fits_7b":  bool,
            "fits_13b": bool,
            "fits_30b": bool,
            "fits_65b": bool,
            "fits_70b": bool,
        }
    """
    # Usable memory budget: available RAM minus a safety margin, plus GPU VRAM.
    ram_budget = max(profile.ram_available_bytes - _MIN_FREE_RAM_BYTES, 0)
    vram_budget = int(profile.gpu_vram_bytes * _USABLE_VRAM_FRACTION) if profile.gpu_detected else 0
    total_budget_bytes = ram_budget + vram_budget

    result: Dict[str, Dict[str, Any]] = {}

    for quant_name, bits in QUANTIZATION_BITS.items():
        bytes_per_param = bits / 8.0
        # Account for overhead (KV-cache, runtime buffers, etc.).
        effective_bytes_per_param = bytes_per_param * _MEMORY_OVERHEAD_FACTOR

        if effective_bytes_per_param > 0:
            max_params = total_budget_bytes / effective_bytes_per_param
        else:
            max_params = 0

        max_params_b = max_params / 1e9  # billions
        max_model_gb = (max_params * bytes_per_param) / (1024 ** 3)

        result[quant_name] = {
            "bits_per_param": bits,
            "max_params_billions": round(max_params_b, 1),
            "max_model_size_gb": round(max_model_gb, 2),
            "fits_7b":  max_params_b >= 7,
            "fits_13b": max_params_b >= 13,
            "fits_30b": max_params_b >= 30,
            "fits_65b": max_params_b >= 65,
            "fits_70b": max_params_b >= 70,
        }

    return result


def recommend_gpu_layers(profile: SystemProfile, model_size_gb: float) -> int:
    """Recommend how many transformer layers to offload to the GPU.

    The heuristic is simple but effective:

    1. If no GPU is detected, return 0.
    2. If the full model fits in usable VRAM, return a sentinel value of
       ``999`` (meaning "all layers" — Ollama interprets any number >=
       total layers as "offload everything").
    3. Otherwise, return the proportion of the model that fits in VRAM,
       translated to a layer count (assuming ~40 layers as a reasonable
       default for 7 B–70 B-class models, scaled linearly by model size).

    Args:
        profile: The detected :class:`SystemProfile`.
        model_size_gb: On-disk size of the quantized model in GiB.

    Returns:
        Recommended number of GPU layers (``num_gpu`` parameter for Ollama).
    """
    if not profile.gpu_detected or profile.gpu_vram_bytes == 0:
        logger.info("No GPU detected; recommending 0 GPU layers.")
        return 0

    if model_size_gb <= 0:
        return 0

    usable_vram_gb = (profile.gpu_vram_bytes * _USABLE_VRAM_FRACTION) / (1024 ** 3)
    model_bytes = model_size_gb * (1024 ** 3)

    # Estimate total layer count from model size (rough heuristic).
    # Small models (~7 B)  -> ~32 layers
    # Medium models (~13 B) -> ~40 layers
    # Large models (~65-70 B) -> ~80 layers
    if model_size_gb < 5:
        estimated_layers = 32
    elif model_size_gb < 10:
        estimated_layers = 32
    elif model_size_gb < 25:
        estimated_layers = 40
    elif model_size_gb < 45:
        estimated_layers = 60
    else:
        estimated_layers = 80

    if usable_vram_gb >= model_size_gb * _MEMORY_OVERHEAD_FACTOR:
        # Entire model fits in VRAM.
        logger.info(
            "Full model (%.1f GiB) fits in VRAM (%.1f GiB usable); "
            "recommending all %d layers on GPU.",
            model_size_gb, usable_vram_gb, estimated_layers,
        )
        return 999  # "all layers"

    # Partial offload: fraction of model that fits in VRAM.
    fraction = usable_vram_gb / (model_size_gb * _MEMORY_OVERHEAD_FACTOR)
    recommended = max(1, math.floor(fraction * estimated_layers))

    logger.info(
        "Partial GPU offload: %.1f GiB usable VRAM / %.1f GiB model "
        "(overhead %.0f%%) -> %d / %d layers on GPU.",
        usable_vram_gb,
        model_size_gb,
        (_MEMORY_OVERHEAD_FACTOR - 1) * 100,
        recommended,
        estimated_layers,
    )
    return recommended


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

def _pretty_print_profile(profile: SystemProfile) -> None:
    """Print a human-readable summary to stdout."""
    print("=" * 60)
    print("  System Profile")
    print("=" * 60)
    print(f"  OS            : {profile.os_type} ({profile.os_version})")
    print(f"  CPU           : {profile.cpu_name}")
    print(f"  Cores         : {profile.cpu_cores_physical} physical / {profile.cpu_cores_logical} logical")
    print(f"  CPU Freq      : {profile.cpu_freq_mhz:.0f} MHz")
    print(f"  RAM Total     : {profile.ram_total_gb:.1f} GiB")
    print(f"  RAM Available : {profile.ram_available_gb:.1f} GiB")
    print(f"  GPU           : {profile.gpu_name}")
    print(f"  GPU VRAM      : {profile.gpu_vram_gb:.1f} GiB")
    print(f"  GPU Detected  : {profile.gpu_detected}")
    if profile.gpu_compute_capability > 0:
        print(f"  Compute Cap   : {profile.gpu_compute_capability:.1f}")
    print(f"  Flash Attn    : {'Yes' if profile.gpu_supports_flash_attn else 'No'}")
    if profile.numa_available:
        print(f"  NUMA          : {profile.numa_node_count} nodes")
    print(f"  Disk Free     : {profile.disk_free_gb:.1f} GiB")
    print("=" * 60)


def _pretty_print_capacity(capacity: Dict[str, Dict[str, Any]]) -> None:
    """Print the capacity table to stdout."""
    print()
    print(f"{'Quant':<10} {'Bits':>5} {'Max Params (B)':>15} {'Max Size (GiB)':>15} "
          f"{'7B':>4} {'13B':>4} {'30B':>4} {'65B':>4} {'70B':>4}")
    print("-" * 78)
    for quant, info in capacity.items():
        yes_no = lambda v: " Yes" if v else "  No"  # noqa: E731
        print(
            f"{quant:<10} {info['bits_per_param']:>5.2f} "
            f"{info['max_params_billions']:>15.1f} "
            f"{info['max_model_size_gb']:>15.2f} "
            f"{yes_no(info['fits_7b'])}{yes_no(info['fits_13b'])}"
            f"{yes_no(info['fits_30b'])}{yes_no(info['fits_65b'])}"
            f"{yes_no(info['fits_70b'])}"
        )
    print()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    profile = detect_system()
    _pretty_print_profile(profile)

    capacity = estimate_model_capacity(profile)
    _pretty_print_capacity(capacity)

    # Example: recommend GPU layers for a ~4 GiB Q4_K_M 7-B model.
    layers = recommend_gpu_layers(profile, model_size_gb=4.0)
    print(f"Recommended GPU layers for a 4 GiB model: {layers}")
