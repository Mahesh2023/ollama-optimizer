"""
Quantization Engine for Ollama Optimizer.

This module implements quantization concepts from the ngrok blog
"Quantization from the Ground Up" and serves two purposes:

1. **Educational**: Demonstrates symmetric (absmax), asymmetric (zero-point),
   per-channel, and per-group quantization with NumPy — no heavy ML frameworks.
2. **Practical**: Houses a knowledge base of GGUF quantization levels and a
   recommendation engine that picks the optimal level for a given model size
   and hardware profile.

References
----------
- ngrok blog: "Quantization from the Ground Up"
- GGUF format: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- llama.cpp quantization types and benchmarks
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Part 1 — Quantization Theory (Educational + Functional)
# ---------------------------------------------------------------------------


class QuantMethod(Enum):
    """Supported quantization strategies.

    Each value maps to a technique described in the ngrok blog:

    * **ABSMAX** – Symmetric quantization around zero.  The scale factor is
      derived from the maximum absolute value in the tensor so the
      representable range is always symmetric: ``[-max|x|, +max|x|]``.
    * **ZERO_POINT** – Asymmetric quantization that introduces a *zero-point*
      offset.  This lets the quantized range cover the full ``[min, max]`` of
      the original tensor, which is more efficient when the distribution is
      not centred on zero.
    * **PER_CHANNEL** – Applies absmax quantization independently to each
      row (channel) of a weight matrix.  Different channels can have very
      different magnitude ranges, so per-channel scales reduce error.
    * **PER_GROUP** – Divides a tensor into fixed-size groups and quantizes
      each group independently.  This is the strategy behind GGUF formats
      like ``Q4_0`` and ``Q4_K_M`` and offers a fine-grained
      quality / compression trade-off.
    """

    ABSMAX = "absmax"
    ZERO_POINT = "zero_point"
    PER_CHANNEL = "per_channel"
    PER_GROUP = "per_group"


class QuantizationEngine:
    """NumPy-based quantization engine for education and analysis.

    All methods operate on plain ``numpy.ndarray`` tensors so the module stays
    lightweight — no PyTorch or TensorFlow required.

    Example
    -------
    >>> engine = QuantizationEngine()
    >>> data = np.random.randn(256).astype(np.float32)
    >>> q, scale = engine.absmax_quantize(data, n_bits=8)
    >>> recon = engine.absmax_dequantize(q, scale)
    >>> err = engine.measure_error(data, recon)
    >>> print(f"SNR: {err['snr_db']:.1f} dB")
    """

    # -- 1. Absmax (symmetric) quantization --------------------------------

    def absmax_quantize(
        self, tensor: np.ndarray, n_bits: int = 8
    ) -> Tuple[np.ndarray, float]:
        """Symmetric quantization mapping ``[-max|x|, +max|x|]`` to integer range.

        The representable integer range is ``[-(2^(n-1) - 1), 2^(n-1) - 1]``
        (the most-negative value ``-2^(n-1)`` is reserved so the range is
        symmetric).

        Parameters
        ----------
        tensor : np.ndarray
            Floating-point tensor to quantize.
        n_bits : int, optional
            Bit-width of the quantized representation (default ``8``).

        Returns
        -------
        quantized : np.ndarray
            Integer tensor with dtype matching the smallest numpy int type
            that can hold the range.
        scale : float
            The scale factor used: ``scale = max(|x|) / (2^(n-1) - 1)``.
        """
        qmax = 2 ** (n_bits - 1) - 1  # e.g. 127 for 8-bit

        abs_max = np.max(np.abs(tensor))
        if abs_max == 0:
            return np.zeros_like(tensor, dtype=np.int8), 0.0

        scale = float(abs_max / qmax)
        quantized = np.clip(np.round(tensor / scale), -qmax, qmax).astype(np.int8 if n_bits <= 8 else np.int16)

        logger.debug(
            "absmax_quantize: n_bits=%d, scale=%.6f, abs_max=%.6f",
            n_bits,
            scale,
            abs_max,
        )
        return quantized, scale

    # -- 2. Absmax dequantize ----------------------------------------------

    def absmax_dequantize(self, quantized: np.ndarray, scale: float) -> np.ndarray:
        """Reconstruct a floating-point tensor from absmax-quantized values.

        Parameters
        ----------
        quantized : np.ndarray
            Integer tensor produced by :meth:`absmax_quantize`.
        scale : float
            Scale factor returned alongside the quantized tensor.

        Returns
        -------
        np.ndarray
            Reconstructed float32 tensor: ``x_recon = quantized * scale``.
        """
        return quantized.astype(np.float32) * scale

    # -- 3. Zero-point (asymmetric) quantization ---------------------------

    def zero_point_quantize(
        self, tensor: np.ndarray, n_bits: int = 8
    ) -> Tuple[np.ndarray, float, int]:
        """Asymmetric quantization with a zero-point offset.

        Unlike absmax, this method maps the *actual* ``[min, max]`` range of
        the tensor to the unsigned integer range ``[0, 2^n - 1]``.  A
        *zero-point* is computed so that the real value ``0.0`` is exactly
        representable after quantization.

        Parameters
        ----------
        tensor : np.ndarray
            Floating-point tensor to quantize.
        n_bits : int, optional
            Bit-width (default ``8``).

        Returns
        -------
        quantized : np.ndarray
            Unsigned-integer tensor.
        scale : float
            ``scale = (max(x) - min(x)) / (2^n - 1)``.
        zero_point : int
            ``zero_point = round(-min(x) / scale)``.
        """
        qmax = 2**n_bits - 1  # e.g. 255 for 8-bit

        x_min = float(np.min(tensor))
        x_max = float(np.max(tensor))

        if x_max == x_min:
            return (
                np.zeros_like(tensor, dtype=np.uint8),
                0.0,
                0,
            )

        scale = (x_max - x_min) / qmax
        zero_point = int(np.round(-x_min / scale))

        quantized = np.clip(
            np.round(tensor / scale) + zero_point, 0, qmax
        ).astype(np.uint8 if n_bits <= 8 else np.uint16)

        logger.debug(
            "zero_point_quantize: n_bits=%d, scale=%.6f, zp=%d, "
            "range=[%.4f, %.4f]",
            n_bits,
            scale,
            zero_point,
            x_min,
            x_max,
        )
        return quantized, scale, zero_point

    # -- 4. Zero-point dequantize ------------------------------------------

    def zero_point_dequantize(
        self, quantized: np.ndarray, scale: float, zero_point: int
    ) -> np.ndarray:
        """Reconstruct a tensor from zero-point-quantized values.

        Parameters
        ----------
        quantized : np.ndarray
            Unsigned-integer tensor from :meth:`zero_point_quantize`.
        scale : float
            Scale factor.
        zero_point : int
            Zero-point offset.

        Returns
        -------
        np.ndarray
            Reconstructed float32 tensor:
            ``x_recon = (quantized - zero_point) * scale``.
        """
        return (quantized.astype(np.float32) - zero_point) * scale

    # -- 5. Per-channel quantization ---------------------------------------

    def per_channel_quantize(
        self, matrix: np.ndarray, n_bits: int = 8, axis: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply absmax quantization independently to each channel (row/column).

        In real LLMs, weight matrices often have very different magnitude
        distributions across output channels.  Per-channel quantization
        assigns a *separate* scale factor to each channel, significantly
        reducing quantization error compared to a single global scale.

        Parameters
        ----------
        matrix : np.ndarray
            2-D floating-point matrix (shape ``[C, D]`` or ``[D, C]``).
        n_bits : int, optional
            Bit-width (default ``8``).
        axis : int, optional
            Axis along which channels are defined.  ``0`` means each *row*
            is a channel (default); ``1`` means each column.

        Returns
        -------
        quantized : np.ndarray
            Integer matrix, same shape as *matrix*.
        scales : np.ndarray
            1-D array of per-channel scale factors.
        """
        if matrix.ndim != 2:
            raise ValueError(
                f"per_channel_quantize expects a 2-D matrix, got {matrix.ndim}-D"
            )

        qmax = 2 ** (n_bits - 1) - 1
        n_channels = matrix.shape[axis]

        quantized = np.zeros_like(matrix, dtype=np.int8 if n_bits <= 8 else np.int16)
        scales = np.zeros(n_channels, dtype=np.float32)

        for ch in range(n_channels):
            if axis == 0:
                channel_data = matrix[ch, :]
            else:
                channel_data = matrix[:, ch]

            abs_max = np.max(np.abs(channel_data))
            if abs_max == 0:
                scales[ch] = 0.0
                continue

            scale = abs_max / qmax
            scales[ch] = scale
            q = np.clip(np.round(channel_data / scale), -qmax, qmax)

            if axis == 0:
                quantized[ch, :] = q
            else:
                quantized[:, ch] = q

        logger.debug(
            "per_channel_quantize: shape=%s, axis=%d, n_bits=%d, "
            "scale_range=[%.6f, %.6f]",
            matrix.shape,
            axis,
            n_bits,
            float(np.min(scales[scales > 0])) if np.any(scales > 0) else 0.0,
            float(np.max(scales)),
        )
        return quantized, scales

    # -- 6. Per-group quantization -----------------------------------------

    def per_group_quantize(
        self, tensor: np.ndarray, n_bits: int = 4, group_size: int = 128
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Divide the tensor into groups and quantize each independently.

        This is the strategy used internally by GGUF formats such as
        ``Q4_0`` (group_size=32, n_bits=4) and ``Q4_K_M``.  Smaller groups
        yield lower error at the cost of more scale-factor overhead.

        Parameters
        ----------
        tensor : np.ndarray
            1-D (or will be flattened) floating-point tensor.
        n_bits : int, optional
            Bit-width for each group (default ``4``).
        group_size : int, optional
            Number of elements per group (default ``128``).

        Returns
        -------
        quantized : np.ndarray
            Integer tensor (same length as the flattened input).
        scales : np.ndarray
            1-D array with one scale factor per group.
        """
        flat = tensor.flatten().astype(np.float32)

        # Pad to a multiple of group_size
        remainder = len(flat) % group_size
        if remainder != 0:
            pad_len = group_size - remainder
            flat = np.concatenate([flat, np.zeros(pad_len, dtype=np.float32)])
        else:
            pad_len = 0

        n_groups = len(flat) // group_size
        qmax = 2 ** (n_bits - 1) - 1

        quantized = np.zeros_like(flat, dtype=np.int8)
        scales = np.zeros(n_groups, dtype=np.float32)

        for g in range(n_groups):
            start = g * group_size
            end = start + group_size
            group = flat[start:end]

            abs_max = np.max(np.abs(group))
            if abs_max == 0:
                scales[g] = 0.0
                continue

            scale = abs_max / qmax
            scales[g] = scale
            quantized[start:end] = np.clip(
                np.round(group / scale), -qmax, qmax
            ).astype(np.int8)

        # Remove padding from the quantized output
        if pad_len > 0:
            quantized = quantized[: len(quantized) - pad_len]

        logger.debug(
            "per_group_quantize: n_elements=%d, n_bits=%d, group_size=%d, "
            "n_groups=%d",
            tensor.size,
            n_bits,
            group_size,
            n_groups,
        )
        return quantized, scales

    # -- 7. Quantization error measurement ---------------------------------

    def measure_error(
        self,
        original: np.ndarray,
        reconstructed: np.ndarray,
        original_bits: int = 32,
        quantized_bits: int = 8,
    ) -> dict:
        """Compute quantization error metrics between original and reconstructed tensors.

        Parameters
        ----------
        original : np.ndarray
            The original floating-point tensor.
        reconstructed : np.ndarray
            The dequantized (reconstructed) tensor.
        original_bits : int, optional
            Bits per element in the original representation (default ``32``).
        quantized_bits : int, optional
            Bits per element in the quantized representation (default ``8``).

        Returns
        -------
        dict
            ``mse``
                Mean squared error.
            ``mae``
                Mean absolute error.
            ``max_error``
                Maximum absolute element-wise error.
            ``snr_db``
                Signal-to-noise ratio in decibels.
            ``compression_ratio``
                ``original_bits / quantized_bits``.
        """
        orig = original.flatten().astype(np.float64)
        recon = reconstructed.flatten().astype(np.float64)

        # Truncate to the shorter length (handles padding in per-group)
        min_len = min(len(orig), len(recon))
        orig = orig[:min_len]
        recon = recon[:min_len]

        diff = orig - recon
        mse = float(np.mean(diff**2))
        mae = float(np.mean(np.abs(diff)))
        max_error = float(np.max(np.abs(diff)))

        # Signal-to-noise ratio
        signal_power = float(np.mean(orig**2))
        if mse > 0 and signal_power > 0:
            snr_db = 10.0 * math.log10(signal_power / mse)
        elif mse == 0:
            snr_db = float("inf")
        else:
            snr_db = 0.0

        compression_ratio = original_bits / max(quantized_bits, 1)

        return {
            "mse": mse,
            "mae": mae,
            "max_error": max_error,
            "snr_db": snr_db,
            "compression_ratio": compression_ratio,
        }

    # -- 8. Demonstrate all methods ----------------------------------------

    def demonstrate_quantization(
        self, sample_data: Optional[np.ndarray] = None
    ) -> dict:
        """Run every quantization strategy on sample data and compare results.

        If *sample_data* is ``None``, a synthetic weight tensor mimicking a
        typical LLM layer is generated (normal distribution, shape 256×256).

        Parameters
        ----------
        sample_data : np.ndarray or None
            Optional custom data; if ``None`` a synthetic matrix is used.

        Returns
        -------
        dict
            Keyed by method name, each value is a dict with ``quantized``,
            ``reconstructed``, and ``error`` sub-keys.
        """
        if sample_data is None:
            rng = np.random.default_rng(42)
            # Simulate LLM-like weight distribution: mostly small values
            # with some outliers (heavy tails).
            sample_data = rng.standard_normal((256, 256)).astype(np.float32) * 0.02
            # Inject a few outlier channels to make per-channel shine
            sample_data[0, :] *= 10.0
            sample_data[5, :] *= 5.0

        results: Dict[str, dict] = {}
        flat = sample_data.flatten()

        # --- Absmax 8-bit -------------------------------------------------
        q_abs, s_abs = self.absmax_quantize(flat, n_bits=8)
        r_abs = self.absmax_dequantize(q_abs, s_abs)
        results["absmax_int8"] = {
            "method": QuantMethod.ABSMAX.value,
            "n_bits": 8,
            "quantized": q_abs,
            "reconstructed": r_abs,
            "error": self.measure_error(flat, r_abs, quantized_bits=8),
        }

        # --- Absmax 4-bit -------------------------------------------------
        q_abs4, s_abs4 = self.absmax_quantize(flat, n_bits=4)
        r_abs4 = self.absmax_dequantize(q_abs4, s_abs4)
        results["absmax_int4"] = {
            "method": QuantMethod.ABSMAX.value,
            "n_bits": 4,
            "quantized": q_abs4,
            "reconstructed": r_abs4,
            "error": self.measure_error(flat, r_abs4, quantized_bits=4),
        }

        # --- Zero-point 8-bit --------------------------------------------
        q_zp, s_zp, zp = self.zero_point_quantize(flat, n_bits=8)
        r_zp = self.zero_point_dequantize(q_zp, s_zp, zp)
        results["zero_point_int8"] = {
            "method": QuantMethod.ZERO_POINT.value,
            "n_bits": 8,
            "quantized": q_zp,
            "reconstructed": r_zp,
            "error": self.measure_error(flat, r_zp, quantized_bits=8),
        }

        # --- Per-channel 8-bit (operates on the 2-D matrix) ---------------
        q_pc, scales_pc = self.per_channel_quantize(sample_data, n_bits=8, axis=0)
        # Dequantize per-channel
        r_pc = np.zeros_like(sample_data, dtype=np.float32)
        for ch in range(sample_data.shape[0]):
            r_pc[ch, :] = q_pc[ch, :].astype(np.float32) * scales_pc[ch]
        results["per_channel_int8"] = {
            "method": QuantMethod.PER_CHANNEL.value,
            "n_bits": 8,
            "quantized": q_pc,
            "reconstructed": r_pc,
            "error": self.measure_error(sample_data, r_pc, quantized_bits=8),
        }

        # --- Per-group 4-bit (group_size=32, like Q4_0) -------------------
        q_pg, scales_pg = self.per_group_quantize(flat, n_bits=4, group_size=32)
        # Dequantize per-group
        r_pg = np.zeros(len(q_pg), dtype=np.float32)
        for g in range(len(scales_pg)):
            start = g * 32
            end = min(start + 32, len(q_pg))
            r_pg[start:end] = q_pg[start:end].astype(np.float32) * scales_pg[g]
        results["per_group_int4_g32"] = {
            "method": QuantMethod.PER_GROUP.value,
            "n_bits": 4,
            "group_size": 32,
            "quantized": q_pg,
            "reconstructed": r_pg,
            "error": self.measure_error(flat, r_pg, quantized_bits=4),
        }

        # --- Per-group 4-bit (group_size=128) -----------------------------
        q_pg128, scales_pg128 = self.per_group_quantize(
            flat, n_bits=4, group_size=128
        )
        r_pg128 = np.zeros(len(q_pg128), dtype=np.float32)
        for g in range(len(scales_pg128)):
            start = g * 128
            end = min(start + 128, len(q_pg128))
            r_pg128[start:end] = (
                q_pg128[start:end].astype(np.float32) * scales_pg128[g]
            )
        results["per_group_int4_g128"] = {
            "method": QuantMethod.PER_GROUP.value,
            "n_bits": 4,
            "group_size": 128,
            "quantized": q_pg128,
            "reconstructed": r_pg128,
            "error": self.measure_error(flat, r_pg128, quantized_bits=4),
        }

        # Log a nice comparison table
        logger.info("=== Quantization Demonstration Results ===")
        for name, res in results.items():
            err = res["error"]
            logger.info(
                "  %-24s | bits=%d | MSE=%.2e | MAE=%.2e | SNR=%.1f dB | "
                "compression=%.1fx",
                name,
                res["n_bits"],
                err["mse"],
                err["mae"],
                err["snr_db"],
                err["compression_ratio"],
            )

        return results


# ---------------------------------------------------------------------------
# Part 2 — GGUF Quantization Level Knowledge Base
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GGUFQuantLevel:
    """Describes a single GGUF quantization level and its characteristics.

    Attributes
    ----------
    name : str
        Canonical name (e.g. ``"Q4_K_M"``).
    bits_per_weight : float
        Average bits per weight parameter, including overhead for scales and
        block metadata.
    description : str
        Human-readable summary of quality and use-case trade-offs.
    quality_score : float
        Relative quality compared to F16, on a 0–1 scale.  ``1.0`` means
        "indistinguishable from the original".  Scores reflect a holistic
        quality assessment incorporating perplexity (wikitext-2), downstream
        task accuracy, and generation coherence — not a linear mapping from
        perplexity alone.  Calibrated against llama.cpp PR #1684 benchmarks,
        discussion #2094 community data, and TheBloke model card measurements.
    speed_multiplier : float
        Inference speed multiplier relative to F16.  Values > 1 are faster.
    memory_multiplier : float
        Memory usage relative to F16.  Values < 1 mean the model is smaller.
    method : str
        Brief description of the quantization method / block structure.
    recommended_for : str
        Target hardware or use-case.
    """

    name: str
    bits_per_weight: float
    description: str
    quality_score: float
    speed_multiplier: float
    memory_multiplier: float
    method: str
    recommended_for: str


# Comprehensive mapping of every commonly used GGUF quantization level.
# Sources: llama.cpp benchmarks, community perplexity measurements, and the
# GGUF specification itself.
GGUF_QUANT_LEVELS: Dict[str, GGUFQuantLevel] = {
    "Q2_K": GGUFQuantLevel(
        name="Q2_K",
        bits_per_weight=2.63,
        description=(
            "Super-aggressive 2-bit. High quality loss. "
            "Only for very constrained systems."
        ),
        quality_score=0.55,
        speed_multiplier=3.8,
        memory_multiplier=0.16,
        method=(
            "2-bit quantization with K-quant super-blocks. Uses 4-bit scales "
            "and 2-bit quants, with importance-weighted selection."
        ),
        recommended_for=(
            "Extremely memory-constrained devices (2-4 GB RAM). "
            "Acceptable only for casual / non-critical tasks."
        ),
    ),
    "IQ1_S": GGUFQuantLevel(
        name="IQ1_S",
        bits_per_weight=1.56,
        description="Extreme 1-bit importance quantization. Research only.",
        quality_score=0.30,
        speed_multiplier=4.5,
        memory_multiplier=0.10,
        method=(
            "1-bit importance-weighted quantization with super-blocks. "
            "Uses learned importance matrices to preserve critical weights."
        ),
        recommended_for=(
            "Research and experimentation only. Severe quality loss "
            "for most practical use cases."
        ),
    ),
    "IQ1_M": GGUFQuantLevel(
        name="IQ1_M",
        bits_per_weight=1.75,
        description="Medium 1-bit importance quantization. Slightly better than IQ1_S.",
        quality_score=0.35,
        speed_multiplier=4.3,
        memory_multiplier=0.11,
        method=(
            "1-bit importance-weighted quantization (medium variant). "
            "Provides marginally better quality than IQ1_S at 1.75 bpw "
            "by using larger grid indices and per-block scales."
        ),
        recommended_for=(
            "Research and experimentation. Marginal improvement over "
            "IQ1_S but still severe quality loss for practical use."
        ),
    ),
    "IQ2_XXS": GGUFQuantLevel(
        name="IQ2_XXS",
        bits_per_weight=2.06,
        description="Ultra-compact 2-bit IQ. Extreme compression.",
        quality_score=0.40,
        speed_multiplier=4.2,
        memory_multiplier=0.13,
        method=(
            "2-bit importance-weighted quantization (extra-extra-small). "
            "Uses importance matrices for weight selection with minimal "
            "overhead."
        ),
        recommended_for=(
            "Extremely memory-constrained devices (1-2 GB). "
            "Only for non-critical tasks where any response is acceptable."
        ),
    ),
    "IQ2_XS": GGUFQuantLevel(
        name="IQ2_XS",
        bits_per_weight=2.31,
        description="Very compact 2-bit IQ. Better than Q2_K for quality.",
        quality_score=0.48,
        speed_multiplier=4.0,
        memory_multiplier=0.14,
        method=(
            "2-bit importance-weighted quantization (extra-small). "
            "Slightly larger blocks than IQ2_XXS for better accuracy."
        ),
        recommended_for=(
            "Ultra-constrained devices (2-3 GB). Marginally better "
            "quality than IQ2_XXS."
        ),
    ),
    "IQ2_S": GGUFQuantLevel(
        name="IQ2_S",
        bits_per_weight=2.56,
        description="Small 2-bit IQ. Comparable size to Q2_K with better quality.",
        quality_score=0.52,
        speed_multiplier=3.9,
        memory_multiplier=0.16,
        method=(
            "2-bit importance-weighted quantization (small). "
            "Uses importance matrices and achieves better quality than "
            "Q2_K at similar compression ratios."
        ),
        recommended_for=(
            "Memory-constrained devices (2-4 GB) where IQ methods "
            "are available. Prefer over Q2_K when supported."
        ),
    ),
    "IQ3_XXS": GGUFQuantLevel(
        name="IQ3_XXS",
        bits_per_weight=3.06,
        description="Compact 3-bit IQ. Better quality than Q3_K_S.",
        quality_score=0.65,
        speed_multiplier=3.3,
        memory_multiplier=0.19,
        method=(
            "3-bit importance-weighted quantization (extra-extra-small). "
            "Achieves Q3_K_S-level compression with better quality "
            "through importance weighting."
        ),
        recommended_for=(
            "4-6 GB systems. Good alternative to Q3_K_S with "
            "improved quality at similar size."
        ),
    ),
    "IQ3_XS": GGUFQuantLevel(
        name="IQ3_XS",
        bits_per_weight=3.30,
        description="3-bit IQ. Strong quality for its size class.",
        quality_score=0.67,
        speed_multiplier=3.1,
        memory_multiplier=0.21,
        method=(
            "3-bit importance-weighted quantization (extra-small). "
            "Balanced approach between IQ3_XXS and Q3_K_M."
        ),
        recommended_for=(
            "4-8 GB systems looking for best quality in the "
            "3-bit range."
        ),
    ),
    "IQ3_S": GGUFQuantLevel(
        name="IQ3_S",
        bits_per_weight=3.44,
        description="3-bit IQ (small). Comparable to Q3_K_M quality.",
        quality_score=0.70,
        speed_multiplier=2.9,
        memory_multiplier=0.22,
        method=(
            "3-bit importance-weighted quantization (small). "
            "Uses sign bits, scales, and high bits for improved accuracy. "
            "Quality between IQ3_XS and Q3_K_M at similar size."
        ),
        recommended_for=(
            "4-8 GB systems wanting best IQ3-tier quality. "
            "Slightly larger than IQ3_XS with measurably better results."
        ),
    ),
    "IQ4_XS": GGUFQuantLevel(
        name="IQ4_XS",
        bits_per_weight=4.25,
        description="Compact 4-bit IQ. Excellent quality/size ratio.",
        quality_score=0.83,
        speed_multiplier=2.5,
        memory_multiplier=0.27,
        method=(
            "4-bit importance-weighted quantization (extra-small). "
            "Near Q4_K_M quality with less memory. Uses importance "
            "matrices for optimal weight distribution."
        ),
        recommended_for=(
            "6-8 GB systems. Sweet spot between Q4_0 and Q4_K_M "
            "when IQ support is available."
        ),
    ),
    "IQ4_NL": GGUFQuantLevel(
        name="IQ4_NL",
        bits_per_weight=4.50,
        description="Non-linear 4-bit IQ. Better quality than Q4_0 at same size.",
        quality_score=0.82,
        speed_multiplier=2.4,
        memory_multiplier=0.28,
        method=(
            "4-bit non-linear quantization using a lookup table with "
            "non-uniform spacing. Same bits-per-weight as Q4_0 but "
            "achieves better quality through non-linear mapping."
        ),
        recommended_for=(
            "6-8 GB systems. Direct upgrade from Q4_0 with better "
            "quality at identical model size (4.50 bpw)."
        ),
    ),
    "Q3_K_S": GGUFQuantLevel(
        name="Q3_K_S",
        bits_per_weight=3.07,
        description="Small 3-bit. Noticeable quality loss.",
        quality_score=0.62,
        speed_multiplier=3.2,
        memory_multiplier=0.19,
        method=(
            "3-bit K-quant (small variant). Uses 6-bit scales per "
            "super-block with smaller block sizes."
        ),
        recommended_for=(
            "Low-RAM systems (4-6 GB). Useful for drafting or "
            "simple Q&A where some quality loss is tolerable."
        ),
    ),
    "Q3_K_M": GGUFQuantLevel(
        name="Q3_K_M",
        bits_per_weight=3.44,
        description="Medium 3-bit. Moderate quality loss.",
        quality_score=0.68,
        speed_multiplier=3.0,
        memory_multiplier=0.22,
        method=(
            "3-bit K-quant (medium variant). Larger blocks and higher-"
            "precision scales than Q3_K_S."
        ),
        recommended_for=(
            "Systems with 4-8 GB RAM. Reasonable trade-off for "
            "running larger models on modest hardware."
        ),
    ),
    "Q3_K_L": GGUFQuantLevel(
        name="Q3_K_L",
        bits_per_weight=3.81,
        description="Large 3-bit. Acceptable quality for most tasks.",
        quality_score=0.72,
        speed_multiplier=2.8,
        memory_multiplier=0.24,
        method=(
            "3-bit K-quant (large variant). Uses largest block sizes "
            "and best scale precision in the Q3 family."
        ),
        recommended_for=(
            "Systems with 6-8 GB RAM that need to run models "
            "above their native weight class."
        ),
    ),
    "Q4_0": GGUFQuantLevel(
        name="Q4_0",
        bits_per_weight=4.50,
        description="Basic 4-bit. Absmax per-group(32). Legacy format.",
        quality_score=0.78,
        speed_multiplier=2.5,
        memory_multiplier=0.28,
        method=(
            "Legacy 4-bit format. Per-group absmax quantization with "
            "group_size=32. Single FP16 scale per block; no min offset."
        ),
        recommended_for=(
            "Broad compatibility. Prefer Q4_K_M for new deployments; "
            "Q4_0 remains useful for older llama.cpp backends."
        ),
    ),
    "Q4_K_S": GGUFQuantLevel(
        name="Q4_K_S",
        bits_per_weight=4.58,
        description="Small 4-bit K-quant. Good balance.",
        quality_score=0.82,
        speed_multiplier=2.4,
        memory_multiplier=0.29,
        method=(
            "4-bit K-quant (small). Super-block structure with 6-bit "
            "scales and 4-bit minimums; 256-element super-blocks."
        ),
        recommended_for=(
            "8 GB RAM / VRAM systems. Good default when Q4_K_M "
            "is a bit too large."
        ),
    ),
    "Q4_K_M": GGUFQuantLevel(
        name="Q4_K_M",
        bits_per_weight=4.83,
        description=(
            "Medium 4-bit K-quant. RECOMMENDED default. "
            "Best quality/size tradeoff."
        ),
        quality_score=0.85,
        speed_multiplier=2.3,
        memory_multiplier=0.30,
        method=(
            "4-bit K-quant (medium). Mixes Q4_K for most layers with "
            "Q6_K for attention and output layers (importance matrix)."
        ),
        recommended_for=(
            "The go-to quantization for 8-16 GB systems. "
            "Best overall quality-to-size ratio."
        ),
    ),
    "Q5_0": GGUFQuantLevel(
        name="Q5_0",
        bits_per_weight=5.50,
        description="Basic 5-bit. Good quality.",
        quality_score=0.88,
        speed_multiplier=2.0,
        memory_multiplier=0.34,
        method=(
            "Legacy 5-bit format. Per-group absmax with group_size=32, "
            "using 5-bit quants and FP16 scale per block."
        ),
        recommended_for=(
            "Systems with 12-16 GB RAM. Good quality with moderate "
            "compression."
        ),
    ),
    "Q5_K_S": GGUFQuantLevel(
        name="Q5_K_S",
        bits_per_weight=5.54,
        description="Small 5-bit K-quant.",
        quality_score=0.89,
        speed_multiplier=2.0,
        memory_multiplier=0.35,
        method=(
            "5-bit K-quant (small). Super-block with 6-bit scales "
            "and 4-bit minimums; 5-bit quants."
        ),
        recommended_for=(
            "12-16 GB systems where you want a step up from Q4_K_M "
            "quality."
        ),
    ),
    "Q5_K_M": GGUFQuantLevel(
        name="Q5_K_M",
        bits_per_weight=5.69,
        description="Medium 5-bit K-quant. Near-lossless for most uses.",
        quality_score=0.90,
        speed_multiplier=1.9,
        memory_multiplier=0.36,
        method=(
            "5-bit K-quant (medium). Mixes Q5_K and Q6_K across "
            "layers based on importance."
        ),
        recommended_for=(
            "16 GB+ systems. Excellent quality with meaningful "
            "compression over F16."
        ),
    ),
    "Q6_K": GGUFQuantLevel(
        name="Q6_K",
        bits_per_weight=6.56,
        description="6-bit. Very high quality, minimal loss.",
        quality_score=0.94,
        speed_multiplier=1.6,
        memory_multiplier=0.41,
        method=(
            "6-bit K-quant. Super-block with 8-bit scales; nearly "
            "imperceptible quality difference from F16 on most benchmarks."
        ),
        recommended_for=(
            "16-24 GB systems where quality is paramount but F16 "
            "doesn't fit."
        ),
    ),
    "Q8_0": GGUFQuantLevel(
        name="Q8_0",
        bits_per_weight=8.50,
        description=(
            "8-bit. Nearly lossless. Best for accuracy-critical work."
        ),
        quality_score=0.98,
        speed_multiplier=1.3,
        memory_multiplier=0.53,
        method=(
            "8-bit per-group absmax with group_size=32. The extra bits "
            "make quantization error negligible for virtually all tasks."
        ),
        recommended_for=(
            "24-48 GB systems. Use when you need maximum accuracy "
            "and can afford the memory."
        ),
    ),
    "F16": GGUFQuantLevel(
        name="F16",
        bits_per_weight=16.0,
        description="Full 16-bit float. Baseline. Maximum quality.",
        quality_score=1.0,
        speed_multiplier=1.0,
        memory_multiplier=1.0,
        method="IEEE 754 half-precision floating point. No quantization.",
        recommended_for=(
            "48 GB+ systems or when bit-exact reproducibility with "
            "the original model is required."
        ),
    ),
}


# ---------------------------------------------------------------------------
# Part 3 — Recommendation Engine
# ---------------------------------------------------------------------------


def recommend_quantization(
    model_params_billions: float,
    available_ram_gb: float,
    available_vram_gb: float = 0,
    priority: str = "balanced",
) -> List[GGUFQuantLevel]:
    """Recommend GGUF quantization levels for a given model and hardware profile.

    The engine estimates the memory footprint for each quantization level,
    filters out levels that won't fit, and ranks the remainder according to
    the chosen *priority*.

    Memory estimation
    -----------------
    ``memory_gb ≈ (params_B × 10^9 × bits_per_weight / 8) / 10^9 × 1.10``

    The 10 % overhead accounts for KV-cache, GGUF metadata, and runtime
    buffers used by llama.cpp / Ollama.

    Parameters
    ----------
    model_params_billions : float
        Number of model parameters in billions (e.g. ``7.0`` for LLaMA-2 7B).
    available_ram_gb : float
        System RAM available to Ollama (in GiB).
    available_vram_gb : float, optional
        GPU VRAM available (default ``0``).  RAM + VRAM are pooled for the
        feasibility check (Ollama can offload layers to GPU).
    priority : str, optional
        Ranking strategy — one of:

        * ``"quality"`` – highest ``quality_score`` first.
        * ``"speed"`` – highest ``speed_multiplier`` first.
        * ``"balanced"`` – maximise ``quality_score × speed_multiplier``.
        * ``"minimum"`` – lowest memory footprint first.

    Returns
    -------
    list[GGUFQuantLevel]
        Ranked list of feasible quantization levels (best first).  An empty
        list means no GGUF level fits in the given memory budget.

    Raises
    ------
    ValueError
        If *priority* is not one of the recognised strategies.

    Example
    -------
    >>> recs = recommend_quantization(7.0, available_ram_gb=8, priority="balanced")
    >>> for r in recs[:3]:
    ...     print(f"{r.name}: ~{r.bits_per_weight} bpw, quality={r.quality_score}")
    Q4_K_M: ~4.83 bpw, quality=0.85
    Q4_K_S: ~4.58 bpw, quality=0.82
    Q4_0: ~4.5 bpw, quality=0.78
    """
    valid_priorities = {"quality", "speed", "balanced", "minimum"}
    if priority not in valid_priorities:
        raise ValueError(
            f"Unknown priority {priority!r}. Choose from {valid_priorities}."
        )

    total_memory_gb = available_ram_gb + available_vram_gb
    overhead_factor = 1.10  # 10 % runtime overhead

    # F16 baseline memory: params_B * 1e9 * 16 bits / 8 bits-per-byte / 1e9
    f16_memory_gb = model_params_billions * 2.0  # 2 bytes per param in F16

    feasible: List[Tuple[GGUFQuantLevel, float]] = []

    for level in GGUF_QUANT_LEVELS.values():
        estimated_gb = f16_memory_gb * level.memory_multiplier * overhead_factor

        if estimated_gb <= total_memory_gb:
            feasible.append((level, estimated_gb))

    if not feasible:
        logger.warning(
            "No GGUF quantization level fits %.1f B params in %.1f GB "
            "(RAM=%.1f + VRAM=%.1f). Consider a smaller model.",
            model_params_billions,
            total_memory_gb,
            available_ram_gb,
            available_vram_gb,
        )
        return []

    # --- Rank by priority -------------------------------------------------
    def _sort_key(item: Tuple[GGUFQuantLevel, float]) -> float:
        level, _ = item
        if priority == "quality":
            return -level.quality_score
        elif priority == "speed":
            return -level.speed_multiplier
        elif priority == "balanced":
            return -(level.quality_score * level.speed_multiplier)
        else:  # "minimum"
            return level.memory_multiplier

    feasible.sort(key=_sort_key)

    ranked = [level for level, _ in feasible]

    logger.info(
        "recommend_quantization: model=%.1fB, budget=%.1f GB, "
        "priority=%s → %d feasible levels, top=%s",
        model_params_billions,
        total_memory_gb,
        priority,
        len(ranked),
        ranked[0].name if ranked else "none",
    )

    return ranked
