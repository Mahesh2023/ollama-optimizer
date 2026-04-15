"""
ollama-optimizer: Cross-platform tool to optimize Ollama models using quantization.

Based on quantization techniques described in ngrok's "Quantization from the Ground Up":
  - Absmax (symmetric) quantization
  - Zero-point (asymmetric) quantization
  - Per-tensor / per-channel / per-group granularity
  - GGUF quantization levels (Q2_K through Q8_0)
  - Hardware-aware optimization and benchmarking
"""

__version__ = "1.0.0"
__author__ = "Ollama Optimizer Contributors"
