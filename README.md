# Ollama Optimizer

**Quantization-powered model optimization for Ollama** — automatically tune every installed model for maximum performance on your hardware.

Built on concepts from ngrok's ["Quantization from the Ground Up"](https://ngrok.com/blog/quantization) blog post.

```
   ___  _ _                        ___        _   _           _
  / _ \| | |__ _ _ __  __ _ ___   / _ \ _ __ | |_(_)_ __ ___ (_)_______ _ __
 | | | | | / _` | '_ \/  _` |___| | | | '_ \| __| | '_ ` _ \| |_  / _ \ '__|
 | |_| | | | (_| | | | | (_| |___| |_| | |_) | |_| | | | | | | |/ /  __/ |
  \___/|_|_|\__,_|_| |_|\__,_|    \___/| .__/ \__|_|_| |_| |_|_/___\___|_|
                                        |_|
```

---

## What It Does

| Feature | Description |
|---------|-------------|
| **Hardware Detection** | Auto-detects CPU, RAM, GPU/VRAM across Windows, Linux, and macOS |
| **Model Analysis** | Scans all installed Ollama models and evaluates their current quantization |
| **Smart Optimization** | Recommends optimal quantization level + runtime parameters per model |
| **Benchmarking** | Measures tokens/sec, latency, memory before AND after optimization |
| **Comparison Reports** | Rich terminal reports showing exact improvement for every model |
| **Quantization Education** | Built-in explainer demonstrating all quantization techniques |

---

## How Quantization Works (The Short Version)

LLM parameters are stored as floating-point numbers. A 70B-parameter model at FP16 needs **140 GB** of RAM. Quantization compresses these numbers:

| Format | Bits/Weight | Memory (70B model) | Quality | Speed |
|--------|-------------|---------------------|---------|-------|
| F16 | 16.0 | 140.0 GB | Baseline | 1.0x |
| Q8_0 | 8.5 | 74.4 GB | ~98% | 1.3x |
| Q6_K | 6.56 | 57.4 GB | ~94% | 1.6x |
| Q5_K_M | 5.69 | 49.8 GB | ~90% | 1.9x |
| **Q4_K_M** | **4.83** | **42.3 GB** | **~85%** | **2.3x** |
| Q4_0 | 4.50 | 39.4 GB | ~78% | 2.5x |
| Q3_K_M | 3.44 | 30.1 GB | ~68% | 3.0x |
| Q2_K | 2.63 | 23.0 GB | ~55% | 3.8x |

**Q4_K_M** is the sweet spot for most hardware — 85% quality at 30% of the memory.

### Techniques Used

- **Absmax (Symmetric) Quantization**: `scale = max(|x|) / 127` — simple, fast, used in Q8_0
- **Zero-Point (Asymmetric) Quantization**: Adds an offset to handle skewed distributions
- **Per-Group Quantization**: Each group of 32-128 weights gets its own scale — what Q4_K_M and K-quants use
- **K-Quants**: Mixed-precision approach where attention layers keep higher precision

Run `ollama-optimizer explain` for a full interactive demo.

---

## Installation

### Prerequisites

- **Python 3.9+**
- **Ollama** installed and running ([download here](https://ollama.ai/download))

### Install from source

```bash
git clone https://github.com/Mahesh2023/ollama-optimizer.git
cd ollama-optimizer
pip install .
```

### Install in development mode

```bash
pip install -e ".[dev]"
```

### GPU detection (optional)

```bash
pip install "ollama-optimizer[gpu]"
```

---

## Quick Start

### 1. Scan your system and models

```bash
ollama-optimizer scan
```

This detects your hardware and analyzes every installed model:

```
┌──────────────────────────────────────────────────────┐
│              System Hardware Profile                  │
├────────────────┬─────────────────────────────────────┤
│ OS             │ Linux (Ubuntu 22.04)                │
│ CPU            │ AMD Ryzen 9 7950X (16c/32t)        │
│ RAM            │ 64.0 GB (58.2 GB available)        │
│ GPU            │ NVIDIA RTX 4090 (24 GB VRAM)       │
│ Disk Free      │ 512 GB                              │
└────────────────┴─────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   Installed Models                           │
├──────────────┬────────┬───────┬────────┬─────────┬──────────┤
│ Model        │ Params │ Quant │ Size   │ Quality │ Speed    │
├──────────────┼────────┼───────┼────────┼─────────┼──────────┤
│ llama3:70b   │ 70B    │ Q4_0  │ 39.4GB │ 78%     │ 2.5x     │
│ mistral:7b   │ 7B     │ Q4_K_M│ 4.4GB  │ 85%     │ 2.3x     │
│ codellama:34b│ 34B    │ F16   │ 68.0GB │ 100%    │ 1.0x     │
└──────────────┴────────┴───────┴────────┴─────────┴──────────┘
```

### 2. Optimize all models

```bash
ollama-optimizer optimize --benchmark
```

This will:
1. Benchmark each model's current performance
2. Calculate the optimal quantization + runtime config for your hardware
3. Create optimized model variants
4. Re-benchmark and show the improvement

```
┌─────────────────────── Optimization Plan ────────────────────┐
│                                                              │
│  llama3:70b                                                  │
│    Q4_0 -> Q4_K_M  (better K-quant, +7% quality)           │
│    GPU layers: 0 -> 35 (24GB VRAM utilized)                 │
│    Threads: 1 -> 16, Context: 2048 -> 4096                  │
│    Expected: +40% throughput, +7% quality                    │
│                                                              │
│  codellama:34b                                               │
│    F16 -> Q5_K_M  (saves 37GB RAM, 90% quality retained)   │
│    GPU layers: 0 -> 40 (fits in VRAM at Q5_K_M)            │
│    Expected: +90% throughput, 68GB -> 24GB memory           │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 3. Benchmark specific models

```bash
ollama-optimizer benchmark --model llama3:70b
ollama-optimizer benchmark --compare llama3:70b llama3:70b-optimized
```

### 4. Prioritize what matters to you

```bash
# Maximum quality (pick highest quant that fits)
ollama-optimizer optimize --priority quality

# Maximum speed (pick most aggressive quant)
ollama-optimizer optimize --priority speed

# Best tradeoff (default)
ollama-optimizer optimize --priority balanced

# Fit on minimal hardware
ollama-optimizer optimize --priority minimum
```

### 5. Learn about quantization

```bash
ollama-optimizer explain
```

Shows an interactive explainer with all quantization techniques, error measurements, and the full GGUF quantization level reference table.

---

## Commands Reference

| Command | Description |
|---------|-------------|
| `scan` | Detect hardware and analyze all installed models |
| `optimize` | Optimize models (with optional benchmarking) |
| `benchmark` | Run performance benchmarks |
| `report` | View the last benchmark report |
| `explain` | Learn about quantization techniques |
| `status` | Check Ollama status and running models |

### Global Options

| Option | Description |
|--------|-------------|
| `--verbose, -v` | Enable debug logging |
| `--version` | Show version |
| `--help` | Show help |

### Optimize Options

| Option | Description |
|--------|-------------|
| `--model, -m` | Optimize specific model (default: all) |
| `--priority, -p` | quality / speed / balanced / minimum |
| `--dry-run, -d` | Show plans without applying |
| `--yes, -y` | Skip confirmation prompts |
| `--benchmark / --no-benchmark` | Run before/after benchmarks (default: on) |

---

## How It Works

### 1. System Profiling
Detects CPU cores/threads, RAM, GPU/VRAM, and disk space using `psutil` and platform-specific APIs.

### 2. Model Analysis
Queries Ollama's API to list all models, then parses each model's metadata to determine current quantization level, parameter count, and family.

### 3. Optimization Planning
For each model, the optimizer:
- Calculates memory requirements at every quantization level
- Filters to levels that fit in your available RAM + VRAM
- Ranks by your priority (quality/speed/balanced/minimum)
- Determines optimal GPU layer offloading
- Sets runtime parameters (threads, context window, batch size)

### 4. Model Creation
Generates an Ollama Modelfile with the optimal configuration and uses `ollama create` to build the optimized variant.

### 5. Benchmarking
Runs 5 diverse prompts (QA, reasoning, creative, code, summarization) against each model, measuring:
- **Tokens/second** (generation throughput)
- **Time to first token** (latency)
- **Memory usage** (RAM consumption)
- **Prompt processing rate** (input throughput)

---

## Cross-Platform Support

| Platform | CPU | RAM | GPU (NVIDIA) | GPU (AMD) | GPU (Apple Silicon) |
|----------|-----|-----|-------------|-----------|-------------------|
| Linux | Full | Full | Full (nvidia-smi) | Partial (rocm-smi) | N/A |
| macOS | Full | Full | N/A | N/A | Full (Metal) |
| Windows | Full | Full | Full (nvidia-smi) | Partial | N/A |

---

## Troubleshooting

### Ollama not detected
```
Ollama is not running. Start it with:
  Linux/Mac: ollama serve
  Windows:   Start Ollama from the Start menu
  Install:   https://ollama.ai/download
```

### No models found
```bash
# Pull some models first
ollama pull llama3:8b
ollama pull mistral:7b
```

### Permission errors on Linux
```bash
# Make sure your user is in the ollama group
sudo usermod -a -G ollama $USER
```

---

## Project Structure

```
ollama-optimizer/
  pyproject.toml              # Package configuration
  README.md                   # This file
  LICENSE                     # MIT License
  ollama_optimizer/
    __init__.py               # Package init
    __main__.py               # python -m entry point
    cli.py                    # Click CLI commands
    system_profiler.py        # Hardware detection
    ollama_client.py          # Ollama REST API client
    quantization.py           # Quantization engine + knowledge base
    optimizer.py              # Optimization logic
    benchmark.py              # Performance benchmarking
    reporter.py               # Rich terminal reports
```

---

## Concepts from "Quantization from the Ground Up"

This tool implements ideas from [ngrok's blog post](https://ngrok.com/blog/quantization):

1. **Parameters are the majority of model size** — a 70B model at FP32 = 280GB
2. **Float precision can be traded** — FP32 (32-bit) -> FP16 (16-bit) -> INT8 (8-bit) -> INT4 (4-bit)
3. **Absmax quantization** — symmetric, maps max absolute value to max int range
4. **Zero-point quantization** — asymmetric, handles skewed distributions better
5. **Per-group quantization** — each group of weights gets its own scale factor (what K-quants use)
6. **Perplexity** — the standard metric for measuring quality loss after quantization
7. **Larger models quantize better** — 70B at Q4 loses less quality than 7B at Q4

---

## License

MIT
