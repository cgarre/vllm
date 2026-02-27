# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is vLLM

vLLM is a high-throughput LLM inference and serving engine. It uses PagedAttention for efficient KV cache memory management, supports continuous batching, CUDA/HIP graphs, various quantization methods (GPTQ, AWQ, FP8, INT8), speculative decoding, and provides an OpenAI-compatible API server.

## Build and Install

```bash
# Editable install (for development)
pip install -e .

# Use precompiled wheels instead of building C++ from source
VLLM_USE_PRECOMPILED=1 pip install -e .

# Install test dependencies
pip install -r requirements/test.txt

# Install lint dependencies
pip install -r requirements/lint.txt
```

The build system uses setuptools + CMake + Ninja. C++/CUDA extensions are built via CMake (`CMakeLists.txt`). The target device (cuda, rocm, cpu, tpu, xpu) is auto-detected. On macOS, it defaults to `cpu`.

## Running Tests

```bash
# Run a single test file
pytest -v -s tests/basic_correctness/test_basic_correctness.py

# Run a specific test
pytest -v -s tests/basic_correctness/test_basic_correctness.py::test_name

# Run with a marker
pytest -v -s -m core_model tests/models/
```

Key pytest markers: `core_model` (runs on every PR), `slow_test`, `cpu_model`, `cpu_test`, `distributed` (multi-GPU only), `optional` (requires `--optional` flag).

## Linting and Formatting

```bash
# Python lint + format (ruff)
ruff check --fix .
ruff format .

# Type checking
mypy vllm/

# C++/CUDA formatting
clang-format --style=file -i <file>

# Spell checking
typos

# Run all pre-commit hooks
pre-commit run --all-files
```

Ruff rules: E (pycodestyle), F (pyflakes), UP (pyupgrade), B (bugbear), ISC (implicit-str-concat), SIM (simplify), I (isort), G (logging-format).

## Code Conventions

- All Python files must have Apache 2.0 SPDX license headers (enforced by pre-commit).
- Commits must include `Signed-off-by:` line (enforced by pre-commit).
- The top-level `vllm/__init__.py` uses lazy imports via `MODULE_ATTRS` dict + `__getattr__`. Follow this pattern when adding new public exports.

## Architecture

### Dual Engine Design

The codebase has two engine generations. **V1** (`vllm/v1/`) is the production engine. The legacy engine lives in `vllm/engine/`.

### V1 Engine (production path)

```
Entrypoints (LLM, AsyncLLM, OpenAI server, gRPC)
    │
    ▼
EngineCore (vllm/v1/engine/core.py)
    │   Runs in a separate process, communicates via ZMQ + msgspec
    │
    ├── Scheduler (vllm/v1/core/sched/) — request scheduling, KV cache block management
    ├── Executor (vllm/v1/executor/) — dispatches to workers
    │       UniProcExecutor, MultiProcExecutor, RayExecutor
    ├── Worker (vllm/v1/worker/) — per-device model execution
    │       gpu_worker.py, cpu_worker.py, model_runner.py
    │       sample/ — sampling (top-k, top-p, penalties)
    │       spec_decode/ — speculative decoding (EAGLE)
    ├── KV Cache Manager (vllm/v1/core/kv_cache_manager.py)
    └── Detokenizer (vllm/v1/engine/detokenizer.py) — separate process
```

### Model Registry

Models are registered in `vllm/model_executor/models/registry.py` as a mapping from HuggingFace architecture names to `(module_name, class_name)` tuples. Uses lazy imports. Capability interfaces (`SupportsLoRA`, `SupportsMultiModal`, `SupportsPP`, etc.) declare what each model supports.

### Attention Backends

Located in `vllm/v1/attention/backends/`. Multiple backends: FlashAttention, FlashInfer, Flex Attention, Triton, ROCm-specific, CPU, and MLA (Multi-Latent Attention for DeepSeek).

### Entrypoints

- `vllm/entrypoints/llm.py` — `LLM` class for offline/batch inference
- `vllm/entrypoints/openai/` — OpenAI-compatible API server (chat, completions, embeddings, responses API)
- `vllm/entrypoints/grpc_server.py` — gRPC server
- `vllm/entrypoints/cli/` — CLI (`vllm serve`, `vllm bench`, `vllm chat`, `vllm complete`)

### C++/CUDA Kernels

`csrc/` contains C++/CUDA source for paged attention, quantization kernels, MoE kernels, and more. Built as Python extensions (`_C`, `_moe_C`) via CMake.

### Key Subsystems

- `vllm/compilation/` — torch.compile integration with custom fusion passes, CUDA graph support
- `vllm/distributed/` — tensor/pipeline/expert/data parallelism, KV transfer
- `vllm/model_executor/layers/quantization/` — quantization implementations
- `vllm/lora/` — LoRA adapter support
- `vllm/multimodal/` — multimodal input processing (images, audio, video)
- `vllm/config/` — all configuration dataclasses

## CI

CI runs on Buildkite (not GitHub Actions). Pipeline configs are in `.buildkite/`. Changes to `CMakeLists.txt`, `csrc/`, `setup.py`, `requirements/`, or `docker/Dockerfile` trigger all test areas.
