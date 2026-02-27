# vLLM Architecture Deep Dive

This document provides a detailed walkthrough of the vLLM codebase — a high-throughput LLM inference and serving engine. After reading this, you should understand how a request flows from the API layer through scheduling, execution, and sampling, and how the major subsystems (attention, quantization, LoRA, multimodal, distributed) fit together.

---

## Table of Contents

1. [High-Level Overview](#1-high-level-overview)
2. [Entrypoints: How Requests Enter](#2-entrypoints-how-requests-enter)
3. [The V1 Engine: Process Architecture](#3-the-v1-engine-process-architecture)
4. [Scheduling: How Requests Are Batched](#4-scheduling-how-requests-are-batched)
5. [Execution: From Schedule to Forward Pass](#5-execution-from-schedule-to-forward-pass)
6. [The Model Layer: Registry, Loading, and Layers](#6-the-model-layer-registry-loading-and-layers)
7. [Attention Backends](#7-attention-backends)
8. [KV Cache Management](#8-kv-cache-management)
9. [Sampling Pipeline](#9-sampling-pipeline)
10. [Speculative Decoding](#10-speculative-decoding)
11. [Quantization System](#11-quantization-system)
12. [torch.compile and CUDA Graphs](#12-torchcompile-and-cuda-graphs)
13. [Distributed Execution](#13-distributed-execution)
14. [LoRA Adapter System](#14-lora-adapter-system)
15. [Multimodal Processing](#15-multimodal-processing)
16. [C++/CUDA Kernels](#16-ccuda-kernels)
17. [End-to-End Request Lifecycle](#17-end-to-end-request-lifecycle)

---

## 1. High-Level Overview

vLLM's architecture is built around a multi-process design where the **API server**, **engine core**, and **GPU workers** run in separate OS processes communicating via ZMQ and shared memory.

```
┌─────────────────────────────────────────────────────────┐
│ API Server Process (FastAPI/gRPC)                       │
│  ┌───────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ OpenAI API    │  │ OutputProc   │  │ Detokenizer  │  │
│  │ Handlers      │  │              │  │              │  │
│  └───────┬───────┘  └──────┬───────┘  └──────────────┘  │
│          │ ZMQ             │ ZMQ                         │
├──────────┼─────────────────┼────────────────────────────┤
│ EngineCore Process         │                             │
│  ┌───────┴───────┐  ┌─────┴────┐                        │
│  │ Input Thread  │  │ Output   │                        │
│  │ (ZMQ recv +   │  │ Thread   │                        │
│  │  preprocess)  │  │ (ZMQ     │                        │
│  └───────┬───────┘  │  send)   │                        │
│          │           └─────┬───┘                        │
│  ┌───────┴─────────────────┴───────┐                    │
│  │ Main Thread                      │                    │
│  │  Scheduler ──► Executor          │                    │
│  └──────────────────┬───────────────┘                    │
│                     │ Shared Memory                      │
├─────────────────────┼───────────────────────────────────┤
│ Worker Process(es)  │                                    │
│  ┌──────────────────┴──────────────────┐                │
│  │ GPUModelRunner                       │                │
│  │  prepare_inputs → model.forward()    │                │
│  │  → sample_tokens → postprocess       │                │
│  └──────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────┘
```

The codebase has two engine generations. **V1** (`vllm/v1/`) is the production engine. The legacy engine in `vllm/engine/` exists for backward compatibility. This document focuses on V1.

---

## 2. Entrypoints: How Requests Enter

### CLI (`vllm/entrypoints/cli/`)

The `vllm` command dispatches to subcommands:

| Command | What it does |
|---------|-------------|
| `vllm serve` | Starts the API server (primary entrypoint) |
| `vllm chat` | Interactive chat client (connects to a running server) |
| `vllm complete` | Interactive completion client |
| `vllm bench` | Benchmarking |
| `vllm run-batch` | Batch processing from file |

`vllm serve` (`vllm/entrypoints/cli/serve.py`) has three modes:
- **Single API server** (default): `uvloop.run(run_server(args))` — starts one FastAPI server with one EngineCore.
- **Multi API server** (`--api-server-count > 1`): Shares a socket across multiple API server processes for higher concurrency.
- **Headless** (`--api-server-count < 1`): Runs EngineCore without any HTTP server — used in multi-node deployments where the API lives elsewhere.

### OpenAI-Compatible API (`vllm/entrypoints/openai/`)

The FastAPI server (`api_server.py`) starts up as:

```
run_server(args)
  → build_async_engine_client(args)    # Creates AsyncLLM (the V1 async engine)
  → build_app(args)                    # FastAPI app with routes
  → init_app_state(engine, app, args)  # Creates serving handlers
  → serve_http(app, sock)              # uvicorn
```

Routes are registered conditionally based on the model's supported tasks. The key serving classes:

- **`OpenAIServingChat`** (`chat_completion/serving.py`): Handles `/v1/chat/completions`. Applies Jinja2 chat templates, manages tool parsing, and converts requests to vLLM's `SamplingParams`.
- **`OpenAIServingCompletion`**: Handles `/v1/completions`.
- **`OpenAIServingResponses`**: Handles the newer `/v1/responses` API.
- **`OpenAIServingModels`**: Manages `/v1/models` and LoRA module listing.

Request/response types are Pydantic models in `chat_completion/protocol.py`, implementing the OpenAI spec plus vLLM extensions (`guided_json`, `repetition_penalty`, `min_p`, `lora_request`, etc.).

### Offline Inference (`vllm/entrypoints/llm.py`)

The `LLM` class provides a synchronous Python API:

```python
llm = LLM(model="meta-llama/Llama-3.1-8B")
outputs = llm.generate(["Hello, world!"], SamplingParams(temperature=0.8))
```

Internally it creates a `LLMEngine` (V1 synchronous engine) and loops `engine.step()` until all requests finish. Key methods: `generate()`, `chat()`, `encode()`, `embed()`, `classify()`, `score()`, `beam_search()`.

### gRPC (`vllm/entrypoints/grpc_server.py`)

`VllmEngineServicer` wraps `AsyncLLM` with 6 RPCs: `Generate` (streaming), `Embed`, `HealthCheck`, `Abort`, `GetModelInfo`, `GetServerInfo`. Proto definition is in `vllm/grpc/vllm_engine.proto`.

---

## 3. The V1 Engine: Process Architecture

### EngineCore (`vllm/v1/engine/core.py`)

The `EngineCore` class is the central loop that coordinates scheduling and execution. It has three variants:

- **`EngineCore`** (base): Used in-process by `InprocClient` for offline inference.
- **`EngineCoreProc(EngineCore)`**: Runs in a separate OS process with ZMQ IPC. This is what the API server uses.
- **`DPEngineCoreProc(EngineCoreProc)`**: Data-parallel variant that synchronizes with DP peers via all-reduce every step.

### Three-Thread Model (EngineCoreProc)

When running as a separate process, EngineCore uses three threads:

1. **Input Thread** — receives ZMQ messages from the API server, deserializes them with `MsgpackDecoder`, preprocesses ADD requests (including structured output grammar initialization), and pushes to `self.input_queue`. Preprocessing overlaps with GPU work.

2. **Main Thread** — the engine loop. Alternates between `_process_input_queue()` (drain pending requests into the scheduler) and `_process_engine_step()` (run one step of schedule → execute → sample → update). Blocks on `input_queue.get()` when idle.

3. **Output Thread** — pulls `EngineCoreOutputs` from `self.output_queue`, serializes with `MsgpackEncoder`, and sends via ZMQ PUSH sockets. Uses zero-copy send with buffer reuse.

### ZMQ Communication

- **Input path**: Client uses `zmq.ROUTER`, engine connects with `zmq.DEALER`. Messages are multipart: `[request_type_frame, *data_frames]`.
- **Output path**: Engine binds `zmq.PUSH`, client connects `zmq.PULL` to receive serialized `EngineCoreOutputs`.
- **Handshake**: On startup, engine sends `HELLO` → receives init config (ZMQ addresses) → sends `READY` with `num_gpu_blocks`.

### EngineCoreClient Hierarchy (`vllm/v1/engine/core_client.py`)

The client-side abstraction for talking to EngineCore:

| Client | When Used | How It Works |
|--------|-----------|-------------|
| `InprocClient` | Offline `LLM` class | Holds `EngineCore` in-process, calls `step()` directly |
| `SyncMPClient` | Sync `LLMEngine` | ZMQ to background `EngineCoreProc` |
| `AsyncMPClient` | `AsyncLLM` (API server) | ZMQ + asyncio for non-blocking communication |
| `DPAsyncMPClient` | Data parallel (external LB) | Per-DP-rank client |
| `DPLBAsyncMPClient` | Data parallel (internal LB) | Distributes requests across DP ranks |

### The Step Function

The core loop in `EngineCore.step()`:

```python
def step(self):
    scheduler_output = self.scheduler.schedule()        # What to run
    executor_output = self.executor.execute_model(scheduler_output)  # Forward pass
    sampler_output = self.executor.sample_tokens(grammar_output)     # Sample
    return self.scheduler.update_from_output(scheduler_output, sampler_output)
```

For pipeline parallelism, `step_with_batch_queue()` uses a deque to overlap execution of multiple batches — while batch N is sampling, batch N+1 starts executing.

---

## 4. Scheduling: How Requests Are Batched

### Scheduler (`vllm/v1/core/sched/scheduler.py`)

The `Scheduler` manages three collections:
- `self.waiting` — a `RequestQueue` (priority queue or FIFO) of requests not yet running.
- `self.running` — list of currently executing requests.
- `self.requests` — dict of all active requests by ID.

### The `schedule()` Method (Two Phases)

**Phase 1: Schedule RUNNING requests.** For each running request:
1. Compute `num_new_tokens` (1 for decode, remaining tokens for ongoing prefill, capped by budget).
2. Call `kv_cache_manager.allocate_slots()` to get new KV cache blocks.
3. If allocation fails → **preempt** the lowest-priority running request (evict its KV blocks, move it back to `waiting`).

**Phase 2: Schedule WAITING requests** (only if no preemptions occurred). For each waiting request:
1. Call `kv_cache_manager.get_computed_blocks()` to find prefix cache hits (skip recomputing already-cached tokens).
2. Compute tokens to schedule, optionally applying chunked prefill threshold.
3. Allocate KV blocks. If allocation fails → stop (no preemption for waiting requests).
4. Move request from `waiting` to `running`.

The output is a `SchedulerOutput` containing: `scheduled_new_reqs`, `scheduled_cached_reqs`, `num_scheduled_tokens` per request, `preempted_req_ids`, `finished_req_ids`.

### After Execution: `update_from_output()`

After the model produces tokens:
1. Append sampled tokens to each request's state.
2. Check stop conditions (max tokens, EOS token, stop strings).
3. Handle speculative decoding rejection (adjust `num_computed_tokens` if draft tokens were rejected).
4. Free KV blocks for finished requests.
5. Return `EngineCoreOutputs` grouped by client.

---

## 5. Execution: From Schedule to Forward Pass

### Executor Hierarchy (`vllm/v1/executor/`)

The `Executor` abstract base dispatches work to GPU workers:

| Executor | When Used | Communication |
|----------|-----------|---------------|
| `UniProcExecutor` | Single GPU | Direct function call |
| `MultiprocExecutor` | Multi-GPU (single node) | Shared memory broadcast (`MessageQueue`) |
| `RayDistributedExecutor` | Multi-node | Ray remote calls |

The factory `Executor.get_class(vllm_config)` selects based on `distributed_executor_backend` config.

**MultiprocExecutor** is the most interesting: it spawns worker processes and broadcasts `SchedulerOutput` via shared memory (much faster than ZMQ for the hot path). Each `WorkerProc` runs a busy loop receiving work and executing it.

### GPU Worker (`vllm/v1/worker/gpu_worker.py`)

The `Worker` class manages per-device state:
- `init_device()` — sets CUDA device, initializes NCCL distributed environment, constructs `GPUModelRunner`.
- `load_model()` — delegates to model runner.
- `determine_available_memory()` — profiles GPU by running a dummy forward pass, calculates available KV cache memory.
- `compile_or_warm_up_model()` — runs torch.compile and captures CUDA graphs at various batch sizes.
- `sleep(level)` / `wake_up()` — power management. Level 1: offload weights to CPU. Level 2: discard all GPU memory.

### GPUModelRunner (`vllm/v1/worker/gpu/model_runner.py`)

This is where the actual model execution happens. The key flow in `execute_model(scheduler_output)`:

1. **Update request states** — finish/free/add/update requests from the scheduler output.
2. **`prepare_inputs()`** — build the input batch:
   - Requests are sorted **decode-first, then prefill** (by ascending `num_scheduled_tokens`). This enables efficient CUDA graph capture for the decode portion.
   - Computes `input_ids`, `positions`, `seq_lens`, `query_start_loc`.
   - For speculative decoding, expands indices to account for multiple token positions per request.
3. **`prepare_attn()`** — build block tables and slot mappings for the attention backend.
4. **Handle multimodal** — run encoder inputs through `EncoderRunner` if needed.
5. **Forward pass** — either CUDA graph replay (`cudagraph_manager.run_fullgraph`) or eager mode (`self.model(**model_inputs)`).

Then `sample_tokens(grammar_output)`:

1. Extract logits via `model.compute_logits(hidden_states)`.
2. Apply grammar bitmask for structured output.
3. Call `self.sampler(logits, ...)`.
4. If draft tokens present, run rejection sampling.
5. If speculator exists, generate draft tokens for the next step.
6. Return `ModelRunnerOutput`.

**Async scheduling optimization**: When enabled, `execute_model` returns an `AsyncOutput` that performs the GPU→CPU copy of results on a separate CUDA stream. The scheduler can begin scheduling the next step while the copy is in progress.

---

## 6. The Model Layer: Registry, Loading, and Layers

### Model Registry (`vllm/model_executor/models/registry.py`)

Models are registered as mappings from HuggingFace architecture names to `(module_name, class_name)` tuples. For example:

```python
"LlamaForCausalLM": ("llama", "LlamaForCausalLM")
# means: vllm.model_executor.models.llama.LlamaForCausalLM
```

There are ~135 text generation models, plus embedding, cross-encoder, multimodal, and speculative decoding models. Many HF architectures alias to the same vLLM implementation (e.g., `AquilaModel`, `InternLMForCausalLM` both map to `LlamaForCausalLM`).

**Lazy loading**: Every model starts as a `_LazyRegisteredModel`. The actual module is only imported when needed (`importlib.import_module`). Model capabilities are inspected via a **subprocess** (using `cloudpickle`) to avoid premature CUDA initialization in the main process. Results are cached in JSON under `VLLM_CACHE_ROOT/modelinfos/`.

### Model Implementation Pattern (LLaMA as canonical example)

`vllm/model_executor/models/llama.py` shows the standard 4-layer hierarchy:

**`LlamaMLP`**: Uses `MergedColumnParallelLinear` for `gate_up_proj` (fuses gate + up into one matmul), `RowParallelLinear` for `down_proj`, `SiluAndMul` activation.

**`LlamaAttention`**: `QKVParallelLinear` fuses Q/K/V into one matmul. `RowParallelLinear` for output projection. Gets RoPE embeddings via `get_rope()`. The `Attention` layer handles KV cache and backend dispatch transparently.

**`LlamaDecoderLayer`**: Pre-norm architecture with fused `RMSNorm + residual add`.

**`LlamaModel`**: Decorated with `@support_torch_compile`. Uses `VocabParallelEmbedding` (parallelized across TP ranks). `make_layers()` creates layers with pipeline-parallelism awareness.

**`LlamaForCausalLM`**: Top-level class implementing `SupportsLoRA`, `SupportsPP`, `SupportsEagle`. Uses `ParallelLMHead` for the output projection and `LogitsProcessor` for logit computation.

### Weight Loading: How HF Weights Map to vLLM Layers

vLLM fuses multiple HF weight matrices into single larger matrices for efficiency. The `stacked_params_mapping` in each model's `load_weights()` defines the mapping:

```python
stacked_params_mapping = [
    (".qkv_proj", ".q_proj", "q"),   # HF q_proj → vLLM qkv_proj shard "q"
    (".qkv_proj", ".k_proj", "k"),   # HF k_proj → vLLM qkv_proj shard "k"
    (".qkv_proj", ".v_proj", "v"),   # HF v_proj → vLLM qkv_proj shard "v"
    (".gate_up_proj", ".gate_proj", 0),  # HF gate_proj → gate_up_proj shard 0
    (".gate_up_proj", ".up_proj", 1),    # HF up_proj → gate_up_proj shard 1
]
```

The full loading pipeline:

```
get_model() [model_loader/__init__.py]
  → ModelRegistry.resolve_model_cls(architectures)       # Find the vLLM class
  → _LazyRegisteredModel.load_model_cls()                # Import the module
  → configure_quant_config(quant_config, model_class)    # Set up quantization
  → model_class(vllm_config=vllm_config)                 # Instantiate (quant_method.create_weights())
  → DefaultModelLoader.load_weights(model)               # Download + load safetensors
      → model.load_weights(weights_iterator)             # Route via stacked_params_mapping
  → process_weights_after_loading(model)                 # quant_method.process_weights_after_loading()
```

### Linear Layer Abstraction (`vllm/model_executor/layers/linear.py`)

The tensor-parallel linear layer hierarchy:

```
LinearBase
  ├── ReplicatedLinear           — no TP sharding
  ├── ColumnParallelLinear       — shards output dim across TP ranks
  │     ├── MergedColumnParallelLinear  — fuses multiple matrices (gate + up)
  │     └── QKVParallelLinear           — fuses Q/K/V with different head counts
  └── RowParallelLinear          — shards input dim, all-reduce on output
```

The key design: `LinearBase.__init__()` gets a `quant_method` from the `quant_config` (or `UnquantizedLinearMethod` if none). The `quant_method` has three hooks:
1. `create_weights()` — allocate parameters (possibly in quantized format).
2. `process_weights_after_loading()` — transform loaded weights (transpose, repack).
3. `apply()` — execute the matmul (called during `forward()`).

Models never deal with quantization directly — they use the standard linear layer types and quantization is injected via the strategy pattern.

---

## 7. Attention Backends

### Architecture (`vllm/v1/attention/`)

The attention system uses three abstract classes:

- **`AttentionBackend`** — static factory. Declares capabilities (supported dtypes, head sizes, block sizes, CUDA graph support).
- **`AttentionMetadataBuilder`** — per-batch metadata construction. Computes block tables, slot mappings, sequence lengths.
- **`AttentionImpl`** — the forward pass: `forward(query, key, value, kv_cache, attn_metadata) → Tensor`.

For Multi-Latent Attention (DeepSeek), there's also `MLAAttentionImpl` with separate `forward_mha()` (prefill with expanded K/V) and `forward_mqa()` (decode with compressed latent vectors).

### Available Backends

There are 20+ backends registered in `AttentionBackendEnum` (`vllm/v1/attention/backends/registry.py`):

| Backend | Use Case |
|---------|----------|
| `FLASH_ATTN` | Default for NVIDIA GPUs (FA2/FA3) |
| `FLASHINFER` | Alternative with advanced features |
| `TRITON_ATTN` | Pure Triton implementation |
| `FLEX_ATTENTION` | PyTorch native flex attention |
| `FLASHMLA` / `CUTLASS_MLA` / `TRITON_MLA` / `FLASHINFER_MLA` | DeepSeek MLA variants |
| `CPU_ATTN` | CPU backend |
| `ROCM_*` (5 variants) | AMD GPU backends |

Selection happens via `get_attn_backend()` → `current_platform.get_attn_backend_cls()`, based on hardware capabilities and config.

### FlashAttention Backend Details

`FlashAttentionBackend` uses KV cache shape `(2, num_blocks, block_size, num_kv_heads, head_size)` with block sizes that are multiples of 16. The `forward()` implementation routes to three paths:

1. **Encoder attention** — direct Q/K/V computation, no cache.
2. **Standard** — calls `flash_attn_varlen_func()` with paged block tables, optional FP8 KV cache, sliding window, alibi slopes, logits soft cap.
3. **Cascade attention** — splits into prefix (common across requests) and suffix (per-request) attention, then merges via `merge_attn_states`. This is a key optimization when many requests share a long common prefix.

---

## 8. KV Cache Management

### KVCacheManager (`vllm/v1/core/kv_cache_manager.py`)

The KV cache uses block-level allocation similar to virtual memory paging (this is the core **PagedAttention** innovation). The manager delegates to a `KVCacheCoordinator` which manages a `BlockPool`.

**Key operations:**

- **`get_computed_blocks(request)`** — prefix cache lookup. Finds the longest previously-computed prefix using content-based block hashing. Always recomputes at least the last token (needed for fresh logits).

- **`allocate_slots(request, num_new_tokens)`** — the core allocation method:
  1. Free blocks outside the sliding window.
  2. Check if enough free blocks exist.
  3. Allocate blocks for prefix cache hits (`allocate_new_computed_blocks`).
  4. Allocate blocks for new tokens + lookahead tokens for speculative decoding.
  5. Commit block hashes for future prefix caching.
  Returns `None` if insufficient memory → triggers preemption in the scheduler.

- **`free(request)`** — frees blocks in reverse order (tail blocks evicted first, preserving prefix cache for other requests sharing the same prefix).

- **`get_num_common_prefix_blocks()`** — finds blocks shared by ALL running requests (ref_count == total). Used for cascade attention optimization.

### KVCacheBlocks

A `KVCacheBlocks` dataclass wraps `tuple[Sequence[KVCacheBlock], ...]` — one sequence per KV cache group (supports multi-group architectures like MLA). `get_block_ids()` extracts integer block IDs for the GPU block table.

---

## 9. Sampling Pipeline

### Two Sampler Implementations

**Standard Sampler** (`vllm/v1/sample/sampler.py`) — a 9-step `nn.Module` pipeline:

1. Compute logprobs (optional)
2. Cast to float32
3. Apply allowed token IDs whitelist (mask with `-inf`)
4. Apply bad words exclusion (n-gram blocking)
5. Non-argmax-invariant logit processors (min_tokens, logit_bias)
6. Penalties (repetition, frequency, presence) via `apply_all_penalties()`
7. **Sample**: temperature scaling → min_p → top-k/top-p → greedy or random
8. Gather logprobs of top-k and sampled tokens
9. Return `SamplerOutput`

**GPU-Optimized Sampler** (`vllm/v1/worker/gpu/sample/sampler.py`) — uses Triton kernels for key operations:
- `_penalties_kernel` — fused repetition + frequency + presence penalties in a single kernel
- `_temperature_kernel` — temperature scaling
- `_min_p_kernel` — threshold-based pruning
- `_gumbel_sample_kernel` — adds Gumbel noise then argmax (avoids `torch.multinomial` CPU-GPU sync)

### Top-K/Top-P Implementation

Three backends in `TopKTopPSampler` (`vllm/v1/sample/ops/topk_topp_sampler.py`):
1. **FlashInfer** — rejection sampling, no sorting needed.
2. **Triton "Qrita"** — pivot-based truncation without full sort. Uses Gaussian sigma-truncation for initial outlier gathering, then ternary search. Used for batches ≥ 8.
3. **PyTorch native** — `torch.sort()` + cumulative probability masking. Used for small batches.

All paths avoid `torch.multinomial` (which causes CPU-GPU sync) by using the Gumbel-max trick: `probs.div_(q.exponential_()).argmax()`.

---

## 10. Speculative Decoding

### EAGLE Speculator (`vllm/v1/worker/gpu/spec_decode/eagle/`)

vLLM implements EAGLE (and EAGLE3/MTP variants) speculative decoding. The `EagleSpeculator` loads a separate **draft model** (EAGLE head) that shares KV cache with the target model.

**`propose()` flow** (after target model forward):
1. Take the target model's hidden states.
2. Run `num_speculative_steps` iterations of the draft model.
3. Use **Gumbel sampling** for draft token generation (enables proper rejection sampling).
4. Return draft tokens to be verified in the next step.

### Rejection Sampling (`rejection_sample.py`)

A Triton kernel (`_rejection_sample_kernel`) runs per-request: compares target-sampled tokens with draft tokens sequentially, accepts while they match, stops at first mismatch. Returns accepted tokens + 1 bonus token from the target model.

---

## 11. Quantization System

### Architecture (`vllm/model_executor/layers/quantization/`)

Two abstract base classes:

**`QuantizationConfig`** — declares `get_quant_method(layer, prefix)` which returns a quantization strategy for each layer (or `None` to skip quantization). Has `packed_modules_mapping` so it knows about fused layers.

**`QuantizeMethodBase`** — the strategy interface with `create_weights()`, `apply()`, `process_weights_after_loading()`.

### Supported Methods

AWQ, FP8 (online + offline), FBGEMM FP8, GPTQ, GPTQ-Marlin, AWQ-Marlin, GGUF, compressed-tensors, bitsandbytes, TorchAO, MXFP4, NVFP4, and more. Custom quantization can be registered via `@register_quantization_config("name")`.

### How Quantization Plugs In

The connection is through `LinearBase.__init__()`:

```python
if quant_config is None:
    self.quant_method = UnquantizedLinearMethod()
else:
    self.quant_method = quant_config.get_quant_method(self, prefix=prefix)

self.quant_method.create_weights(...)  # Allocate quantized parameters
```

During `forward()`: `self.quant_method.apply(self, input_, bias)` — the model code is completely agnostic to quantization.

### FP8 Example

`Fp8Config` supports both offline (pre-quantized checkpoints) and online (quantize at load time) modes, with static or dynamic activation quantization. It returns different method classes depending on context:
- `Fp8LinearMethod` for pre-serialized FP8 weights
- `Fp8OnlineLinearMethod` for online quantization
- `Fp8MoEMethod` / `Fp8OnlineMoEMethod` for MoE layers
- `Fp8KVCacheMethod` for KV cache quantization

---

## 12. torch.compile and CUDA Graphs

### Compilation (`vllm/compilation/`)

The `@support_torch_compile` decorator on model classes enables compilation:

1. On first call, marks dynamic dimensions via `torch._dynamo.mark_dynamic()`.
2. Calls `torch.compile(model, fullgraph=True, dynamic=False)`.
3. **Guard dropping**: After the first compilation, all Dynamo guards are filtered out. The model is compiled once and never retraced — shapes are handled by pre-compiling for multiple shape ranges.

### Piecewise Compilation

`PiecewiseBackend` splits the computation graph into pieces at "splitting ops" (custom CUDA kernels that can't be fused). Each piece is compiled separately for different shape ranges defined in `compilation_config.compile_sizes`.

### Fusion Passes (`vllm/compilation/passes/`)

Custom post-grad passes run after Inductor optimization, fusing operations to eliminate memory round-trips:

| Pass | What It Fuses |
|------|--------------|
| `RMSNormQuantFusionPass` | RMSNorm + FP8 quantization → single kernel |
| `ActivationQuantFusionPass` | Activation function + quantization |
| `RopeKVCacheFusionPass` | RoPE + KV cache updates |
| `AllReduceFusionPass` | All-reduce + subsequent ops |
| `SequenceParallelismPass` | GEMM + collective communications |
| `AttnFusionPass` | Attention + quantization |

These passes use `torch._inductor.pattern_matcher` to find and replace subgraphs in the FX IR.

### CUDA Graphs

`CUDAGraphWrapper` wraps compiled runnables with CUDA graph capture/replay. On first encounter of a batch shape, it captures a CUDA graph. On subsequent calls with the same shape, it replays the captured graph (eliminating kernel launch overhead). Combined with piecewise compilation, this gives `PIECEWISE` CUDA graph mode where each compiled piece gets its own graph.

---

## 13. Distributed Execution

### GroupCoordinator (`vllm/distributed/parallel_state.py`)

The fundamental distributed communication primitive. Wraps PyTorch `ProcessGroup` with both CPU (Gloo) and GPU (NCCL) groups.

Key communication methods:
- `all_reduce()`, `all_gather()`, `reduce_scatter()` — registered as custom PyTorch ops for torch.compile compatibility.
- `broadcast_tensor_dict()` — broadcasts dicts of tensors + metadata, uses shared memory `MessageQueue` when available.
- `send()` / `recv()` — point-to-point via device communicator.
- `graph_capture()` — context manager for CUDA graph capture with stream management.

### Parallel Groups

Initialized by `initialize_model_parallel()` from a 5D rank tensor `[ExternalDP, DP, PP, PCP, TP]`:

| Group | Accessor | Purpose |
|-------|----------|---------|
| `_WORLD` | `get_world_group()` | All processes |
| `_TP` | `get_tp_group()` | Tensor parallelism — shards weights across GPUs |
| `_PP` | `get_pp_group()` | Pipeline parallelism — shards layers across GPUs |
| `_DP` | `get_dp_group()` | Data parallelism — replicates model, shards requests |
| `_EP` | `get_ep_group()` | Expert parallelism — for MoE models |
| `_DCP` | `get_dcp_group()` | Decode context parallelism |
| `_PCP` | `get_pcp_group()` | Prefill context parallelism |

### How TP Works in Practice

Tensor parallelism is built into the layer abstractions:
- `ColumnParallelLinear` shards the output dimension across TP ranks. Each rank holds `output_dim / tp_size` columns.
- `RowParallelLinear` shards the input dimension, then does `all_reduce` on the output to combine partial sums.
- `VocabParallelEmbedding` shards the vocabulary across ranks.

Models never handle TP explicitly — they just use these layer types.

---

## 14. LoRA Adapter System

### How LoRA Works in vLLM (`vllm/lora/`)

**Request-level**: Each request can specify a `LoRARequest(lora_name, lora_int_id, lora_path)`.

**Loading** (`worker_manager.py`): `WorkerLoRAManager._load_adapter()` downloads/loads adapter files, creates a `LoRAModel` via `LoRAModel.from_local_checkpoint()` which parses `adapter_config.json` and extracts `lora_a`/`lora_b` weight matrices.

**Module wrapping** (`model_manager.py`): `LoRAModelManager._create_lora_modules()` iterates all model modules. For each matching target module, it wraps the layer with a LoRA-capable version via `replace_submodule()`.

**Activation** (`model_manager.py`): `activate_adapter(lora_id)` assigns a GPU slot and calls `module.set_lora(index, lora_a, lora_b)` for each wrapped module. Active adapters are managed via LRU cache — when GPU slots are full, the least-recently-used adapter is deactivated.

**Execution**: During forward passes, `PunicaWrapper` dispatches batched LoRA computation — it knows which tokens in the batch belong to which adapter and computes `output += lora_b @ lora_a @ input` for each.

**For fused layers** (QKV, gate_up), `_create_merged_loras_inplace()` combines individual lora_a/lora_b matrices into the fused format.

---

## 15. Multimodal Processing

### Architecture (`vllm/multimodal/`)

The multimodal system handles images, video, and audio through a processor pipeline.

**Registry** (`registry.py`): `MULTIMODAL_REGISTRY` is a singleton. Models register their processor via `@register_processor(processor, info, dummy_inputs)` decorator. The registry provides `create_processor(model_config, tokenizer)` to instantiate the right processor.

**Processor** (`processing/processor.py`): `BaseMultiModalProcessor` is the abstract base class. Models implement:
- `_get_mm_fields_config()` — defines how to batch HF-processed data.
- `_get_prompt_updates()` — defines placeholder replacement rules.

**Prompt update system**: `PromptReplacement` replaces placeholder tokens. For example, a single `<image>` token gets replaced with N image feature tokens (where N depends on image resolution).

### Data Flow

```
User: {"prompt": "Describe: <image>", "multi_modal_data": {"image": PIL.Image}}
  ↓
HF Processor: tokenize text, process image → pixel_values tensor
  ↓
Prompt updates: replace <image> with [IMG]*576 tokens
  ↓
find_mm_placeholders(): scan token IDs for placeholder positions
  ↓
MultiModalInputs:
  prompt_token_ids: [101, ..., IMG, IMG, ..., IMG, ..., 102]
  mm_kwargs: {"pixel_values": tensor(...)}
  mm_placeholders: {"image": [PlaceholderRange(offset=3, length=576)]}
  ↓
Model Runner: at placeholder positions, replaces token embeddings
  with vision encoder outputs (pixel_values → vision_tower → projection)
```

**Caching**: Multimodal inputs can be cached across requests via shared memory (`"shm"` mode), LRU cache, or processor-only cache. Content hashes (`mm_hashes`) enable deduplication.

---

## 16. C++/CUDA Kernels

### Organization (`csrc/`)

C++ and CUDA kernels are organized by function:

| Directory/File | Purpose |
|----------------|---------|
| `csrc/attention/` | Paged attention v1 (single-pass) and v2 (split-KV with reduction), merge_attn_states, MLA decode |
| `csrc/cache_kernels.cu` | `swap_blocks` (GPU↔CPU), `reshape_and_cache`, `concat_and_cache_mla` |
| `csrc/quantization/` | ~45 .cu files: AWQ, GPTQ, Marlin, FP8, FP4, GGUF, CUTLASS w8a8, Machete |
| `csrc/moe/` | Top-k softmax, token alignment, permute/unpermute, grouped top-k, DeepSeek V3 router |
| `csrc/activation_kernels.cu` | Fused SiLU, GELU, FATReLU |
| `csrc/layernorm_kernels.cu` | RMS norm, fused add + RMS norm |
| `csrc/layernorm_quant_kernels.cu` | Fused layernorm + quantization |
| `csrc/pos_encoding_kernels.cu` | Rotary positional embedding (RoPE) |
| `csrc/custom_all_reduce.cu` | Custom multi-GPU all-reduce |
| `csrc/sparse/cutlass/` | 2:4 structured sparsity GEMM |

### Python Bridge

Kernels are exposed via PyTorch's `TORCH_LIBRARY_EXPAND` macro in `csrc/torch_bindings.cpp`, organized into four library groups:
- **`_C`** (main ops): ~60 ops including attention, activation, norm, RoPE, quantization GEMMs.
- **`_cache_ops`**: swap_blocks, reshape_and_cache, convert_fp8.
- **`_cuda_utils`**: device attribute queries.
- **`_custom_ar`**: custom all-reduce operations.

Plus `_moe_C` in `csrc/moe/torch_bindings.cpp` for MoE-specific ops.

The Python side (`vllm/_custom_ops.py`) provides thin wrappers calling `torch.ops._C.<op_name>(...)`, with `@register_fake` meta functions for torch.compile compatibility.

### PluggableLayer System

`PluggableLayer` (`vllm/model_executor/custom_op.py`) uses `__new__` to intercept layer instantiation. Out-of-tree replacements can be registered via `@PluggableLayer.register_oot(name="column_parallel_linear")`, enabling platform-specific layer overrides without changing model code.

---

## 17. End-to-End Request Lifecycle

Putting it all together, here's the complete path of a chat completion request:

```
1. HTTP POST /v1/chat/completions
   → OpenAIServingChat: apply chat template, create SamplingParams
   → AsyncLLM.add_request()

2. AsyncLLM (API server process)
   → InputProcessor.process_inputs(): tokenize, process multimodal
   → Create EngineCoreRequest
   → Serialize with MsgpackEncoder, send via ZMQ DEALER socket

3. EngineCoreProc Input Thread
   → Receive ZMQ message, deserialize
   → preprocess_add_request(): create Request, init structured output grammar
   → Push to input_queue

4. EngineCoreProc Main Thread
   → _process_input_queue() → scheduler.add_request() → waiting queue

5. EngineCore.step()
   a. scheduler.schedule()
      - Find prefix cache hits via kv_cache_manager.get_computed_blocks()
      - Allocate KV blocks via kv_cache_manager.allocate_slots()
      - Return SchedulerOutput

   b. executor.execute_model(scheduler_output)
      - Broadcast SchedulerOutput to workers via shared memory
      - Worker: GPUModelRunner.execute_model()
        → prepare_inputs(): sort decode-first, build input tensors
        → prepare_attn(): build block tables, slot mappings
        → model(**inputs) or CUDA graph replay

   c. executor.sample_tokens(grammar_output)
      - Worker: GPUModelRunner.sample_tokens()
        → compute_logits(hidden_states)
        → sampler(logits, sampling_params)
        → rejection_sample() if speculative decoding
        → speculator.propose() → draft tokens for next step
      - Return ModelRunnerOutput (async D2H copy)

   d. scheduler.update_from_output()
      - Append sampled tokens, check stop conditions
      - Free KV blocks for finished requests
      - Return EngineCoreOutputs

6. EngineCoreProc Output Thread
   → Serialize EngineCoreOutputs, send via ZMQ PUSH

7. AsyncLLM Output Handler (API server process)
   → Receive via ZMQ PULL, deserialize
   → OutputProcessor: IncrementalDetokenizer.update(new_token_ids)
   → Create RequestOutput, push to streaming queue

8. HTTP SSE stream / final JSON response
```

Steps 5a–5d repeat in a loop until the request finishes (hits max tokens, EOS, or stop string). For streaming responses, each iteration's sampled tokens are sent back incrementally.
