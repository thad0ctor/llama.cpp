# Blackwell (SM_120) Optimizations

This document tracks specific optimizations targeted at the NVIDIA Blackwell architecture (RTX 5090 / GB200).

## Implemented Optimizations

### 1. Flash Attention Blackwell Kernels

**Files:** `ggml/src/ggml-cuda/fattn-blackwell-f16.cuh`, `ggml/src/ggml-cuda/attention_v5.cu`

Two optimized Flash Attention kernels for Blackwell:

| Kernel | Input Types | Head Dims | Notes |
|--------|-------------|-----------|-------|
| **blackwell_f16** | Q/K/V all F16 | 64, 128 | Most common path (Q cast to F16) |
| **attention_v5** | Q/K/V all BF16 | 128 | BF16-native path |

**Key Features:**
- Uses m16n8k16 MMA tensor core instructions
- `cp.async` for efficient memory loading
- Online softmax with row-wise tracking
- 128 threads (4 warps) per block for register pressure management
- Supports ALiBi bias and logit softcap (Gemma2)

**Kernel Selection:** Automatic based on tensor types. Falls back to `mma_f16`/`vec` if Blackwell kernels not supported.

**Environment Variables:**
- `GGML_CUDA_NO_ATTENTION_V5`: Disable BF16 kernel
- `GGML_CUDA_NO_BLACKWELL_F16`: Disable F16 kernel

See `docs/TENSOR_MAP_FLASH_ATTN_V5.md` for detailed tensor flow documentation.

---

### 2. TMA (Tensor Memory Accelerator) Primitives

**File:** `ggml/src/ggml-cuda/tma.cuh`

Host-side tensor map creation for TMA bulk loads:
- `ggml_cuda_create_tensor_map_2d()` - Standard 2D tensor map with 128B swizzle
- `ggml_cuda_create_tensor_map_2d_ex()` - Extended with configurable swizzle

Device-side TMA primitives:
- `tma_load_2d()` - Async bulk load: global → shared via tensor map
- `tma_fence_acquire()` - Fence for TMA descriptor visibility
- `mbarrier_init/arrive/wait` - Memory barrier operations for async coordination

**L2 Cache Promotion Options:**
- `CU_TENSOR_MAP_L2_PROMOTION_L2_256B` (default) - Optimal for large tile loads
- Configurable per tensor map for K/V cache residency during decode

---

### 3. L2 Cache Persistence

**File:** `ggml/src/ggml-cuda/common.cuh`

RAII guard for L2 cache persistence during decode (RTX 5090 has 96MB L2):

```cpp
ggml_cuda_l2_persist_guard guard(stream, ptr, size, cc, n_queries);
```

**Environment Variables:**
| Variable | Default | Description |
|----------|---------|-------------|
| `GGML_CUDA_L2_PERSIST` | OFF | Master enable (set to "1" to enable) |
| `GGML_CUDA_L2_PERSIST_SIZE` | 48 | Max size in MB to persist |
| `GGML_CUDA_L2_PERSIST_RATIO` | 1.0 | Hit ratio (0.0-1.0) |
| `GGML_CUDA_L2_PERSIST_DECODE_ONLY` | OFF | Only persist during decode (Q tokens == 1) |

---

### 4. WGMMA (Warp Group Matrix Multiply-Accumulate)

**File:** `ggml/src/ggml-cuda/mma.cuh` (namespace `ggml_cuda_wgmma`)

Blackwell WGMMA primitives for larger matrix operations using warp groups (4 consecutive warps):

- `make_smem_desc()` - Create shared memory descriptor for WGMMA operand B
- `wgmma_m64n64k16_f16_f32()` - 64x64x16 FP16→FP32 accumulation
- `wgmma_commit_group()` / `wgmma_wait_group()` - Async commit/wait
- `wgmma_fence()` - Memory fence for WGMMA operations

**Guarded by:** `BLACKWELL_WGMMA_AVAILABLE` (defined for SM_120+)

---

### 5. MMQ Q8_0 Burst Pipeline
**Target Kernel:** `mul_mat_q` (Q8_0 quantization)

To saturate the massive compute throughput of Blackwell and hide the latency of global memory loads, we implemented a **2-tile burst pipeline** using a 4-buffer shared memory layout.

*   **Logic:**
    *   **Buffer A (Tiles 0 & 1):** Loaded and ready for compute.
    *   **Buffer B (Tiles 2 & 3):** Asynchronously loaded (`cp.async`) while Buffer A is being computed.
    *   **Pipeline:** 
        1.  Load Tiles 0 & 1 into Buffer A.
        2.  Issue async loads for Tiles 2 & 3 into Buffer B.
        3.  Compute Tile 0 & Tile 1 from Buffer A (doubling compute duration per wait).
        4.  Wait for Buffer B.
        5.  Issue async loads for next tiles into Buffer A.
        6.  Compute Tile 2 & Tile 3 from Buffer B.
        7.  Repeat.
*   **Shared Memory:** Allocated 4 full Y-tile buffers per block to support this double-buffered burst mode.
*   **Synchronization:** Leveraged `cp_async_wait_group<N>` to wait specifically for the compute-ready buffer while keeping the next prefetch in flight.

## Future Optimization Targets

Profiling indicates the following kernels may still be bottlenecks or fail to fully utilize Blackwell's capabilities:

| Kernel | Issue | Potential Optimization |
| :--- | :--- | :--- |
| **soft_max_f32** | Critical path latency | Warp shuffle reductions, larger tile sizes to reduce memory round-trips. |
| **rope_*** | High memory traffic | Fuse with attention kernels; explore TMA for weight loading. |
| **add_f32 / mul_f32** | Bandwidth bound | Increase block sizes, use vectorized loads (128-bit). |
| **quantize_row_q8_0** | Activation quantization overhead | Use TMA for input loading; overlap with `cp.async`. |
| **dequantize_*** | Weight loading bandwidth | Verify throughput; ensure 128-bit vectorization is active. |
| ~~**flash_attn**~~ | ~~Standard attention performance~~ | ✅ **DONE** - See Section 1 above |

## Profiling

Use the `profile_mmq_targeted.sh` script to collect specific metrics on these kernels:

```bash
./profile_mmq_targeted.sh
```

Key metrics watched:
*   `long_scoreboard` stalls (Global Memory Latency)
*   `sm__pipe_tensor_cycles_active` (Tensor Core Utilization)
*   `gpu__compute_memory_throughput` (DRAM Bandwidth)

---

## Environment Variables Summary

### Flash Attention Control
| Variable | Description |
|----------|-------------|
| `GGML_CUDA_NO_ATTENTION_V5` | Disable Blackwell BF16 attention kernel |
| `GGML_CUDA_NO_BLACKWELL_F16` | Disable Blackwell F16 attention kernel |

### L2 Cache Persistence
| Variable | Default | Description |
|----------|---------|-------------|
| `GGML_CUDA_L2_PERSIST` | OFF | Master enable for L2 persistence |
| `GGML_CUDA_L2_PERSIST_SIZE` | 48 | Max persist size in MB |
| `GGML_CUDA_L2_PERSIST_RATIO` | 1.0 | L2 hit ratio (0.0-1.0) |
| `GGML_CUDA_L2_PERSIST_DECODE_ONLY` | OFF | Only persist during decode phase |

### Debug
| Variable | Description |
|----------|-------------|
| `GGML_CUDA_DEBUG_MMQ_BUFFERS` | Log MMQ buffer allocations/resizes |

---

## Build Configuration

### Register Limit for Occupancy

Blackwell (SM_120) has strict register limits. The build system automatically applies `--maxrregcount=224` **only when SM_120 is the sole target architecture**:

```bash
# Optimal for Blackwell-only builds (register limit applied)
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=120a-real

# Multi-arch builds (register limit NOT applied to avoid penalizing other GPUs)
cmake -B build -DCMAKE_CUDA_ARCHITECTURES="86;120"
```

**CMake Variable:** `GGML_CUDA_SM120_MAXRREGCOUNT` (default: 224)

---

## Architecture Guards

The following compile-time guards control Blackwell-specific code:

| Guard | Defined When | Enables |
|-------|--------------|---------|
| `GGML_CUDA_CC_BLACKWELL` | SM_120+ | Blackwell compute capability checks |
| `BLACKWELL_TMA_AVAILABLE` | SM_120+ | TMA primitives in `tma.cuh` |
| `BLACKWELL_WGMMA_AVAILABLE` | SM_120+ | WGMMA primitives in `mma.cuh` |
| `CP_ASYNC_AVAILABLE` | SM_80+ | `cp.async` for async memory copies |