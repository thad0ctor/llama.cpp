# Blackwell Flash Attention - Complete Tensor Processing Chain

## Overview

This document traces tensors through the complete Blackwell Flash Attention pipeline, from model layer Q/K/V creation through the optimized CUDA kernel execution. The focus is on critical dimension semantics and transformations at each step.

**Blackwell Kernel Variants:**
- **attention_v5**: BF16-optimized kernel (requires all BF16 inputs)
- **blackwell_f16**: F16-optimized kernel (requires all F16 inputs) - most commonly used since Q is cast to F16

**Example Configuration (Qwen3):**
- D (head_dim) = 128
- n_heads = 32
- n_heads_kv = 32 (no GQA)
- seq_q = 1 (decoding)
- seq_kv = 768
- batch = 1

---

## Step 1: KV Cache Retrieval

**Function:** `llama_kv_cache::get_k()`, `llama_kv_cache::get_v()`
**File:** `src/llama-kv-cache.cpp:1008`

**Input:**
- K cache buffer: raw allocated tensor
- Layer index: il

**Transformation:**
- Extract slice of KV cache for current batch slot via `ggml_view_4d`

**Output (K, V from cache):**
```
Shape: [D, n_heads_kv, n_kv, batch]
  ne[0] = 128  (head dimension)
  ne[1] = 32   (number of KV heads)
  ne[2] = 768  (KV sequence length)
  ne[3] = 1    (batch)
```

---

## Step 2: build_attn_mha - Permute & Type Cast

**Function:** `llm_graph_context::build_attn_mha()`
**File:** `src/llama-graph.cpp:1400`

**Input (before permute):**
```
Q: [D, n_heads, seq_q, batch] = [128, 32, 1, 1]
K: [D, n_heads_kv, seq_kv, batch] = [128, 32, 768, 1]
V: [D, n_heads_kv, seq_kv, batch] = [128, 32, 768, 1]
```

**Transformation:**
1. `ggml_permute(ctx0, x, 0, 2, 1, 3)` - Swaps dimensions 1 and 2
2. V transpose if needed: `if (v_trans) v = ggml_transpose(ctx0, v)`
3. Type cast Q/K/V to F16 if F32 (enables optimized Blackwell kernels)

**Type Casting (lines 1432-1445):**
```cpp
// Q is typically F32 from computation, cast to F16 for Blackwell kernels
if (q->type == GGML_TYPE_F32) {
    q = ggml_cast(ctx0, q, GGML_TYPE_F16);
}
// K/V typically already F16 from KV cache, but cast if F32
if (k->type == GGML_TYPE_F32) {
    k = ggml_cast(ctx0, k, GGML_TYPE_F16);
}
if (v->type == GGML_TYPE_F32) {
    v = ggml_cast(ctx0, v, GGML_TYPE_F16);
}
```

**Output (after permute):**
```
Q: [D, seq_q, n_heads, batch] = [128, 1, 32, 1]
  ne[0] = 128, ne[1] = 1, ne[2] = 32, ne[3] = 1
  nb[1] = D * sizeof(T)        (stride to next seq)
  nb[2] = D * seq * sizeof(T)  (stride to next head)

K: [D, seq_kv, n_heads_kv, batch] = [128, 768, 32, 1]
V: [D, seq_kv, n_heads_kv, batch] = [128, 768, 32, 1]
```

**Key Insight:** After permute, `ne[1] = seq`, `ne[2] = heads`. No `ggml_cont()` needed here.

---

## Step 3: ggml_flash_attn_ext

**Function:** `ggml_flash_attn_ext()`
**File:** `ggml/src/ggml.c:5246`

**Input:**
```
q: [D, seq_q, n_heads, batch]
k: [D, seq_kv, n_heads_kv, batch]
v: [D, seq_kv, n_heads_kv, batch]
mask: [n_kv, n_tokens, 1, batch]
```

**Transformation:**
Creates output tensor with **SWAPPED dimensions**:
```c
int64_t ne[4] = { v->ne[0], q->ne[2], q->ne[1], q->ne[3] };
```

**Output tensor shape:**
```
O: [D, n_heads, seq_q, batch] = [128, 32, 1, 1]
  ne[0] = D = 128
  ne[1] = n_heads = 32   <-- SWAPPED from input ne[2]!
  ne[2] = seq_q = 1      <-- SWAPPED from input ne[1]!
  ne[3] = batch = 1
```

**Critical:** Output layout differs from input layout!

**Post-FA Processing (lines 1452-1472):**
```cpp
ggml_flash_attn_ext_add_sinks(cur, sinks);  // Add attention sinks if configured
ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);  // Set accumulator precision

// MLA (Multi-head Latent Attention) projection if applicable (e.g., DeepSeek)
if (v_mla) {
    cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
    cur = ggml_mul_mat(ctx0, v_mla, cur);
    cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
    cur = ggml_cont(ctx0, cur);
}

// Flatten output: [D, n_heads, seq_q, batch] -> [D*n_heads, seq_q*batch]
cur = ggml_reshape_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);
```

---

## Step 4: CUDA Dispatch

**Function:** `ggml_cuda_flash_attn_ext()`
**File:** `ggml/src/ggml-cuda/fattn.cu:398`

**Transformation:**
- Checks compute capability (Blackwell = cc >= 1200)
- For Blackwell GPUs, selects kernel based on tensor types:
  - **attention_v5**: Requires Q/K/V all BF16, head_dim=128
  - **blackwell_f16**: Requires Q/K/V all F16, head_dim=64 or 128
- Falls back to `mma_f16` or `vec` kernels if Blackwell kernels not supported

**Blackwell Kernel Selection (lines 340-350):**
```cpp
if (cc >= GGML_CUDA_CC_BLACKWELL) {
    // BF16 path
    if (!ggml_cuda_no_attention_v5 && ggml_cuda_flash_attn_ext_attention_v5_supported(dst)) {
        return BEST_FATTN_KERNEL_ATTENTION_V5;
    }
    // F16 path (most common after Q cast to F16)
    if (!ggml_cuda_no_blackwell_f16 && ggml_cuda_flash_attn_ext_blackwell_f16_supported(dst)) {
        return BEST_FATTN_KERNEL_BLACKWELL_F16;
    }
}
```

**Environment Variables:**
- `GGML_CUDA_NO_ATTENTION_V5`: Disable BF16 kernel, fall back to MMA/vec
- `GGML_CUDA_NO_BLACKWELL_F16`: Disable F16 kernel, fall back to MMA/vec

---

## Step 5: attention_v5.cu Wrapper

**Function:** `ggml_cuda_flash_attn_ext_attention_v5()`
**File:** `ggml/src/ggml-cuda/attention_v5.cu:1717`

**Input tensors (from dst->src[]):**
```
Q: [D, seq_q, n_heads, batch] = [128, 1, 32, 1]
K: [D, seq_kv, n_heads_kv, batch] = [128, 768, 32, 1]
V: [D, seq_kv, n_heads_kv, batch] = [128, 768, 32, 1]
```

**Dimension Extraction:**
```cpp
ne01 = Q->ne[2] = 32   // n_heads (from dim 2 after permute)
ne02 = Q->ne[1] = 1    // seq_q (from dim 1 after permute)
ne11 = K->ne[2] = 32   // n_heads_kv
ne12 = K->ne[1] = 768  // seq_kv
```

**Input Stride Extraction (for [D, seq, heads, batch]):**
```cpp
stride_Q_row  = Q->nb[1] / sizeof(T)  // stride to next seq (dim 1)
stride_Q_head = Q->nb[2] / sizeof(T)  // stride to next head (dim 2)
```

**Output Stride Extraction (for [D, heads, seq, batch] - SWAPPED!):**
```cpp
stride_O_row  = dst->nb[2] / sizeof(To)  // seq is dim 2 in output
stride_O_head = dst->nb[1] / sizeof(To)  // heads is dim 1 in output
```

**Kernel Configuration:**
```
BLOCK_Q = 64, BLOCK_KV = 64, DIM = 128, NUM_WARPS = 4
num_blocks = n_heads * n_batch * ceil(seq_q / BLOCK_Q) = 32
```

---

## Step 5b: blackwell_f16 Wrapper (F16 Path)

**Function:** `ggml_cuda_flash_attn_ext_blackwell_f16()`
**File:** `ggml/src/ggml-cuda/fattn-blackwell-f16.cuh:786`

This is the **most common path** since Q is cast from F32 to F16 in `build_attn_mha()`.

**Input tensors (same as attention_v5):**
```
Q: [D, seq_q, n_heads, batch] = [128, 1, 32, 1] (F16)
K: [D, seq_kv, n_heads_kv, batch] = [128, 768, 32, 1] (F16)
V: [D, seq_kv, n_heads_kv, batch] = [128, 768, 32, 1] (F16)
```

**Parameter Extraction:**
```cpp
scale = op_params[0]        // Attention scale (default: 1/sqrt(head_dim))
max_bias = op_params[1]     // ALiBi max bias
logit_softcap = op_params[2] // Logit softcap (e.g., Gemma2)
```

**Kernel Launch by Head Dimension:**
```cpp
switch (ne00) {  // head_dim
    case 64:
        launch_flash_attn_blackwell_f16<64, 64, 1, 1>(...);
        break;
    case 128:
        launch_flash_attn_blackwell_f16<128, 128, 1, 1>(...);
        break;
}
```

**Kernel Configuration (F16):**
```
BLOCK_M = 64 (Q rows per block)
BLOCK_N = 64 (K/V block size)
NUM_WARPS = 4
NUM_THREADS = 128
Uses m16n8k16 MMA tensor core instructions
```

---

## Step 6: Kernel - Thread Block Assignment

**Function:** `attention_v5_kernel<>()` (template)
**File:** `ggml/src/ggml-cuda/attention_v5.cu:255`

**Block ID Decomposition:**
```cpp
bs = n_heads * n_batch = 32
num_q_blocks = ceil(len_q / BLOCK_Q) = 1
bs_id = bid / num_q_blocks
q_block_id = bid % num_q_blocks

head_id = bs_id % n_heads      // 0-31
batch_id = bs_id / n_heads     // 0
kv_head_id = head_id / gqa_ratio  // = head_id for non-GQA
```

**Pointer Setup:**
```cpp
Q += batch_id * stride_Q_batch + head_id * stride_Q_head + q_block_id * BLOCK_Q * stride_Q_row
K += batch_id * stride_K_batch + kv_head_id * stride_K_head
V += batch_id * stride_V_batch + kv_head_id * stride_V_head
O += batch_id * stride_O_batch + head_id * stride_O_head + q_block_id * BLOCK_Q * stride_O_row
```

---

## Step 7: Kernel - KV Loop

**Function:** `attention_v5_kernel<>()` (KV iteration loop)
**File:** `ggml/src/ggml-cuda/attention_v5.cu:475`

**Iteration:**
```
num_kv_iter = ceil(len_kv / BLOCK_KV) = ceil(768 / 64) = 12
```

**Per Iteration:**
1. Load K block [BLOCK_KV, D] to shared memory
2. Load V block [BLOCK_KV, D] to shared memory
3. Compute S = Q @ K.T using MMA (m16n8k16)
4. Apply mask, scale, softmax
5. Compute O += softmax(S) @ V using MMA
6. Track running max and sum for online softmax

---

## Step 8: Kernel - Output Write

**Function:** `attention_v5_kernel<>()` (output write) or `attention_v5_splitk_reduce<>()` (for Split-K)
**File:** `ggml/src/ggml-cuda/attention_v5.cu:650` (direct) / `1292` (Split-K reduce)

**Transformation:**
1. Normalize by softmax denominator
2. Convert to output type (F32/F16/BF16)
3. Write to global memory using output strides

**Output Write Pattern:**
```cpp
O[row * stride_O_row + col] = normalized_value
// where stride_O_row = dst->nb[2] / sizeof(To) for [D, heads, seq, batch] output
```

---

## Stride Summary Table

| Tensor | Layout | ne[1] | ne[2] | stride_row | stride_head |
|--------|--------|-------|-------|------------|-------------|
| Q (input) | [D, seq, heads, batch] | seq | heads | nb[1] | nb[2] |
| K (input) | [D, seq, heads, batch] | seq | heads | nb[1] | nb[2] |
| V (input) | [D, seq, heads, batch] | seq | heads | nb[1] | nb[2] |
| O (output) | [D, heads, seq, batch] | **heads** | **seq** | **nb[2]** | **nb[1]** |

**Critical:** Output strides are SWAPPED because output layout differs from input!

---

## Quick Reference: Dimension Semantics by Location

| Location | Q dimensions | K dimensions | Notes |
|----------|-------------|--------------|-------|
| KV cache | [D, heads, seq, batch] | [D, heads_kv, seq_kv, batch] | Original layout |
| After permute | [D, seq, heads, batch] | [D, seq_kv, heads_kv, batch] | ne[1]=seq, ne[2]=heads |
| Output tensor | [D, heads, seq, batch] | N/A | ne[1]=heads, ne[2]=seq (swapped!) |
