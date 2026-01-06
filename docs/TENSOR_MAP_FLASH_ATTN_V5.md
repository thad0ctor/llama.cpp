# Blackwell Flash Attention V5 - Complete Tensor Processing Chain

## Overview

This document traces tensors through the complete Blackwell Flash Attention v5 pipeline, from model layer Q/K/V creation through the optimized CUDA kernel execution. The focus is on critical dimension semantics and transformations at each step.

**Example Configuration (Qwen3):**
- D (head_dim) = 128
- n_heads = 32
- n_heads_kv = 32 (no GQA)
- seq_q = 1 (decoding)
- seq_kv = 768
- batch = 1

---

## Step 1: KV Cache Retrieval

**File:** `src/llama-kv-cache.cpp:get_k()`

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

## Step 2: build_attn_mha - Permute & Contiguous

**File:** `src/llama-graph.cpp:1446-1452`

**Input (before permute):**
```
Q: [D, n_heads, seq_q, batch] = [128, 32, 1, 1]
K: [D, n_heads_kv, seq_kv, batch] = [128, 32, 768, 1]
V: [D, n_heads_kv, seq_kv, batch] = [128, 32, 768, 1]
```

**Transformation:**
1. `ggml_permute(ctx0, x, 0, 2, 1, 3)` - Swaps dimensions 1 and 2
2. `ggml_cont(ctx0, x)` - Makes contiguous in new order
3. Type cast to BF16/F16 if needed

**Output (after permute + cont):**
```
Q: [D, seq_q, n_heads, batch] = [128, 1, 32, 1]
  ne[0] = 128, ne[1] = 1, ne[2] = 32, ne[3] = 1
  nb[1] = D * sizeof(T)        (stride to next seq)
  nb[2] = D * seq * sizeof(T)  (stride to next head)

K: [D, seq_kv, n_heads_kv, batch] = [128, 768, 32, 1]
V: [D, seq_kv, n_heads_kv, batch] = [128, 768, 32, 1]
```

**Key Insight:** After permute, `ne[1] = seq`, `ne[2] = heads`

---

## Step 3: ggml_flash_attn_ext

**File:** `ggml/src/ggml.c:5246-5287`

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

---

## Step 4: CUDA Dispatch

**File:** `ggml/src/ggml-cuda/fattn.cu:399-419`

**Transformation:**
- Checks compute capability (Blackwell = cc >= 1200)
- Validates head_dim == 128, types match
- Dispatches to `ggml_cuda_flash_attn_ext_attention_v5()`

---

## Step 5: attention_v5.cu Wrapper

**File:** `ggml/src/ggml-cuda/attention_v5.cu:611-835`

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

## Step 6: Kernel - Thread Block Assignment

**File:** `ggml/src/ggml-cuda/attention_v5.cu:274-291`

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

**File:** `ggml/src/ggml-cuda/attention_v5.cu:377-562`

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

**File:** `ggml/src/ggml-cuda/attention_v5.cu:564-599`

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
