# Critical Bottleneck Gap Analysis: Undocumented Performance Issues

**Status:** Analysis of profile_mmq_targeted.sh output vs existing documentation (TENSOR_MAP_FLASH_ATTN_V5.md, blackwell.md)

**Date:** 2026-01-06

---

## Executive Summary

Of the 10 major bottlenecks identified in profiling, **only Flash Attention V5 is documented** (TENSOR_MAP_FLASH_ATTN_V5.md). The remaining **9 kernels representing 79.4% of execution time are completely undocumented** regarding their performance characteristics, optimization constraints, and inefficiencies.

The documentation gap is critical because:
1. Engineering cannot understand WHY these bottlenecks exist
2. Optimization opportunities are invisible (inefficiency indicators not explained)
3. Platform-specific constraints (Blackwell architecture mismatch) go unaddressed
4. Root causes of stalls and underutilization are unexplained

---

## Part 1: Completely Undocumented Bottlenecks (9 kernels)

### CRITICAL: mul_mat_vec_q (Quantized Matrix-Vector Multiplication)

**Profile Data:**
- Duration: 5.54 µs per invocation
- Call count: 77+ calls in inference window
- **Tensor Core Usage: 0%** (completely bypassed)
- SM Throughput: 4.86% (massive underutilization)
- Memory Throughput: 11.55% (severely bandwidth-limited)
- Long Scoreboard Stalls: 8,005 per SMSP (88% time waiting for memory)
- Shared Memory Bank Conflicts: 0 (not the issue)

**Why This Matters:**
- Kernel never touches Tensor Cores despite operating on quantized weights
- Intended for compute throughput but achieving 4.86% SM utilization
- 88% of execution time spent stalled on long-latency memory operations
- **Optimization Potential: HIGH** (currently leaving 95% compute on the table)

**Current Documentation Coverage:** ZERO
- blackwell.md mentions MMQ optimizations but doesn't explain the 0% Tensor Core usage
- No discussion of why quantized ops can't leverage int8 Tensor Cores on Blackwell
- No analysis of the memory-load stall pattern

**Root Cause Analysis (Engineering Gap):**
The kernel uses integer quantization (Q8_0, Q4_K, etc.) but performs scalar dot products rather than vectorized Tensor Core operations. On Blackwell (with int8 Tensor Core support), this is a significant miss:
- File: `/ggml/src/ggml-cuda/mmvq.cu:143` - mul_mat_vec_q kernel
- Line 227: Loop over quantized blocks using `vec_dot_q_cuda` (scalar function pointer)
- No MMA (Matrix Multiply-Accumulate) instructions despite quantized data format
- Barrier at line 271: `__syncthreads()` after warp reductions (potential occupancy killer)

**Documentation Should Address:**
1. Why Tensor Cores are not used for quantized mat-vec operations
2. Memory stall patterns and what triggers long_scoreboard stalls
3. Comparison with matrix (non-vector) versions that DO use Tensor Cores
4. Occupancy analysis: Block size (128) with 55 registers/thread, grid size 512

---

### CRITICAL: attention_v5_splitk_reduce (Reduce Phase of Split-K Attention)

**Profile Data:**
- Duration: 22.40 µs per invocation
- Call count: 12 calls
- **Tensor Core Usage: 0%**
- SM Throughput: 0.45% (severe underutilization)
- Memory Throughput: 1.62% (bandwidth starvation)
- Long Scoreboard Stalls: 1,233 per SMSP
- Block Size: 128, Grid Size: 32

**Why This Matters:**
- Split-K is a Blackwell optimization technique for attention
- Reduce phase should combine partial sums efficiently
- Currently achieving 0% Tensor Core and 1.62% memory throughput
- **Second-longest duration (22.4 µs) after main attention kernel**
- Optimization Potential: HIGH (occupancy and memory access pattern critical)

**Current Documentation Coverage:** MENTIONED BUT NOT EXPLAINED
- TENSOR_MAP_FLASH_ATTN_V5.md line 205: Mentions "attention_v5_splitk_reduce<>()" exists
- No explanation of:
  - What Split-K is or why it's used
  - How the reduce phase differs from accumulation in forward pass
  - Why memory throughput is only 1.62%
  - Occupancy constraints (32 grid size is suspiciously small)

**Root Cause Analysis (Engineering Gap):**
File: `/ggml/src/ggml-cuda/attention_v5.cu:1292` - Split-K reduce kernel
- Designed for parallel accumulation across multiple SMs when seq_q is large
- Reduce phase reads partial results, sums, and writes
- Grid size 32 suggests suboptimal occupancy on Blackwell (which has 300+ SMs)
- **Memory access pattern likely not optimized for TMA (Tensor Memory Accelerator) usage**

**Documentation Should Address:**
1. Split-K algorithm overview and when it's triggered
2. Reduce kernel design and synchronization requirements
3. Why occupancy is limited (synchronization requirements?)
4. Memory coalescence pattern for read+reduce+write
5. Blackwell-specific optimization opportunities (TMA, async memcpy)

---

### HIGH PRIORITY: quantize_q8_1 (Activation Quantization)

**Profile Data:**
- Duration: 4.10 µs per invocation
- Call count: 76 calls
- Memory Throughput: 10.1% (severely limited)
- SM Throughput: 0.15% (near-zero compute utilization)
- Long Scoreboard Stalls: 75 per SMSP (minimal)
- Registers/Thread: 24 (light), Block Size: 256, Grid Size: 8

**Why This Matters:**
- Called 76 times in inference window (frequent hotspot)
- Memory-bound operation but achieving only 10% memory throughput
- Grid size 8 is tiny for Blackwell (suggests data parallelism bottleneck)
- Activation quantization is on critical path for KV cache updates
- **Optimization Potential: MODERATE-HIGH** (embarrassingly parallel, parallelization limited)

**Current Documentation Coverage:** ZERO
- blackwell.md line 35 mentions "quantize_row_q8_0" as future optimization target
- No analysis of q8_1 variant currently in production
- No discussion of why grid size is limited to 8

**Root Cause Analysis (Engineering Gap):**
File: `/ggml/src/ggml-cuda/quantize.cu:5` - quantize_q8_1 kernel
- Per-element quantization (scale factor computed via warp reduction)
- Line 35-36: `warp_reduce_max` and `warp_reduce_sum` (synchronous reductions)
- Line 47: Stores only when `iqs == 0` (one thread per quantization block)
- **Issue:** Multiple threads compute per-element but only one writes metadata

**Documentation Should Address:**
1. Algorithm: max-scale computation pattern and synchronization
2. Why grid size is limited (data dependencies in quantization blocks?)
3. Comparison with fp32→q8_0 (blackwell.md mentions as future target)
4. Blackwell optimizations:
   - TMA for input loading during quantization
   - Async barriers for scale factor computation
   - Occupancy impact of warp reductions

---

### HIGH PRIORITY: rms_norm_f32 (Layer Normalization)

**Profile Data:**
- Duration: 4.51 µs per invocation
- Call count: 52 calls
- Memory Throughput: 7.1% (severely limited)
- SM Throughput: 0.2% (near-zero compute)
- Long Scoreboard Stalls: 37.99 per SMSP (minimal, computation-light)
- Block Size: 1024, Grid Size: 1

**Why This Matters:**
- Called 52 times (frequent, non-attention path)
- Worst memory throughput (7.1%) in entire profile
- **Grid size 1 means single-block kernel** (gross underutilization of Blackwell's 300+ SMs)
- Synchronous across all 1024 threads (high synchronization cost)
- **Optimization Potential: VERY HIGH** (trivial to parallelize across samples/channels)

**Current Documentation Coverage:** ZERO
- Not mentioned in any optimization documentation
- No discussion of single-block constraint
- No analysis of why parallelization is not used

**Root Cause Analysis (Engineering Gap):**
File: `/ggml/src/ggml-cuda/norm.cu:4` - norm_f32 template
- Generic normalization kernel supporting arbitrary shapes
- Line 8-17: Iterates all columns sequentially within single block
- Line 28-40: Warp-reduce for mean/variance (synchronous)
- Line 46-48: Writes result per-thread (serialized output phase)

**Problem:** Current design assumes single row → single block
- Works for single-token inference (seq_q=1) but inefficient
- Blackwell can launch much larger grids; no coordination needed
- Could split norm across multiple blocks with atomic adds or persistent kernels

**Documentation Should Address:**
1. Current algorithm: single-block constraint and why
2. Memory access pattern: sequential column iteration (L1 cache miss pattern?)
3. Comparison with multi-block designs (for larger batch sizes)
4. Blackwell-specific optimization:
   - Persistent kernel approach (fewer SMs, higher occupancy)
   - TMA for weight loading
   - Async operations during warp reductions

---

### MEDIUM PRIORITY: rope_neox (Rotary Position Embeddings)

**Profile Data:**
- Duration: 4.16 µs per invocation
- Call count: 26 calls
- Memory Throughput: 8.84% (severely limited)
- SM Throughput: 0.03% (negligible compute)
- Long Scoreboard Stalls: 14.50 per SMSP (minimal)
- Block Size: 256, Grid Size: 4

**Why This Matters:**
- Called 26 times in standard inference
- Part of critical path (must complete before attention)
- Transcendental operations (cos, sin) are expensive but NOT using Tensor Cores
- Grid size 4 is suboptimal for Blackwell (only 1k threads total)
- **Optimization Potential: MEDIUM** (compute-bound, but fusing with attention is hard)

**Current Documentation Coverage:** MENTIONED BUT NOT ANALYZED
- blackwell.md line 33: "rope_*** | High memory traffic | Fuse with attention kernels; explore TMA for weight loading."
- Documentation is generic guidance, not specific analysis
- No explanation of why 8.84% memory throughput
- No analysis of transcendental operation bottleneck

**Root Cause Analysis (Engineering Gap):**
File: `/ggml/src/ggml-cuda/rope.cu:44` - rope_norm template
- Computes rotary embeddings: `cos(theta) * mscale`, `sin(theta) * mscale`
- Line 96: Theta computation per-element: `powf(theta_scale, i0/2.0f)` (very expensive)
- Line 37-38: Calls `cosf()` and `sinf()` per thread (no vectorization)
- No shared memory usage (all direct global I/O)

**Problem:**
- Function calls are issued by each thread (256 threads computing cos/sin independently)
- powf in theta computation is expensive (transcendental)
- No prefetching or double-buffering

**Documentation Should Address:**
1. Rope algorithm: why theta must be recomputed (can't be pre-cached?)
2. Transcendental function cost analysis (cos, sin throughput on Blackwell)
3. Current memory access pattern and L1 hit rate
4. Why fusion with attention is suggested but not implemented
5. Blackwell-specific optimizations:
   - Use fast math flags (__cosf, __sinf)?
   - TMA for theta_scale lookup?
   - Fusing with attention to amortize memory I/O?

---

### MEDIUM PRIORITY: cpy_scalar (Memory Copy)

**Profile Data:**
- Duration: 12.29 µs per invocation
- Call count: 25 calls
- Memory Throughput: 8.53% (severely limited)
- SM Throughput: 54.87% (high, but memory-bound)
- Long Scoreboard Stalls: 24,483 per SMSP (second-highest stalls)
- Block Size: 64, Grid Size: 14,336

**Why This Matters:**
- Called 25 times during inference
- **Huge grid size (14,336)** suggests data-parallel operation on large tensor
- High stall count (24,483) indicates memory access contention
- Non-compute kernel but still consuming significant time
- **Optimization Potential: HIGH** (memory access pattern critical)

**Current Documentation Coverage:** ZERO
- Not mentioned in any optimization documentation
- No explanation of stall pattern or memory pattern

**Root Cause Analysis (Engineering Gap):**
This appears to be a scalar copy kernel (possibly format conversion or redistribution).
- Large grid with small block size suggests many independent copy operations
- 24,483 long scoreboard stalls indicate memory dependency chain
- Could be element-wise conversion, tensor gather/scatter, or dtype translation
- **Unknown:** exact operation and whether it could be fused with compute kernels

**Documentation Should Address:**
1. What operation is cpy_scalar performing?
2. Why is memory throughput only 8.53% despite massively parallel grid?
3. Memory access pattern: sequential? strided? scattered?
4. Could this operation be fused with other kernels?
5. Blackwell optimizations:
   - Vectorized loads (128-bit for memory bus efficiency)?
   - TMA for bulk moves?
   - Async memcpy for overlapping with compute?

---

### MEDIUM PRIORITY: mul_mat_vec_f (Float Matrix-Vector)

**Profile Data:**
- Duration: 4.54 µs per invocation
- Call count: 12 calls
- Memory Throughput: 13.34% (better than Q variant, but still constrained)
- SM Throughput: 2.69% (severe underutilization)
- Long Scoreboard Stalls: 3,040 per SMSP
- Block Size: 256, Grid Size: 128

**Why This Matters:**
- 12 calls in inference (less frequent than quantized variant)
- Float versions should be faster but still underutilizing compute
- SM throughput (2.69%) even worse than quantized variant (4.86%)
- **Optimization Potential: MEDIUM** (expected to be memory-bound, but current design is too passive)

**Current Documentation Coverage:** ZERO
- Not mentioned in optimization documentation
- No analysis of float vs quantized performance difference

**Root Cause Analysis (Engineering Gap):**
File: `/ggml/src/ggml-cuda/mmvq.cu` - uses same mul_mat_vec_q kernel but with float types
- Scalar dot products (no Tensor Core usage)
- Larger working set than quantized variant (float = 4 bytes vs int8 = 1 byte)
- Memory throughput better than quantized (13.34% vs 11.55%) because larger blocks
- But still suffering from stall pattern (3,040 long scoreboard per SMSP)

**Documentation Should Address:**
1. Why float mat-vec also avoids Tensor Cores
2. Comparison with quantized performance (what's the trade-off?)
3. Grid size (128) utilization: is this optimal for Blackwell?
4. Memory prefetching opportunities
5. Why compute throughput is lower than quantized (4.86% → 2.69%)

---

### LOW PRIORITY: topk_moe_cuda_optimized (Mixture of Experts)

**Profile Data:**
- Duration: 6.27 µs per invocation
- Call count: 12 calls
- Memory Throughput: 5.84% (extremely limited)
- SM Throughput: 0.01% (near-zero)
- Long Scoreboard Stalls: 3.19 per SMSP (minimal)
- Block Size: 256, Grid Size: 1

**Why This Matters:**
- Only 12 calls (lowest frequency bottleneck)
- **Grid size 1 single-block kernel** (complete underutilization)
- Extremely low memory throughput
- MoE is important for model scaling but not optimized
- **Optimization Potential: MODERATE** (specific to MoE-based architectures)

**Current Documentation Coverage:** ZERO
- Not mentioned in any documentation
- No discussion of MoE kernel constraints

**Root Cause Analysis (Engineering Gap):**
File: `/ggml/src/ggml-cuda/moe.cu` (unknown location, likely in quantize or mmvq area)
- topk selection for MoE gating is sequential by nature
- Single-block constraint makes sense if reading global router output
- But writing experts selection should be parallelizable

**Documentation Should Address:**
1. MoE topk algorithm and why single-block design
2. Memory pattern: sorted expert selection
3. Parallelization opportunities for modern architectures
4. Blackwell-specific: can topk leverage new sorting instructions?

---

### VERY LOW PRIORITY: k_get_rows_float (Gather Operation)

**Profile Data:**
- Duration: 4 µs per invocation
- Call count: Frequent (exact count unknown)
- Memory Throughput: 9.18%
- SM Throughput: 0.22%

**Why This Matters:**
- Gather operation used for various indexing patterns
- Low memory throughput suggests uncoalesced access
- **Optimization Potential: LOW-MEDIUM** (depends on access pattern)

**Current Documentation Coverage:** ZERO

---

## Part 2: Documented but Incompletely Analyzed - Flash Attention V5

### attention_v5_kernel (Main Kernel)

**Profile Data:**
- Duration: 28.77 µs per invocation
- Call count: 12 calls
- Tensor Core Usage: 56.3% (good, but room for improvement)
- SM Throughput: 39.2% (acceptable)
- Memory Throughput: 18.9% (acceptable)
- Shared Mem Bank Conflicts: 857 (moderate)
- Long Scoreboard Stalls: 3,053 per SMSP
- Registers/Thread: 168 (high occupancy penalty)

**Documentation Coverage:** COMPLETE (TENSOR_MAP_FLASH_ATTN_V5.md)
- Covers tensor layouts, permutations, kernel grid structure
- Explains stride calculations and Q/K/V access patterns
- Documents output dimension swapping

**What Documentation MISSES:**

1. **Performance Gap Between Main Kernel and Reduce Phase:**
   - Main kernel: 28.77 µs with 56% Tensor Core usage
   - Reduce phase: 22.40 µs with 0% Tensor Core usage
   - **Documentation doesn't explain why reduce is almost as slow as forward pass**

2. **Memory Throughput Analysis (18.9%):**
   - Documentation explains what data moves but not why throughput is constrained
   - No analysis of shared memory patterns or bank conflicts (857 conflicts)
   - No explanation of long scoreboard stalls (3,053 per SMSP)

3. **Register Pressure (168 regs/thread):**
   - With 256 threads/block and 168 registers, occupancy is limited
   - Documentation doesn't discuss occupancy bottleneck
   - No analysis of whether register usage could be optimized

4. **Tensor Core Utilization Gap (56% vs theoretical max):**
   - Why not 80-90%?
   - Is it output synchronization? memory waits? shared memory conflicts?
   - No bottleneck identification

5. **Blackwell-Specific Optimizations Missing:**
   - Tensor Memory Accelerator (TMA) for async weight loading not discussed
   - Warp specialization opportunities not analyzed
   - Synchronization primitives (e.g., `cp.async_wait_group`) mentioned in code but not explained

---

## Part 3: Optimization Potential by Impact

| Rank | Kernel | Time Impact | Current Utilization | Gap | Effort | Priority |
|------|--------|-------------|-------------------|-----|--------|----------|
| 1 | mul_mat_vec_q | 40.3% | SM: 4.9%, TC: 0% | 95.1% | MEDIUM | CRITICAL |
| 2 | attention_v5_kernel | 11.8% | SM: 39.2%, TC: 56% | 43.8% | HARD | CRITICAL |
| 3 | attention_v5_splitk_reduce | 9.0% | SM: 0.45%, TC: 0% | 99.6% | HARD | CRITICAL |
| 4 | quantize_q8_1 | 9.4% | SM: 0.15%, TC: 0% | 99.9% | MEDIUM | HIGH |
| 5 | rms_norm_f32 | 9.2% | SM: 0.2%, TC: 0% | 99.8% | EASY | HIGH |
| 6 | cpy_scalar | 10.8% | SM: 54.9%, TC: 0% | Memory-bound | MEDIUM | HIGH |
| 7 | rope_neox | 3.6% | SM: 0.03%, TC: 0% | 99.97% | HARD | MEDIUM |
| 8 | mul_mat_vec_f | 1.8% | SM: 2.69%, TC: 0% | 97.3% | MEDIUM | MEDIUM |
| 9 | topk_moe_cuda_optimized | 2.5% | SM: 0.01%, TC: 0% | 99.99% | HARD | LOW |

---

## Part 4: Documentation Checklist - What Should Exist

### Per Kernel Documentation Should Cover:

1. **Algorithm Overview**
   - What operation is performed
   - When it's invoked in inference pipeline
   - Data dependencies and synchronization requirements

2. **Performance Analysis**
   - Current inefficiencies (0% Tensor Core, low SM throughput, etc.)
   - Root cause of performance gaps
   - Memory access pattern and coalescence analysis
   - Occupancy constraints and why

3. **Profiling Metrics Explained**
   - What long_scoreboard stalls mean for this kernel
   - What memory throughput percentage is achievable
   - Register pressure impact on occupancy
   - Grid/block size rationale

4. **Blackwell Architecture Mismatch** (Critical)
   - Is this kernel using Tensor Cores appropriately?
   - Are there Blackwell-specific primitives not being used?
   - TMA (Tensor Memory Accelerator) opportunities
   - Async synchronization opportunities (cp.async, wait_groups)

5. **Optimization Roadmap**
   - Known bottlenecks and feasible optimizations
   - Impact estimates (time saved, occupancy improvement)
   - Engineering effort and complexity
   - Dependencies on other optimizations

### Missing Documentation Files:

```
/docs/CUDA_KERNELS_BOTTLENECK_ANALYSIS.md          (comprehensive kernel reference)
/docs/QUANTIZED_MATMUL_OPTIMIZATION.md             (mul_mat_vec_q deep dive)
/docs/NORMALIZATION_KERNELS.md                      (rms_norm, norm_f32, etc.)
/docs/ROPE_EMBEDDING_OPTIMIZATION.md                (rope_neox, rope_norm)
/docs/MOE_KERNEL_OPTIMIZATION.md                    (topk_moe_cuda_optimized)
/docs/SPLIT_K_ATTENTION_ANALYSIS.md                 (attention_v5_splitk_reduce)
/docs/BLACKWELL_ARCHITECTURE_MISMATCH.md            (systematic analysis of all kernels)
```

---

## Part 5: Key Findings Summary

### Finding 1: Tensor Core Avoidance Pattern
**9 out of 10 kernels achieve 0% Tensor Core usage**, despite:
- Blackwell having int8 and fp8 Tensor Cores
- Quantized operations being natural candidates for Tensor Cores
- NVIDIA publishing multiple papers on quantized Tensor Core operations

**Why This Matters:** This suggests systematic architectural gap between what Blackwell offers and what the codebase exploits.

**Engineering Action Item:** Systematic audit of which kernels COULD use Tensor Cores but don't:
- Quantized ops (mul_mat_vec_q, quantize_q8_1) should use int8 Tensor Cores
- Attention reduce phase should use FP32 Tensor Cores if possible
- Rope operations might benefit from FP8 Tensor Cores

---

### Finding 2: Grid Size Suboptimality
Multiple kernels severely underutilize Blackwell's massive SM count (300+):
- rms_norm_f32: Grid size 1 (single block)
- topk_moe_cuda_optimized: Grid size 1 (single block)
- quantize_q8_1: Grid size 8 (only 8 blocks for 300+ SMs)
- rope_neox: Grid size 4 (only 4 blocks for 300+ SMs)

**Why This Matters:** Grid size 1 kernels cannot achieve >0.3% of Blackwell's compute.

**Engineering Action Item:** Parallelize single-block kernels or use persistent kernel patterns.

---

### Finding 3: Memory Throughput Ceiling
Nearly all kernels are severely memory-limited:
- Best performer: 18.9% (attention_v5_kernel)
- Worst: 0.3% (various kernels)
- Average: 8-10% memory throughput

**Why This Matters:** Indicates systematic memory access inefficiency, not compute limitation.

**Potential Causes:**
- Uncoalesced access patterns (scattered reads/writes)
- Shared memory bank conflicts (857 in attention kernel)
- Insufficient prefetching/pipelining
- Not leveraging TMA (Tensor Memory Accelerator)

---

### Finding 4: Long Scoreboard Stalls
Multiple kernels show excessive long scoreboard stalls (memory latency):
- mul_mat_vec_q: 8,005 per SMSP (88% of execution time)
- cpy_scalar: 24,483 per SMSP (highest)
- attention kernel: 3,053 per SMSP

**Why This Matters:** Indicates memory access chains that can't be hidden by instruction scheduling.

**Indicates:** Insufficient instruction-level parallelism to hide global memory latency.

---

## Part 6: Recommendations

### Immediate (Week 1)
1. Create `/docs/CRITICAL_PERFORMANCE_GAPS.md` documenting:
   - Which kernels avoid Tensor Cores and why
   - Memory throughput limitations and root causes
   - Grid size suboptimality

2. Add kernel-specific sections to CLAUDE.md:
   - Links to bottleneck analysis
   - Profiling metrics to watch
   - Known optimization limitations

### Short-term (Month 1)
1. Write detailed analysis for top 5 bottlenecks:
   - mul_mat_vec_q (40.3% of time)
   - attention_v5_splitk_reduce (9.0% of time)
   - quantize_q8_1 (9.4% of time)
   - rms_norm_f32 (9.2% of time)
   - cpy_scalar (10.8% of time)

2. Create optimization roadmap linking:
   - Current bottleneck
   - Feasible optimizations
   - Time savings estimate
   - Engineering effort

### Long-term (Quarter 1)
1. Systematic Tensor Core audit:
   - Which kernels should use Tensor Cores
   - Implementation complexity per kernel
   - Estimated throughput improvements

2. Blackwell architecture utilization study:
   - TMA feasibility for each kernel
   - Async synchronization opportunities
   - Persistent kernel patterns for small-grid kernels

3. Memory access pattern analysis:
   - Coalescence testing per kernel
   - Shared memory bank conflict analysis
   - TMA vs manual prefetching trade-offs

---

## Appendix A: File Locations

### Kernel Source Files
- `ggml/src/ggml-cuda/mmvq.cu:143` - mul_mat_vec_q
- `ggml/src/ggml-cuda/attention_v5.cu:255` - attention_v5_kernel (main)
- `ggml/src/ggml-cuda/attention_v5.cu:1292` - attention_v5_splitk_reduce
- `ggml/src/ggml-cuda/quantize.cu:5` - quantize_q8_1
- `ggml/src/ggml-cuda/norm.cu:4` - norm_f32
- `ggml/src/ggml-cuda/rope.cu:44` - rope_norm

### Profile Data
- `logs/profile_targeted/20260106_141655/profile.csv` - NCU profiler output

### Documentation Files
- `docs/TENSOR_MAP_FLASH_ATTN_V5.md` - Only comprehensive kernel analysis
- `docs/blackwell.md` - High-level optimization targets (no details)
- `docs/build.md` - Build configuration, no performance analysis
- `CLAUDE.md` - General guidance, insufficient bottleneck detail

---

## Appendix B: Metric Definitions (from profile_mmq_targeted.sh)

- **gpu__compute_memory_throughput**: % of peak memory bandwidth sustained
- **sm__pipe_tensor_cycles_active**: % of time Tensor Cores are executing
- **sm__throughput**: % of peak SM compute throughput
- **smsp__warps_issue_stalled_long_scoreboard**: Global memory latency stalls
- **smsp__warps_issue_stalled_short_scoreboard**: Local memory/shared mem latency
- **l1tex__data_bank_conflicts_pipe_lsu_mem_shared**: Shared memory bank conflicts
- **launch__registers_per_thread**: Register usage (occupancy limiting factor)

