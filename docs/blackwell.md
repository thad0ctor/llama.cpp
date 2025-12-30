# Blackwell (SM_120) Optimizations

This document tracks specific optimizations targeted at the NVIDIA Blackwell architecture (RTX 5090 / GB200).

## implemented Optimizations

### 1. MMQ Q8_0 Burst Pipeline
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
| **flash_attn** | Standard attention performance | Check if custom MLA optimizations can apply or if specific Blackwell `fa_` kernels are needed. |

## Profiling

Use the `profile_mmq_targeted.sh` script to collect specific metrics on these kernels:

```bash
./profile_mmq_targeted.sh
```

Key metrics watched:
*   `long_scoreboard` stalls (Global Memory Latency)
*   `sm__pipe_tensor_cycles_active` (Tensor Core Utilization)
*   `gpu__compute_memory_throughput` (DRAM Bandwidth)