# Flash Attention Debugging Session - Blackwell sm_120

## Problem Summary
Flash attention is broken on NVIDIA Blackwell GPUs (RTX 5090, sm_120, compute capability 12.0). When flash attention is **disabled**, inference works correctly. When **enabled**, the server hangs during prompt processing (~60% through the first batch).

## Hardware
- 2x NVIDIA GeForce RTX 5090 (sm_120, compute 12.0)
- CUDA 13.1

## Symptoms
1. Server starts and loads model successfully
2. First prompt processing begins: `prompt processing progress, n_tokens = 2048, batch.n_tokens = 2048, progress = 0.604665`
3. Server hangs indefinitely - no errors, no crash, just stuck
4. With `--flash-attn off`, everything works fine

## What Was Already Fixed
We fixed a **compilation error** in `ggml/src/ggml-cuda/fattn-mma-f16.cuh` where `ggml_cuda_fattn_mma_get_config_blackwell()` was ignoring the `ncols` parameter, causing zero-sized arrays for certain template instantiations (ncols <= 16).

The fix added per-ncols configurations:
- ncols <= 16: Use `nbatch_fa=128` (occupancy=2)
- ncols >= 32: Use `nbatch_fa=64` (occupancy=3)

This fixed compilation, but there's now a **runtime hang**.

## Build Configuration
```bash
cmake .. \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_FORCE_MMQ=ON \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_CUDA_ARCHITECTURES="120a-real" \
    -DCMAKE_BUILD_TYPE=Release
```

## Starting Point
1. First, review my pending git changes: `git diff` to see what's been modified
2. Focus on `ggml/src/ggml-cuda/fattn-mma-f16.cuh` - the Blackwell flash attention implementation
3. Key areas to investigate:
   - `flash_attn_ext_f16_blackwell` kernel (line ~2026)
   - TMA (Tensor Memory Accelerator) code paths
   - Warp specialization (1 producer + 7 consumers pattern)
   - `mbarrier` synchronization primitives
   - The `fattn_producer_loop` and consumer logic

## Likely Causes
1. **Deadlock in mbarrier synchronization** - producer/consumer not properly synchronized
2. **TMA descriptor issues** - tensor maps may be incorrectly configured
3. **Warp divergence** - some warps may be stuck waiting
4. **Shared memory bank conflicts** - causing extreme slowdown appearing as hang
5. **Pipeline state machine bug** - `fattn_pipeline_state` handling

## Debug Commands
```bash
# Test with flash attention disabled (works)
./launch.sh  # Currently has --flash-attn off

# Test with flash attention enabled (hangs)
# Edit launch.sh to use --flash-attn on

# For debugging, can add to launch.sh:
export CUDA_LAUNCH_BLOCKING=1
```

## Files to Examine
- `ggml/src/ggml-cuda/fattn-mma-f16.cuh` - Main flash attention MMA implementation
- `ggml/src/ggml-cuda/fattn-blackwell-f16.cuh` - Blackwell-specific code (if exists)
- `ggml/src/ggml-cuda/tma.cuh` - TMA helpers
- `ggml/src/ggml-cuda/mma.cuh` - MMA tile operations
- `ggml/src/ggml-cuda/common.cuh` - Common CUDA utilities

Please start by running `git diff` to see all pending changes, then investigate why the Blackwell flash attention kernel hangs during execution.
