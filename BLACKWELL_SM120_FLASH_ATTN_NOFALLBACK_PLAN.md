### Goal
Fix the immediate `llama-server` crash on **RTX 5090 (Blackwell, sm_120 / cc 12.0)** when `--flash-attn on` by ensuring the **Blackwell FA kernel always has non-zero occupancy** (no fallback on cc 12.0), while keeping performance as the priority.

### Confirmed hardware constraints (from your poll)
- **Compute capability**: `12.0` (sm_120) on RTX 5090.
- **Registers per block limit**: `cudaDevAttrMaxRegistersPerBlock = 65536`.
- **Shared memory per block opt-in**: `cudaDevAttrMaxSharedMemoryPerBlockOptin = 101376`.
- You have a mixed system (cc 12.0 and cc 8.6). “No fallback” is interpreted as: **no fallback on cc 12.0**. Non-Blackwell GPUs must use non-Blackwell kernels.

### Root cause (why `max_blocks_per_sm == 0`)
The failing assert is in `ggml/src/ggml-cuda/fattn-common.cuh` where `cudaOccupancyMaxActiveBlocksPerMultiprocessor()` returns 0.

From your logs:
- `block_dim = 32, 9, 1` ⇒ **288 threads/block**
- `kernel regs = 255` ⇒ **255 regs/thread**
- Required registers per block \(= 255 × 288 = 73440\) which **exceeds 65536**, so occupancy must be **0** and the assert fires.

### Key constraint: Blackwell kernel currently hard-requires 9 warps
In `ggml/src/ggml-cuda/fattn-mma-f16.cuh`, `flash_attn_ext_f16_blackwell`:
- Has `__launch_bounds__(288, 1)`
- Contains logic that hardcodes **9 total warps** during Q loading (`nwarps_total = 9`)
- Uses a **producer (warp 0)** + **consumer warps** model with barriers; reducing warps is not safe without redesign.

### Decision (performance-first, cc 12.0 “no fallback”)
Do **not** use forced register caps (`__maxnregcount__`) and do **not** rely on runtime fallback on cc 12.0.
Instead, **refactor the Blackwell kernel** to reduce register pressure in the problematic specialization(s) until:
- `attrs.numRegs <= floor(65536 / 288) = 227` regs/thread (practically target **≤ 224** for safety),
- and occupancy becomes **≥ 1**.

### Implementation plan (targeted)
#### 1) Identify the register hot spots (Blackwell-only path)
File: `ggml/src/ggml-cuda/fattn-mma-f16.cuh`
- Focus on the Blackwell kernel path that corresponds to the failing configuration (reported as DKQ=128, DV=128, ncols=64).
- The Q-loading phase is a prime suspect because it currently runs multi-level loops across **all 9 warps**, which can inflate live ranges and per-thread temporaries.

#### 2) Refactor Q-loading to reduce live ranges and per-thread work
File: `ggml/src/ggml-cuda/fattn-mma-f16.cuh`
- Make Q loading a distinct, tightly-scoped phase that minimizes per-thread state.
- Prefer shifting Q loading responsibility toward the producer warp (or otherwise shrinking the amount of work each consumer warp does during Q-load) while keeping barrier semantics correct.
- Explicit objective: reduce the kernel’s `numRegs` in the failing specialization below the 227 threshold.

#### 3) Reduce unrolling / shrink temporaries in the Blackwell kernel path
File: `ggml/src/ggml-cuda/fattn-mma-f16.cuh`
- Where safe, reduce `#pragma unroll` in Blackwell-only code blocks that create large live ranges.
- Eliminate or restructure large per-thread temporaries (arrays, multiple accumulators held simultaneously).
- If needed, use shared memory for data that is currently replicated in registers across warps/threads.

#### 4) Validation
- Rebuild and run the same request that triggers the crash with `--flash-attn on`.
- Confirm on cc 12.0:
  - `cudaOccupancyMaxActiveBlocksPerMultiprocessor(...) >= 1` for the Blackwell kernel.
  - No assertion / no core dump.
  - Blackwell path remains selected on RTX 5090.

### Notes / boundaries
- This plan intentionally avoids adding a “fallback when occupancy==0” for cc 12.0 per your requirement.
- Because you have cc 8.6 GPUs in the box, non-Blackwell devices will still run the Ampere/legacy kernels as required by architecture.




