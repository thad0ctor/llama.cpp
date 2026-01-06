// attention_v5.cu - SM_120 (Blackwell) optimized Flash Attention kernel
// Uses BF16/F16 tensor cores with m16n8k16 MMA instructions
//
// This kernel is dispatched at runtime when cc >= 1200 (Blackwell)
// Supports both BF16 (native) and F16 (for Q8_0 and other quantized models)
//
// TWO KERNEL VARIANTS:
// 1. attention_v5_kernel: 3-stage software pipeline with memory prefetching
// 2. attention_v5_warp_specialized_kernel: Producer-Consumer warp specialization
//    - Warps 0-1 (Producer): Load K, compute Q@K^T, softmax, write P to shared
//    - Warps 2-3 (Consumer): Read P, load V, compute P@V, accumulate O
//    - Uses __syncthreads() for correct block-wide synchronization
//    - Achieves compute overlap in addition to memory prefetch overlap

#include "common.cuh"
#include "tma.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <float.h>
#include <type_traits>

// =============================================================================
// WARP SPECIALIZATION: Synchronization Notes
// =============================================================================
// The warp-specialized kernel uses __syncthreads() for producer-consumer
// synchronization. This ensures all 128 threads participate in each barrier,
// which is required for correct CUDA barrier semantics.
//
// Previous implementation used named barriers (bar.sync/bar.arrive) with
// ALL_THREADS=128, but only 64 threads (producers OR consumers) would arrive
// at each barrier call, causing undefined behavior due to warp divergence.
//
// The current implementation uses __syncthreads() which is safe because:
// 1. All threads in the block participate in every barrier
// 2. Double-buffering via (kv_id % 2) prevents data races on P buffer
// 3. __threadfence_block() ensures memory visibility before barriers

// Use WARP_SIZE from common.cuh
// Use CUDA_CHECK from common.cuh

// Ceiling division helper
__device__ __host__ constexpr
int attn_v5_cdiv(int a, int b) { return (a + b - 1) / b; }

// Rename cdiv usages to attn_v5_cdiv to avoid conflicts
#define cdiv attn_v5_cdiv

// NOTE: stride in bytes
template <int STRIDE>
__device__
uint32_t swizzle(uint32_t index) {
  // no need swizzling
  if constexpr (STRIDE == 16)
    return index;

  uint32_t row_idx = (index / STRIDE) % 8;
  uint32_t bits_to_xor = row_idx / max(64 / STRIDE, 1);
  return index ^ (bits_to_xor << 4);
}

template <int HEIGHT, int WIDTH, int TB_SIZE, typename T>
__device__ inline
void global_to_shared(uint32_t dst, const T *src, int src_stride, int tid, int max_rows = HEIGHT) {
  constexpr int num_elems = 16 / sizeof(T);  // 8 elements for 16-bit types
  constexpr int num_iters = HEIGHT * WIDTH / (TB_SIZE * num_elems);

  for (int iter = 0; iter < num_iters; iter++) {
    const int idx = (iter * TB_SIZE + tid) * num_elems;
    const int row = idx / WIDTH;
    const int col = idx % WIDTH;

    if (row < max_rows) {
        const uint32_t dst_addr = dst + (row * WIDTH + col) * sizeof(T);
        const T *src_addr = src + (row * src_stride + col);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst_addr), "l"(src_addr));
    }
  }
}

template <int HEIGHT, int WIDTH, int TB_SIZE, typename T>
__device__ inline
void global_to_shared_swizzle(uint32_t dst, const T *src, int src_stride, int tid, int max_rows = HEIGHT) {
  constexpr int num_elems = 16 / sizeof(T);  // 8 elements for 16-bit types
  constexpr int num_iters = HEIGHT * WIDTH / (TB_SIZE * num_elems);

  for (int iter = 0; iter < num_iters; iter++) {
    const int idx = (iter * TB_SIZE + tid) * num_elems;
    const int row = idx / WIDTH;
    const int col = idx % WIDTH;

    const uint32_t dst_addr = swizzle<WIDTH * sizeof(T)>(dst + (row * WIDTH + col) * sizeof(T));
    if (row < max_rows) {
        const T *src_addr = src + (row * src_stride + col);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst_addr), "l"(src_addr));
    } else {
        // Zero-initialize rows beyond max_rows to prevent garbage in MMA computations
        // Write 16 bytes of zeros (8 x 16-bit elements)
        T * smem_ptr = reinterpret_cast<T*>(__cvta_shared_to_generic(dst_addr));
        #pragma unroll
        for (int i = 0; i < num_elems; i++) {
            smem_ptr[i] = T(0);
        }
    }
  }
}

__device__ inline
void ldmatrix_x2(uint32_t regs[2], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
              : "=r"(regs[0]), "=r"(regs[1])
              : "r"(addr));
}

__device__ inline
void ldmatrix_x4(uint32_t regs[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
              : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
              : "r"(addr));
}

__device__ inline
void ldmatrix_x2_trans(uint32_t regs[2], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.b16 {%0, %1}, [%2];"
              : "=r"(regs[0]), "=r"(regs[1])
              : "r"(addr));
}

__device__ inline
void ldmatrix_x4_trans(uint32_t regs[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.b16 {%0, %1, %2, %3}, [%4];"
              : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
              : "r"(addr));
}

// BF16 MMA m16n8k16 instruction
__device__ inline
void mma_m16n8k16_bf16(uint32_t A[4], uint32_t B[2], float D[4]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
              "{%0, %1, %2, %3}, "
              "{%4, %5, %6, %7}, "
              "{%8, %9}, "
              "{%10, %11, %12, %13};"
              : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
              : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                "r"(B[0]), "r"(B[1]),
                "f"(D[0]), "f"(D[1]), "f"(D[2]), "f"(D[3]));
}

// F16 MMA m16n8k16 instruction
__device__ inline
void mma_m16n8k16_f16(uint32_t A[4], uint32_t B[2], float D[4]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
              "{%0, %1, %2, %3}, "
              "{%4, %5, %6, %7}, "
              "{%8, %9}, "
              "{%10, %11, %12, %13};"
              : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
              : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                "r"(B[0]), "r"(B[1]),
                "f"(D[0]), "f"(D[1]), "f"(D[2]), "f"(D[3]));
}

// Type-dispatched MMA instruction
template<typename T>
__device__ inline void mma_m16n8k16(uint32_t A[4], uint32_t B[2], float D[4]) {
  if constexpr (std::is_same_v<T, nv_bfloat16>) {
    mma_m16n8k16_bf16(A, B, D);
  } else {
    mma_m16n8k16_f16(A, B, D);
  }
}

// Type-dispatched float2 to packed half/bfloat16 conversion
template<typename T>
__device__ inline auto float2_to_vec2(float a, float b) {
  if constexpr (std::is_same_v<T, nv_bfloat16>) {
    return __float22bfloat162_rn({a, b});
  } else {
    return __float22half2_rn(make_float2(a, b));
  }
}

// Type alias for packed type (half2 or bfloat162)
template<typename T>
using packed_t = std::conditional_t<std::is_same_v<T, nv_bfloat16>, nv_bfloat162, half2>;

template <typename T, typename... Args>
void launch_kernel(
  T *kernel,
  int num_blocks,
  int block_size,
  int smem_size,
  Args... args) {
  if (smem_size > 48'000)
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  kernel<<<num_blocks, block_size, smem_size>>>(args...);
  CUDA_CHECK(cudaGetLastError());
}

// ALiBi slope calculation (same as fattn-common.cuh)
__device__ __forceinline__ float get_alibi_slope(
    float max_bias, int h, uint32_t n_head_log2, float m0, float m1) {
  if (max_bias <= 0.0f) {
    return 1.0f;  // No ALiBi, slope = 1.0 means mask is added directly
  }
  // ALiBi slope calculation
  const float base = h < int(n_head_log2) ? m0 : m1;
  const int   exp  = h < int(n_head_log2) ? h + 1 : 2*(h - n_head_log2) + 1;
  return powf(base, float(exp));
}

/*
 * TENSOR LAYOUT DOCUMENTATION
 * ===========================
 *
 * TENSOR FLOW:
 *   1. KV cache returns K,V as [D, heads, seq, batch] via get_k()/get_v()
 *   2. llama-graph.cpp applies permute(0,2,1,3) which swaps dims 1<->2
 *   3. ggml_cont() makes contiguous in the NEW order
 *   4. Result: INPUT tensors are [D, seq, heads, batch]
 *
 * INPUT TENSORS (from llama-graph.cpp after permute + ggml_cont):
 *   Q, K, V: shape = [D, seq, n_heads, batch] - CONTIGUOUS
 *   - ne[0] = D (head dimension, 128)
 *   - ne[1] = seq_len (seq_q for Q, seq_kv for K/V)
 *   - ne[2] = n_heads (Q) or n_heads_kv (K, V for GQA)
 *   - ne[3] = batch
 *   - nb[1] = D * sizeof(T)        (stride to next seq position)
 *   - nb[2] = D * seq * sizeof(T)  (stride to next head)
 *
 * OUTPUT TENSOR (created by ggml_flash_attn_ext with ne = {v.ne[0], q.ne[2], q.ne[1], q.ne[3]}):
 *   O: shape = [D, n_heads, seq_q, batch] - **DIFFERENT FROM INPUT!**
 *   - ne[0] = D
 *   - ne[1] = n_heads   <-- NOTE: swapped from input!
 *   - ne[2] = seq_q     <-- NOTE: swapped from input!
 *   - ne[3] = batch
 *   - nb[1] = D * sizeof(To)           (stride to next HEAD)
 *   - nb[2] = D * n_heads * sizeof(To) (stride to next seq position)
 *
 * STRIDE PARAMETER SEMANTICS (kernel always uses seq/head semantics):
 *   stride_X_row   = elements to advance by 1 in SEQUENCE dimension
 *   stride_X_head  = elements to advance by 1 in HEADS dimension
 *   stride_X_batch = elements to advance by 1 in BATCH dimension
 *
 * For INPUT [D, seq, heads, batch]:  stride_row = nb[1], stride_head = nb[2]
 * For OUTPUT [D, heads, seq, batch]: stride_row = nb[2], stride_head = nb[1] (SWAPPED!)
 */
template<int BLOCK_Q, int BLOCK_KV, int DIM, int NUM_WARPS, typename T, typename To>
__launch_bounds__(NUM_WARPS * WARP_SIZE)
__global__
void attention_v5_kernel(
  const T *Q,           // [D, seq_q, n_heads, batch] - contiguous after permute+cont
  const T *K,           // [D, seq_kv, n_heads_kv, batch]
  const T *V,           // [D, seq_kv, n_heads_kv, batch]
  const half *mask,     // [seq_kv, seq_q, 1, batch] - F16, nullptr if no mask
  To *O,                // [D, seq_q, n_heads, batch] - same layout as inputs
  int n_heads,          // Q->ne[2] - number of query heads
  int n_heads_kv,       // K->ne[2] - number of KV heads (for GQA)
  int n_batch,          // Q->ne[3] - batch size
  int len_q,            // Q->ne[1] - query sequence length
  int len_kv,           // K->ne[1] - KV sequence length
  float softmax_scale,  // from op_params[0]
  float max_bias,       // ALiBi max_bias from op_params[1]
  float m0,             // ALiBi base for h < n_head_log2
  float m1,             // ALiBi base for h >= n_head_log2
  uint32_t n_head_log2, // for ALiBi slope calculation
  float logit_softcap,  // from op_params[2], 0 if disabled
  // Input strides (in elements) - for [D, seq, heads, batch] layout
  int stride_Q_row,     // Q->nb[1] / sizeof(T) - stride to next seq position
  int stride_Q_head,    // Q->nb[2] / sizeof(T) - stride to next head
  int stride_Q_batch,   // Q->nb[3] / sizeof(T)
  int stride_K_row,     // K->nb[1] / sizeof(T)
  int stride_K_head,    // K->nb[2] / sizeof(T)
  int stride_K_batch,   // K->nb[3] / sizeof(T)
  int stride_V_row,     // V->nb[1] / sizeof(T)
  int stride_V_head,    // V->nb[2] / sizeof(T)
  int stride_V_batch,   // V->nb[3] / sizeof(T)
  // Output strides (in elements) - same [D, seq, heads, batch] layout
  int stride_O_row,     // dst->nb[1] / sizeof(To) - stride to next seq position
  int stride_O_head,    // dst->nb[2] / sizeof(To) - stride to next head
  int stride_O_batch,   // dst->nb[3] / sizeof(To)
  // Mask strides (in elements, F16)
  int stride_mask_row,  // mask->nb[1] / sizeof(half) - stride between q positions
  int stride_mask_batch,// mask->nb[3] / sizeof(half) - stride between batches
  // ===========================================================================
  // Split-K Parallelism Parameters
  // ===========================================================================
  // Split-K divides the KV sequence across multiple blocks for better GPU occupancy
  // With split_k=8 and 32 heads: 256 blocks instead of 32 blocks
  int split_k,          // Number of KV splits per head (1 = no split, original behavior)
  float * partial_O,    // Workspace for partial outputs [total_tiles * split_k * BLOCK_Q * DIM]
  float2 * partial_meta)// Workspace for partial metadata [total_tiles * split_k * BLOCK_Q] (rowmax, rowsumexp)
{
  constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;

  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;

  // ===========================================================================
  // Block Decomposition with Split-K
  // ===========================================================================
  // Block indexing: [batch, head, q_block, kv_split]
  // Total blocks = n_batch * n_heads * num_q_blocks * split_k
  const int num_q_blocks = cdiv(len_q, BLOCK_Q);
  const int blocks_per_head = num_q_blocks * split_k;
  const int blocks_per_batch = n_heads * blocks_per_head;

  const int batch_id = bid / blocks_per_batch;
  const int bid_in_batch = bid % blocks_per_batch;
  const int head_id = bid_in_batch / blocks_per_head;
  const int bid_in_head = bid_in_batch % blocks_per_head;
  const int q_block_id = bid_in_head / split_k;
  const int kv_split_id = bid_in_head % split_k;

  // Calculate KV range for this split
  const int total_kv_iters = cdiv(len_kv, BLOCK_KV);
  const int kv_iters_per_split = cdiv(total_kv_iters, split_k);
  const int kv_iter_start = kv_split_id * kv_iters_per_split;
  const int kv_iter_end = min(kv_iter_start + kv_iters_per_split, total_kv_iters);

  // Early exit if this split has no work (when split_k > total_kv_iters)
  if (kv_iter_start >= total_kv_iters) {
    // Write zero partial results for reduction
    if (split_k > 1) {
      const int tile_idx = batch_id * (n_heads * num_q_blocks) + head_id * num_q_blocks + q_block_id;
      const int split_offset = tile_idx * split_k + kv_split_id;
      float2 * meta_ptr = partial_meta + split_offset * BLOCK_Q;
      for (int i = tid; i < BLOCK_Q; i += TB_SIZE) {
        meta_ptr[i] = make_float2(-FLT_MAX, 0.0f);  // rowmax=-inf, rowsumexp=0
      }
    }
    return;
  }

  // GQA: map query head to KV head
  const int gqa_ratio = n_heads / n_heads_kv;
  const int kv_head_id = head_id / gqa_ratio;

  // Compute base pointers using proper strides (supports non-contiguous layout)
  Q += batch_id * stride_Q_batch + head_id * stride_Q_head + q_block_id * BLOCK_Q * stride_Q_row;
  K += batch_id * stride_K_batch + kv_head_id * stride_K_head;
  V += batch_id * stride_V_batch + kv_head_id * stride_V_head;
  O += batch_id * stride_O_batch + head_id * stride_O_head + q_block_id * BLOCK_Q * stride_O_row;

  // Mask base pointer for this batch and q_block (mask is shared across heads)
  const half *mask_block = mask ? (mask + batch_id * stride_mask_batch + q_block_id * BLOCK_Q * stride_mask_row) : nullptr;

  // ALiBi slope for this head
  const float alibi_slope = get_alibi_slope(max_bias, head_id, n_head_log2, m0, m1);

  // Starting q position for bounds checking
  const int q_start = q_block_id * BLOCK_Q;

  // Software Pipelining: Triple-buffer K, Double-buffer V for 3-stage pipeline
  // Shared memory layout:
  //   - K triple-buffer: 3 * BLOCK_KV * DIM * sizeof(T) = 3 * 64 * 128 * 2 = 48KB
  //   - V double-buffer: 2 * BLOCK_KV * DIM * sizeof(T) = 2 * 64 * 128 * 2 = 32KB
  //   - Mask F32 buffer: BLOCK_Q * BLOCK_KV * sizeof(float) = 64 * 64 * 4 = 16KB
  //   - Total: 48KB + 32KB + 16KB = 96KB (within 99KB SM_120 limit)
  // Note: T is either half or nv_bfloat16, both are 16 bits
  extern __shared__ char smem_raw[];
  T * smem = reinterpret_cast<T *>(smem_raw);
  const uint32_t Q_smem = __cvta_generic_to_shared(smem);
  const uint32_t K_smem = Q_smem;  // TRIPLE buffer for K (overlaps with Q_smem initially)
  constexpr int K_BUF_SIZE = BLOCK_KV * DIM * sizeof(T);  // 16KB per K buffer
  const uint32_t V_smem = K_smem + 3 * K_BUF_SIZE;  // After K triple-buffer
  constexpr int V_BUF_SIZE = BLOCK_KV * DIM * sizeof(T);  // 16KB per V buffer
  // Pre-converted mask buffer: BLOCK_Q * BLOCK_KV floats = 64 * 64 * 4 = 16KB
  const uint32_t mask_f32_smem = V_smem + 2 * V_BUF_SIZE;  // After V double-buffer

  // FA2: shard BLOCK_Q among all warps
  // replicate K and V on all warps
  constexpr int WARP_Q = BLOCK_Q / NUM_WARPS;

  // mma.m16n8k16
  constexpr int MMA_M = 16;
  constexpr int MMA_N = 8;
  constexpr int MMA_K = 16;

  // set up registers
  uint32_t Q_rmem[WARP_Q / MMA_M][DIM / MMA_K][4];
  uint32_t K_rmem[BLOCK_KV / MMA_N][DIM / MMA_K][2];

  // let compiler decide register reuse?
  uint32_t P_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_K][4];
  uint32_t V_rmem[BLOCK_KV / MMA_K][DIM / MMA_N][2];

  // rescale O_rmem once we obtain new rowmax, then accumulate to O_rmem for P @ V
  float O_rmem[WARP_Q / MMA_M][DIM / MMA_N][4] = {};

  // pre-compute address and swizzling for ldmatrix
  // Note: sizeof(T) == 2 for both half and nv_bfloat16
  uint32_t Q_smem_thread, K_smem_thread, V_smem_thread;
  {
    // A tile
    const int row_off = warp_id * WARP_Q + (lane_id % 16);
    const int col_off = lane_id / 16 * 8;
    Q_smem_thread = swizzle<DIM * sizeof(T)>(Q_smem + (row_off * DIM + col_off) * sizeof(T));
  }
  {
    // B tile
    const int row_off = lane_id % 8;
    const int col_off = lane_id / 8 * 8;
    K_smem_thread = swizzle<DIM * sizeof(T)>(K_smem + (row_off * DIM + col_off) * sizeof(T));
  }
  {
    // B tile trans
    const int row_off = lane_id % 16;
    const int col_off = lane_id / 16 * 8;
    V_smem_thread = swizzle<DIM * sizeof(T)>(V_smem + (row_off * DIM + col_off) * sizeof(T));
  }

  float rowmax[WARP_Q / MMA_M][2];
  float rowsumexp[WARP_Q / MMA_M][2] = {};

  for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
    rowmax[mma_id_q][0] = -FLT_MAX;
    rowmax[mma_id_q][1] = -FLT_MAX;
  }

  // load Q [BLOCK_Q, DIM] - use stride_Q_row for non-contiguous support
  global_to_shared_swizzle<BLOCK_Q, DIM, TB_SIZE, T>(Q_smem, Q, stride_Q_row, tid, len_q - q_block_id * BLOCK_Q);
  asm volatile("cp.async.commit_group;");
  asm volatile("cp.async.wait_all;");
  __syncthreads();

  // shared -> registers (shared memory is always contiguous with stride=DIM)
  for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
    for (int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d++) {
      uint32_t addr = Q_smem_thread;
      addr += mma_id_q * MMA_M * DIM * sizeof(T);  // row
      addr ^= mma_id_d * MMA_K * sizeof(T);  // col
      ldmatrix_x4(Q_rmem[mma_id_q][mma_id_d], addr);
    }
  // we need a syncthreads() here so that we don't load K global->shared
  // before finishing loading Q shared->reg
  __syncthreads();

  // ===========================================================================
  // Split-K: Use range [kv_iter_start, kv_iter_end) for this split
  // ===========================================================================
  const int num_kv_iter = kv_iter_end - kv_iter_start;

  // Track K and V pointers separately for pipelining
  // Advance pointers to the starting KV position for this split
  const T * K_ptr = K + kv_iter_start * BLOCK_KV * stride_K_row;
  const T * V_ptr = V + kv_iter_start * BLOCK_KV * stride_V_row;

  // Load K to a specific buffer slot (0, 1, or 2 for triple-buffer)
  auto load_K = [&](int kv_id, int buf_idx) {
    if (kv_id < num_kv_iter) {
      // Triple buffer for K - use stride_K_row for non-contiguous support
      const uint32_t dst = K_smem + (buf_idx % 3) * K_BUF_SIZE;
      global_to_shared_swizzle<BLOCK_KV, DIM, TB_SIZE, T>(dst, K_ptr + kv_id * BLOCK_KV * stride_K_row, stride_K_row, tid, len_kv - kv_id * BLOCK_KV);
    }
    asm volatile("cp.async.commit_group;");
  };

  // Load V to a specific buffer slot (0 or 1 for double-buffer)
  auto load_V = [&](int kv_id, int buf_idx) {
    if (kv_id < num_kv_iter) {
      // Double buffer for V - use stride_V_row for non-contiguous support
      const uint32_t dst = V_smem + (buf_idx % 2) * V_BUF_SIZE;
      global_to_shared_swizzle<BLOCK_KV, DIM, TB_SIZE, T>(dst, V_ptr + kv_id * BLOCK_KV * stride_V_row, stride_V_row, tid, len_kv - kv_id * BLOCK_KV);
    }
    asm volatile("cp.async.commit_group;");
  };

  // ===========================================================================
  // 3-STAGE SOFTWARE PIPELINE: Prologue
  // ===========================================================================
  // Pipeline structure:
  //   - K triple-buffer: slots 0, 1, 2
  //   - V double-buffer: slots 0, 1
  //   - Load K[i+2], V[i+1] while computing with K[i], V[i]
  //
  // Commit group management:
  //   Group 0: K[0] (oldest, consumed first)
  //   Group 1: K[1] (next K)
  //   Group 2: V[0] (first V)
  //   ... pattern continues in steady state

  // Buffer index tracking
  int k_load_buf = 0;   // Next K buffer to load into (0, 1, 2, 0, 1, 2, ...)
  int k_use_buf = 0;    // Current K buffer to read from
  int v_load_buf = 0;   // Next V buffer to load into (0, 1, 0, 1, ...)
  int v_use_buf = 0;    // Current V buffer to read from

  // Prologue: Fill the pipeline with K[0], K[1], V[0]
  load_K(0, k_load_buf++);  // K[0] -> buffer 0, commit group

  if (num_kv_iter > 1) {
    load_K(1, k_load_buf++);  // K[1] -> buffer 1, commit group
  } else {
    // Dummy commit to maintain group count
    asm volatile("cp.async.commit_group;");
  }

  load_V(0, v_load_buf++);  // V[0] -> buffer 0, commit group

  // Wait for K[0] to be ready (2 groups in flight: K[1], V[0])
  asm volatile("cp.async.wait_group 2;");
  __syncthreads();

  // ===========================================================================
  // 3-STAGE SOFTWARE PIPELINE: Main Loop
  // ===========================================================================
  for (int kv_id = 0; kv_id < num_kv_iter; kv_id++) {
    float S_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_N][4] = {};

    // =========================================================================
    // STAGE A: Prefetch - Issue loads for future iterations
    // =========================================================================

    // Load K[kv_id+2] if valid (2 iterations ahead)
    if (kv_id + 2 < num_kv_iter) {
      load_K(kv_id + 2, k_load_buf % 3);
      k_load_buf++;
    } else {
      // Dummy commit to maintain group count
      asm volatile("cp.async.commit_group;");
    }

    // Load V[kv_id+1] if valid (1 iteration ahead)
    // Need syncthreads to protect V buffer from previous iteration's use
    __syncthreads();
    if (kv_id + 1 < num_kv_iter) {
      load_V(kv_id + 1, v_load_buf % 2);
      v_load_buf++;
    } else {
      asm volatile("cp.async.commit_group;");
    }

    // =========================================================================
    // STAGE B: Compute S = Q @ K^T
    // =========================================================================

    // Wait for K[kv_id] - should have up to 2 groups in flight (K[kv_id+1], K[kv_id+2] or V)
    asm volatile("cp.async.wait_group 2;");
    __syncthreads();

    // K[kv_id] smem -> registers (using k_use_buf)
    for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)
      for (int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d += 2) {
        uint32_t addr = K_smem_thread + (k_use_buf % 3) * K_BUF_SIZE;
        addr += mma_id_kv * MMA_N * DIM * sizeof(T);  // row
        addr ^= mma_id_d * MMA_K * sizeof(T);  // col
        ldmatrix_x4(K_rmem[mma_id_kv][mma_id_d], addr);
      }
    k_use_buf++;  // Advance to next K buffer for next iteration

    // Pre-convert mask from F16 to F32 in shared memory (once per KV iteration)
    // This eliminates ~12,288 redundant __half2float conversions per iteration
    // Uses __half22float2 to process pairs of F16 values for 2x conversion throughput
    float * mask_f32 = reinterpret_cast<float *>(__cvta_shared_to_generic(mask_f32_smem));
    if (mask_block) {
      // Split-K: use global KV position for mask access
      const int kv_base = (kv_iter_start + kv_id) * BLOCK_KV;
      // Process pairs of mask values using __half22float2 for better throughput
      // Each thread processes 2 consecutive KV positions at once
      constexpr int num_pairs = (BLOCK_Q * BLOCK_KV) / 2;  // 2048 pairs for 64x64
      for (int pair_idx = tid; pair_idx < num_pairs; pair_idx += TB_SIZE) {
        const int i = pair_idx * 2;  // Base index for this pair
        const int q_idx = i / BLOCK_KV;
        const int kv_idx = i % BLOCK_KV;
        const int q_pos = q_start + q_idx;
        const int kv_pos0 = kv_base + kv_idx;
        const int kv_pos1 = kv_base + kv_idx + 1;

        // Both elements in a pair are from the same q_idx row (consecutive in KV dim)
        // since kv_idx is always even (i = pair_idx * 2, and BLOCK_KV is 64)
        if (q_pos < len_q && kv_pos1 < len_kv) {
          // Both positions valid - use paired conversion
          const half * src = &mask_block[q_idx * stride_mask_row + kv_pos0];
          const half2 mask_pair = *reinterpret_cast<const half2 *>(src);
          const float2 vals = __half22float2(mask_pair);
          mask_f32[i] = vals.x;
          mask_f32[i + 1] = vals.y;
        } else if (q_pos < len_q && kv_pos0 < len_kv) {
          // Only first position valid
          mask_f32[i] = __half2float(mask_block[q_idx * stride_mask_row + kv_pos0]);
          mask_f32[i + 1] = -FLT_MAX;
        } else {
          // Both positions invalid
          mask_f32[i] = -FLT_MAX;
          mask_f32[i + 1] = -FLT_MAX;
        }
      }
      __syncthreads();
    }

    // MMA S = Q @ K.T [BLOCK_Q, BLOCK_KV]
    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
      for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)
        for (int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d++)
          mma_m16n8k16<T>(Q_rmem[mma_id_q][mma_id_d],
                          K_rmem[mma_id_kv][mma_id_d],
                          S_rmem[mma_id_q][mma_id_kv]);

    // NOTE: K prefetch for (kv_id + 1) now happens earlier (after K->registers load)
    // to better overlap memory latency with compute

    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
      // Apply softmax scale, logit softcap, and mask to attention scores
      // MMA m16n8k16 output layout:
      //   - Each thread holds 4 values: reg[0], reg[1] are in row (lane_id/4), columns 2*(lane_id%4) and 2*(lane_id%4)+1
      //   - reg[2], reg[3] are in row (lane_id/4 + 8), same columns
      for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
        float *regs = S_rmem[mma_id_q][mma_id_kv];

        // Calculate actual q and kv positions for this thread's registers
        // Split-K: use global KV position
        const int q_row_base = warp_id * WARP_Q + mma_id_q * MMA_M + (lane_id / 4);
        const int kv_col_base = (kv_iter_start + kv_id) * BLOCK_KV + mma_id_kv * MMA_N + (lane_id % 4) * 2;

        // Process all 4 registers
        #pragma unroll
        for (int r = 0; r < 4; r++) {
          // Determine actual q_row and kv_col for this register
          const int q_row = q_row_base + (r >= 2 ? 8 : 0);  // reg 2,3 are +8 rows
          const int kv_col = kv_col_base + (r % 2);          // reg 1,3 are +1 column

          // Apply softmax scale
          float score = regs[r] * softmax_scale;

          // Apply logit softcap if enabled: score = softcap * tanh(score / softcap)
          // Note: scale was already divided by softcap in wrapper if softcap != 0
          if (logit_softcap != 0.0f) {
            score = logit_softcap * tanhf(score);
          }

          // Apply mask with ALiBi slope
          // Mask values were pre-converted from F16 to F32 in shared memory
          if (kv_col >= len_kv) {
             score = -FLT_MAX;
          } else if (mask_block && (q_start + q_row) < len_q) {
            // Read from pre-converted F32 mask in shared memory
            // kv_col is absolute position, need local offset within this KV block
            // Must account for kv_iter_start in Split-K mode
            const int kv_local = kv_col - (kv_iter_start + kv_id) * BLOCK_KV;
            const float mask_val = mask_f32[q_row * BLOCK_KV + kv_local];
            score += alibi_slope * mask_val;
          }

          regs[r] = score;
        }
      }

      // rowmax
      float this_rowmax[2];
      for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
        float *regs = S_rmem[mma_id_q][mma_id_kv];
        if (mma_id_kv == 0) {
          this_rowmax[0] = max(regs[0], regs[1]);  // c0 and c1
          this_rowmax[1] = max(regs[2], regs[3]);  // c2 and c3
        } else {
          this_rowmax[0] = max(this_rowmax[0], max(regs[0], regs[1]));  // c0 and c1
          this_rowmax[1] = max(this_rowmax[1], max(regs[2], regs[3]));  // c2 and c3
        }
      }

      // butterfly reduction within 4 threads
      this_rowmax[0] = max(this_rowmax[0], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[0], 1));
      this_rowmax[0] = max(this_rowmax[0], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[0], 2));
      this_rowmax[1] = max(this_rowmax[1], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[1], 1));
      this_rowmax[1] = max(this_rowmax[1], __shfl_xor_sync(0xFFFF'FFFF, this_rowmax[1], 2));

      // new rowmax
      this_rowmax[0] = max(this_rowmax[0], rowmax[mma_id_q][0]);
      this_rowmax[1] = max(this_rowmax[1], rowmax[mma_id_q][1]);

      // rescale for previous O
      float rescale[2];
      rescale[0] = __expf(rowmax[mma_id_q][0] - this_rowmax[0]);
      rescale[1] = __expf(rowmax[mma_id_q][1] - this_rowmax[1]);
      for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++) {
        O_rmem[mma_id_q][mma_id_d][0] *= rescale[0];
        O_rmem[mma_id_q][mma_id_d][1] *= rescale[0];
        O_rmem[mma_id_q][mma_id_d][2] *= rescale[1];
        O_rmem[mma_id_q][mma_id_d][3] *= rescale[1];
      }

      // save new rowmax
      rowmax[mma_id_q][0] = this_rowmax[0];
      rowmax[mma_id_q][1] = this_rowmax[1];

      // rowsumexp
      float this_rowsumexp[2];
      for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
        float *regs = S_rmem[mma_id_q][mma_id_kv];
        regs[0] = __expf(regs[0] - rowmax[mma_id_q][0]);  // c0
        regs[1] = __expf(regs[1] - rowmax[mma_id_q][0]);  // c1
        regs[2] = __expf(regs[2] - rowmax[mma_id_q][1]);  // c2
        regs[3] = __expf(regs[3] - rowmax[mma_id_q][1]);  // c3

        if (mma_id_kv == 0) {
          this_rowsumexp[0] = regs[0] + regs[1];
          this_rowsumexp[1] = regs[2] + regs[3];
        } else {
          this_rowsumexp[0] += regs[0] + regs[1];
          this_rowsumexp[1] += regs[2] + regs[3];
        }

        // pack to P registers for next MMA
        // we need to change from m16n8 to m16k16
        packed_t<T> *this_P_rmem = reinterpret_cast<packed_t<T> *>(P_rmem[mma_id_q][mma_id_kv / 2]);
        this_P_rmem[(mma_id_kv % 2) * 2]     = float2_to_vec2<T>(regs[0], regs[1]);
        this_P_rmem[(mma_id_kv % 2) * 2 + 1] = float2_to_vec2<T>(regs[2], regs[3]);
      }

      // butterfly reduction within 4 threads
      this_rowsumexp[0] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[0], 1);
      this_rowsumexp[0] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[0], 2);
      this_rowsumexp[1] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[1], 1);
      this_rowsumexp[1] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[1], 2);

      // accumulate to total rowsumexp
      rowsumexp[mma_id_q][0] = rowsumexp[mma_id_q][0] * rescale[0] + this_rowsumexp[0];
      rowsumexp[mma_id_q][1] = rowsumexp[mma_id_q][1] * rescale[1] + this_rowsumexp[1];
    }

    // =========================================================================
    // STAGE C: Compute O += P @ V
    // =========================================================================

    // Wait for V[kv_id] - should have 1 group in flight (V[kv_id+1] or next K)
    asm volatile("cp.async.wait_group 1;");
    __syncthreads();

    // V[kv_id] smem -> registers (using v_use_buf for double buffer)
    for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++)
      for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d += 2) {
        uint32_t addr = V_smem_thread + (v_use_buf % 2) * V_BUF_SIZE;
        addr += mma_id_kv * MMA_K * DIM * sizeof(T);  // row
        addr ^= mma_id_d * MMA_N * sizeof(T);  // col
        ldmatrix_x4_trans(V_rmem[mma_id_kv][mma_id_d], addr);
      }
    v_use_buf++;  // Advance to next V buffer for next iteration

    // MMA O += P @ V [BLOCK_Q, DIM]
    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
      for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++)
        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++)
          mma_m16n8k16<T>(P_rmem[mma_id_q][mma_id_kv],
                          V_rmem[mma_id_kv][mma_id_d],
                          O_rmem[mma_id_q][mma_id_d]);
  }

  // ===========================================================================
  // 3-STAGE SOFTWARE PIPELINE: Epilogue
  // ===========================================================================
  // Ensure all async operations are complete before writing output
  asm volatile("cp.async.wait_all;");
  __syncthreads();

  // ===========================================================================
  // Split-K: Conditional Output (Partial vs Final)
  // ===========================================================================
  if (split_k == 1) {
    // Original behavior: write normalized output directly to O
    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
      for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++) {
        const int row = warp_id * WARP_Q + mma_id_q * MMA_M + (lane_id / 4);
        const int col = mma_id_d * MMA_N + (lane_id % 4) * 2;

        // divide by softmax denominator (use reciprocal multiply for performance)
        float *regs = O_rmem[mma_id_q][mma_id_d];
        const float inv_rowsumexp0 = 1.0f / rowsumexp[mma_id_q][0];
        const float inv_rowsumexp1 = 1.0f / rowsumexp[mma_id_q][1];
        regs[0] *= inv_rowsumexp0;
        regs[1] *= inv_rowsumexp0;
        regs[2] *= inv_rowsumexp1;
        regs[3] *= inv_rowsumexp1;

        // Write output with type-aware conversion (To can be float, half, or nv_bfloat16)
        // Boundary check: ensure we don't write past len_q
        if ((q_block_id * BLOCK_Q + row) < len_q) {
            if constexpr (std::is_same_v<To, float>) {
                reinterpret_cast<float2 *>(O + (row + 0) * stride_O_row + col)[0] = make_float2(regs[0], regs[1]);
            } else if constexpr (std::is_same_v<To, half>) {
                reinterpret_cast<half2 *>(O + (row + 0) * stride_O_row + col)[0] = __float22half2_rn(make_float2(regs[0], regs[1]));
            } else if constexpr (std::is_same_v<To, nv_bfloat16>) {
                reinterpret_cast<nv_bfloat162 *>(O + (row + 0) * stride_O_row + col)[0] = __float22bfloat162_rn(make_float2(regs[0], regs[1]));
            }
        }
        if ((q_block_id * BLOCK_Q + row + 8) < len_q) {
            if constexpr (std::is_same_v<To, float>) {
                reinterpret_cast<float2 *>(O + (row + 8) * stride_O_row + col)[0] = make_float2(regs[2], regs[3]);
            } else if constexpr (std::is_same_v<To, half>) {
                reinterpret_cast<half2 *>(O + (row + 8) * stride_O_row + col)[0] = __float22half2_rn(make_float2(regs[2], regs[3]));
            } else if constexpr (std::is_same_v<To, nv_bfloat16>) {
                reinterpret_cast<nv_bfloat162 *>(O + (row + 8) * stride_O_row + col)[0] = __float22bfloat162_rn(make_float2(regs[2], regs[3]));
            }
        }
      }
  } else {
    // Split-K mode: write UNNORMALIZED partial results to workspace
    // Reduction kernel will combine partial results and normalize
    const int tile_idx = batch_id * (n_heads * num_q_blocks) + head_id * num_q_blocks + q_block_id;
    const int split_offset = tile_idx * split_k + kv_split_id;

    // Write partial_O (UNNORMALIZED - do NOT divide by rowsumexp)
    float * partial_O_ptr = partial_O + split_offset * BLOCK_Q * DIM;

    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
      for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++) {
        const int row = warp_id * WARP_Q + mma_id_q * MMA_M + (lane_id / 4);
        const int col = mma_id_d * MMA_N + (lane_id % 4) * 2;

        float *regs = O_rmem[mma_id_q][mma_id_d];

        // Write unnormalized partial O (no division by rowsumexp)
        if ((q_block_id * BLOCK_Q + row) < len_q) {
          reinterpret_cast<float2 *>(partial_O_ptr + (row + 0) * DIM + col)[0] = make_float2(regs[0], regs[1]);
        }
        if ((q_block_id * BLOCK_Q + row + 8) < len_q) {
          reinterpret_cast<float2 *>(partial_O_ptr + (row + 8) * DIM + col)[0] = make_float2(regs[2], regs[3]);
        }
      }

    // Write partial metadata (rowmax, rowsumexp) - one per query row
    // Only one thread per row should write metadata (avoid duplicate writes)
    float2 * meta_ptr = partial_meta + split_offset * BLOCK_Q;

    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
      const int row0 = warp_id * WARP_Q + mma_id_q * MMA_M + (lane_id / 4);
      const int row8 = row0 + 8;

      // Only lane 0 in each group of 4 writes metadata (avoid redundant writes)
      if ((lane_id % 4) == 0) {
        if ((q_block_id * BLOCK_Q + row0) < len_q) {
          meta_ptr[row0] = make_float2(rowmax[mma_id_q][0], rowsumexp[mma_id_q][0]);
        }
        if ((q_block_id * BLOCK_Q + row8) < len_q) {
          meta_ptr[row8] = make_float2(rowmax[mma_id_q][1], rowsumexp[mma_id_q][1]);
        }
      }
    }
  }
}

// =============================================================================
// TWO-PHASE KERNEL SPLIT: Phase 1 - Q@K^T + Softmax
// =============================================================================
// This kernel computes the attention scores (Q @ K^T), applies mask and softmax,
// and outputs the P matrix (attention weights) along with numerical stability metadata.
//
// REGISTER BUDGET (~212 registers, fits within 255 limit):
//   - Q_rmem: (WARP_Q/16) * (DIM/16) * 4 = 1 * 8 * 4 = 32 registers
//   - K_rmem: (BLOCK_KV/8) * (DIM/16) * 2 = 8 * 8 * 2 = 128 registers
//   - S_rmem: (WARP_Q/16) * (BLOCK_KV/8) * 4 = 1 * 8 * 4 = 32 registers
//   - misc (rowmax, rowsumexp, addresses, etc.): ~20 registers
//   Total: ~212 registers
//
// NO V_rmem or O_rmem in this kernel - those are in Phase 2
// =============================================================================
template<int BLOCK_Q, int BLOCK_KV, int DIM, int NUM_WARPS, typename T>
__launch_bounds__(NUM_WARPS * WARP_SIZE, 2)
__global__
void attention_v5_phase1_qk_softmax(
    const T * __restrict__ Q,           // [D, seq_q, n_heads, batch]
    const T * __restrict__ K,           // [D, seq_kv, n_heads_kv, batch]
    const half * __restrict__ mask,     // [seq_kv, seq_q, 1, batch]
    float * __restrict__ P_out,         // Output: [batch, heads, num_q_blocks, num_kv_iters, BLOCK_Q, BLOCK_KV]
    float * __restrict__ meta_out,      // Output: [batch, heads, num_q_blocks, num_kv_iters, BLOCK_Q, 2] (rowmax, rowsumexp)
    int n_heads,
    int n_heads_kv,
    int n_batch,
    int len_q,
    int len_kv,
    float softmax_scale,
    float max_bias,
    float m0,
    float m1,
    uint32_t n_head_log2,
    float logit_softcap,
    int stride_Q_row,
    int stride_Q_head,
    int stride_Q_batch,
    int stride_K_row,
    int stride_K_head,
    int stride_K_batch,
    int stride_mask_row,
    int stride_mask_batch
) {
    constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;
    constexpr int WARP_Q = BLOCK_Q / NUM_WARPS;
    constexpr int MMA_M = 16, MMA_N = 8, MMA_K = 16;

    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    // Block decomposition: [batch, head, q_block, kv_iter]
    const int num_q_blocks = cdiv(len_q, BLOCK_Q);
    const int num_kv_iters = cdiv(len_kv, BLOCK_KV);
    const int blocks_per_q_block = num_kv_iters;
    const int blocks_per_head = num_q_blocks * blocks_per_q_block;
    const int blocks_per_batch = n_heads * blocks_per_head;

    const int batch_id = bid / blocks_per_batch;
    const int bid_in_batch = bid % blocks_per_batch;
    const int head_id = bid_in_batch / blocks_per_head;
    const int bid_in_head = bid_in_batch % blocks_per_head;
    const int q_block_id = bid_in_head / blocks_per_q_block;
    const int kv_iter_id = bid_in_head % blocks_per_q_block;

    // GQA: map query head to KV head
    const int gqa_ratio = n_heads / n_heads_kv;
    const int kv_head_id = head_id / gqa_ratio;

    // Base pointers
    Q += batch_id * stride_Q_batch + head_id * stride_Q_head + q_block_id * BLOCK_Q * stride_Q_row;
    K += batch_id * stride_K_batch + kv_head_id * stride_K_head + kv_iter_id * BLOCK_KV * stride_K_row;

    const half *mask_block = mask ? (mask + batch_id * stride_mask_batch + q_block_id * BLOCK_Q * stride_mask_row) : nullptr;
    const float alibi_slope = get_alibi_slope(max_bias, head_id, n_head_log2, m0, m1);
    const int q_start = q_block_id * BLOCK_Q;
    const int kv_start = kv_iter_id * BLOCK_KV;

    // Shared memory layout:
    // - Q buffer: BLOCK_Q * DIM * sizeof(T) = 64 * 128 * 2 = 16KB
    // - K buffer: BLOCK_KV * DIM * sizeof(T) = 64 * 128 * 2 = 16KB
    // - Mask F32: BLOCK_Q * BLOCK_KV * sizeof(float) = 64 * 64 * 4 = 16KB
    // Total: 48KB
    extern __shared__ char smem_raw[];
    T * smem = reinterpret_cast<T *>(smem_raw);
    const uint32_t Q_smem = __cvta_generic_to_shared(smem);
    const uint32_t K_smem = Q_smem + BLOCK_Q * DIM * sizeof(T);
    const uint32_t mask_f32_smem = K_smem + BLOCK_KV * DIM * sizeof(T);

    // Register allocation - ONLY Q, K, S (no V or O)
    uint32_t Q_rmem[WARP_Q / MMA_M][DIM / MMA_K][4];
    uint32_t K_rmem[BLOCK_KV / MMA_N][DIM / MMA_K][2];

    // Pre-compute swizzled addresses
    uint32_t Q_smem_thread, K_smem_thread;
    {
        const int row_off = warp_id * WARP_Q + (lane_id % 16);
        const int col_off = lane_id / 16 * 8;
        Q_smem_thread = swizzle<DIM * sizeof(T)>(Q_smem + (row_off * DIM + col_off) * sizeof(T));
    }
    {
        const int row_off = lane_id % 8;
        const int col_off = lane_id / 8 * 8;
        K_smem_thread = swizzle<DIM * sizeof(T)>(K_smem + (row_off * DIM + col_off) * sizeof(T));
    }

    // Load Q to shared memory
    global_to_shared_swizzle<BLOCK_Q, DIM, TB_SIZE, T>(Q_smem, Q, stride_Q_row, tid, len_q - q_block_id * BLOCK_Q);
    asm volatile("cp.async.commit_group;");

    // Load K to shared memory
    global_to_shared_swizzle<BLOCK_KV, DIM, TB_SIZE, T>(K_smem, K, stride_K_row, tid, len_kv - kv_iter_id * BLOCK_KV);
    asm volatile("cp.async.commit_group;");

    // Wait for Q
    asm volatile("cp.async.wait_group 1;");
    __syncthreads();

    // Q shared -> registers
    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
        for (int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d++) {
            uint32_t addr = Q_smem_thread;
            addr += mma_id_q * MMA_M * DIM * sizeof(T);
            addr ^= mma_id_d * MMA_K * sizeof(T);
            ldmatrix_x4(Q_rmem[mma_id_q][mma_id_d], addr);
        }

    // Wait for K
    asm volatile("cp.async.wait_all;");
    __syncthreads();

    // K shared -> registers
    for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)
        for (int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d += 2) {
            uint32_t addr = K_smem_thread;
            addr += mma_id_kv * MMA_N * DIM * sizeof(T);
            addr ^= mma_id_d * MMA_K * sizeof(T);
            ldmatrix_x4(K_rmem[mma_id_kv][mma_id_d], addr);
        }

    // Pre-convert mask from F16 to F32 in shared memory
    float * mask_f32 = reinterpret_cast<float *>(__cvta_shared_to_generic(mask_f32_smem));
    if (mask_block) {
        constexpr int num_pairs = (BLOCK_Q * BLOCK_KV) / 2;
        for (int pair_idx = tid; pair_idx < num_pairs; pair_idx += TB_SIZE) {
            const int i = pair_idx * 2;
            const int q_idx = i / BLOCK_KV;
            const int kv_idx = i % BLOCK_KV;
            const int q_pos = q_start + q_idx;
            const int kv_pos0 = kv_start + kv_idx;
            const int kv_pos1 = kv_start + kv_idx + 1;

            if (q_pos < len_q && kv_pos1 < len_kv) {
                const half * src = &mask_block[q_idx * stride_mask_row + kv_pos0];
                const half2 mask_pair = *reinterpret_cast<const half2 *>(src);
                const float2 vals = __half22float2(mask_pair);
                mask_f32[i] = vals.x;
                mask_f32[i + 1] = vals.y;
            } else if (q_pos < len_q && kv_pos0 < len_kv) {
                mask_f32[i] = __half2float(mask_block[q_idx * stride_mask_row + kv_pos0]);
                mask_f32[i + 1] = -FLT_MAX;
            } else {
                mask_f32[i] = -FLT_MAX;
                mask_f32[i + 1] = -FLT_MAX;
            }
        }
        __syncthreads();
    }

    // Compute S = Q @ K^T
    float S_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_N][4] = {};

    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)
            for (int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d++)
                mma_m16n8k16<T>(Q_rmem[mma_id_q][mma_id_d], K_rmem[mma_id_kv][mma_id_d], S_rmem[mma_id_q][mma_id_kv]);

    // Apply softmax scale, mask, and compute P
    float rowmax[WARP_Q / MMA_M][2];
    float rowsumexp[WARP_Q / MMA_M][2] = {};

    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
        rowmax[mma_id_q][0] = -FLT_MAX;
        rowmax[mma_id_q][1] = -FLT_MAX;

        // Apply scale and mask
        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
            float *regs = S_rmem[mma_id_q][mma_id_kv];
            const int q_row_base = warp_id * WARP_Q + mma_id_q * MMA_M + (lane_id / 4);
            const int kv_col_base = kv_start + mma_id_kv * MMA_N + (lane_id % 4) * 2;

            #pragma unroll
            for (int r = 0; r < 4; r++) {
                const int q_row = q_row_base + (r >= 2 ? 8 : 0);
                const int kv_col = kv_col_base + (r % 2);

                float score = regs[r] * softmax_scale;
                if (logit_softcap != 0.0f) {
                    score = logit_softcap * tanhf(score);
                }

                if (kv_col >= len_kv) {
                    score = -FLT_MAX;
                } else if (mask_block && (q_start + q_row) < len_q) {
                    const int kv_local = kv_col - kv_start;
                    const float mask_val = mask_f32[q_row * BLOCK_KV + kv_local];
                    score += alibi_slope * mask_val;
                }
                regs[r] = score;
            }
        }

        // Compute rowmax
        float this_rowmax[2];
        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
            float *regs = S_rmem[mma_id_q][mma_id_kv];
            if (mma_id_kv == 0) {
                this_rowmax[0] = max(regs[0], regs[1]);
                this_rowmax[1] = max(regs[2], regs[3]);
            } else {
                this_rowmax[0] = max(this_rowmax[0], max(regs[0], regs[1]));
                this_rowmax[1] = max(this_rowmax[1], max(regs[2], regs[3]));
            }
        }

        // Butterfly reduction
        this_rowmax[0] = max(this_rowmax[0], __shfl_xor_sync(0xFFFFFFFF, this_rowmax[0], 1));
        this_rowmax[0] = max(this_rowmax[0], __shfl_xor_sync(0xFFFFFFFF, this_rowmax[0], 2));
        this_rowmax[1] = max(this_rowmax[1], __shfl_xor_sync(0xFFFFFFFF, this_rowmax[1], 1));
        this_rowmax[1] = max(this_rowmax[1], __shfl_xor_sync(0xFFFFFFFF, this_rowmax[1], 2));

        rowmax[mma_id_q][0] = this_rowmax[0];
        rowmax[mma_id_q][1] = this_rowmax[1];

        // Compute exp(score - rowmax) and rowsumexp
        float this_rowsumexp[2] = {0.0f, 0.0f};
        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
            float *regs = S_rmem[mma_id_q][mma_id_kv];
            regs[0] = __expf(regs[0] - rowmax[mma_id_q][0]);
            regs[1] = __expf(regs[1] - rowmax[mma_id_q][0]);
            regs[2] = __expf(regs[2] - rowmax[mma_id_q][1]);
            regs[3] = __expf(regs[3] - rowmax[mma_id_q][1]);

            this_rowsumexp[0] += regs[0] + regs[1];
            this_rowsumexp[1] += regs[2] + regs[3];
        }

        // Butterfly reduction for rowsumexp
        this_rowsumexp[0] += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[0], 1);
        this_rowsumexp[0] += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[0], 2);
        this_rowsumexp[1] += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[1], 1);
        this_rowsumexp[1] += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[1], 2);

        rowsumexp[mma_id_q][0] = this_rowsumexp[0];
        rowsumexp[mma_id_q][1] = this_rowsumexp[1];
    }

    // Output P matrix and metadata
    // P_out layout: [batch, heads, num_q_blocks, num_kv_iters, BLOCK_Q, BLOCK_KV]
    const int p_tile_offset = ((batch_id * n_heads + head_id) * num_q_blocks + q_block_id) * num_kv_iters + kv_iter_id;
    float * P_tile = P_out + p_tile_offset * BLOCK_Q * BLOCK_KV;
    float * meta_tile = meta_out + p_tile_offset * BLOCK_Q * 2;

    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
            float *regs = S_rmem[mma_id_q][mma_id_kv];
            const int row0 = warp_id * WARP_Q + mma_id_q * MMA_M + (lane_id / 4);
            const int row8 = row0 + 8;
            const int col = mma_id_kv * MMA_N + (lane_id % 4) * 2;

            if ((q_start + row0) < len_q) {
                P_tile[row0 * BLOCK_KV + col] = regs[0];
                P_tile[row0 * BLOCK_KV + col + 1] = regs[1];
            }
            if ((q_start + row8) < len_q) {
                P_tile[row8 * BLOCK_KV + col] = regs[2];
                P_tile[row8 * BLOCK_KV + col + 1] = regs[3];
            }
        }

        // Write metadata (rowmax, rowsumexp) - one per row, only lane 0 writes
        if ((lane_id % 4) == 0) {
            const int row0 = warp_id * WARP_Q + mma_id_q * MMA_M + (lane_id / 4);
            const int row8 = row0 + 8;
            if ((q_start + row0) < len_q) {
                meta_tile[row0 * 2] = rowmax[mma_id_q][0];
                meta_tile[row0 * 2 + 1] = rowsumexp[mma_id_q][0];
            }
            if ((q_start + row8) < len_q) {
                meta_tile[row8 * 2] = rowmax[mma_id_q][1];
                meta_tile[row8 * 2 + 1] = rowsumexp[mma_id_q][1];
            }
        }
    }
}

// =============================================================================
// TWO-PHASE KERNEL SPLIT: Phase 2 - P@V Accumulation
// =============================================================================
// This kernel reads P matrix and metadata from Phase 1, computes P @ V,
// and accumulates across all KV iterations with proper numerical rescaling.
//
// REGISTER BUDGET (~228 registers, fits within 255 limit):
//   - P_rmem: (WARP_Q/16) * (BLOCK_KV/16) * 4 = 1 * 4 * 4 = 16 registers (as uint32)
//   - V_rmem: (BLOCK_KV/16) * (DIM/8) * 2 = 4 * 16 * 2 = 128 registers
//   - O_rmem: (WARP_Q/16) * (DIM/8) * 4 = 1 * 16 * 4 = 64 registers
//   - misc (rowmax, rowsumexp, addresses, rescale): ~20 registers
//   Total: ~228 registers
//
// NO Q_rmem or K_rmem in this kernel - those were in Phase 1
// =============================================================================
template<int BLOCK_Q, int BLOCK_KV, int DIM, int NUM_WARPS, typename T, typename To>
__launch_bounds__(NUM_WARPS * WARP_SIZE, 2)
__global__
void attention_v5_phase2_pv_accum(
    const float * __restrict__ P_in,    // Input: [batch, heads, num_q_blocks, num_kv_iters, BLOCK_Q, BLOCK_KV]
    const float * __restrict__ meta_in, // Input: [batch, heads, num_q_blocks, num_kv_iters, BLOCK_Q, 2]
    const T * __restrict__ V,           // [D, seq_kv, n_heads_kv, batch]
    To * __restrict__ O,                // Output: [D, n_heads, seq_q, batch]
    int n_heads,
    int n_heads_kv,
    int n_batch,
    int len_q,
    int len_kv,
    int num_kv_iters,
    int stride_V_row,
    int stride_V_head,
    int stride_V_batch,
    int stride_O_row,
    int stride_O_head,
    int stride_O_batch
) {
    constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;
    constexpr int WARP_Q = BLOCK_Q / NUM_WARPS;
    constexpr int MMA_M = 16, MMA_N = 8, MMA_K = 16;

    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    // Block decomposition: [batch, head, q_block]
    const int num_q_blocks = cdiv(len_q, BLOCK_Q);
    const int blocks_per_batch = n_heads * num_q_blocks;

    const int batch_id = bid / blocks_per_batch;
    const int bid_in_batch = bid % blocks_per_batch;
    const int head_id = bid_in_batch / num_q_blocks;
    const int q_block_id = bid_in_batch % num_q_blocks;

    // GQA
    const int gqa_ratio = n_heads / n_heads_kv;
    const int kv_head_id = head_id / gqa_ratio;

    const int q_start = q_block_id * BLOCK_Q;

    // Shared memory layout:
    // - V buffer: BLOCK_KV * DIM * sizeof(T) = 64 * 128 * 2 = 16KB
    // - P buffer (F32): BLOCK_Q * BLOCK_KV * sizeof(float) = 64 * 64 * 4 = 16KB
    // Total: 32KB
    extern __shared__ char smem_raw[];
    T * smem = reinterpret_cast<T *>(smem_raw);
    const uint32_t V_smem = __cvta_generic_to_shared(smem);
    float * P_smem = reinterpret_cast<float *>(smem_raw + BLOCK_KV * DIM * sizeof(T));

    // Register allocation - ONLY P, V, O (no Q or K)
    uint32_t P_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_K][4];
    uint32_t V_rmem[BLOCK_KV / MMA_K][DIM / MMA_N][2];
    float O_rmem[WARP_Q / MMA_M][DIM / MMA_N][4] = {};

    float rowmax[WARP_Q / MMA_M][2];
    float rowsumexp[WARP_Q / MMA_M][2] = {};

    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
        rowmax[mma_id_q][0] = -FLT_MAX;
        rowmax[mma_id_q][1] = -FLT_MAX;
    }

    // Pre-compute swizzled V address
    uint32_t V_smem_thread;
    {
        const int row_off = lane_id % 16;
        const int col_off = lane_id / 16 * 8;
        V_smem_thread = swizzle<DIM * sizeof(T)>(V_smem + (row_off * DIM + col_off) * sizeof(T));
    }

    // Base tile offset for this (batch, head, q_block)
    const int base_tile_offset = ((batch_id * n_heads + head_id) * num_q_blocks + q_block_id) * num_kv_iters;

    // Iterate over all KV blocks
    for (int kv_iter = 0; kv_iter < num_kv_iters; kv_iter++) {
        const int kv_start = kv_iter * BLOCK_KV;

        // Load V to shared memory
        const T * V_ptr = V + batch_id * stride_V_batch + kv_head_id * stride_V_head + kv_start * stride_V_row;
        global_to_shared_swizzle<BLOCK_KV, DIM, TB_SIZE, T>(V_smem, V_ptr, stride_V_row, tid, len_kv - kv_start);
        asm volatile("cp.async.commit_group;");

        // Load P tile to shared memory (cooperative load)
        const float * P_tile = P_in + (base_tile_offset + kv_iter) * BLOCK_Q * BLOCK_KV;
        for (int i = tid; i < BLOCK_Q * BLOCK_KV; i += TB_SIZE) {
            P_smem[i] = P_tile[i];
        }

        // Wait for both loads
        asm volatile("cp.async.wait_all;");
        __syncthreads();

        // Load metadata for this KV iteration
        const float * meta_tile = meta_in + (base_tile_offset + kv_iter) * BLOCK_Q * 2;

        // V shared -> registers
        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++)
            for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d += 2) {
                uint32_t addr = V_smem_thread;
                addr += mma_id_kv * MMA_K * DIM * sizeof(T);
                addr ^= mma_id_d * MMA_N * sizeof(T);
                ldmatrix_x4_trans(V_rmem[mma_id_kv][mma_id_d], addr);
            }

        // Process each Q row in this warp's chunk
        for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
            const int row0 = warp_id * WARP_Q + mma_id_q * MMA_M + (lane_id / 4);
            const int row8 = row0 + 8;

            // Read rowmax/rowsumexp for this iteration from metadata
            // Use shuffle to broadcast from lane 0 in each group
            float iter_rowmax0 = -FLT_MAX, iter_rowmax1 = -FLT_MAX;
            float iter_rowsumexp0 = 0.0f, iter_rowsumexp1 = 0.0f;

            if ((q_start + row0) < len_q) {
                iter_rowmax0 = meta_tile[row0 * 2];
                iter_rowsumexp0 = meta_tile[row0 * 2 + 1];
            }
            if ((q_start + row8) < len_q) {
                iter_rowmax1 = meta_tile[row8 * 2];
                iter_rowsumexp1 = meta_tile[row8 * 2 + 1];
            }

            // Broadcast from lane 0 within each group of 4
            iter_rowmax0 = __shfl_sync(0xFFFFFFFF, iter_rowmax0, (lane_id / 4) * 4);
            iter_rowsumexp0 = __shfl_sync(0xFFFFFFFF, iter_rowsumexp0, (lane_id / 4) * 4);
            iter_rowmax1 = __shfl_sync(0xFFFFFFFF, iter_rowmax1, (lane_id / 4) * 4);
            iter_rowsumexp1 = __shfl_sync(0xFFFFFFFF, iter_rowsumexp1, (lane_id / 4) * 4);

            // Compute new global rowmax
            float new_rowmax0 = max(rowmax[mma_id_q][0], iter_rowmax0);
            float new_rowmax1 = max(rowmax[mma_id_q][1], iter_rowmax1);

            // Rescale existing O accumulator
            float rescale0 = __expf(rowmax[mma_id_q][0] - new_rowmax0);
            float rescale1 = __expf(rowmax[mma_id_q][1] - new_rowmax1);
            for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++) {
                O_rmem[mma_id_q][mma_id_d][0] *= rescale0;
                O_rmem[mma_id_q][mma_id_d][1] *= rescale0;
                O_rmem[mma_id_q][mma_id_d][2] *= rescale1;
                O_rmem[mma_id_q][mma_id_d][3] *= rescale1;
            }

            // Load P from shared memory and scale by exp(iter_rowmax - new_rowmax)
            float iter_scale0 = __expf(iter_rowmax0 - new_rowmax0);
            float iter_scale1 = __expf(iter_rowmax1 - new_rowmax1);

            for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++) {
                const int kv_col = mma_id_kv * MMA_K + (lane_id % 4) * 2;
                const int kv_col_8 = kv_col + 8;

                // Load P values for this row
                float p0_0 = P_smem[row0 * BLOCK_KV + kv_col] * iter_scale0;
                float p0_1 = P_smem[row0 * BLOCK_KV + kv_col + 1] * iter_scale0;
                float p8_0 = P_smem[row8 * BLOCK_KV + kv_col] * iter_scale1;
                float p8_1 = P_smem[row8 * BLOCK_KV + kv_col + 1] * iter_scale1;

                float p0_8 = P_smem[row0 * BLOCK_KV + kv_col_8] * iter_scale0;
                float p0_9 = P_smem[row0 * BLOCK_KV + kv_col_8 + 1] * iter_scale0;
                float p8_8 = P_smem[row8 * BLOCK_KV + kv_col_8] * iter_scale1;
                float p8_9 = P_smem[row8 * BLOCK_KV + kv_col_8 + 1] * iter_scale1;

                // Pack into P_rmem format for MMA
                packed_t<T> *this_P_rmem = reinterpret_cast<packed_t<T> *>(P_rmem[mma_id_q][mma_id_kv]);
                this_P_rmem[0] = float2_to_vec2<T>(p0_0, p0_1);
                this_P_rmem[1] = float2_to_vec2<T>(p8_0, p8_1);
                this_P_rmem[2] = float2_to_vec2<T>(p0_8, p0_9);
                this_P_rmem[3] = float2_to_vec2<T>(p8_8, p8_9);
            }

            // Update rowmax and rowsumexp
            rowsumexp[mma_id_q][0] = rowsumexp[mma_id_q][0] * rescale0 + iter_rowsumexp0 * iter_scale0;
            rowsumexp[mma_id_q][1] = rowsumexp[mma_id_q][1] * rescale1 + iter_rowsumexp1 * iter_scale1;
            rowmax[mma_id_q][0] = new_rowmax0;
            rowmax[mma_id_q][1] = new_rowmax1;

            // MMA O += P @ V
            for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++)
                for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++)
                    mma_m16n8k16<T>(P_rmem[mma_id_q][mma_id_kv], V_rmem[mma_id_kv][mma_id_d], O_rmem[mma_id_q][mma_id_d]);
        }

        __syncthreads();
    }

    // Write final output (normalized)
    O += batch_id * stride_O_batch + head_id * stride_O_head + q_start * stride_O_row;

    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
        for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++) {
            const int row = warp_id * WARP_Q + mma_id_q * MMA_M + (lane_id / 4);
            const int col = mma_id_d * MMA_N + (lane_id % 4) * 2;

            float *regs = O_rmem[mma_id_q][mma_id_d];
            regs[0] /= rowsumexp[mma_id_q][0];
            regs[1] /= rowsumexp[mma_id_q][0];
            regs[2] /= rowsumexp[mma_id_q][1];
            regs[3] /= rowsumexp[mma_id_q][1];

            if ((q_start + row) < len_q) {
                if constexpr (std::is_same_v<To, float>) {
                    reinterpret_cast<float2 *>(O + row * stride_O_row + col)[0] = make_float2(regs[0], regs[1]);
                } else if constexpr (std::is_same_v<To, half>) {
                    reinterpret_cast<half2 *>(O + row * stride_O_row + col)[0] = __float22half2_rn(make_float2(regs[0], regs[1]));
                } else if constexpr (std::is_same_v<To, nv_bfloat16>) {
                    reinterpret_cast<nv_bfloat162 *>(O + row * stride_O_row + col)[0] = __float22bfloat162_rn(make_float2(regs[0], regs[1]));
                }
            }
            if ((q_start + row + 8) < len_q) {
                if constexpr (std::is_same_v<To, float>) {
                    reinterpret_cast<float2 *>(O + (row + 8) * stride_O_row + col)[0] = make_float2(regs[2], regs[3]);
                } else if constexpr (std::is_same_v<To, half>) {
                    reinterpret_cast<half2 *>(O + (row + 8) * stride_O_row + col)[0] = __float22half2_rn(make_float2(regs[2], regs[3]));
                } else if constexpr (std::is_same_v<To, nv_bfloat16>) {
                    reinterpret_cast<nv_bfloat162 *>(O + (row + 8) * stride_O_row + col)[0] = __float22bfloat162_rn(make_float2(regs[2], regs[3]));
                }
            }
        }
}

// =============================================================================
// Split-K Reduction Kernel
// =============================================================================
// Combines partial results from multiple splits using numerically stable
// softmax combination. Each thread handles one element of the output.
//
// Algorithm:
//   global_max = max(partial_rowmax[0..split_k-1])
//   For each split k:
//     scale[k] = exp(partial_rowmax[k] - global_max)
//   global_sumexp = sum(scale[k] * partial_rowsumexp[k])
//   O_final = sum(scale[k] * partial_O[k]) / global_sumexp
// =============================================================================
template<int DIM, int BLOCK_Q, typename To>
__global__ void attention_v5_splitk_reduce(
    const float * __restrict__ partial_O,     // [total_tiles * split_k * BLOCK_Q * DIM]
    const float2 * __restrict__ partial_meta, // [total_tiles * split_k * BLOCK_Q]
    To * __restrict__ O,                      // Final output
    int n_heads,
    int n_batch,
    int len_q,
    int split_k,
    int stride_O_row,
    int stride_O_head,
    int stride_O_batch
) {
    // Thread block configuration: (tile_id) x (q_row, d_elem)
    // One block per tile, threads process (q_row, d_elem) pairs
    const int num_q_blocks = (len_q + BLOCK_Q - 1) / BLOCK_Q;
    const int tile_id = blockIdx.x;

    // Decompose tile_id into batch, head, q_block
    const int batch_id = tile_id / (n_heads * num_q_blocks);
    const int tile_in_batch = tile_id % (n_heads * num_q_blocks);
    const int head_id = tile_in_batch / num_q_blocks;
    const int q_block_id = tile_in_batch % num_q_blocks;

    // Thread indexing: each thread handles one (q_row, d_elem) pair
    // blockDim.x = BLOCK_Q, blockDim.y = DIM/8 (process 8 elements per thread to stay within 1024 thread limit)
    // 64 x 16 = 1024 threads (exactly SM_120 max)
    const int q_row_local = threadIdx.x;  // 0..BLOCK_Q-1
    const int d_base = threadIdx.y * 8;   // 0, 8, 16, ..., DIM-8

    const int q_pos = q_block_id * BLOCK_Q + q_row_local;

    // Early exit if this row is beyond len_q
    if (q_pos >= len_q) return;

    // Find global maximum across all splits for this row
    float global_max = -FLT_MAX;
    for (int k = 0; k < split_k; k++) {
        const int meta_idx = tile_id * split_k * BLOCK_Q + k * BLOCK_Q + q_row_local;
        const float2 meta = partial_meta[meta_idx];
        global_max = fmaxf(global_max, meta.x);
    }

    // Compute scaled accumulation (8 elements per thread)
    float O_accum[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float denom_accum = 0.0f;

    for (int k = 0; k < split_k; k++) {
        const int meta_idx = tile_id * split_k * BLOCK_Q + k * BLOCK_Q + q_row_local;
        const float2 meta = partial_meta[meta_idx];
        const float scale = __expf(meta.x - global_max);
        denom_accum += scale * meta.y;

        // Load 8 elements of partial_O for this split
        const float * partial_O_k = partial_O +
            (tile_id * split_k + k) * BLOCK_Q * DIM + q_row_local * DIM + d_base;

        // Use float4 vector loads for better memory throughput (4x fewer transactions)
        const float4 partial_0 = *reinterpret_cast<const float4*>(&partial_O_k[0]);
        const float4 partial_1 = *reinterpret_cast<const float4*>(&partial_O_k[4]);
        O_accum[0] += scale * partial_0.x;
        O_accum[1] += scale * partial_0.y;
        O_accum[2] += scale * partial_0.z;
        O_accum[3] += scale * partial_0.w;
        O_accum[4] += scale * partial_1.x;
        O_accum[5] += scale * partial_1.y;
        O_accum[6] += scale * partial_1.z;
        O_accum[7] += scale * partial_1.w;
    }

    // Normalize and write final output
    // Output layout: [D, heads, seq, batch] - note heads and seq are swapped from input!
    To * O_out = O + batch_id * stride_O_batch + head_id * stride_O_head +
                 q_pos * stride_O_row + d_base;

    const float inv_denom = 1.0f / denom_accum;

    if constexpr (std::is_same_v<To, float>) {
        O_out[0] = O_accum[0] * inv_denom;
        O_out[1] = O_accum[1] * inv_denom;
        O_out[2] = O_accum[2] * inv_denom;
        O_out[3] = O_accum[3] * inv_denom;
        O_out[4] = O_accum[4] * inv_denom;
        O_out[5] = O_accum[5] * inv_denom;
        O_out[6] = O_accum[6] * inv_denom;
        O_out[7] = O_accum[7] * inv_denom;
    } else if constexpr (std::is_same_v<To, half>) {
        O_out[0] = __float2half(O_accum[0] * inv_denom);
        O_out[1] = __float2half(O_accum[1] * inv_denom);
        O_out[2] = __float2half(O_accum[2] * inv_denom);
        O_out[3] = __float2half(O_accum[3] * inv_denom);
        O_out[4] = __float2half(O_accum[4] * inv_denom);
        O_out[5] = __float2half(O_accum[5] * inv_denom);
        O_out[6] = __float2half(O_accum[6] * inv_denom);
        O_out[7] = __float2half(O_accum[7] * inv_denom);
    } else if constexpr (std::is_same_v<To, nv_bfloat16>) {
        O_out[0] = __float2bfloat16(O_accum[0] * inv_denom);
        O_out[1] = __float2bfloat16(O_accum[1] * inv_denom);
        O_out[2] = __float2bfloat16(O_accum[2] * inv_denom);
        O_out[3] = __float2bfloat16(O_accum[3] * inv_denom);
        O_out[4] = __float2bfloat16(O_accum[4] * inv_denom);
        O_out[5] = __float2bfloat16(O_accum[5] * inv_denom);
        O_out[6] = __float2bfloat16(O_accum[6] * inv_denom);
        O_out[7] = __float2bfloat16(O_accum[7] * inv_denom);
    }
}

// =============================================================================
// WARP SPECIALIZED KERNEL: Producer-Consumer Model
// =============================================================================
// Achieves compute overlap by running Q@K^T (producers) and P@V (consumers) in parallel
// - Warps 0-1 (Producers): Load K, compute Q@K^T, softmax, write P to shared
// - Warps 2-3 (Consumers): Read P from shared, load V, compute P@V, accumulate O
// Uses named barriers (bar.sync/bar.arrive) for low-latency synchronization
//
// Expected speedup: 15-25% over 3-stage pipeline for long sequences (>4K tokens)

template<int BLOCK_Q, int BLOCK_KV, int DIM, int NUM_WARPS, typename T, typename To>
__launch_bounds__(NUM_WARPS * WARP_SIZE)
__global__
void attention_v5_warp_specialized_kernel(
  const T *Q, const T *K, const T *V, const half *mask, To *O,
  int n_heads, int n_heads_kv, int n_batch, int len_q, int len_kv,
  float softmax_scale, float max_bias, float m0, float m1, uint32_t n_head_log2, float logit_softcap,
  int stride_Q_row, int stride_Q_head, int stride_Q_batch,
  int stride_K_row, int stride_K_head, int stride_K_batch,
  int stride_V_row, int stride_V_head, int stride_V_batch,
  int stride_O_row, int stride_O_head, int stride_O_batch,
  int stride_mask_row, int stride_mask_batch)
{
  constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;
  constexpr int PRODUCER_WARPS = NUM_WARPS / 2;
  constexpr int WARP_Q = BLOCK_Q / NUM_WARPS;
  constexpr int MMA_M = 16, MMA_N = 8, MMA_K = 16;

  const int bid = blockIdx.x, tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE, lane_id = tid % WARP_SIZE;
  const bool is_producer = (warp_id < PRODUCER_WARPS);

  // Block/head decomposition
  const int num_q_blocks = cdiv(len_q, BLOCK_Q);
  const int bs_id = bid / num_q_blocks, q_block_id = bid % num_q_blocks;
  const int head_id = bs_id % n_heads, batch_id = bs_id / n_heads;
  const int kv_head_id = head_id / (n_heads / n_heads_kv);

  Q += batch_id * stride_Q_batch + head_id * stride_Q_head + q_block_id * BLOCK_Q * stride_Q_row;
  K += batch_id * stride_K_batch + kv_head_id * stride_K_head;
  V += batch_id * stride_V_batch + kv_head_id * stride_V_head;
  O += batch_id * stride_O_batch + head_id * stride_O_head + q_block_id * BLOCK_Q * stride_O_row;

  const half *mask_block = mask ? (mask + batch_id * stride_mask_batch + q_block_id * BLOCK_Q * stride_mask_row) : nullptr;
  const float alibi_slope = get_alibi_slope(max_bias, head_id, n_head_log2, m0, m1);
  const int q_start = q_block_id * BLOCK_Q;

  // Shared memory layout
  extern __shared__ char smem_raw[];
  T * smem = reinterpret_cast<T *>(smem_raw);
  const uint32_t Q_smem = __cvta_generic_to_shared(smem);
  const uint32_t K_smem = Q_smem;
  constexpr int K_BUF_SIZE = BLOCK_KV * DIM * sizeof(T);
  const uint32_t V_smem = K_smem + 2 * K_BUF_SIZE;
  constexpr int V_BUF_SIZE = BLOCK_KV * DIM * sizeof(T);
  const uint32_t P_smem = V_smem + V_BUF_SIZE + BLOCK_Q * BLOCK_KV * sizeof(float);  // After V and mask_f32
  constexpr int P_BUF_SIZE = BLOCK_Q * BLOCK_KV * sizeof(T);

  // Softmax state sharing
  float * softmax_state = reinterpret_cast<float *>(__cvta_shared_to_generic(P_smem + 2 * P_BUF_SIZE));
  float * rowmax_smem = softmax_state, * rowsumexp_smem = rowmax_smem + BLOCK_Q;

  // Registers
  uint32_t Q_rmem[WARP_Q / MMA_M][DIM / MMA_K][4];
  float O_rmem[WARP_Q / MMA_M][DIM / MMA_N][4] = {};
  float rowmax[WARP_Q / MMA_M][2], rowsumexp[WARP_Q / MMA_M][2] = {};

  // Pre-compute swizzled addresses
  uint32_t Q_smem_thread = swizzle<DIM * sizeof(T)>(Q_smem + (warp_id * WARP_Q + (lane_id % 16)) * DIM * sizeof(T) + (lane_id / 16 * 8) * sizeof(T));
  uint32_t K_smem_thread = swizzle<DIM * sizeof(T)>(K_smem + (lane_id % 8) * DIM * sizeof(T) + (lane_id / 8 * 8) * sizeof(T));
  uint32_t V_smem_thread = swizzle<DIM * sizeof(T)>(V_smem + (lane_id % 16) * DIM * sizeof(T) + (lane_id / 16 * 8) * sizeof(T));
  uint32_t P_smem_thread = swizzle<BLOCK_KV * sizeof(T)>(P_smem + (warp_id * WARP_Q + (lane_id % 16)) * BLOCK_KV * sizeof(T) + (lane_id / 16 * 8) * sizeof(T));

  for (int i = 0; i < WARP_Q / MMA_M; i++) { rowmax[i][0] = rowmax[i][1] = -FLT_MAX; }

  // Prologue: Load Q
  global_to_shared_swizzle<BLOCK_Q, DIM, TB_SIZE, T>(Q_smem, Q, stride_Q_row, tid, len_q - q_block_id * BLOCK_Q);
  asm volatile("cp.async.commit_group;"); asm volatile("cp.async.wait_all;"); __syncthreads();

  for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
    for (int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d++) {
      uint32_t addr = Q_smem_thread + mma_id_q * MMA_M * DIM * sizeof(T);
      addr ^= mma_id_d * MMA_K * sizeof(T);
      ldmatrix_x4(Q_rmem[mma_id_q][mma_id_d], addr);
    }
  __syncthreads();

  const int num_kv_iter = cdiv(len_kv, BLOCK_KV);

  // Main loop - uses __syncthreads() for correct producer-consumer synchronization
  // All 128 threads participate in every barrier to satisfy CUDA barrier semantics
  for (int kv_id = 0; kv_id < num_kv_iter; kv_id++) {
    // Phase 1: Buffer protection - wait for consumers to finish with P[kv_id-2]
    // This ensures producers don't overwrite P buffer that consumers are still reading
    if (kv_id >= 2) __syncthreads();

    if (is_producer) {
      // Load K
      const uint32_t K_dst = K_smem + (kv_id % 2) * K_BUF_SIZE;
      constexpr int num_elems = 16 / sizeof(T);
      constexpr int PROD_THREADS = PRODUCER_WARPS * WARP_SIZE;
      #pragma unroll
      for (int iter = 0; iter < (BLOCK_KV * DIM) / (PROD_THREADS * num_elems); iter++) {
        const int idx = (iter * PROD_THREADS + tid) * num_elems;
        const int row = idx / DIM, col = idx % DIM;
        const uint32_t dst_addr = swizzle<DIM * sizeof(T)>(K_dst + (row * DIM + col) * sizeof(T));
        if (row < len_kv - kv_id * BLOCK_KV) {
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst_addr), "l"(K + kv_id * BLOCK_KV * stride_K_row + row * stride_K_row + col));
        } else {
          T * p = reinterpret_cast<T*>(__cvta_shared_to_generic(dst_addr));
          for (int i = 0; i < num_elems; i++) p[i] = T(0);
        }
      }
      asm volatile("cp.async.commit_group;"); asm volatile("cp.async.wait_all;"); __syncwarp();

      // K -> registers
      uint32_t K_rmem[BLOCK_KV / MMA_N][DIM / MMA_K][2];
      for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)
        for (int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d += 2) {
          uint32_t addr = K_smem_thread + (kv_id % 2) * K_BUF_SIZE + mma_id_kv * MMA_N * DIM * sizeof(T);
          addr ^= mma_id_d * MMA_K * sizeof(T);
          ldmatrix_x4(K_rmem[mma_id_kv][mma_id_d], addr);
        }

      // Compute S = Q @ K^T and softmax
      float S_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_N][4] = {};
      for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)
          for (int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d++)
            mma_m16n8k16<T>(Q_rmem[mma_id_q][mma_id_d], K_rmem[mma_id_kv][mma_id_d], S_rmem[mma_id_q][mma_id_kv]);

      // Softmax processing
      for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
        // Apply scale, mask
        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
          float *regs = S_rmem[mma_id_q][mma_id_kv];
          const int q_row_base = warp_id * WARP_Q + mma_id_q * MMA_M + (lane_id / 4);
          const int kv_col_base = kv_id * BLOCK_KV + mma_id_kv * MMA_N + (lane_id % 4) * 2;
          #pragma unroll
          for (int r = 0; r < 4; r++) {
            float score = regs[r] * softmax_scale;
            if (logit_softcap != 0.0f) score = logit_softcap * tanhf(score);
            const int kv_col = kv_col_base + (r % 2);
            if (kv_col >= len_kv) score = -FLT_MAX;
            else if (mask_block) {
              const int q_row = q_row_base + (r >= 2 ? 8 : 0);
              if ((q_start + q_row) < len_q)
                score += alibi_slope * __half2float(mask_block[q_row * stride_mask_row + kv_col - kv_id * BLOCK_KV]);
            }
            regs[r] = score;
          }
        }

        // Rowmax and softmax
        float this_rowmax[2] = {-FLT_MAX, -FLT_MAX};
        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
          float *regs = S_rmem[mma_id_q][mma_id_kv];
          this_rowmax[0] = max(this_rowmax[0], max(regs[0], regs[1]));
          this_rowmax[1] = max(this_rowmax[1], max(regs[2], regs[3]));
        }
        this_rowmax[0] = max(this_rowmax[0], __shfl_xor_sync(0xFFFFFFFF, this_rowmax[0], 1));
        this_rowmax[0] = max(this_rowmax[0], __shfl_xor_sync(0xFFFFFFFF, this_rowmax[0], 2));
        this_rowmax[1] = max(this_rowmax[1], __shfl_xor_sync(0xFFFFFFFF, this_rowmax[1], 1));
        this_rowmax[1] = max(this_rowmax[1], __shfl_xor_sync(0xFFFFFFFF, this_rowmax[1], 2));

        float new_rowmax0 = max(this_rowmax[0], rowmax[mma_id_q][0]);
        float new_rowmax1 = max(this_rowmax[1], rowmax[mma_id_q][1]);
        float rescale0 = __expf(rowmax[mma_id_q][0] - new_rowmax0);
        float rescale1 = __expf(rowmax[mma_id_q][1] - new_rowmax1);

        float this_rowsumexp[2] = {0.0f, 0.0f};
        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
          float *regs = S_rmem[mma_id_q][mma_id_kv];
          regs[0] = __expf(regs[0] - new_rowmax0); regs[1] = __expf(regs[1] - new_rowmax0);
          regs[2] = __expf(regs[2] - new_rowmax1); regs[3] = __expf(regs[3] - new_rowmax1);
          this_rowsumexp[0] += regs[0] + regs[1]; this_rowsumexp[1] += regs[2] + regs[3];
        }
        this_rowsumexp[0] += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[0], 1);
        this_rowsumexp[0] += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[0], 2);
        this_rowsumexp[1] += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[1], 1);
        this_rowsumexp[1] += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[1], 2);

        rowsumexp[mma_id_q][0] = rowsumexp[mma_id_q][0] * rescale0 + this_rowsumexp[0];
        rowsumexp[mma_id_q][1] = rowsumexp[mma_id_q][1] * rescale1 + this_rowsumexp[1];
        rowmax[mma_id_q][0] = new_rowmax0; rowmax[mma_id_q][1] = new_rowmax1;

        // Store softmax state
        if (lane_id == 0) {
          const int q_row0 = warp_id * WARP_Q + mma_id_q * MMA_M;
          rowmax_smem[q_row0] = new_rowmax0; rowmax_smem[q_row0 + 8] = new_rowmax1;
          rowsumexp_smem[q_row0] = rowsumexp[mma_id_q][0]; rowsumexp_smem[q_row0 + 8] = rowsumexp[mma_id_q][1];
        }
      }

      // Write P to shared
      T * P_buf = reinterpret_cast<T *>(__cvta_shared_to_generic(P_smem + (kv_id % 2) * P_BUF_SIZE));
      for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
          float *regs = S_rmem[mma_id_q][mma_id_kv];
          const int q_row = warp_id * WARP_Q + mma_id_q * MMA_M + (lane_id / 4);
          const int kv_col = mma_id_kv * MMA_N + (lane_id % 4) * 2;
          *reinterpret_cast<packed_t<T> *>(&P_buf[(q_row + 0) * BLOCK_KV + kv_col]) = float2_to_vec2<T>(regs[0], regs[1]);
          *reinterpret_cast<packed_t<T> *>(&P_buf[(q_row + 8) * BLOCK_KV + kv_col]) = float2_to_vec2<T>(regs[2], regs[3]);
        }

      __threadfence_block();  // Ensure P writes are visible before barrier
    }

    // Phase 2: All 128 threads synchronize - P is now ready for consumers
    __syncthreads();

    if (!is_producer) {
      // Load P
      uint32_t P_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_K][4];
      for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++) {
          uint32_t addr = P_smem_thread + (kv_id % 2) * P_BUF_SIZE + mma_id_q * MMA_M * BLOCK_KV * sizeof(T);
          addr ^= mma_id_kv * MMA_K * sizeof(T);
          ldmatrix_x4(P_rmem[mma_id_q][mma_id_kv], addr);
        }

      // Read and apply rescale
      if (lane_id == 0) {
        for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
          const int q_row0 = warp_id * WARP_Q + mma_id_q * MMA_M;
          float new_max0 = rowmax_smem[q_row0], new_max1 = rowmax_smem[q_row0 + 8];
          float rescale0 = __expf(rowmax[mma_id_q][0] - new_max0);
          float rescale1 = __expf(rowmax[mma_id_q][1] - new_max1);
          for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++) {
            O_rmem[mma_id_q][mma_id_d][0] *= rescale0; O_rmem[mma_id_q][mma_id_d][1] *= rescale0;
            O_rmem[mma_id_q][mma_id_d][2] *= rescale1; O_rmem[mma_id_q][mma_id_d][3] *= rescale1;
          }
          rowmax[mma_id_q][0] = new_max0; rowmax[mma_id_q][1] = new_max1;
          rowsumexp[mma_id_q][0] = rowsumexp_smem[q_row0]; rowsumexp[mma_id_q][1] = rowsumexp_smem[q_row0 + 8];
        }
      }

      // Load V
      constexpr int num_elems = 16 / sizeof(T);
      constexpr int CONS_THREADS = (NUM_WARPS - PRODUCER_WARPS) * WARP_SIZE;
      const int consumer_tid = tid - PRODUCER_WARPS * WARP_SIZE;
      #pragma unroll
      for (int iter = 0; iter < (BLOCK_KV * DIM) / (CONS_THREADS * num_elems); iter++) {
        const int idx = (iter * CONS_THREADS + consumer_tid) * num_elems;
        const int row = idx / DIM, col = idx % DIM;
        const uint32_t dst_addr = swizzle<DIM * sizeof(T)>(V_smem + (row * DIM + col) * sizeof(T));
        if (row < len_kv - kv_id * BLOCK_KV) {
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst_addr), "l"(V + kv_id * BLOCK_KV * stride_V_row + row * stride_V_row + col));
        } else {
          T * p = reinterpret_cast<T*>(__cvta_shared_to_generic(dst_addr));
          for (int i = 0; i < num_elems; i++) p[i] = T(0);
        }
      }
      asm volatile("cp.async.commit_group;"); asm volatile("cp.async.wait_all;"); __syncwarp();

      // V -> registers
      uint32_t V_rmem[BLOCK_KV / MMA_K][DIM / MMA_N][2];
      for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++)
        for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d += 2) {
          uint32_t addr = V_smem_thread + mma_id_kv * MMA_K * DIM * sizeof(T);
          addr ^= mma_id_d * MMA_N * sizeof(T);
          ldmatrix_x4_trans(V_rmem[mma_id_kv][mma_id_d], addr);
        }

      // O += P @ V
      for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
        for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++)
          for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++)
            mma_m16n8k16<T>(P_rmem[mma_id_q][mma_id_kv], V_rmem[mma_id_kv][mma_id_d], O_rmem[mma_id_q][mma_id_d]);
    }

    // Phase 3: All 128 threads synchronize - safe for next iteration
    // This ensures consumers are done reading P[kv_id] before producers overwrite it
    __syncthreads();
  }

  // Epilogue: Consumer writes output
  __syncthreads();
  if (!is_producer) {
    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
      for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++) {
        const int row = warp_id * WARP_Q + mma_id_q * MMA_M + (lane_id / 4);
        const int col = mma_id_d * MMA_N + (lane_id % 4) * 2;
        float *regs = O_rmem[mma_id_q][mma_id_d];
        // Use reciprocal multiply for performance (division ~25 cycles, multiply ~4 cycles)
        const float inv_rowsumexp0 = 1.0f / rowsumexp[mma_id_q][0];
        const float inv_rowsumexp1 = 1.0f / rowsumexp[mma_id_q][1];
        regs[0] *= inv_rowsumexp0; regs[1] *= inv_rowsumexp0;
        regs[2] *= inv_rowsumexp1; regs[3] *= inv_rowsumexp1;

        if ((q_block_id * BLOCK_Q + row) < len_q) {
          if constexpr (std::is_same_v<To, float>) reinterpret_cast<float2 *>(O + row * stride_O_row + col)[0] = make_float2(regs[0], regs[1]);
          else if constexpr (std::is_same_v<To, half>) reinterpret_cast<half2 *>(O + row * stride_O_row + col)[0] = __float22half2_rn(make_float2(regs[0], regs[1]));
          else if constexpr (std::is_same_v<To, nv_bfloat16>) reinterpret_cast<nv_bfloat162 *>(O + row * stride_O_row + col)[0] = __float22bfloat162_rn(make_float2(regs[0], regs[1]));
        }
        if ((q_block_id * BLOCK_Q + row + 8) < len_q) {
          if constexpr (std::is_same_v<To, float>) reinterpret_cast<float2 *>(O + (row + 8) * stride_O_row + col)[0] = make_float2(regs[2], regs[3]);
          else if constexpr (std::is_same_v<To, half>) reinterpret_cast<half2 *>(O + (row + 8) * stride_O_row + col)[0] = __float22half2_rn(make_float2(regs[2], regs[3]));
          else if constexpr (std::is_same_v<To, nv_bfloat16>) reinterpret_cast<nv_bfloat162 *>(O + (row + 8) * stride_O_row + col)[0] = __float22bfloat162_rn(make_float2(regs[2], regs[3]));
        }
      }
  }
}

// =============================================================================
// GGML Integration Wrapper for SM_120 (Blackwell) Flash Attention
// Supports both BF16 (native) and F16 (for Q8_0 and other quantized models)
// with full stride support for non-contiguous tensors after ggml_permute()
// =============================================================================

#include "fattn-common.cuh"

// GGML wrapper that dispatches to appropriate kernel based on input type
// Extracts proper strides to support non-contiguous tensor layouts
void ggml_cuda_flash_attn_ext_attention_v5(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];  // Causal mask, can be nullptr

    /*
     * TENSOR LAYOUT CHECKPOINT: attention_v5 wrapper
     * ===============================================
     *
     * TENSOR FLOW from llama-graph.cpp:
     *   1. KV cache returns K,V as [D, heads, seq_kv, batch] via get_k()/get_v()
     *   2. Q starts as [D, heads, seq_q, batch] from model layers
     *   3. permute(0, 2, 1, 3) swaps dims 1<->2: [D, heads, seq, batch] -> [D, seq, heads, batch]
     *   4. ggml_cont() makes contiguous in the NEW dimension order
     *
     * AFTER permute + ggml_cont (what we receive here):
     *   Q: shape=[D, seq_q, n_heads_q, batch] - CONTIGUOUS
     *      - ne[0] = D (head dimension, e.g., 128)
     *      - ne[1] = seq_q (query sequence length) <-- WAS heads before permute
     *      - ne[2] = n_heads_q (number of heads)   <-- WAS seq before permute
     *      - ne[3] = batch
     *
     *   K: shape=[D, seq_kv, n_heads_kv, batch] - CONTIGUOUS
     *   V: shape=[D, seq_kv, n_heads_kv, batch] - CONTIGUOUS
     *
     * Strides after ggml_cont (contiguous):
     *   nb[0] = sizeof(T)
     *   nb[1] = D * sizeof(T)          (stride to next seq position)
     *   nb[2] = D * seq * sizeof(T)    (stride to next head)
     *   nb[3] = D * seq * heads * sizeof(T)
     */

    // Extract dimensions - after permute(0,2,1,3), layout is [D, seq, heads, batch]
    const int64_t ne00 = Q->ne[0];  // head_dim (D)
    const int64_t ne01 = Q->ne[2];  // n_heads_q - in dimension 2 after permute
    const int64_t ne02 = Q->ne[1];  // seq_q - in dimension 1 after permute
    const int64_t ne03 = Q->ne[3];  // batch

    const int64_t ne11 = K->ne[2];  // n_heads_kv - in dimension 2 after permute
    const int64_t ne12 = K->ne[1];  // seq_kv - in dimension 1 after permute

    // Calculate dimensions for kernel
    const int n_heads = ne01;
    const int n_heads_kv = ne11;
    const int n_batch = ne03;
    const int len_q = ne02;
    const int len_kv = ne12;

    // Extract parameters from op_params (same as fattn-common.cuh)
    float scale         = 1.0f;
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;

    memcpy(&scale,         (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    // If logit_softcap is enabled, adjust scale (kernel applies: softcap * tanh(score * scale))
    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    // ALiBi parameters
    const uint32_t n_head_log2 = 1u << uint32_t(floorf(log2f(float(n_heads))));
    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    // Mask strides (if mask is present)
    const half * mask_data = mask ? (const half *)mask->data : nullptr;
    const int stride_mask_row   = mask ? (int)(mask->nb[1] / sizeof(half)) : 0;
    const int stride_mask_batch = mask ? (int)(mask->nb[3] / sizeof(half)) : 0;

    // Kernel configuration (Blackwell optimized)
    constexpr int BLOCK_Q = 64;
    constexpr int BLOCK_KV = 64;
    constexpr int DIM = 128;
    constexpr int NUM_WARPS = 4;
    constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;

    // Warp specialization feature flag - set to true to use producer-consumer pipeline
    // Expected 15-25% speedup for long sequences (>4K tokens) when split_k == 1
    constexpr bool USE_WARP_SPECIALIZED = true;

    // Two-phase kernel split feature flag - addresses register pressure on SM_120
    // When enabled, splits the attention computation into two kernels:
    // - Phase 1: Q@K^T + Softmax (~212 registers) - fits within 255 limit
    // - Phase 2: P@V accumulation (~228 registers) - fits within 255 limit
    // This avoids the ~428 register requirement of the fused kernel and eliminates
    // the need for --maxrregcount=224 which causes 200+ registers to spill to local memory
    constexpr bool USE_TWO_PHASE_KERNEL = true;

    const int num_q_blocks = (len_q + BLOCK_Q - 1) / BLOCK_Q;
    const int total_kv_iters = (len_kv + BLOCK_KV - 1) / BLOCK_KV;

    // ===========================================================================
    // Split-K Determination Heuristic
    // ===========================================================================
    // Goal: Maximize GPU occupancy by ensuring enough blocks for all SMs
    // Blackwell typically has 144-256 SMs depending on variant
    const int device = ggml_cuda_get_device();
    const int nsm = ggml_cuda_info().devices[device].nsm;
    const int base_blocks = n_heads * n_batch * num_q_blocks;

    // Target at least 2 waves of blocks for good occupancy (2 blocks per SM)
    const int target_blocks = 2 * nsm;

    int split_k = 1;  // Default: no split-k
    if (base_blocks < target_blocks && total_kv_iters > 1) {
        // Calculate how many splits needed
        int split_k_needed = (target_blocks + base_blocks - 1) / base_blocks;
        // Cap split_k at total_kv_iters (can't split more than we have iterations)
        split_k_needed = min(split_k_needed, total_kv_iters);
        // Cap at 32 to limit workspace size
        split_k_needed = min(split_k_needed, 32);
        // Round down to power of 2 for efficient reduction
        if (split_k_needed >= 16) split_k = 16;
        else if (split_k_needed >= 8) split_k = 8;
        else if (split_k_needed >= 4) split_k = 4;
        else if (split_k_needed >= 2) split_k = 2;
        else split_k = 1;
    }

    const int num_blocks = base_blocks * split_k;

    // Get the CUDA stream from context (critical for multi-GPU correctness)
    cudaStream_t main_stream = ctx.stream();

    // ===========================================================================
    // Split-K Workspace Allocation
    // ===========================================================================
    // Workspace: partial_O [total_tiles * split_k * BLOCK_Q * DIM] float
    //            partial_meta [total_tiles * split_k * BLOCK_Q] float2
    float * partial_O = nullptr;
    float2 * partial_meta = nullptr;

    const int total_tiles = n_batch * n_heads * num_q_blocks;
    const size_t partial_O_size = (split_k > 1) ? (size_t)total_tiles * split_k * BLOCK_Q * DIM * sizeof(float) : 0;
    const size_t partial_meta_size = (split_k > 1) ? (size_t)total_tiles * split_k * BLOCK_Q * sizeof(float2) : 0;

    // Use pool allocation for temporary workspace
    ggml_cuda_pool & pool = ctx.pool();
    ggml_cuda_pool_alloc<float> partial_O_alloc(pool);
    ggml_cuda_pool_alloc<float2> partial_meta_alloc(pool);

    if (split_k > 1) {
        partial_O_alloc.alloc(total_tiles * split_k * BLOCK_Q * DIM);
        partial_meta_alloc.alloc(total_tiles * split_k * BLOCK_Q);
        partial_O = partial_O_alloc.ptr;
        partial_meta = partial_meta_alloc.ptr;
    }

    // ===========================================================================
    // Two-Phase Kernel Workspace Allocation
    // ===========================================================================
    // P buffer: [batch, heads, num_q_blocks, num_kv_iters, BLOCK_Q, BLOCK_KV] floats
    // Meta buffer: [batch, heads, num_q_blocks, num_kv_iters, BLOCK_Q, 2] floats (rowmax, rowsumexp)
    float * phase_P_buffer = nullptr;
    float * phase_meta_buffer = nullptr;

    ggml_cuda_pool_alloc<float> phase_P_alloc(pool);
    ggml_cuda_pool_alloc<float> phase_meta_alloc(pool);

    // Determine if we should use two-phase kernel
    const bool use_two_phase = USE_TWO_PHASE_KERNEL && (split_k == 1);

    if (use_two_phase) {
        const size_t phase_P_size = (size_t)total_tiles * total_kv_iters * BLOCK_Q * BLOCK_KV;
        const size_t phase_meta_size = (size_t)total_tiles * total_kv_iters * BLOCK_Q * 2;

        phase_P_alloc.alloc(phase_P_size);
        phase_meta_alloc.alloc(phase_meta_size);
        phase_P_buffer = phase_P_alloc.ptr;
        phase_meta_buffer = phase_meta_alloc.ptr;
    }

    // Lambda to launch kernel with specific output type
    auto launch_kernel_with_output_type = [&](auto input_type_tag) {
        using T = typename decltype(input_type_tag)::type;
        const T * Q_data = (const T *)Q->data;
        const T * K_data = (const T *)K->data;
        const T * V_data = (const T *)V->data;

        /*
         * STRIDE MAPPING for [D, seq, heads, batch] layout (after permute):
         * =================================================================
         * Tensor dimensions: ne[0]=D, ne[1]=seq, ne[2]=heads, ne[3]=batch
         * Contiguous strides: nb[0]=sizeof(T), nb[1]=D*sizeof(T), nb[2]=D*seq*sizeof(T)
         *
         * Kernel stride semantics:
         *   stride_X_row   = stride between SEQUENCE positions (to advance by 1 in seq dim)
         *   stride_X_head  = stride between HEAD groups (to advance by 1 in heads dim)
         *   stride_X_batch = stride between BATCHES
         *
         * Mapping (contiguous tensor in [D, seq, heads, batch] order):
         *   stride_row  = nb[1] / sizeof(T) = D            (seq is dim 1)
         *   stride_head = nb[2] / sizeof(T) = D * seq      (heads is dim 2)
         *   stride_batch = nb[3] / sizeof(T)
         */
        const int stride_Q_row   = (int)(Q->nb[1] / sizeof(T));  // stride between seq positions (dim 1)
        const int stride_Q_head  = (int)(Q->nb[2] / sizeof(T));  // stride between heads (dim 2)
        const int stride_Q_batch = Q->nb[3] / sizeof(T);
        const int stride_K_row   = (int)(K->nb[1] / sizeof(T));
        const int stride_K_head  = (int)(K->nb[2] / sizeof(T));
        const int stride_K_batch = K->nb[3] / sizeof(T);
        const int stride_V_row   = (int)(V->nb[1] / sizeof(T));  // stride between seq positions (dim 1)
        const int stride_V_head  = (int)(V->nb[2] / sizeof(T));  // stride between heads (dim 2)
        const int stride_V_batch = V->nb[3] / sizeof(T);

        // Shared memory layout for standard kernel (3-stage software pipelining):
        // - K TRIPLE buffer: 3 * BLOCK_KV * DIM * sizeof(T) = 3 * 64 * 128 * 2 = 48KB
        // - V DOUBLE buffer: 2 * BLOCK_KV * DIM * sizeof(T) = 2 * 64 * 128 * 2 = 32KB
        // - Mask F32 buffer: BLOCK_Q * BLOCK_KV * sizeof(float) = 64 * 64 * 4 = 16KB
        // Total: 48KB + 32KB + 16KB = 96KB (within 99KB SM_120 limit)
        const int smem_size = (3 * BLOCK_KV + 2 * BLOCK_KV) * DIM * sizeof(T) + BLOCK_Q * BLOCK_KV * sizeof(float);

        // Shared memory layout for warp-specialized kernel (producer-consumer):
        // - K double-buffer: 2 * BLOCK_KV * DIM * sizeof(T) = 32KB
        // - V buffer: BLOCK_KV * DIM * sizeof(T) = 16KB
        // - Mask F32: BLOCK_Q * BLOCK_KV * sizeof(float) = 16KB
        // - P double-buffer: 2 * BLOCK_Q * BLOCK_KV * sizeof(float) = 32KB
        // - Softmax state (rowmax + rowsumexp): 2 * BLOCK_Q * sizeof(float) = 0.5KB
        // Total: ~96.5KB (within 99KB SM_120 limit)
        const int smem_size_warp_spec = (2 * BLOCK_KV + BLOCK_KV) * DIM * sizeof(T)  // K + V
                                      + BLOCK_Q * BLOCK_KV * sizeof(float)            // Mask
                                      + 2 * BLOCK_Q * BLOCK_KV * sizeof(float)        // P double-buffer
                                      + 2 * BLOCK_Q * sizeof(float);                  // Softmax state

        // Determine which kernel to use
        const bool use_warp_spec = USE_WARP_SPECIALIZED && (split_k == 1) && !use_two_phase;

        // Shared memory sizes for two-phase kernels
        // Phase 1: Q buffer (16KB) + K buffer (16KB) + Mask F32 (16KB) = 48KB
        const int smem_size_phase1 = BLOCK_Q * DIM * sizeof(T) + BLOCK_KV * DIM * sizeof(T) + BLOCK_Q * BLOCK_KV * sizeof(float);
        // Phase 2: V buffer (16KB) + P buffer F32 (16KB) = 32KB
        const int smem_size_phase2 = BLOCK_KV * DIM * sizeof(T) + BLOCK_Q * BLOCK_KV * sizeof(float);

        if (dst->type == GGML_TYPE_F32) {
            using To = float;
            To * dst_data = (To *)dst->data;
            /*
             * OUTPUT LAYOUT: [D, heads, seq, batch] (from ggml_flash_attn_ext)
             * This is DIFFERENT from input layout [D, seq, heads, batch]!
             *   - ne[1] = heads, ne[2] = seq (swapped from input)
             *   - nb[1] = D * sizeof(To)         (stride to next head)
             *   - nb[2] = D * heads * sizeof(To) (stride to next seq position)
             *
             * Kernel expects: stride_O_row = seq stride, stride_O_head = head stride
             */
            const int stride_O_row   = dst->nb[2] / sizeof(To);  // seq is dim 2 in output
            const int stride_O_head  = dst->nb[1] / sizeof(To);  // heads is dim 1 in output
            const int stride_O_batch = dst->nb[3] / sizeof(To);

            if (use_two_phase) {
                // ===========================================================================
                // TWO-PHASE KERNEL: Eliminates register spilling on SM_120
                // ===========================================================================
                // Phase 1: Q@K^T + Softmax - ~212 registers, no spilling
                // Phase 2: P@V accumulation - ~228 registers, no spilling
                const int phase1_blocks = n_batch * n_heads * num_q_blocks * total_kv_iters;
                const int phase2_blocks = base_blocks;  // n_batch * n_heads * num_q_blocks

                // Phase 1: Compute attention scores and softmax
                auto kernel_phase1 = attention_v5_phase1_qk_softmax<BLOCK_Q, BLOCK_KV, DIM, NUM_WARPS, T>;
                if (smem_size_phase1 > 48000) {
                    CUDA_CHECK(cudaFuncSetAttribute(kernel_phase1, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_phase1));
                }
                kernel_phase1<<<phase1_blocks, TB_SIZE, smem_size_phase1, main_stream>>>(
                    Q_data, K_data, mask_data,
                    phase_P_buffer, phase_meta_buffer,
                    n_heads, n_heads_kv, n_batch, len_q, len_kv,
                    scale, max_bias, m0, m1, n_head_log2, logit_softcap,
                    stride_Q_row, stride_Q_head, stride_Q_batch,
                    stride_K_row, stride_K_head, stride_K_batch,
                    stride_mask_row, stride_mask_batch);

                // Phase 2: P@V accumulation and output
                auto kernel_phase2 = attention_v5_phase2_pv_accum<BLOCK_Q, BLOCK_KV, DIM, NUM_WARPS, T, To>;
                if (smem_size_phase2 > 48000) {
                    CUDA_CHECK(cudaFuncSetAttribute(kernel_phase2, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_phase2));
                }
                kernel_phase2<<<phase2_blocks, TB_SIZE, smem_size_phase2, main_stream>>>(
                    phase_P_buffer, phase_meta_buffer, V_data, dst_data,
                    n_heads, n_heads_kv, n_batch, len_q, len_kv, total_kv_iters,
                    stride_V_row, stride_V_head, stride_V_batch,
                    stride_O_row, stride_O_head, stride_O_batch);

            } else if (use_warp_spec) {
                // Warp-specialized kernel: producer-consumer pipeline, no split-k support
                auto kernel_ws = attention_v5_warp_specialized_kernel<BLOCK_Q, BLOCK_KV, DIM, NUM_WARPS, T, To>;
                if (smem_size_warp_spec > 48000) {
                    CUDA_CHECK(cudaFuncSetAttribute(kernel_ws, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_warp_spec));
                }
                kernel_ws<<<base_blocks, TB_SIZE, smem_size_warp_spec, main_stream>>>(
                    Q_data, K_data, V_data, mask_data, dst_data,
                    n_heads, n_heads_kv, n_batch, len_q, len_kv,
                    scale, max_bias, m0, m1, n_head_log2, logit_softcap,
                    stride_Q_row, stride_Q_head, stride_Q_batch,
                    stride_K_row, stride_K_head, stride_K_batch,
                    stride_V_row, stride_V_head, stride_V_batch,
                    stride_O_row, stride_O_head, stride_O_batch,
                    stride_mask_row, stride_mask_batch);
            } else {
                // Standard kernel with optional split-k
                auto kernel = attention_v5_kernel<BLOCK_Q, BLOCK_KV, DIM, NUM_WARPS, T, To>;
                if (smem_size > 48000) {
                    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
                }
                kernel<<<num_blocks, TB_SIZE, smem_size, main_stream>>>(
                    Q_data, K_data, V_data, mask_data, dst_data,
                    n_heads, n_heads_kv, n_batch, len_q, len_kv,
                    scale, max_bias, m0, m1, n_head_log2, logit_softcap,
                    stride_Q_row, stride_Q_head, stride_Q_batch,
                    stride_K_row, stride_K_head, stride_K_batch,
                    stride_V_row, stride_V_head, stride_V_batch,
                    stride_O_row, stride_O_head, stride_O_batch,
                    stride_mask_row, stride_mask_batch,
                    split_k, partial_O, partial_meta);

                // Launch reduction kernel if split_k > 1
                if (split_k > 1) {
                    dim3 reduce_blocks(total_tiles);
                    dim3 reduce_threads(BLOCK_Q, DIM / 8);  // 64 x 16 = 1024 threads (SM_120 max)
                    attention_v5_splitk_reduce<DIM, BLOCK_Q, To>
                        <<<reduce_blocks, reduce_threads, 0, main_stream>>>(
                            partial_O, partial_meta, dst_data,
                            n_heads, n_batch, len_q, split_k,
                            stride_O_row, stride_O_head, stride_O_batch);
                }
            }

        } else if (dst->type == GGML_TYPE_BF16) {
            using To = nv_bfloat16;
            To * dst_data = (To *)dst->data;
            // OUTPUT LAYOUT: [D, heads, seq, batch] - swapped from input [D, seq, heads, batch]
            const int stride_O_row   = dst->nb[2] / sizeof(To);  // seq is dim 2 in output
            const int stride_O_head  = dst->nb[1] / sizeof(To);  // heads is dim 1 in output
            const int stride_O_batch = dst->nb[3] / sizeof(To);

            if (use_two_phase) {
                // Two-phase kernel for BF16 output
                const int phase1_blocks = n_batch * n_heads * num_q_blocks * total_kv_iters;
                const int phase2_blocks = base_blocks;

                auto kernel_phase1 = attention_v5_phase1_qk_softmax<BLOCK_Q, BLOCK_KV, DIM, NUM_WARPS, T>;
                if (smem_size_phase1 > 48000) {
                    CUDA_CHECK(cudaFuncSetAttribute(kernel_phase1, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_phase1));
                }
                kernel_phase1<<<phase1_blocks, TB_SIZE, smem_size_phase1, main_stream>>>(
                    Q_data, K_data, mask_data,
                    phase_P_buffer, phase_meta_buffer,
                    n_heads, n_heads_kv, n_batch, len_q, len_kv,
                    scale, max_bias, m0, m1, n_head_log2, logit_softcap,
                    stride_Q_row, stride_Q_head, stride_Q_batch,
                    stride_K_row, stride_K_head, stride_K_batch,
                    stride_mask_row, stride_mask_batch);

                auto kernel_phase2 = attention_v5_phase2_pv_accum<BLOCK_Q, BLOCK_KV, DIM, NUM_WARPS, T, To>;
                if (smem_size_phase2 > 48000) {
                    CUDA_CHECK(cudaFuncSetAttribute(kernel_phase2, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_phase2));
                }
                kernel_phase2<<<phase2_blocks, TB_SIZE, smem_size_phase2, main_stream>>>(
                    phase_P_buffer, phase_meta_buffer, V_data, dst_data,
                    n_heads, n_heads_kv, n_batch, len_q, len_kv, total_kv_iters,
                    stride_V_row, stride_V_head, stride_V_batch,
                    stride_O_row, stride_O_head, stride_O_batch);

            } else if (use_warp_spec) {
                // Warp-specialized kernel: producer-consumer pipeline, no split-k support
                auto kernel_ws = attention_v5_warp_specialized_kernel<BLOCK_Q, BLOCK_KV, DIM, NUM_WARPS, T, To>;
                if (smem_size_warp_spec > 48000) {
                    CUDA_CHECK(cudaFuncSetAttribute(kernel_ws, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_warp_spec));
                }
                kernel_ws<<<base_blocks, TB_SIZE, smem_size_warp_spec, main_stream>>>(
                    Q_data, K_data, V_data, mask_data, dst_data,
                    n_heads, n_heads_kv, n_batch, len_q, len_kv,
                    scale, max_bias, m0, m1, n_head_log2, logit_softcap,
                    stride_Q_row, stride_Q_head, stride_Q_batch,
                    stride_K_row, stride_K_head, stride_K_batch,
                    stride_V_row, stride_V_head, stride_V_batch,
                    stride_O_row, stride_O_head, stride_O_batch,
                    stride_mask_row, stride_mask_batch);
            } else {
                // Standard kernel with optional split-k
                auto kernel = attention_v5_kernel<BLOCK_Q, BLOCK_KV, DIM, NUM_WARPS, T, To>;
                if (smem_size > 48000) {
                    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
                }
                kernel<<<num_blocks, TB_SIZE, smem_size, main_stream>>>(
                    Q_data, K_data, V_data, mask_data, dst_data,
                    n_heads, n_heads_kv, n_batch, len_q, len_kv,
                    scale, max_bias, m0, m1, n_head_log2, logit_softcap,
                    stride_Q_row, stride_Q_head, stride_Q_batch,
                    stride_K_row, stride_K_head, stride_K_batch,
                    stride_V_row, stride_V_head, stride_V_batch,
                    stride_O_row, stride_O_head, stride_O_batch,
                    stride_mask_row, stride_mask_batch,
                    split_k, partial_O, partial_meta);

                // Launch reduction kernel if split_k > 1
                if (split_k > 1) {
                    dim3 reduce_blocks(total_tiles);
                    dim3 reduce_threads(BLOCK_Q, DIM / 8);  // 64 x 16 = 1024 threads (SM_120 max)
                    attention_v5_splitk_reduce<DIM, BLOCK_Q, To>
                        <<<reduce_blocks, reduce_threads, 0, main_stream>>>(
                            partial_O, partial_meta, dst_data,
                            n_heads, n_batch, len_q, split_k,
                            stride_O_row, stride_O_head, stride_O_batch);
                }
            }

        } else if (dst->type == GGML_TYPE_F16) {
            using To = half;
            To * dst_data = (To *)dst->data;
            // OUTPUT LAYOUT: [D, heads, seq, batch] - swapped from input [D, seq, heads, batch]
            const int stride_O_row   = dst->nb[2] / sizeof(To);  // seq is dim 2 in output
            const int stride_O_head  = dst->nb[1] / sizeof(To);  // heads is dim 1 in output
            const int stride_O_batch = dst->nb[3] / sizeof(To);

            if (use_two_phase) {
                // Two-phase kernel for F16 output
                const int phase1_blocks = n_batch * n_heads * num_q_blocks * total_kv_iters;
                const int phase2_blocks = base_blocks;

                auto kernel_phase1 = attention_v5_phase1_qk_softmax<BLOCK_Q, BLOCK_KV, DIM, NUM_WARPS, T>;
                if (smem_size_phase1 > 48000) {
                    CUDA_CHECK(cudaFuncSetAttribute(kernel_phase1, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_phase1));
                }
                kernel_phase1<<<phase1_blocks, TB_SIZE, smem_size_phase1, main_stream>>>(
                    Q_data, K_data, mask_data,
                    phase_P_buffer, phase_meta_buffer,
                    n_heads, n_heads_kv, n_batch, len_q, len_kv,
                    scale, max_bias, m0, m1, n_head_log2, logit_softcap,
                    stride_Q_row, stride_Q_head, stride_Q_batch,
                    stride_K_row, stride_K_head, stride_K_batch,
                    stride_mask_row, stride_mask_batch);

                auto kernel_phase2 = attention_v5_phase2_pv_accum<BLOCK_Q, BLOCK_KV, DIM, NUM_WARPS, T, To>;
                if (smem_size_phase2 > 48000) {
                    CUDA_CHECK(cudaFuncSetAttribute(kernel_phase2, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_phase2));
                }
                kernel_phase2<<<phase2_blocks, TB_SIZE, smem_size_phase2, main_stream>>>(
                    phase_P_buffer, phase_meta_buffer, V_data, dst_data,
                    n_heads, n_heads_kv, n_batch, len_q, len_kv, total_kv_iters,
                    stride_V_row, stride_V_head, stride_V_batch,
                    stride_O_row, stride_O_head, stride_O_batch);

            } else if (use_warp_spec) {
                // Warp-specialized kernel: producer-consumer pipeline, no split-k support
                auto kernel_ws = attention_v5_warp_specialized_kernel<BLOCK_Q, BLOCK_KV, DIM, NUM_WARPS, T, To>;
                if (smem_size_warp_spec > 48000) {
                    CUDA_CHECK(cudaFuncSetAttribute(kernel_ws, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_warp_spec));
                }
                kernel_ws<<<base_blocks, TB_SIZE, smem_size_warp_spec, main_stream>>>(
                    Q_data, K_data, V_data, mask_data, dst_data,
                    n_heads, n_heads_kv, n_batch, len_q, len_kv,
                    scale, max_bias, m0, m1, n_head_log2, logit_softcap,
                    stride_Q_row, stride_Q_head, stride_Q_batch,
                    stride_K_row, stride_K_head, stride_K_batch,
                    stride_V_row, stride_V_head, stride_V_batch,
                    stride_O_row, stride_O_head, stride_O_batch,
                    stride_mask_row, stride_mask_batch);
            } else {
                // Standard kernel with optional split-k
                auto kernel = attention_v5_kernel<BLOCK_Q, BLOCK_KV, DIM, NUM_WARPS, T, To>;
                if (smem_size > 48000) {
                    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
                }
                kernel<<<num_blocks, TB_SIZE, smem_size, main_stream>>>(
                    Q_data, K_data, V_data, mask_data, dst_data,
                    n_heads, n_heads_kv, n_batch, len_q, len_kv,
                    scale, max_bias, m0, m1, n_head_log2, logit_softcap,
                    stride_Q_row, stride_Q_head, stride_Q_batch,
                    stride_K_row, stride_K_head, stride_K_batch,
                    stride_V_row, stride_V_head, stride_V_batch,
                    stride_O_row, stride_O_head, stride_O_batch,
                    stride_mask_row, stride_mask_batch,
                    split_k, partial_O, partial_meta);

                // Launch reduction kernel if split_k > 1
                if (split_k > 1) {
                    dim3 reduce_blocks(total_tiles);
                    dim3 reduce_threads(BLOCK_Q, DIM / 8);  // 64 x 16 = 1024 threads (SM_120 max)
                    attention_v5_splitk_reduce<DIM, BLOCK_Q, To>
                        <<<reduce_blocks, reduce_threads, 0, main_stream>>>(
                            partial_O, partial_meta, dst_data,
                            n_heads, n_batch, len_q, split_k,
                            stride_O_row, stride_O_head, stride_O_batch);
                }
            }
        } else {
            GGML_ABORT("attention_v5: unsupported output type");
        }
    };

    struct type_tag_bf16 { using type = nv_bfloat16; };
    struct type_tag_f16 { using type = half; };

    // Dispatch based on input type
    // Note: Type consistency (Q/K/V same type, BF16 or F16) is validated by
    // ggml_cuda_flash_attn_ext_attention_v5_supported() before dispatch
    if (Q->type == GGML_TYPE_BF16) {
        launch_kernel_with_output_type(type_tag_bf16{});
    } else {
        launch_kernel_with_output_type(type_tag_f16{});
    }

    CUDA_CHECK(cudaGetLastError());
}

// Check if attention_v5 kernel is supported for this configuration
// Note: Blackwell (sm_120) check is done in fattn.cu before calling this
bool ggml_cuda_flash_attn_ext_attention_v5_supported(const ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    // Only supports head_dim=128
    if (Q->ne[0] != 128) {
        return false;
    }

    // All tensors must have the same type
    if (Q->type != K->type || K->type != V->type) {
        return false;
    }

    // Supports both BF16 (native Blackwell) and F16 (for quantized models like Q8_0)
    if (Q->type != GGML_TYPE_BF16 && Q->type != GGML_TYPE_F16) {
        return false;
    }

    // Output must be supported
    if (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_F16 && dst->type != GGML_TYPE_BF16) {
        return false;
    }

    // Innermost dimension must be contiguous for vectorized loads (cp.async.cg 16-byte)
    if (Q->nb[0] != ggml_type_size(Q->type) ||
        K->nb[0] != ggml_type_size(K->type) ||
        V->nb[0] != ggml_type_size(V->type)) {
        return false;
    }

    // GQA support: n_heads_q must be divisible by n_heads_kv
    // After permute(0,2,1,3), heads is in dimension 2
    if (Q->ne[2] % K->ne[2] != 0) {
        return false;
    }

    return true;
}
