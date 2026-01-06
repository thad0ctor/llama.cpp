// attention_v5.cu - SM_120 (Blackwell) optimized Flash Attention kernel
// Uses BF16/F16 tensor cores with m16n8k16 MMA instructions
//
// This kernel is dispatched at runtime when cc >= 1200 (Blackwell)
// Supports both BF16 (native) and F16 (for Q8_0 and other quantized models)

#include "common.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <float.h>
#include <type_traits>

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
  int stride_mask_batch)// mask->nb[3] / sizeof(half) - stride between batches
{
  constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;

  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;

  // each threadblock handles 1 BLOCK_Q
  const int bs = n_heads * n_batch;
  const int num_q_blocks = cdiv(len_q, BLOCK_Q);
  const int bs_id = bid / num_q_blocks;
  const int q_block_id = bid % num_q_blocks;

  // Decompose bs_id into head and batch indices
  const int head_id = bs_id % n_heads;
  const int batch_id = bs_id / n_heads;

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

  // we overlap Q_smem with (K_smem + V_smem), since we only need to load Q_smem once
  // Note: T is either half or nv_bfloat16, both are 16 bits
  extern __shared__ char smem_raw[];
  T * smem = reinterpret_cast<T *>(smem_raw);
  const uint32_t Q_smem = __cvta_generic_to_shared(smem);
  const uint32_t K_smem = Q_smem;  // double buffer for K
  const uint32_t V_smem = K_smem + 2 * BLOCK_KV * DIM * sizeof(T);
  // Pre-converted mask buffer: BLOCK_Q * BLOCK_KV floats = 64 * 64 * 4 = 16KB
  // Placed after V_smem (which is BLOCK_KV * DIM * sizeof(T) = 64 * 128 * 2 = 16KB)
  const uint32_t mask_f32_smem = V_smem + BLOCK_KV * DIM * sizeof(T);

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

  const int num_kv_iter = cdiv(len_kv, BLOCK_KV);

  auto load_K = [&](int kv_id) {
    if (kv_id < num_kv_iter) {
      // double buffer for K - use stride_K_row for non-contiguous support
      const uint32_t dst = K_smem + (kv_id % 2) * (BLOCK_KV * DIM * sizeof(T));
      global_to_shared_swizzle<BLOCK_KV, DIM, TB_SIZE, T>(dst, K, stride_K_row, tid, len_kv - kv_id * BLOCK_KV);
      K += BLOCK_KV * stride_K_row;  // advance by actual stride, not assumed contiguous
    }
    asm volatile("cp.async.commit_group;");
  };
  auto load_V = [&](int kv_id) {
    // single buffer for V - use stride_V_row for non-contiguous support
    const uint32_t dst = V_smem;
    global_to_shared_swizzle<BLOCK_KV, DIM, TB_SIZE, T>(dst, V, stride_V_row, tid, len_kv - kv_id * BLOCK_KV);
    V += BLOCK_KV * stride_V_row;  // advance by actual stride, not assumed contiguous
    asm volatile("cp.async.commit_group;");
  };

  // prefetch K
  load_K(0);

  for (int kv_id = 0; kv_id < num_kv_iter; kv_id++) {
    float S_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_N][4] = {};

    // prefetch V
    // __syncthreads() here is required to make sure we finish using V_smem
    // from the previous iteration, since there is only 1 shared buffer for V.
    __syncthreads();
    load_V(kv_id);

    // K shared -> registers
    asm volatile("cp.async.wait_group 1;");
    __syncthreads();

    // PREFETCH: Start loading next K block early to hide memory latency
    // This overlaps next K load with current iteration's compute
    if (kv_id + 1 < num_kv_iter) {
      load_K(kv_id + 1);
    }

    for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)
      for (int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d += 2) {
        uint32_t addr = K_smem_thread + (kv_id % 2) * (BLOCK_KV * DIM * sizeof(T));
        addr += mma_id_kv * MMA_N * DIM * sizeof(T);  // row
        addr ^= mma_id_d * MMA_K * sizeof(T);  // col
        ldmatrix_x4(K_rmem[mma_id_kv][mma_id_d], addr);
      }

    // Pre-convert mask from F16 to F32 in shared memory (once per KV iteration)
    // This eliminates ~12,288 redundant __half2float conversions per iteration
    // Uses __half22float2 to process pairs of F16 values for 2x conversion throughput
    float * mask_f32 = reinterpret_cast<float *>(__cvta_shared_to_generic(mask_f32_smem));
    if (mask_block) {
      const int kv_base = kv_id * BLOCK_KV;
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
        const int q_row_base = warp_id * WARP_Q + mma_id_q * MMA_M + (lane_id / 4);
        const int kv_col_base = kv_id * BLOCK_KV + mma_id_kv * MMA_N + (lane_id % 4) * 2;

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
            const int kv_local = kv_col - kv_id * BLOCK_KV;
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

    // V shared -> registers
    asm volatile("cp.async.wait_group 1;");
    __syncthreads();
    for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++)
      for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d += 2) {
        uint32_t addr = V_smem_thread;
        addr += mma_id_kv * MMA_K * DIM * sizeof(T);  // row
        addr ^= mma_id_d * MMA_N * sizeof(T);  // col
        ldmatrix_x4_trans(V_rmem[mma_id_kv][mma_id_d], addr);
      }

    // MMA O += P @ V [BLOCK_Q, DIM]
    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
      for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++)
        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++)
          mma_m16n8k16<T>(P_rmem[mma_id_q][mma_id_kv],
                          V_rmem[mma_id_kv][mma_id_d],
                          O_rmem[mma_id_q][mma_id_d]);
  }

  // write to O (output is always F32 for flash attention)
  // Use stride_O_row for non-contiguous output support
  for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
    for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++) {
      const int row = warp_id * WARP_Q + mma_id_q * MMA_M + (lane_id / 4);
      const int col = mma_id_d * MMA_N + (lane_id % 4) * 2;

      // divide by softmax denominator
      float *regs = O_rmem[mma_id_q][mma_id_d];
      regs[0] /= rowsumexp[mma_id_q][0];
      regs[1] /= rowsumexp[mma_id_q][0];
      regs[2] /= rowsumexp[mma_id_q][1];
      regs[3] /= rowsumexp[mma_id_q][1];

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

    const int bs = n_heads * n_batch;
    const int num_blocks = bs * ((len_q + BLOCK_Q - 1) / BLOCK_Q);
    const int TB_SIZE = NUM_WARPS * WARP_SIZE;

    // Get the CUDA stream from context (critical for multi-GPU correctness)
    cudaStream_t main_stream = ctx.stream();

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

        // Shared memory layout:
        // - K double buffer: 2 * BLOCK_KV * DIM * sizeof(T) = 2 * 64 * 128 * 2 = 32KB
        // - V buffer: BLOCK_KV * DIM * sizeof(T) = 64 * 128 * 2 = 16KB
        // - Mask F32 buffer: BLOCK_Q * BLOCK_KV * sizeof(float) = 64 * 64 * 4 = 16KB
        // Total: 32KB + 16KB + 16KB = 64KB (well within 99KB limit)
        const int smem_size = max(BLOCK_Q, BLOCK_KV * 3) * DIM * sizeof(T) + BLOCK_Q * BLOCK_KV * sizeof(float);

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
                stride_mask_row, stride_mask_batch);

        } else if (dst->type == GGML_TYPE_BF16) {
            using To = nv_bfloat16;
            To * dst_data = (To *)dst->data;
            // OUTPUT LAYOUT: [D, heads, seq, batch] - swapped from input [D, seq, heads, batch]
            const int stride_O_row   = dst->nb[2] / sizeof(To);  // seq is dim 2 in output
            const int stride_O_head  = dst->nb[1] / sizeof(To);  // heads is dim 1 in output
            const int stride_O_batch = dst->nb[3] / sizeof(To);

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
                stride_mask_row, stride_mask_batch);

        } else if (dst->type == GGML_TYPE_F16) {
            using To = half;
            To * dst_data = (To *)dst->data;
            // OUTPUT LAYOUT: [D, heads, seq, batch] - swapped from input [D, seq, heads, batch]
            const int stride_O_row   = dst->nb[2] / sizeof(To);  // seq is dim 2 in output
            const int stride_O_head  = dst->nb[1] / sizeof(To);  // heads is dim 1 in output
            const int stride_O_batch = dst->nb[3] / sizeof(To);

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
                stride_mask_row, stride_mask_batch);
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
