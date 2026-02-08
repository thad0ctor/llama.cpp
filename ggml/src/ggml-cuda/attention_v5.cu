// attention_v5.cu - SM_120 (Blackwell) optimized Flash Attention kernel
// Uses BF16 tensor cores with m16n8k16 MMA instructions
//
// This kernel is dispatched at runtime when cc >= 1200 (Blackwell)

#include "common.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdlib>
#include <float.h>

// Use WARP_SIZE from common.cuh
// Use CUDA_CHECK from common.cuh

// Ceiling division helper
__device__ __host__ constexpr
int attn_v5_cdiv(int a, int b) { return (a + b - 1) / b; }

// Rename cdiv usages to attn_v5_cdiv to avoid conflicts
#define cdiv attn_v5_cdiv

// Fast tanh approximation using polynomial (7th order, accurate to ~1e-5 in [-4,4])
// Avoids expensive tanhf() in critical path when logit_softcap is enabled
__device__ __forceinline__
float tanh_fast(float x) {
    // Clamp to avoid overflow in polynomial
    x = fmaxf(-4.0f, fminf(4.0f, x));
    const float x2 = x * x;
    // Coefficients for tanh Taylor series approximation
    return x * (1.0f + x2 * (-0.333333333f + x2 * (0.133333333f + x2 * (-0.053968254f))));
}

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

template <int HEIGHT, int WIDTH, int TB_SIZE>
__device__ inline
void global_to_shared(uint32_t dst, const nv_bfloat16 *src, int src_stride, int tid) {
  constexpr int num_elems = 16 / sizeof(nv_bfloat16);
  constexpr int num_iters = HEIGHT * WIDTH / (TB_SIZE * num_elems);

  for (int iter = 0; iter < num_iters; iter++) {
    const int idx = (iter * TB_SIZE + tid) * num_elems;
    const int row = idx / WIDTH;
    const int col = idx % WIDTH;

    const uint32_t dst_addr = dst + (row * WIDTH + col) * sizeof(nv_bfloat16);
    const nv_bfloat16 *src_addr = src + (row * src_stride + col);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst_addr), "l"(src_addr));
  }
}

template <int HEIGHT, int WIDTH, int TB_SIZE>
__device__ inline
void global_to_shared_swizzle(uint32_t dst, const nv_bfloat16 *src, int src_stride, int tid) {
  constexpr int num_elems = 16 / sizeof(nv_bfloat16);
  constexpr int num_iters = HEIGHT * WIDTH / (TB_SIZE * num_elems);

  for (int iter = 0; iter < num_iters; iter++) {
    const int idx = (iter * TB_SIZE + tid) * num_elems;
    const int row = idx / WIDTH;
    const int col = idx % WIDTH;

    const uint32_t dst_addr = swizzle<WIDTH * sizeof(nv_bfloat16)>(dst + (row * WIDTH + col) * sizeof(nv_bfloat16));
    const nv_bfloat16 *src_addr = src + (row * src_stride + col);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst_addr), "l"(src_addr));
  }
}

// Variant that pre-applies scale during Q load (fused load + scale)
// Uses cp.async to preserve overlap, then scales in shared memory.
// This amortizes the scale multiplication cost since Q is loaded once and used many times.
template <int HEIGHT, int WIDTH, int TB_SIZE>
__device__ inline
void global_to_shared_swizzle_scaled(uint32_t dst, const nv_bfloat16 *src, int src_stride, int tid, float scale) {
  constexpr int num_elems = 16 / sizeof(nv_bfloat16);
  constexpr int num_iters = HEIGHT * WIDTH / (TB_SIZE * num_elems);
  const nv_bfloat162 scale_bf2 = __float2bfloat162_rn(scale);

  // Async load to shared (same layout as global_to_shared_swizzle)
  for (int iter = 0; iter < num_iters; iter++) {
    const int idx = (iter * TB_SIZE + tid) * num_elems;
    const int row = idx / WIDTH;
    const int col = idx % WIDTH;

    const uint32_t dst_addr = swizzle<WIDTH * sizeof(nv_bfloat16)>(dst + (row * WIDTH + col) * sizeof(nv_bfloat16));
    const nv_bfloat16 *src_addr = src + (row * src_stride + col);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst_addr), "l"(src_addr));
  }
  asm volatile("cp.async.commit_group;");
  asm volatile("cp.async.wait_all;");
  __syncthreads();

  // Scale in shared memory (vectorized)
  #pragma unroll
  for (int iter = 0; iter < num_iters; iter++) {
    const int idx = (iter * TB_SIZE + tid) * num_elems;
    const int row = idx / WIDTH;
    const int col = idx % WIDTH;

    const uint32_t dst_addr = swizzle<WIDTH * sizeof(nv_bfloat16)>(dst + (row * WIDTH + col) * sizeof(nv_bfloat16));
    nv_bfloat16 *dst_ptr = reinterpret_cast<nv_bfloat16 *>(__cvta_shared_to_generic(dst_addr));

    nv_bfloat162 data0 = *reinterpret_cast<const nv_bfloat162*>(dst_ptr + 0);
    nv_bfloat162 data1 = *reinterpret_cast<const nv_bfloat162*>(dst_ptr + 2);
    nv_bfloat162 data2 = *reinterpret_cast<const nv_bfloat162*>(dst_ptr + 4);
    nv_bfloat162 data3 = *reinterpret_cast<const nv_bfloat162*>(dst_ptr + 6);

    data0 = __hmul2(data0, scale_bf2);
    data1 = __hmul2(data1, scale_bf2);
    data2 = __hmul2(data2, scale_bf2);
    data3 = __hmul2(data3, scale_bf2);

    *reinterpret_cast<nv_bfloat162*>(dst_ptr + 0) = data0;
    *reinterpret_cast<nv_bfloat162*>(dst_ptr + 2) = data1;
    *reinterpret_cast<nv_bfloat162*>(dst_ptr + 4) = data2;
    *reinterpret_cast<nv_bfloat162*>(dst_ptr + 6) = data3;
  }
  __syncthreads();
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

__device__ inline
void mma_m16n8k16(uint32_t A[4], uint32_t B[2], float D[4]) {
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

template<int BLOCK_Q, int BLOCK_KV, int DIM, int NUM_WARPS>
__launch_bounds__(NUM_WARPS * WARP_SIZE, 2)
__global__
void attention_v5_kernel(
  const nv_bfloat16 * __restrict__ Q,  // [bs, len_q, DIM]
  const nv_bfloat16 * __restrict__ K,  // [bs, len_kv, DIM]
  const nv_bfloat16 * __restrict__ V,  // [bs, len_kv, DIM]
  float * __restrict__ O,              // [bs, len_q, DIM]
  const char * __restrict__ mask,      // [len_q, len_kv]
  int bs,
  int len_q,
  int len_kv,
  int n_head,
  float scale,
  float max_bias,
  float m0,
  float m1,
  uint32_t n_head_log2,
  float logit_softcap,
  int ne32,
  int64_t nb32,
  int ne33,
  int64_t nb33,
  int stride_mask,
  int mask_is_f16,
  int is_causal) {

  constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;

  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;

  // each threadblock handles 1 BLOCK_Q
  const int num_q_blocks = cdiv(len_q, BLOCK_Q);
  const int bs_id = bid / num_q_blocks;
  const int q_block_id = bid % num_q_blocks;

  const int head_id  = bs_id % n_head;
  const int batch_id = bs_id / n_head;

  Q += (bs_id * num_q_blocks + q_block_id) * BLOCK_Q * DIM;
  K += bs_id * len_kv * DIM;
  V += bs_id * len_kv * DIM;
  // O uses layout [batch, seq, head, dim] (i.e. [D, heads, seq, batch] in ggml).

  // we overlap Q_smem with (K_smem + V_smem), since we only need to load Q_smem once
  extern __shared__ nv_bfloat16 smem[];
  const uint32_t Q_smem = __cvta_generic_to_shared(smem);
  const uint32_t K_smem = Q_smem;  // double buffer for K
  const uint32_t V_smem = K_smem + 2 * BLOCK_KV * DIM * sizeof(nv_bfloat16);
  constexpr int smem_q_bytes = BLOCK_Q * DIM * sizeof(nv_bfloat16);
  constexpr int smem_kv_bytes = 4 * BLOCK_KV * DIM * sizeof(nv_bfloat16);
  constexpr int smem_base_bytes = smem_q_bytes > smem_kv_bytes ? smem_q_bytes : smem_kv_bytes;
  uint8_t * const mask_smem_base = reinterpret_cast<uint8_t *>(smem) + smem_base_bytes;
  half  * const mask_smem_h = reinterpret_cast<half *>(mask_smem_base);
  float * const mask_smem_f = reinterpret_cast<float *>(mask_smem_base);

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
  uint32_t Q_smem_thread, K_smem_thread, V_smem_thread;
  {
    // A tile
    const int row_off = warp_id * WARP_Q + (lane_id % 16);
    const int col_off = lane_id / 16 * 8;
    Q_smem_thread = swizzle<DIM * sizeof(nv_bfloat16)>(Q_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
  }
  {
    // B tile
    const int row_off = lane_id % 8;
    const int col_off = lane_id / 8 * 8;
    K_smem_thread = swizzle<DIM * sizeof(nv_bfloat16)>(K_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
  }
  {
    // B tile trans
    const int row_off = lane_id % 16;
    const int col_off = lane_id / 16 * 8;
    V_smem_thread = swizzle<DIM * sizeof(nv_bfloat16)>(V_smem + (row_off * DIM + col_off) * sizeof(nv_bfloat16));
  }

  const float softmax_scale = scale;

  float rowmax[WARP_Q / MMA_M][2];
  float rowsumexp[WARP_Q / MMA_M][2] = {};

  for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {
    rowmax[mma_id_q][0] = -FLT_MAX;
    rowmax[mma_id_q][1] = -FLT_MAX;
  }

  // load Q [BLOCK_Q, DIM] with scale pre-applied (fused)
  global_to_shared_swizzle_scaled<BLOCK_Q, DIM, TB_SIZE>(Q_smem, Q, DIM, tid, softmax_scale);

  // shared -> registers
  #pragma unroll
  for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
    #pragma unroll
    for (int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d++) {
      uint32_t addr = Q_smem_thread;
      addr += mma_id_q * MMA_M * DIM * sizeof(nv_bfloat16);  // row
      addr ^= mma_id_d * MMA_K * sizeof(nv_bfloat16);  // col
      ldmatrix_x4(Q_rmem[mma_id_q][mma_id_d], addr);
    }
  // we need a syncthreads() here so that we don't load K global->shared
  // before finishing loading Q shared->reg
  __syncthreads();

  const int num_kv_iter = cdiv(len_kv, BLOCK_KV);

  auto load_K = [&](int kv_id) {
    if (kv_id < num_kv_iter) {
      // double buffer for K
      const uint32_t dst = K_smem + (kv_id % 2) * (BLOCK_KV * DIM * sizeof(nv_bfloat16));
      global_to_shared_swizzle<BLOCK_KV, DIM, TB_SIZE>(dst, K, DIM, tid);
      K += BLOCK_KV * DIM;
    }
    asm volatile("cp.async.commit_group;");
  };
  auto load_V = [&](int kv_id) {
    // double buffer for V
    if (kv_id < num_kv_iter) {
      const uint32_t dst = V_smem + (kv_id % 2) * (BLOCK_KV * DIM * sizeof(nv_bfloat16));
      global_to_shared_swizzle<BLOCK_KV, DIM, TB_SIZE>(dst, V, DIM, tid);
      V += BLOCK_KV * DIM;
    }
    asm volatile("cp.async.commit_group;");
  };

  // prefetch K
  load_K(0);
  // prefetch V
  load_V(0);

  const bool has_mask = (mask != nullptr);
  const char * mask_base = has_mask ?
      (mask + nb33 * (batch_id % ne33) + nb32 * (head_id % ne32)) :
      nullptr;
  const float slope = (max_bias != 0.0f) ? get_alibi_slope(max_bias, head_id, n_head_log2, m0, m1) : 1.0f;

  for (int kv_id = 0; kv_id < num_kv_iter; kv_id++) {
    // Causal tile skipping optimization:
    // For causal attention, position (q, k) is masked if k > q.
    // If the entire KV block starts after the last Q position in this block,
    // all elements would be masked (-inf), so we can skip the entire tile.
    // This saves up to 2x compute for causal attention.
    if (is_causal) {
      const int kv_block_start = kv_id * BLOCK_KV;
      const int q_block_end = q_block_id * BLOCK_Q + BLOCK_Q - 1;  // last Q row in this block
      if (kv_block_start > q_block_end) {
        // All K positions are > all Q positions in this block, skip.
        // Keep K/V pointers and prefetch pipeline in sync.
        V += BLOCK_KV * DIM;
        load_K(kv_id + 1);
        continue;
      }
    }

    float S_rmem[WARP_Q / MMA_M][BLOCK_KV / MMA_N][4] = {};

    // K shared -> registers
    asm volatile("cp.async.wait_group 1;");
    __syncthreads();
    #pragma unroll
    for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)
      #pragma unroll
      for (int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d += 2) {
        uint32_t addr = K_smem_thread + (kv_id % 2) * (BLOCK_KV * DIM * sizeof(nv_bfloat16));
        addr += mma_id_kv * MMA_N * DIM * sizeof(nv_bfloat16);  // row
        addr ^= mma_id_d * MMA_K * sizeof(nv_bfloat16);  // col
        ldmatrix_x4(K_rmem[mma_id_kv][mma_id_d], addr);
      }

    // MMA S = Q @ K.T [BLOCK_Q, BLOCK_KV]
    #pragma unroll
    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
      #pragma unroll
      for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++)
        #pragma unroll
        for (int mma_id_d = 0; mma_id_d < DIM / MMA_K; mma_id_d++)
          mma_m16n8k16(Q_rmem[mma_id_q][mma_id_d],
                       K_rmem[mma_id_kv][mma_id_d],
                       S_rmem[mma_id_q][mma_id_kv]);

    // prefetch K
    load_K(kv_id + 1);

    // Note: softmax scale is now pre-applied to Q during load (fused),
    // so no scale multiplication needed here after GEMM1

    // apply logit softcap if enabled (before mask addition)
    // Uses fast polynomial approximation instead of expensive tanhf()
    if (logit_softcap != 0.0f) {
      #pragma unroll
      for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
        #pragma unroll
        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
          float *regs = S_rmem[mma_id_q][mma_id_kv];
          regs[0] = logit_softcap * tanh_fast(regs[0]);
          regs[1] = logit_softcap * tanh_fast(regs[1]);
          regs[2] = logit_softcap * tanh_fast(regs[2]);
          regs[3] = logit_softcap * tanh_fast(regs[3]);
        }
    }

    // load mask tile (if present)
    if (has_mask) {
      const int total = BLOCK_Q * BLOCK_KV;
      if (mask_is_f16) {
        const uint32_t mask_smem_addr = __cvta_generic_to_shared(mask_smem_h);
        constexpr int num_elems = 16 / sizeof(half);  // 8 half per 16B
        for (int idx = tid * num_elems; idx < total; idx += TB_SIZE * num_elems) {
          const int q_local = idx / BLOCK_KV;
          const int k_local = idx % BLOCK_KV;
          const int q = q_block_id * BLOCK_Q + q_local;
          const int k = kv_id * BLOCK_KV + k_local;
          const uint32_t dst_addr = mask_smem_addr + idx * sizeof(half);
          if (k_local <= BLOCK_KV - num_elems && q < len_q && k + (num_elems - 1) < len_kv) {
            const half *src = reinterpret_cast<const half *>(mask_base) + q * stride_mask + k;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst_addr), "l"(src));
          } else {
            for (int i = 0; i < num_elems; ++i) {
              const int kk = k + i;
              const int dst_idx = idx + i;
              if (dst_idx < total && q < len_q && kk < len_kv) {
                mask_smem_h[dst_idx] = reinterpret_cast<const half *>(mask_base)[q * stride_mask + kk];
              } else if (dst_idx < total) {
                mask_smem_h[dst_idx] = __float2half(0.0f);
              }
            }
          }
        }
        asm volatile("cp.async.commit_group;");
      } else {
        for (int idx = tid * 4; idx < total; idx += TB_SIZE * 4) {
          const int q_local = idx / BLOCK_KV;
          const int k_local = idx % BLOCK_KV;
          const int q = q_block_id * BLOCK_Q + q_local;
          const int k = kv_id * BLOCK_KV + k_local;
          if (k_local <= BLOCK_KV - 4 && q < len_q && k + 3 < len_kv) {
            const float4 v4 = *reinterpret_cast<const float4 *>(reinterpret_cast<const float *>(mask_base) + q * stride_mask + k);
            *reinterpret_cast<float4 *>(mask_smem_f + idx) = v4;
          } else {
            for (int i = 0; i < 4; ++i) {
              const int kk = k + i;
              const int dst_idx = idx + i;
              if (dst_idx < total && q < len_q && kk < len_kv) {
                mask_smem_f[dst_idx] = reinterpret_cast<const float *>(mask_base)[q * stride_mask + kk];
              } else if (dst_idx < total) {
                mask_smem_f[dst_idx] = 0.0f;
              }
            }
          }
        }
      }
    }

    if (has_mask && mask_is_f16) {
      // Keep at most the two newest async groups pending (e.g., V + K prefetch),
      // while ensuring the mask tile (older group) is resident in shared memory.
      asm volatile("cp.async.wait_group 2;");
    }
    __syncthreads();

    // apply mask / OOB (additive bias)
    #pragma unroll
    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
      #pragma unroll
      for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_N; mma_id_kv++) {
        float *regs = S_rmem[mma_id_q][mma_id_kv];

        const int row0 = warp_id * WARP_Q + mma_id_q * MMA_M + (lane_id / 4);
        const int row1 = row0 + 8;
        const int col0 = mma_id_kv * MMA_N + (lane_id % 4) * 2;
        const int col1 = col0 + 1;

        const int q0 = q_block_id * BLOCK_Q + row0;
        const int q1 = q0 + 8;
        const int k0 = kv_id * BLOCK_KV + col0;
        const int k1 = k0 + 1;

        float m00 = (q0 < len_q && k0 < len_kv) ? 0.0f : -INFINITY;
        float m01 = (q0 < len_q && k1 < len_kv) ? 0.0f : -INFINITY;
        float m10 = (q1 < len_q && k0 < len_kv) ? 0.0f : -INFINITY;
        float m11 = (q1 < len_q && k1 < len_kv) ? 0.0f : -INFINITY;

        if (has_mask) {
          if (q0 < len_q && k0 < len_kv) {
            m00 = mask_is_f16 ? __half2float(mask_smem_h[row0 * BLOCK_KV + col0]) :
                                mask_smem_f[row0 * BLOCK_KV + col0];
          }
          if (q0 < len_q && k1 < len_kv) {
            m01 = mask_is_f16 ? __half2float(mask_smem_h[row0 * BLOCK_KV + col1]) :
                                mask_smem_f[row0 * BLOCK_KV + col1];
          }
          if (q1 < len_q && k0 < len_kv) {
            m10 = mask_is_f16 ? __half2float(mask_smem_h[(row0 + 8) * BLOCK_KV + col0]) :
                                mask_smem_f[(row0 + 8) * BLOCK_KV + col0];
          }
          if (q1 < len_q && k1 < len_kv) {
            m11 = mask_is_f16 ? __half2float(mask_smem_h[(row0 + 8) * BLOCK_KV + col1]) :
                                mask_smem_f[(row0 + 8) * BLOCK_KV + col1];
          }
        }

        regs[0] += slope * m00;
        regs[1] += slope * m01;
        regs[2] += slope * m10;
        regs[3] += slope * m11;
      }

    #pragma unroll
    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++) {

      // rowmax
      float this_rowmax[2];
      #pragma unroll
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
      // On first iteration, rowmax is -FLT_MAX so exp(-FLT_MAX - x) = 0
      // which would incorrectly zero out the output. Skip rescaling on first iter.
      float rescale[2];
      if (kv_id == 0) {
        // First iteration: no previous output to rescale
        rescale[0] = 1.0f;
        rescale[1] = 1.0f;
      } else {
        rescale[0] = __expf(rowmax[mma_id_q][0] - this_rowmax[0]);
        rescale[1] = __expf(rowmax[mma_id_q][1] - this_rowmax[1]);
      }
      #pragma unroll
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
      #pragma unroll
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
        nv_bfloat162 *this_P_rmem = reinterpret_cast<nv_bfloat162 *>(P_rmem[mma_id_q][mma_id_kv / 2]);
        this_P_rmem[(mma_id_kv % 2) * 2]     = __float22bfloat162_rn({regs[0], regs[1]});
        this_P_rmem[(mma_id_kv % 2) * 2 + 1] = __float22bfloat162_rn({regs[2], regs[3]});
      }

      // butterfly reduction within 4 threads
      this_rowsumexp[0] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[0], 1);
      this_rowsumexp[0] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[0], 2);
      this_rowsumexp[1] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[1], 1);
      this_rowsumexp[1] += __shfl_xor_sync(0xFFFF'FFFF, this_rowsumexp[1], 2);

      // accumulate to total rowsumexp
      // On first iteration, rowsumexp is 0, so just assign this_rowsumexp directly
      if (kv_id == 0) {
        rowsumexp[mma_id_q][0] = this_rowsumexp[0];
        rowsumexp[mma_id_q][1] = this_rowsumexp[1];
      } else {
        rowsumexp[mma_id_q][0] = rowsumexp[mma_id_q][0] * rescale[0] + this_rowsumexp[0];
        rowsumexp[mma_id_q][1] = rowsumexp[mma_id_q][1] * rescale[1] + this_rowsumexp[1];
      }
    }

    // V shared -> registers
    asm volatile("cp.async.wait_group 1;");
    __syncthreads();
    #pragma unroll
    for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++)
      #pragma unroll
      for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d += 2) {
        uint32_t addr = V_smem_thread + (kv_id % 2) * (BLOCK_KV * DIM * sizeof(nv_bfloat16));
        addr += mma_id_kv * MMA_K * DIM * sizeof(nv_bfloat16);  // row
        addr ^= mma_id_d * MMA_N * sizeof(nv_bfloat16);  // col
        ldmatrix_x4_trans(V_rmem[mma_id_kv][mma_id_d], addr);
      }

    // prefetch V
    load_V(kv_id + 1);

    // MMA O += P @ V [BLOCK_Q, DIM]
    #pragma unroll
    for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
      #pragma unroll
      for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++)
        #pragma unroll
        for (int mma_id_kv = 0; mma_id_kv < BLOCK_KV / MMA_K; mma_id_kv++)
          mma_m16n8k16(P_rmem[mma_id_q][mma_id_kv],
                       V_rmem[mma_id_kv][mma_id_d],
                       O_rmem[mma_id_q][mma_id_d]);
  }

  // write to O
  #pragma unroll
  for (int mma_id_q = 0; mma_id_q < WARP_Q / MMA_M; mma_id_q++)
    #pragma unroll
    for (int mma_id_d = 0; mma_id_d < DIM / MMA_N; mma_id_d++) {
      const int row = warp_id * WARP_Q + mma_id_q * MMA_M + (lane_id / 4);
      const int col = mma_id_d * MMA_N + (lane_id % 4) * 2;

      // divide by softmax denominator
      float *regs = O_rmem[mma_id_q][mma_id_d];
      const float denom0 = fmaxf(rowsumexp[mma_id_q][0], 1e-6f);
      const float denom1 = fmaxf(rowsumexp[mma_id_q][1], 1e-6f);
      regs[0] /= denom0;
      regs[1] /= denom0;
      regs[2] /= denom1;
      regs[3] /= denom1;

      const int global_row0 = q_block_id * BLOCK_Q + row;
      const int global_row1 = global_row0 + 8;
      if (global_row0 < len_q) {
        float *dst_row0 = O + ((batch_id * len_q + global_row0) * n_head + head_id) * DIM + col;
        dst_row0[0] = regs[0];
        dst_row0[1] = regs[1];
      }
      if (global_row1 < len_q) {
        float *dst_row1 = O + ((batch_id * len_q + global_row1) * n_head + head_id) * DIM + col;
        dst_row1[0] = regs[2];
        dst_row1[1] = regs[3];
      }
    }
}

void attention_v5(
  const nv_bfloat16 *Q,  // [bs, len_q, DIM]
  const nv_bfloat16 *K,  // [bs, len_kv, DIM]
  const nv_bfloat16 *V,  // [bs, len_kv, DIM]
  float *O,              // [bs, len_q, DIM]
  const char *mask,      // [len_q, len_kv]
  int bs,
  int len_q,
  int len_kv,
  int dim,
  int n_head,
  float scale,
  float max_bias,
  float m0,
  float m1,
  uint32_t n_head_log2,
  float logit_softcap,
  int ne32,
  int64_t nb32,
  int ne33,
  int64_t nb33,
  int stride_mask,
  int mask_is_f16,
  int is_causal) {

  GGML_ASSERT(dim == 128 && "attention_v5 only supports dim=128");

  const int BLOCK_Q = 64;
  const int BLOCK_KV = 128;
  const int DIM = 128;
  const int NUM_WARPS = 4;

  const int num_blocks = bs * cdiv(len_q, BLOCK_Q);
  const int TB_SIZE = NUM_WARPS * WARP_SIZE;
  const int smem_kv_bytes = 4 * BLOCK_KV * DIM * sizeof(nv_bfloat16);
  const int smem_q_bytes  = BLOCK_Q * DIM * sizeof(nv_bfloat16);
  const int smem_base_bytes = max(smem_kv_bytes, smem_q_bytes);
  const int mask_bytes = mask ? (BLOCK_Q * BLOCK_KV * (mask_is_f16 ? (int)sizeof(half) : (int)sizeof(float))) : 0;
  const int smem_size = smem_base_bytes + mask_bytes;

  auto kernel = attention_v5_kernel<BLOCK_Q, BLOCK_KV, DIM, NUM_WARPS>;
  launch_kernel(kernel, num_blocks, TB_SIZE, smem_size, Q, K, V, O, mask, bs, len_q, len_kv,
                n_head, scale, max_bias, m0, m1, n_head_log2, logit_softcap,
                ne32, nb32, ne33, nb33, stride_mask, mask_is_f16, is_causal);
}

// =============================================================================
// GGML Integration Wrapper for SM_120 Flash Attention
// =============================================================================

#include "fattn-common.cuh"

// Forward declaration for the ggml wrapper
void ggml_cuda_flash_attn_ext_attention_v5(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];

    const int64_t ne00 = Q->ne[0];  // head_dim (D)
    const int64_t ne01 = Q->ne[1];  // n_queries
    const int64_t ne02 = Q->ne[2];  // n_heads_q
    const int64_t ne03 = Q->ne[3];  // batch

    const int64_t ne10 = K->ne[0];  // head_dim (D)
    const int64_t ne11 = K->ne[1];  // n_kv (sequence length)
    const int64_t ne12 = K->ne[2];  // n_heads_kv

    GGML_ASSERT(ne00 == 128 && "attention_v5 only supports head_dim=128");
    GGML_ASSERT(Q->type == GGML_TYPE_BF16 && "attention_v5 requires BF16 input");
    GGML_ASSERT(K->type == GGML_TYPE_BF16 && "attention_v5 requires BF16 input");
    GGML_ASSERT(V->type == GGML_TYPE_BF16 && "attention_v5 requires BF16 input");
    GGML_ASSERT(dst->type == GGML_TYPE_F32 && "attention_v5 requires F32 output");
    GGML_ASSERT(!mask || mask->type == GGML_TYPE_F16 || mask->type == GGML_TYPE_F32);

    // Get raw pointers
    const nv_bfloat16 * Q_data = (const nv_bfloat16 *)Q->data;
    const nv_bfloat16 * K_data = (const nv_bfloat16 *)K->data;
    const nv_bfloat16 * V_data = (const nv_bfloat16 *)V->data;
    float * dst_data = (float *)dst->data;
    const char * mask_data = mask ? (const char *)mask->data : nullptr;

    // Calculate dimensions
    const int bs = ne02 * ne03;  // batch * n_heads
    const int len_q = ne01;
    const int len_kv = ne11;
    const int dim = ne00;

    float scale = 1.0f;
    float max_bias = 0.0f;
    float logit_softcap = 0.0f;

    memcpy(&scale,         (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }
    if (scale == 0.0f) {
        scale = 1.0f / sqrtf(float(ne00));
    }

    const uint32_t n_head = ne02;
    const uint32_t n_head_log2 = 1u << uint32_t(floorf(log2f(float(n_head))));
    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    const bool mask_is_f16 = mask && mask->type == GGML_TYPE_F16;
    const int stride_mask = mask ? (mask_is_f16 ? (mask->nb[1] / (int)sizeof(half)) : (mask->nb[1] / (int)sizeof(float))) : 0;
    const int ne32 = mask ? (int)mask->ne[2] : 1;
    const int ne33 = mask ? (int)mask->ne[3] : 1;
    const int64_t nb32 = mask ? mask->nb[2] : 0;
    const int64_t nb33 = mask ? mask->nb[3] : 0;

    // Causal tile skip only valid when Q and KV are position-aligned (len_q >= len_kv),
    // i.e. during prefill. During decode (len_q=1, len_kv=context_len), local Q
    // index 0 maps to the END of the sequence, so all KV positions are valid.
    const int is_causal = (mask != nullptr && len_q >= len_kv && getenv("GGML_CUDA_CAUSAL_SKIP") != nullptr) ? 1 : 0;

    // Call the kernel
    attention_v5(Q_data, K_data, V_data, dst_data, mask_data, bs, len_q, len_kv, dim,
                 n_head, scale, max_bias, m0, m1, n_head_log2, logit_softcap,
                 ne32, nb32, ne33, nb33, stride_mask, mask_is_f16 ? 1 : 0, is_causal);

    CUDA_CHECK(cudaGetLastError());
}

// Check if attention_v5 kernel is supported for this configuration
bool ggml_cuda_flash_attn_ext_attention_v5_supported(const ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    // Only supports head_dim=128
    if (Q->ne[0] != 128) {
        return false;
    }

    // Only supports BF16 (for now)
    if (Q->type != GGML_TYPE_BF16 || K->type != GGML_TYPE_BF16 || V->type != GGML_TYPE_BF16) {
        return false;
    }

    // Output must be F32
    if (dst->type != GGML_TYPE_F32) {
        return false;
    }

    // Mask type must be supported (if present)
    if (mask && mask->type != GGML_TYPE_F16 && mask->type != GGML_TYPE_F32) {
        return false;
    }

    return true;
}
