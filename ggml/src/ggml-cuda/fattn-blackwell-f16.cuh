#pragma once

#include "common.cuh"
#include "fattn-common.cuh"

// Blackwell-specific Flash Attention Kernel for F16 inputs (sm_120+)
// Architecture:
//   - Uses m16n8k16 MMA tensor core instructions (available sm_75+)
//   - Uses cp.async for efficient memory loading
//   - F16 inputs, F32 accumulators
//   - Online softmax with row-wise tracking
//
// This kernel handles F16 K/V inputs on Blackwell GPUs, complementing
// the attention_v5 kernel which handles BF16 inputs.

#if defined(CP_ASYNC_AVAILABLE) || !defined(__CUDA_ARCH__)

// Ceiling division helper
__device__ __host__ constexpr
int fattn_blackwell_cdiv(int a, int b) { return (a + b - 1) / b; }

// Fast tanh approximation using polynomial (7th order, accurate to ~1e-5 in [-4,4])
// Avoids expensive tanhf() in critical path when logit_softcap is enabled
__device__ __forceinline__
float tanh_fast(float x) {
    x = fmaxf(-4.0f, fminf(4.0f, x));
    const float x2 = x * x;
    return x * (1.0f + x2 * (-0.333333333f + x2 * (0.133333333f + x2 * (-0.053968254f))));
}

// ----------------------------------------------------------------------------
// Swizzle for bank-conflict-free shared memory access
// ----------------------------------------------------------------------------

// NOTE: stride in bytes
template <int STRIDE>
__device__
uint32_t fattn_swizzle(uint32_t index) {
    // No swizzling needed for small strides
    if constexpr (STRIDE == 16)
        return index;

    uint32_t row_idx = (index / STRIDE) % 8;
    uint32_t bits_to_xor = row_idx / max(64 / STRIDE, 1);
    return index ^ (bits_to_xor << 4);
}

// ----------------------------------------------------------------------------
// Memory loading helpers
// ----------------------------------------------------------------------------

template <int HEIGHT, int WIDTH, int TB_SIZE>
__device__ inline
void fattn_global_to_shared_swizzle(uint32_t dst, const half *src, int src_stride, int tid, int valid_rows) {
    constexpr int num_elems = 16 / sizeof(half);  // 8 half elements per 16-byte load
    constexpr int num_iters = HEIGHT * WIDTH / (TB_SIZE * num_elems);

    #pragma unroll
    for (int iter = 0; iter < num_iters; iter++) {
        const int idx = (iter * TB_SIZE + tid) * num_elems;
        const int row = idx / WIDTH;
        const int col = idx % WIDTH;

        const uint32_t dst_addr = fattn_swizzle<WIDTH * sizeof(half)>(dst + (row * WIDTH + col) * sizeof(half));
        if (row < valid_rows) {
            const half *src_addr = src + (row * src_stride + col);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst_addr), "l"(src_addr));
        } else {
            half *dst_ptr = reinterpret_cast<half *>(__cvta_shared_to_generic(dst_addr));
            *reinterpret_cast<half2*>(dst_ptr + 0) = __float2half2_rn(0.0f);
            *reinterpret_cast<half2*>(dst_ptr + 2) = __float2half2_rn(0.0f);
            *reinterpret_cast<half2*>(dst_ptr + 4) = __float2half2_rn(0.0f);
            *reinterpret_cast<half2*>(dst_ptr + 6) = __float2half2_rn(0.0f);
        }
    }
}

// Variant for loading with scale applied
template <int HEIGHT, int WIDTH, int TB_SIZE>
__device__ inline
void fattn_global_to_shared_swizzle_scaled(uint32_t dst, const half *src, int src_stride, int tid, half2 scale) {
    constexpr int num_elems = 16 / sizeof(half);
    constexpr int num_iters = HEIGHT * WIDTH / (TB_SIZE * num_elems);

    #pragma unroll
    for (int iter = 0; iter < num_iters; iter++) {
        const int idx = (iter * TB_SIZE + tid) * num_elems;
        const int row = idx / WIDTH;
        const int col = idx % WIDTH;

        // Load 8 halfs, scale, then store
        const half *src_addr = src + (row * src_stride + col);
        half2 data[4];
        data[0] = *reinterpret_cast<const half2*>(src_addr + 0);
        data[1] = *reinterpret_cast<const half2*>(src_addr + 2);
        data[2] = *reinterpret_cast<const half2*>(src_addr + 4);
        data[3] = *reinterpret_cast<const half2*>(src_addr + 6);

        data[0] = __hmul2(data[0], scale);
        data[1] = __hmul2(data[1], scale);
        data[2] = __hmul2(data[2], scale);
        data[3] = __hmul2(data[3], scale);

        const uint32_t dst_addr = fattn_swizzle<WIDTH * sizeof(half)>(dst + (row * WIDTH + col) * sizeof(half));
        half *dst_ptr = reinterpret_cast<half *>(__cvta_shared_to_generic(dst_addr));
        *reinterpret_cast<half2*>(dst_ptr + 0) = data[0];
        *reinterpret_cast<half2*>(dst_ptr + 2) = data[1];
        *reinterpret_cast<half2*>(dst_ptr + 4) = data[2];
        *reinterpret_cast<half2*>(dst_ptr + 6) = data[3];
    }
}

// ----------------------------------------------------------------------------
// ldmatrix helpers for loading from shared memory to registers
// ----------------------------------------------------------------------------

__device__ inline
void fattn_ldmatrix_x4(uint32_t regs[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
                : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
                : "r"(addr));
}

__device__ inline
void fattn_ldmatrix_x4_trans(uint32_t regs[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.b16 {%0, %1, %2, %3}, [%4];"
                : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
                : "r"(addr));
}

// ----------------------------------------------------------------------------
// MMA helper for m16n8k16 F16->F32 accumulation
// ----------------------------------------------------------------------------

__device__ inline
void fattn_mma_m16n8k16_f16_f32(uint32_t A[4], uint32_t B[2], float D[4]) {
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

// ----------------------------------------------------------------------------
// Configuration
// ----------------------------------------------------------------------------

template<int DKQ, int DV>
struct blackwell_f16_config {
    // Tile sizes
    static constexpr int BLOCK_M  = 64;   // Q rows per block
    static constexpr int BLOCK_N  = 64;   // K/V block size

    // Dimensions
    static constexpr int HEAD_SIZE_Q = DKQ;
    static constexpr int HEAD_SIZE_V = DV;

    // Thread configuration - use 4 warps like attention_v5
    static constexpr int NUM_WARPS   = 4;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_SIZE;

    // MMA tile sizes (m16n8k16)
    static constexpr int MMA_M = 16;
    static constexpr int MMA_N = 8;
    static constexpr int MMA_K = 16;

    // Work per warp
    static constexpr int WARP_Q = BLOCK_M / NUM_WARPS;  // 16 Q rows per warp
};

// ----------------------------------------------------------------------------
// Kernel
// ----------------------------------------------------------------------------

template<int DKQ, int DV, int ncols1, int ncols2, bool mla, int MIN_BLOCKS>
__launch_bounds__(128, MIN_BLOCKS)
__global__ void flash_attn_blackwell_f16(
    const char * __restrict__ Q,
    const char * __restrict__ K,
    const char * __restrict__ V,
    const char * __restrict__ mask,
    float      * __restrict__ dst,
    const float scale,
    const float max_bias,
    const float m0,
    const float m1,
    const uint32_t n_head_log2,
    const float logit_softcap,
    const int ne00,         // head_dim Q
    const int ne01,         // n_queries
    const int ne02,         // n_heads_q
    const int ne03,         // batch
    const int ne10,         // head_dim K
    const int ne11,         // n_kv (sequence length)
    const int ne12,         // n_heads_kv
    const int nb01, const int nb02, const int nb03,  // Q strides
    const int nb11, const int nb12, const int64_t nb13,  // K strides
    const int nb21, const int nb22, const int64_t nb23,  // V strides
    const int ne32, const int64_t nb32,                  // mask head dims
    const int ne33, const int64_t nb33,                  // mask batch dims
    const int stride_mask, const int mask_is_f16,         // mask stride/type
    const int is_causal                                   // causal tile skipping
) {
    using Config = blackwell_f16_config<DKQ, DV>;

    const int tid     = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    // Grid indexing
    const int q_block_id  = blockIdx.x;
    const int head_group  = blockIdx.y;
    const int batch_id    = blockIdx.z;

    // Calculate number of Q blocks
    const int num_q_blocks = fattn_blackwell_cdiv(ne01, Config::BLOCK_M);
    if (q_block_id >= num_q_blocks) return;
    const int q_valid = min(Config::BLOCK_M, ne01 - q_block_id * Config::BLOCK_M);

    // GQA head grouping: group multiple Q heads that share the same KV head
    const int gqa_ratio = ne02 / ne12;
    const int head_groups_per_kv = (gqa_ratio + ncols2 - 1) / ncols2;
    const int kv_head = head_group / head_groups_per_kv;
    if (kv_head >= ne12) return;
    const int head_group_in_kv = head_group % head_groups_per_kv;
    const int head_base = kv_head * gqa_ratio + head_group_in_kv * ncols2;

    const half * K_ptr = (const half *)(K + batch_id * nb13 + kv_head * nb12);
    const half * V_ptr = (const half *)(V + batch_id * nb23 + kv_head * nb22);

    int head_id[ncols2];
    const half * Q_ptr[ncols2];
    bool head_active[ncols2];
    #pragma unroll
    for (int h = 0; h < ncols2; ++h) {
        head_id[h] = head_base + h;
        head_active[h] = head_id[h] < ne02;
        Q_ptr[h] = head_active[h]
            ? (const half *)(Q + batch_id * nb03 + head_id[h] * nb02 + q_block_id * Config::BLOCK_M * nb01)
            : nullptr;
    }

    // Shared memory layout: Q and K alias the same region (Q is loaded to registers before K starts)
    extern __shared__ half smem[];
    const uint32_t Q_smem = __cvta_generic_to_shared(smem);
    const uint32_t K_smem = Q_smem;  // K reuses Q's shared memory after Q is loaded to registers
    const uint32_t V_smem = K_smem + 2 * Config::BLOCK_N * DKQ * sizeof(half);
    constexpr size_t q_bytes  = Config::BLOCK_M * DKQ * sizeof(half);
    constexpr size_t kv_bytes = (2 * Config::BLOCK_N * DKQ + 2 * Config::BLOCK_N * DV) * sizeof(half);
    constexpr size_t smem_base_bytes = q_bytes > kv_bytes ? q_bytes : kv_bytes;
    uint8_t * const mask_smem_base = reinterpret_cast<uint8_t *>(smem) + smem_base_bytes;
    half  * const mask_smem_h = reinterpret_cast<half *>(mask_smem_base);
    float * const mask_smem_f = reinterpret_cast<float *>(mask_smem_base);

    // Register storage for MMA operands
    // Q is persistent in registers
    uint32_t Q_regs[ncols2][Config::WARP_Q / Config::MMA_M][DKQ / Config::MMA_K][4];
    // P (softmax output) for GEMM2
    uint32_t P_regs[ncols2][Config::WARP_Q / Config::MMA_M][Config::BLOCK_N / Config::MMA_K][4] = {};
    // Output accumulator
    float O_regs[ncols2][Config::WARP_Q / Config::MMA_M][DV / Config::MMA_N][4] = {};

    // Pre-compute swizzled addresses for ldmatrix
    uint32_t Q_smem_thread, K_smem_thread, V_smem_thread;
    {
        // A tile (Q): row = warp_id*WARP_Q + lane%16, col = lane/16 * 8
        const int row_off = warp_id * Config::WARP_Q + (lane_id % 16);
        const int col_off = lane_id / 16 * 8;
        Q_smem_thread = fattn_swizzle<DKQ * sizeof(half)>(Q_smem + (row_off * DKQ + col_off) * sizeof(half));
    }
    {
        // B tile (K): row = lane%8, col = lane/8 * 8
        const int row_off = lane_id % 8;
        const int col_off = lane_id / 8 * 8;
        K_smem_thread = fattn_swizzle<DKQ * sizeof(half)>(K_smem + (row_off * DKQ + col_off) * sizeof(half));
    }
    {
        // B tile transposed (V): row = lane%16, col = lane/16 * 8
        const int row_off = lane_id % 16;
        const int col_off = lane_id / 16 * 8;
        V_smem_thread = fattn_swizzle<DV * sizeof(half)>(V_smem + (row_off * DV + col_off) * sizeof(half));
    }

    // Softmax tracking
    const float softmax_scale = scale;
    float rowmax[ncols2][Config::WARP_Q / Config::MMA_M][2];
    float rowsumexp[ncols2][Config::WARP_Q / Config::MMA_M][2] = {};

    #pragma unroll
    for (int h = 0; h < ncols2; ++h) {
        #pragma unroll
        for (int mma_id_q = 0; mma_id_q < Config::WARP_Q / Config::MMA_M; mma_id_q++) {
            rowmax[h][mma_id_q][0] = -FLT_MAX;
            rowmax[h][mma_id_q][1] = -FLT_MAX;
        }
    }

    // ============================================================
    // Load Q [BLOCK_M, DKQ] to shared memory, then to registers (scale applied to registers)
    // ============================================================
    #pragma unroll
    for (int h = 0; h < ncols2; ++h) {
        if (head_active[h]) {
            fattn_global_to_shared_swizzle<Config::BLOCK_M, DKQ, Config::NUM_THREADS>(
                Q_smem, Q_ptr[h], nb01 / sizeof(half), tid, q_valid);
            asm volatile("cp.async.commit_group;");
        } else {
            // Zero-fill for inactive heads
            fattn_global_to_shared_swizzle<Config::BLOCK_M, DKQ, Config::NUM_THREADS>(
                Q_smem, Q_ptr[0], nb01 / sizeof(half), tid, 0);
            asm volatile("cp.async.commit_group;");
        }
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();

        #pragma unroll
        for (int mma_id_q = 0; mma_id_q < Config::WARP_Q / Config::MMA_M; mma_id_q++) {
            #pragma unroll
            for (int mma_id_d = 0; mma_id_d < DKQ / Config::MMA_K; mma_id_d++) {
                uint32_t addr = Q_smem_thread;
                addr += mma_id_q * Config::MMA_M * DKQ * sizeof(half);  // Row offset
                addr ^= mma_id_d * Config::MMA_K * sizeof(half);        // Column offset (XOR for swizzle)
                fattn_ldmatrix_x4(Q_regs[h][mma_id_q][mma_id_d], addr);
            }
        }

        // Apply softmax_scale to Q registers once (avoids per-KV-iteration scaling of S_regs)
        const half  scale_h  = __float2half_rn(softmax_scale);
        const half2 scale_h2 = make_half2(scale_h, scale_h);
        #pragma unroll
        for (int mma_id_q = 0; mma_id_q < Config::WARP_Q / Config::MMA_M; mma_id_q++) {
            #pragma unroll
            for (int mma_id_d = 0; mma_id_d < DKQ / Config::MMA_K; mma_id_d++) {
                half2 *q_h2 = reinterpret_cast<half2*>(Q_regs[h][mma_id_q][mma_id_d]);
                q_h2[0] = __hmul2(q_h2[0], scale_h2);
                q_h2[1] = __hmul2(q_h2[1], scale_h2);
                q_h2[2] = __hmul2(q_h2[2], scale_h2);
                q_h2[3] = __hmul2(q_h2[3], scale_h2);
            }
        }

        __syncthreads();
    }

    // ============================================================
    // K/V iteration loop
    // ============================================================
    const int num_kv_iter = fattn_blackwell_cdiv(ne11, Config::BLOCK_N);

    // Lambda for loading K
    auto load_K = [&](int kv_id) {
        if (kv_id < num_kv_iter) {
            const int k_valid = min(Config::BLOCK_N, ne11 - kv_id * Config::BLOCK_N);
            const uint32_t dst = K_smem + (kv_id % 2) * (Config::BLOCK_N * DKQ * sizeof(half));
            const half *src = K_ptr + kv_id * Config::BLOCK_N * (nb11 / sizeof(half));
            fattn_global_to_shared_swizzle<Config::BLOCK_N, DKQ, Config::NUM_THREADS>(dst, src, nb11 / sizeof(half), tid, k_valid);
            asm volatile("cp.async.commit_group;");
        }
    };

    // Lambda for loading V (double-buffered)
    auto load_V = [&](int kv_id) {
        if (kv_id < num_kv_iter) {
            const int k_valid = min(Config::BLOCK_N, ne11 - kv_id * Config::BLOCK_N);
            const uint32_t dst = V_smem + (kv_id % 2) * (Config::BLOCK_N * DV * sizeof(half));
            const half *src = V_ptr + kv_id * Config::BLOCK_N * (nb21 / sizeof(half));
            fattn_global_to_shared_swizzle<Config::BLOCK_N, DV, Config::NUM_THREADS>(dst, src, nb21 / sizeof(half), tid, k_valid);
            asm volatile("cp.async.commit_group;");
        }
    };

    // Prefetch first K and V blocks
    load_K(0);
    load_V(0);

    const bool has_mask = (mask != nullptr);
    const bool mask_shared = has_mask && (ne32 == 1);
    const char * mask_base[ncols2];
    const char * mask_base_shared = nullptr;
    float slope[ncols2];
    #pragma unroll
    for (int h = 0; h < ncols2; ++h) {
        mask_base[h] = (has_mask && head_active[h]) ?
            (mask + nb33 * (batch_id % ne33) + nb32 * (head_id[h] % ne32)) :
            nullptr;
        if (mask_base_shared == nullptr && mask_base[h] != nullptr) {
            mask_base_shared = mask_base[h];
        }
        slope[h] = (max_bias != 0.0f && head_active[h])
            ? get_alibi_slope(max_bias, head_id[h], n_head_log2, m0, m1)
            : 1.0f;
    }

    for (int kv_id = 0; kv_id < num_kv_iter; kv_id++) {
        // Causal tile skipping: if entire KV block is beyond all Q positions,
        // all mask values would be -inf. Once one tile is skippable, all
        // subsequent tiles are too, so break out of the loop entirely.
        if (is_causal) {
            const int kv_block_start = kv_id * Config::BLOCK_N;
            const int q_block_end = q_block_id * Config::BLOCK_M + q_valid - 1;
            if (kv_block_start > q_block_end) {
                // Drain pending async copies before exiting loop
                asm volatile("cp.async.wait_group 0;");
                __syncthreads();
                break;
            }
        }

        // ============================================================
        // Wait for K[kv_id] to be in shared memory
        // Pending groups: K[kv_id] (older), V[kv_id] (newer)
        // wait_group 1: keeps V[kv_id] in flight, waits for K[kv_id]
        // ============================================================
        asm volatile("cp.async.wait_group 1;");
        __syncthreads();

        // ============================================================
        // Load mask tile into shared memory (if present)
        // Committed BEFORE K[kv_id+1] so that wait_group 1 correctly
        // keeps K[kv_id+1] in flight and waits for V[kv_id] + mask.
        // ============================================================
        // Prefetch mask tile into shared memory (if present) without cp.async to avoid pipeline interference
        auto load_mask_tile = [&](const char * mask_base_ptr) {
            const int total = Config::BLOCK_M * Config::BLOCK_N;
            if (mask_is_f16) {
                constexpr int num_elems = 16 / sizeof(half);  // 8 half per 16B
                for (int idx = tid * num_elems; idx < total; idx += Config::NUM_THREADS * num_elems) {
                    const int q_local = idx / Config::BLOCK_N;
                    const int k_local = idx % Config::BLOCK_N;
                    const int q = q_block_id * Config::BLOCK_M + q_local;
                    const int k = kv_id * Config::BLOCK_N + k_local;
                    if (q < ne01 && k + (num_elems - 1) < ne11 && k_local <= Config::BLOCK_N - num_elems) {
                        const half *src = reinterpret_cast<const half *>(mask_base_ptr) + q * stride_mask + k;
                        *reinterpret_cast<half2 *>(mask_smem_h + idx + 0) = *reinterpret_cast<const half2 *>(src + 0);
                        *reinterpret_cast<half2 *>(mask_smem_h + idx + 2) = *reinterpret_cast<const half2 *>(src + 2);
                        *reinterpret_cast<half2 *>(mask_smem_h + idx + 4) = *reinterpret_cast<const half2 *>(src + 4);
                        *reinterpret_cast<half2 *>(mask_smem_h + idx + 6) = *reinterpret_cast<const half2 *>(src + 6);
                    } else {
                        // Fallback scalar for tail or OOB
                        for (int i = 0; i < num_elems; ++i) {
                            const int kk = k + i;
                            const int dst_idx = idx + i;
                            if (dst_idx < total && q < ne01 && kk < ne11) {
                                mask_smem_h[dst_idx] = reinterpret_cast<const half *>(mask_base_ptr)[q * stride_mask + kk];
                            } else if (dst_idx < total) {
                                mask_smem_h[dst_idx] = __float2half(0.0f);
                            }
                        }
                    }
                }
            } else {
                // Vectorized float4 loads for F32 mask
                for (int idx = tid * 4; idx < total; idx += Config::NUM_THREADS * 4) {
                    const int q_local = idx / Config::BLOCK_N;
                    const int k_local = idx % Config::BLOCK_N;
                    const int q = q_block_id * Config::BLOCK_M + q_local;
                    const int k = kv_id * Config::BLOCK_N + k_local;
                    if (q < ne01 && k + 3 < ne11 && k_local <= Config::BLOCK_N - 4) {
                        const float4 v4 = *reinterpret_cast<const float4 *>(reinterpret_cast<const float *>(mask_base_ptr) + q * stride_mask + k);
                        *reinterpret_cast<float4 *>(mask_smem_f + idx) = v4;
                    } else {
                        for (int i = 0; i < 4; ++i) {
                            const int kk = k + i;
                            const int dst_idx = idx + i;
                            if (dst_idx < total && q < ne01 && kk < ne11) {
                                mask_smem_f[dst_idx] = reinterpret_cast<const float *>(mask_base_ptr)[q * stride_mask + kk];
                            } else if (dst_idx < total) {
                                mask_smem_f[dst_idx] = 0.0f;
                            }
                        }
                    }
                }
            }
        };

        if (mask_shared && mask_base_shared != nullptr) {
            load_mask_tile(mask_base_shared);
            __syncthreads();
        }

        // Prefetch next K (committed AFTER mask so it's the newest group)
        load_K(kv_id + 1);

        // ============================================================
        // Wait for V[kv_id], keep K[kv_id+1] in flight
        // Group order: V[kv_id] (oldest), K[kv_id+1] (newest)
        // wait_group 1: keeps K[kv_id+1], waits for V
        // ============================================================
        asm volatile("cp.async.wait_group 1;");
        __syncthreads();

        #pragma unroll
        for (int h = 0; h < ncols2; ++h) {
            if (!head_active[h]) {
                continue;
            }

            // S accumulator for Q @ K^T
            float S_regs[Config::WARP_Q / Config::MMA_M][Config::BLOCK_N / Config::MMA_N][4] = {};

            // GEMM1: S = Q @ K^T (interleaved load + compute)
            #pragma unroll
            for (int mma_id_kv = 0; mma_id_kv < Config::BLOCK_N / Config::MMA_N; mma_id_kv++) {
                // Load one K tile from shared memory
                uint32_t K_local[DKQ / Config::MMA_K][2];
                #pragma unroll
                for (int mma_id_d = 0; mma_id_d < DKQ / Config::MMA_K; mma_id_d += 2) {
                    uint32_t addr = K_smem_thread + (kv_id % 2) * (Config::BLOCK_N * DKQ * sizeof(half));
                    addr += mma_id_kv * Config::MMA_N * DKQ * sizeof(half);
                    addr ^= mma_id_d * Config::MMA_K * sizeof(half);
                    uint32_t tmp[4];
                    fattn_ldmatrix_x4(tmp, addr);
                    K_local[mma_id_d + 0][0] = tmp[0];
                    K_local[mma_id_d + 0][1] = tmp[1];
                    K_local[mma_id_d + 1][0] = tmp[2];
                    K_local[mma_id_d + 1][1] = tmp[3];
                }
                // Use immediately in MMA
                #pragma unroll
                for (int mma_id_q = 0; mma_id_q < Config::WARP_Q / Config::MMA_M; mma_id_q++) {
                    #pragma unroll
                    for (int mma_id_d = 0; mma_id_d < DKQ / Config::MMA_K; mma_id_d++) {
                        fattn_mma_m16n8k16_f16_f32(
                            Q_regs[h][mma_id_q][mma_id_d],
                            K_local[mma_id_d],
                            S_regs[mma_id_q][mma_id_kv]);
                    }
                }
            }

            // Apply logit softcap if enabled (before mask addition)
            if (logit_softcap != 0.0f) {
                #pragma unroll
                for (int mma_id_q = 0; mma_id_q < Config::WARP_Q / Config::MMA_M; mma_id_q++) {
                    #pragma unroll
                    for (int mma_id_kv = 0; mma_id_kv < Config::BLOCK_N / Config::MMA_N; mma_id_kv++) {
                        float *regs = S_regs[mma_id_q][mma_id_kv];
                        regs[0] = logit_softcap * tanh_fast(regs[0]);
                        regs[1] = logit_softcap * tanh_fast(regs[1]);
                        regs[2] = logit_softcap * tanh_fast(regs[2]);
                        regs[3] = logit_softcap * tanh_fast(regs[3]);
                    }
                }
            }

            // Apply mask / OOB (additive bias)
            if (!mask_shared && has_mask && mask_base[h] != nullptr) {
                load_mask_tile(mask_base[h]);
                __syncthreads();
            }
            #pragma unroll
            for (int mma_id_q = 0; mma_id_q < Config::WARP_Q / Config::MMA_M; mma_id_q++) {
                #pragma unroll
                for (int mma_id_kv = 0; mma_id_kv < Config::BLOCK_N / Config::MMA_N; mma_id_kv++) {
                    float *regs = S_regs[mma_id_q][mma_id_kv];

                    const int row0 = warp_id * Config::WARP_Q + mma_id_q * Config::MMA_M + (lane_id / 4);
                    const int row1 = row0 + 8;
                    const int col0 = mma_id_kv * Config::MMA_N + (lane_id % 4) * 2;
                    const int col1 = col0 + 1;

                    const int q0 = q_block_id * Config::BLOCK_M + row0;
                    const int q1 = q0 + 8;
                    const int k0 = kv_id * Config::BLOCK_N + col0;
                    const int k1 = k0 + 1;

                    float m00 = (q0 < ne01 && k0 < ne11) ? 0.0f : -INFINITY;
                    float m01 = (q0 < ne01 && k1 < ne11) ? 0.0f : -INFINITY;
                    float m10 = (q1 < ne01 && k0 < ne11) ? 0.0f : -INFINITY;
                    float m11 = (q1 < ne01 && k1 < ne11) ? 0.0f : -INFINITY;

                    if (has_mask && mask_base[h] != nullptr) {
                        if (q0 < ne01 && k0 < ne11) {
                            m00 = mask_is_f16 ? __half2float(mask_smem_h[row0 * Config::BLOCK_N + col0]) :
                                                mask_smem_f[row0 * Config::BLOCK_N + col0];
                        }
                        if (q0 < ne01 && k1 < ne11) {
                            m01 = mask_is_f16 ? __half2float(mask_smem_h[row0 * Config::BLOCK_N + col1]) :
                                                mask_smem_f[row0 * Config::BLOCK_N + col1];
                        }
                        if (q1 < ne01 && k0 < ne11) {
                            m10 = mask_is_f16 ? __half2float(mask_smem_h[(row0 + 8) * Config::BLOCK_N + col0]) :
                                                mask_smem_f[(row0 + 8) * Config::BLOCK_N + col0];
                        }
                        if (q1 < ne01 && k1 < ne11) {
                            m11 = mask_is_f16 ? __half2float(mask_smem_h[(row0 + 8) * Config::BLOCK_N + col1]) :
                                                mask_smem_f[(row0 + 8) * Config::BLOCK_N + col1];
                        }
                    }

                    regs[0] += slope[h] * m00;
                    regs[1] += slope[h] * m01;
                    regs[2] += slope[h] * m10;
                    regs[3] += slope[h] * m11;
                }
            }

            // Online Softmax
            #pragma unroll
            for (int mma_id_q = 0; mma_id_q < Config::WARP_Q / Config::MMA_M; mma_id_q++) {
                float this_rowmax[2];
                #pragma unroll
                for (int mma_id_kv = 0; mma_id_kv < Config::BLOCK_N / Config::MMA_N; mma_id_kv++) {
                    float *regs = S_regs[mma_id_q][mma_id_kv];
                    if (mma_id_kv == 0) {
                        this_rowmax[0] = fmaxf(regs[0], regs[1]);
                        this_rowmax[1] = fmaxf(regs[2], regs[3]);
                    } else {
                        this_rowmax[0] = fmaxf(this_rowmax[0], fmaxf(regs[0], regs[1]));
                        this_rowmax[1] = fmaxf(this_rowmax[1], fmaxf(regs[2], regs[3]));
                    }
                }

                this_rowmax[0] = fmaxf(this_rowmax[0], __shfl_xor_sync(0xFFFFFFFF, this_rowmax[0], 1));
                this_rowmax[0] = fmaxf(this_rowmax[0], __shfl_xor_sync(0xFFFFFFFF, this_rowmax[0], 2));
                this_rowmax[1] = fmaxf(this_rowmax[1], __shfl_xor_sync(0xFFFFFFFF, this_rowmax[1], 1));
                this_rowmax[1] = fmaxf(this_rowmax[1], __shfl_xor_sync(0xFFFFFFFF, this_rowmax[1], 2));

                this_rowmax[0] = fmaxf(this_rowmax[0], rowmax[h][mma_id_q][0]);
                this_rowmax[1] = fmaxf(this_rowmax[1], rowmax[h][mma_id_q][1]);

                float rescale[2];
                if (kv_id == 0) {
                    rescale[0] = 1.0f;
                    rescale[1] = 1.0f;
                } else {
                    rescale[0] = __expf(rowmax[h][mma_id_q][0] - this_rowmax[0]);
                    rescale[1] = __expf(rowmax[h][mma_id_q][1] - this_rowmax[1]);
                }

                #pragma unroll
                for (int mma_id_d = 0; mma_id_d < DV / Config::MMA_N; mma_id_d++) {
                    O_regs[h][mma_id_q][mma_id_d][0] *= rescale[0];
                    O_regs[h][mma_id_q][mma_id_d][1] *= rescale[0];
                    O_regs[h][mma_id_q][mma_id_d][2] *= rescale[1];
                    O_regs[h][mma_id_q][mma_id_d][3] *= rescale[1];
                }

                rowmax[h][mma_id_q][0] = this_rowmax[0];
                rowmax[h][mma_id_q][1] = this_rowmax[1];

                float this_rowsumexp[2];
                #pragma unroll
                for (int mma_id_kv = 0; mma_id_kv < Config::BLOCK_N / Config::MMA_N; mma_id_kv++) {
                    float *regs = S_regs[mma_id_q][mma_id_kv];
                    regs[0] = __expf(regs[0] - rowmax[h][mma_id_q][0]);
                    regs[1] = __expf(regs[1] - rowmax[h][mma_id_q][0]);
                    regs[2] = __expf(regs[2] - rowmax[h][mma_id_q][1]);
                    regs[3] = __expf(regs[3] - rowmax[h][mma_id_q][1]);

                    if (mma_id_kv == 0) {
                        this_rowsumexp[0] = regs[0] + regs[1];
                        this_rowsumexp[1] = regs[2] + regs[3];
                    } else {
                        this_rowsumexp[0] += regs[0] + regs[1];
                        this_rowsumexp[1] += regs[2] + regs[3];
                    }

                    half2 *this_P_regs = reinterpret_cast<half2*>(P_regs[h][mma_id_q][mma_id_kv / 2]);
                    this_P_regs[(mma_id_kv % 2) * 2]     = __float22half2_rn(make_float2(regs[0], regs[1]));
                    this_P_regs[(mma_id_kv % 2) * 2 + 1] = __float22half2_rn(make_float2(regs[2], regs[3]));
                }

                this_rowsumexp[0] += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[0], 1);
                this_rowsumexp[0] += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[0], 2);
                this_rowsumexp[1] += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[1], 1);
                this_rowsumexp[1] += __shfl_xor_sync(0xFFFFFFFF, this_rowsumexp[1], 2);

                if (kv_id == 0) {
                    rowsumexp[h][mma_id_q][0] = this_rowsumexp[0];
                    rowsumexp[h][mma_id_q][1] = this_rowsumexp[1];
                } else {
                    rowsumexp[h][mma_id_q][0] = rowsumexp[h][mma_id_q][0] * rescale[0] + this_rowsumexp[0];
                    rowsumexp[h][mma_id_q][1] = rowsumexp[h][mma_id_q][1] * rescale[1] + this_rowsumexp[1];
                }
            }
        }

        // ============================================================
        // GEMM2: O += P @ V (interleaved load + compute)
        // V[kv_id] already waited for above
        // ============================================================
        #pragma unroll
        for (int mma_id_kv = 0; mma_id_kv < Config::BLOCK_N / Config::MMA_K; mma_id_kv++) {
            // Load one V slice from shared memory
            uint32_t V_local[DV / Config::MMA_N][2];
            #pragma unroll
            for (int mma_id_d = 0; mma_id_d < DV / Config::MMA_N; mma_id_d += 2) {
                uint32_t addr = V_smem_thread + (kv_id % 2) * (Config::BLOCK_N * DV * sizeof(half));
                addr += mma_id_kv * Config::MMA_K * DV * sizeof(half);
                addr ^= mma_id_d * Config::MMA_N * sizeof(half);
                uint32_t tmp[4];
                fattn_ldmatrix_x4_trans(tmp, addr);
                V_local[mma_id_d + 0][0] = tmp[0];
                V_local[mma_id_d + 0][1] = tmp[1];
                V_local[mma_id_d + 1][0] = tmp[2];
                V_local[mma_id_d + 1][1] = tmp[3];
            }
            // Use immediately in MMA for all heads
            #pragma unroll
            for (int h = 0; h < ncols2; ++h) {
                if (!head_active[h]) continue;
                #pragma unroll
                for (int mma_id_q = 0; mma_id_q < Config::WARP_Q / Config::MMA_M; mma_id_q++) {
                    #pragma unroll
                    for (int mma_id_d = 0; mma_id_d < DV / Config::MMA_N; mma_id_d++) {
                        fattn_mma_m16n8k16_f16_f32(
                            P_regs[h][mma_id_q][mma_id_kv],
                            V_local[mma_id_d],
                            O_regs[h][mma_id_q][mma_id_d]);
                    }
                }
            }
        }

        // Prefetch next V into the other double-buffer slot
        load_V(kv_id + 1);
    }

    // Write output: O / rowsumexp
    #pragma unroll
    for (int h = 0; h < ncols2; ++h) {
        if (!head_active[h]) {
            continue;
        }
        #pragma unroll
        for (int mma_id_q = 0; mma_id_q < Config::WARP_Q / Config::MMA_M; mma_id_q++) {
            #pragma unroll
            for (int mma_id_d = 0; mma_id_d < DV / Config::MMA_N; mma_id_d++) {
                const int row = warp_id * Config::WARP_Q + mma_id_q * Config::MMA_M + (lane_id / 4);
                const int col = mma_id_d * Config::MMA_N + (lane_id % 4) * 2;

                float *regs = O_regs[h][mma_id_q][mma_id_d];
                const float denom0 = fmaxf(rowsumexp[h][mma_id_q][0], 1e-6f);
                const float denom1 = fmaxf(rowsumexp[h][mma_id_q][1], 1e-6f);
                regs[0] /= denom0;
                regs[1] /= denom0;
                regs[2] /= denom1;
                regs[3] /= denom1;

                const int global_row = q_block_id * Config::BLOCK_M + row;
                if (global_row < ne01) {
                    float *dst_row0 = dst + ((batch_id * ne01 + (global_row + 0)) * ne02 + head_id[h]) * DV + col;
                    float *dst_row8 = dst + ((batch_id * ne01 + (global_row + 8)) * ne02 + head_id[h]) * DV + col;

                    *reinterpret_cast<float2 *>(dst_row0) = make_float2(regs[0], regs[1]);
                    if (global_row + 8 < ne01) {
                        *reinterpret_cast<float2 *>(dst_row8) = make_float2(regs[2], regs[3]);
                    }
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Host-side wrapper
// ----------------------------------------------------------------------------

template<int DKQ, int DV, int ncols1, int ncols2>
void launch_flash_attn_blackwell_f16(
    ggml_backend_cuda_context & ctx,
    ggml_tensor * dst,
    const ggml_tensor * Q,
    const ggml_tensor * K,
    const ggml_tensor * V,
    const ggml_tensor * mask,
    float scale,
    float max_bias,
    float m0,
    float m1,
    uint32_t n_head_log2,
    float logit_softcap,
    int is_causal) {

    using Config = blackwell_f16_config<DKQ, DV>;

    const int64_t ne00 = Q->ne[0];  // head_dim Q
    const int64_t ne01 = Q->ne[1];  // n_queries
    const int64_t ne02 = Q->ne[2];  // n_heads_q
    const int64_t ne03 = Q->ne[3];  // batch

    const int64_t ne10 = K->ne[0];  // head_dim K
    const int64_t ne11 = K->ne[1];  // n_kv
    const int64_t ne12 = K->ne[2];  // n_heads_kv

    // Grid dimensions (group Q heads by shared KV head when ncols2 > 1)
    const int num_q_blocks = fattn_blackwell_cdiv(ne01, Config::BLOCK_M);
    const int gqa_ratio = (int)(ne02 / ne12);
    const int head_groups_per_kv = (gqa_ratio + ncols2 - 1) / ncols2;
    dim3 grid(num_q_blocks, ne12 * head_groups_per_kv, ne03);
    dim3 block(Config::NUM_THREADS);

    // Shared memory: max(Q, K double buffer + V) -- Q and K/V alias the same region
    const size_t q_bytes  = Config::BLOCK_M * DKQ * sizeof(half);
    const size_t kv_bytes = (2 * Config::BLOCK_N * DKQ + 2 * Config::BLOCK_N * DV) * sizeof(half);
    const size_t smem_base_bytes = q_bytes > kv_bytes ? q_bytes : kv_bytes;
    const size_t mask_bytes = (mask != nullptr)
        ? (Config::BLOCK_M * Config::BLOCK_N * (mask->type == GGML_TYPE_F16 ? sizeof(half) : sizeof(float)))
        : 0;
    const int smem_size = (int) (smem_base_bytes + mask_bytes);

    constexpr int min_blocks = (ncols2 >= 4) ? 1 : 2;

    // Set shared memory limit if needed
    if (smem_size > 48000) {
        CUDA_CHECK(cudaFuncSetAttribute(
            flash_attn_blackwell_f16<DKQ, DV, ncols1, ncols2, false, min_blocks>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size));
    }

    const bool mask_is_f16 = mask && mask->type == GGML_TYPE_F16;
    const int  stride_mask = mask_is_f16 ? (mask->nb[1] / (int)sizeof(half)) : (mask ? (mask->nb[1] / (int)sizeof(float)) : 0);
    const int  ne32 = mask ? (int)mask->ne[2] : 1;
    const int  ne33 = mask ? (int)mask->ne[3] : 1;
    const int64_t nb32 = mask ? mask->nb[2] : 0;
    const int64_t nb33 = mask ? mask->nb[3] : 0;

    flash_attn_blackwell_f16<DKQ, DV, ncols1, ncols2, false, min_blocks>
        <<<grid, block, smem_size, ctx.stream()>>>(
            (const char *)Q->data,
            (const char *)K->data,
            (const char *)V->data,
            mask ? (const char *)mask->data : nullptr,
            (float *)dst->data,
            scale,
            max_bias,
            m0,
            m1,
            n_head_log2,
            logit_softcap,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12,
            Q->nb[1], Q->nb[2], Q->nb[3],
            K->nb[1], K->nb[2], K->nb[3],
            V->nb[1], V->nb[2], V->nb[3],
            ne32, nb32, ne33, nb33,
            stride_mask, mask_is_f16 ? 1 : 0,
            is_causal);

    CUDA_CHECK(cudaGetLastError());
}

#endif // CP_ASYNC_AVAILABLE || !__CUDA_ARCH__

// ----------------------------------------------------------------------------
// External interface
// ----------------------------------------------------------------------------

void ggml_cuda_flash_attn_ext_blackwell_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
#if defined(CP_ASYNC_AVAILABLE) || !defined(__CUDA_ARCH__)
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    // Allow disabling mask for testing (set GGML_CUDA_NO_MASK_BLACKWELL=1)
    const bool no_mask = getenv("GGML_CUDA_NO_MASK_BLACKWELL") != nullptr;
    if (no_mask) {
        mask = nullptr;
    }

    const int64_t ne00 = Q->ne[0];  // head_dim

    float scale = 1.0f;
    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&scale, (const float *)dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *)dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *)dst->op_params + 2, sizeof(float));

    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    // If scale is 0, use default 1/sqrt(head_dim)
    if (scale == 0.0f) {
        scale = 1.0f / sqrtf(float(ne00));
    }

    const uint32_t n_head      = Q->ne[2];
    const uint32_t n_head_log2 = 1u << uint32_t(floorf(log2f(float(n_head))));
    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    const bool gqa_divisible = (Q->ne[2] % K->ne[2]) == 0;
    const int gqa_ratio = gqa_divisible ? (int)(Q->ne[2] / K->ne[2]) : 1;
    const bool mask_all_heads = (mask == nullptr) || (mask->ne[2] == 1);
    const bool use_gqa_group = gqa_divisible && mask_all_heads && (gqa_ratio >= 2);

    // Causal skip only valid when Q and KV are position-aligned (ne01 >= ne11),
    // i.e. during prefill. During decode (ne01=1, ne11=context_len), local Q
    // index 0 maps to the END of the sequence, so all KV positions are valid.
    const int64_t ne01 = Q->ne[1];
    const int64_t ne11 = K->ne[1];
    const int is_causal = (mask != nullptr && ne01 >= ne11 && getenv("GGML_CUDA_CAUSAL_SKIP") != nullptr) ? 1 : 0;

    switch (ne00) {
        case 64:
            if (use_gqa_group && gqa_ratio >= 4) {
                launch_flash_attn_blackwell_f16<64, 64, 1, 4>(ctx, dst, Q, K, V, mask, scale, max_bias, m0, m1, n_head_log2, logit_softcap, is_causal);
            } else if (use_gqa_group) {
                launch_flash_attn_blackwell_f16<64, 64, 1, 2>(ctx, dst, Q, K, V, mask, scale, max_bias, m0, m1, n_head_log2, logit_softcap, is_causal);
            } else {
                launch_flash_attn_blackwell_f16<64, 64, 1, 1>(ctx, dst, Q, K, V, mask, scale, max_bias, m0, m1, n_head_log2, logit_softcap, is_causal);
            }
            break;
        case 128:
            // DKQ=128: always ncols2=1 (GQA grouping causes register spilling at this head size)
            launch_flash_attn_blackwell_f16<128, 128, 1, 1>(ctx, dst, Q, K, V, mask, scale, max_bias, m0, m1, n_head_log2, logit_softcap, is_causal);
            break;
        default:
            GGML_ABORT("Unsupported head dimension for Blackwell F16 kernel");
    }
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_ABORT("Blackwell F16 kernel requires MMA support");
#endif
}

bool ggml_cuda_flash_attn_ext_blackwell_f16_supported(const ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    // Only support specific head dimensions
    if (Q->ne[0] != 64 && Q->ne[0] != 128) {
        return false;
    }

    // Must have matching head dimensions
    if (Q->ne[0] != V->ne[0] || K->ne[0] != Q->ne[0]) {
        return false;
    }

    // Must be F16 inputs
    if (Q->type != GGML_TYPE_F16 || K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16) {
        return false;
    }

    // Output must be F32
    if (dst->type != GGML_TYPE_F32) {
        return false;
    }

    // Mask type must be supported (F16 or F32), if present
    if (mask != nullptr && mask->type != GGML_TYPE_F16 && mask->type != GGML_TYPE_F32) {
        return false;
    }

    return true;
}
