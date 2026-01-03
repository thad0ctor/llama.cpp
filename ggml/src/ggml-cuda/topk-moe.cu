#include "ggml-cuda/common.cuh"
#include "ggml.h"
#include "topk-moe.cuh"

#include <cmath>
#include <initializer_list>

// Optimization settings for different architectures
// Blackwell (sm_120): 228KB shared memory, can use 8 rows/block efficiently
// Ampere (sm_80-89): 164KB shared memory, use 8 rows/block
// Legacy: 4 rows/block for older architectures
#define MOE_ROWS_PER_BLOCK_OPTIMIZED 8
#define MOE_ROWS_PER_BLOCK_DEFAULT 4

// Use shared memory for expert logits when n_experts >= threshold
#define MOE_SMEM_THRESHOLD 32

// Warp-local softmax used for both the pre-top-k logits and the post-top-k delayed path.
template <int experts_per_thread, bool use_limit>
__device__ void softmax_warp_inplace(float (&vals)[experts_per_thread], const int limit, const int lane) {
    float max_val = -INFINITY;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            max_val = max(max_val, vals[i]);
        }
    }

    max_val = warp_reduce_max(max_val);

    float sum = 0.f;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            const float val = expf(vals[i] - max_val);
            vals[i]         = val;
            sum += val;
        } else {
            vals[i] = 0.f;
        }
    }

    sum = warp_reduce_sum(sum);

    const float inv_sum = 1.0f / sum;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            vals[i] *= inv_sum;
        }
    }
}

// Optimized argmax reduction with index tracking
__device__ __forceinline__ float warp_reduce_max_with_idx(float val, int idx, int * max_expert) {
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
        const float other_val = __shfl_xor_sync(0xFFFFFFFF, val, mask, WARP_SIZE);
        const int   other_idx = __shfl_xor_sync(0xFFFFFFFF, idx, mask, WARP_SIZE);
        if (other_val > val || (other_val == val && other_idx < idx)) {
            val = other_val;
            idx = other_idx;
        }
    }
    *max_expert = idx;
    return val;
}

/*
    Legacy MoE top-k kernel (4 rows per block, no shared memory caching)
    Used for older architectures or small expert counts
*/
template <int n_experts, bool with_norm, bool delayed_softmax = false>
__launch_bounds__(4 * WARP_SIZE, 1) __global__ void topk_moe_cuda(const float * logits,
                                                                  float *       weights,
                                                                  int32_t *     ids,
                                                                  const int     n_rows,
                                                                  const int     n_expert_used,
                                                                  const float   clamp_val) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= n_rows) {
        return;
    }

    logits += n_experts * row;
    weights += n_expert_used * row;
    ids += n_experts * row;

    constexpr int experts_per_thread = (n_experts > WARP_SIZE) ? n_experts / WARP_SIZE : 1;

    float wt[experts_per_thread];

#pragma unroll
    for (int i = 0; i < n_experts; i += WARP_SIZE) {
        const int expert  = i + threadIdx.x;
        wt[i / WARP_SIZE] = (n_experts % WARP_SIZE == 0 || expert < n_experts) ? logits[expert] : -INFINITY;
    }

    if constexpr (!delayed_softmax) {
        softmax_warp_inplace<experts_per_thread, false>(wt, n_experts, threadIdx.x);
    }

    float wt_sum = 0.f;
    float output_weights[experts_per_thread];

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        output_weights[i] = 0.f;
    }

    for (int k = 0; k < n_expert_used; k++) {
        float max_val    = wt[0];
        int   max_expert = threadIdx.x;

#pragma unroll
        for (int i = 1; i < experts_per_thread; i++) {
            const int expert = threadIdx.x + i * WARP_SIZE;
            if ((n_experts % WARP_SIZE == 0 || expert < n_experts) && wt[i] > max_val) {
                max_val    = wt[i];
                max_expert = expert;
            }
        }

#pragma unroll
        for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
            const float val    = __shfl_xor_sync(0xFFFFFFFF, max_val, mask, WARP_SIZE);
            const int   expert = __shfl_xor_sync(0xFFFFFFFF, max_expert, mask, WARP_SIZE);
            if (val > max_val || (val == max_val && expert < max_expert)) {
                max_val    = val;
                max_expert = expert;
            }
        }

        if ((k & (WARP_SIZE - 1)) == threadIdx.x) {
            output_weights[k / WARP_SIZE] = max_val;
        }

        if ((max_expert & (WARP_SIZE - 1)) == threadIdx.x) {
            wt[max_expert / WARP_SIZE] = -INFINITY;

            ids[k] = max_expert;
            if constexpr (with_norm) {
                wt_sum += max_val;
            }
        }
    }

    if constexpr (with_norm) {
        wt_sum              = warp_reduce_sum(wt_sum);
        wt_sum              = max(wt_sum, clamp_val);
        const float inv_sum = 1.0f / wt_sum;

        for (int i = 0; i < experts_per_thread; i++) {
            output_weights[i] *= inv_sum;
        }
    }

    if constexpr (delayed_softmax) {
        softmax_warp_inplace<experts_per_thread, true>(output_weights, n_expert_used, threadIdx.x);
    }

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int idx = i * WARP_SIZE + threadIdx.x;
        if (idx < n_expert_used) {
            weights[idx] = output_weights[i];
        }
    }

    if (!with_norm) {
        GGML_UNUSED(clamp_val);
    }
}

/*
    Optimized MoE top-k kernel for Ampere+ (sm_80+) with:
    1. Shared memory caching for repeated argmax iterations
    2. 8 rows per block for better SM utilization
    3. Optimized warp-level reduction

    For Blackwell (sm_120), the larger shared memory (228KB) allows efficient
    operation even with 512 experts and 8 rows per block.
*/
template <int n_experts, int rows_per_block, bool with_norm, bool delayed_softmax = false>
__launch_bounds__(rows_per_block * WARP_SIZE, 2)
__global__ void topk_moe_cuda_optimized(
    const float * __restrict__ logits,
    float *       __restrict__ weights,
    int32_t *     __restrict__ ids,
    const int     n_rows,
    const int     n_expert_used,
    const float   clamp_val
) {
    // Shared memory for caching logits per row (enables efficient repeated argmax)
    extern __shared__ float smem[];
    float * row_logits = smem + threadIdx.y * n_experts;

    const int row = blockIdx.x * rows_per_block + threadIdx.y;
    if (row >= n_rows) {
        return;
    }

    const float * row_logits_global = logits + n_experts * row;
    float *       row_weights = weights + n_expert_used * row;
    int32_t *     row_ids = ids + n_experts * row;

    const int lane = threadIdx.x;
    constexpr int experts_per_thread = (n_experts > WARP_SIZE) ? n_experts / WARP_SIZE : 1;

    float wt[experts_per_thread];

    // Load logits to shared memory (cooperative load by warp)
#pragma unroll
    for (int i = lane; i < n_experts; i += WARP_SIZE) {
        row_logits[i] = row_logits_global[i];
    }
    __syncwarp();

    // Copy to registers
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int expert = lane + i * WARP_SIZE;
        wt[i] = (expert < n_experts) ? row_logits[expert] : -INFINITY;
    }

    // Apply softmax if not delayed
    if constexpr (!delayed_softmax) {
        softmax_warp_inplace<experts_per_thread, false>(wt, n_experts, lane);

        // Update shared memory with softmax values
#pragma unroll
        for (int i = 0; i < experts_per_thread; i++) {
            const int expert = lane + i * WARP_SIZE;
            if (expert < n_experts) {
                row_logits[expert] = wt[i];
            }
        }
        __syncwarp();
    }

    float wt_sum = 0.f;
    float output_weights[experts_per_thread];

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        output_weights[i] = 0.f;
    }

    // Top-k selection with optimized reduction
    for (int k = 0; k < n_expert_used; k++) {
        float max_val = wt[0];
        int   max_expert_local = lane;

        // Find local max across this thread's experts
#pragma unroll
        for (int i = 1; i < experts_per_thread; i++) {
            const int expert = lane + i * WARP_SIZE;
            if ((n_experts % WARP_SIZE == 0 || expert < n_experts) && wt[i] > max_val) {
                max_val = wt[i];
                max_expert_local = expert;
            }
        }

        // Warp-wide reduction to find global max
        int max_expert;
        max_val = warp_reduce_max_with_idx(max_val, max_expert_local, &max_expert);

        // Store result
        if ((k & (WARP_SIZE - 1)) == lane) {
            output_weights[k / WARP_SIZE] = max_val;
        }

        if (blockIdx.x == 0 && threadIdx.y == 0 && lane == 0) {
             printf("topk_opt: row=%d k=%d max_expert=%d max_val=%f wt[0]=%f logits[0]=%f\n",
                    row, k, max_expert, max_val, wt[0], row_logits_global[0]);
        }

        // Mark selected expert as used
        if ((max_expert & (WARP_SIZE - 1)) == lane) {
            wt[max_expert / WARP_SIZE] = -INFINITY;
            row_ids[k] = max_expert;

            if constexpr (with_norm) {
                wt_sum += max_val;
            }

            // Update shared memory
            row_logits[max_expert] = -INFINITY;
        }
        __syncwarp();
    }

    // Apply normalization
    if constexpr (with_norm) {
        wt_sum = warp_reduce_sum(wt_sum);
        wt_sum = max(wt_sum, clamp_val);
        const float inv_sum = 1.0f / wt_sum;

#pragma unroll
        for (int i = 0; i < experts_per_thread; i++) {
            output_weights[i] *= inv_sum;
        }
    }

    // Apply delayed softmax if needed
    if constexpr (delayed_softmax) {
        softmax_warp_inplace<experts_per_thread, true>(output_weights, n_expert_used, lane);
    }

    // Write output weights
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int idx = i * WARP_SIZE + lane;
        if (idx < n_expert_used) {
            row_weights[idx] = output_weights[i];
        }
    }

    if constexpr (!with_norm) {
        GGML_UNUSED(clamp_val);
    }
}

// Dispatch macro for optimized path
#define LAUNCH_MOE_OPTIMIZED(N_EXPERTS)                                                                     \
    topk_moe_cuda_optimized<N_EXPERTS, MOE_ROWS_PER_BLOCK_OPTIMIZED, with_norm, delayed_softmax>            \
        <<<grid_dims, block_dims, smem_size, stream>>>(logits, weights, ids, n_rows, n_expert_used, clamp_val)

// Dispatch macro for legacy path
#define LAUNCH_MOE_LEGACY(N_EXPERTS)                                                                        \
    topk_moe_cuda<N_EXPERTS, with_norm, delayed_softmax>                                                    \
        <<<grid_dims, block_dims, 0, stream>>>(logits, weights, ids, n_rows, n_expert_used, clamp_val)

template <bool with_norm, bool delayed_softmax = false>
static void launch_topk_moe_cuda(ggml_backend_cuda_context & ctx,
                                 const float *               logits,
                                 float *                     weights,
                                 int32_t *                   ids,
                                 const int                   n_rows,
                                 const int                   n_expert,
                                 const int                   n_expert_used,
                                 const float                 clamp_val) {
    static_assert(!(with_norm && delayed_softmax), "delayed softmax is not supported with weight normalization");

    cudaStream_t stream = ctx.stream();
    const int    cc = ggml_cuda_info().devices[ctx.device].cc;

    // Use optimized path for Ampere+ (sm_80+) with shared memory caching
    // Blackwell (sm_120) also uses this path but benefits from larger shared memory
    const bool use_optimized_path = (cc >= GGML_CUDA_CC_AMPERE) && (n_expert >= MOE_SMEM_THRESHOLD);

    if (use_optimized_path) {
        // Optimized: 8 rows/block, shared memory caching
        constexpr int rows_per_block = MOE_ROWS_PER_BLOCK_OPTIMIZED;
        const dim3    grid_dims((n_rows + rows_per_block - 1) / rows_per_block, 1, 1);
        const dim3    block_dims(WARP_SIZE, rows_per_block, 1);
        const size_t  smem_size = rows_per_block * n_expert * sizeof(float);

        switch (n_expert) {
            case 32:  LAUNCH_MOE_OPTIMIZED(32);  break;
            case 64:  LAUNCH_MOE_OPTIMIZED(64);  break;
            case 128: LAUNCH_MOE_OPTIMIZED(128); break;
            case 256: LAUNCH_MOE_OPTIMIZED(256); break;
            case 512: LAUNCH_MOE_OPTIMIZED(512); break;
            default:
                GGML_ASSERT(false && "Unsupported n_expert for optimized path");
                break;
        }
    } else {
        // Legacy path: 4 rows/block, no shared memory
        constexpr int rows_per_block = MOE_ROWS_PER_BLOCK_DEFAULT;
        const dim3    grid_dims((n_rows + rows_per_block - 1) / rows_per_block, 1, 1);
        const dim3    block_dims(WARP_SIZE, rows_per_block, 1);

        switch (n_expert) {
            case 1:   LAUNCH_MOE_LEGACY(1);   break;
            case 2:   LAUNCH_MOE_LEGACY(2);   break;
            case 4:   LAUNCH_MOE_LEGACY(4);   break;
            case 8:   LAUNCH_MOE_LEGACY(8);   break;
            case 16:  LAUNCH_MOE_LEGACY(16);  break;
            case 32:  LAUNCH_MOE_LEGACY(32);  break;
            case 64:  LAUNCH_MOE_LEGACY(64);  break;
            case 128: LAUNCH_MOE_LEGACY(128); break;
            case 256: LAUNCH_MOE_LEGACY(256); break;
            case 512: LAUNCH_MOE_LEGACY(512); break;
            default:
                GGML_ASSERT(false && "Unsupported n_expert");
                break;
        }
    }
}

#undef LAUNCH_MOE_OPTIMIZED
#undef LAUNCH_MOE_LEGACY

void ggml_cuda_op_topk_moe(ggml_backend_cuda_context & ctx,
                           const ggml_tensor *         logits,
                           ggml_tensor *               weights,
                           ggml_tensor *               ids,
                           const bool                  with_norm,
                           const bool                  delayed_softmax,
                           ggml_tensor *               clamp) {
    GGML_ASSERT(logits->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    const int n_experts = logits->ne[0];
    const int n_rows    = logits->ne[1];

    const float * logits_d  = (const float *) logits->data;
    float *       weights_d = (float *) weights->data;
    int32_t *     ids_d     = (int32_t *) ids->data;

    GGML_ASSERT(ids->nb[1] / ggml_type_size(ids->type) == (size_t) n_experts);

    const int n_expert_used = weights->ne[1];

    float clamp_val = -INFINITY;
    if (with_norm) {
        if (clamp) {
            clamp_val = ggml_get_op_params_f32(clamp, 0);
        }
        launch_topk_moe_cuda<true>(ctx, logits_d, weights_d, ids_d, n_rows, n_experts, n_expert_used, clamp_val);
    } else {
        GGML_ASSERT(clamp == nullptr);
        if (delayed_softmax) {
            launch_topk_moe_cuda<false, true>(ctx, logits_d, weights_d, ids_d, n_rows, n_experts, n_expert_used,
                                              clamp_val);
        } else {
            launch_topk_moe_cuda<false, false>(ctx, logits_d, weights_d, ids_d, n_rows, n_experts, n_expert_used,
                                               clamp_val);
        }
    }
}

bool ggml_cuda_should_use_topk_moe(const ggml_tensor * softmax,
                                   const ggml_tensor * weights,
                                   const ggml_tensor * get_rows,
                                   const ggml_tensor * argsort,
                                   const ggml_tensor * clamp,
                                   int n_expert) {
    ggml_tensor * probs = get_rows->src[0];
    if (probs->op != GGML_OP_RESHAPE) {
        return false;
    }
    probs = probs->src[0];
    ggml_tensor * selection_probs = argsort->src[0];

    if (probs != selection_probs) {
        return false;
    }

    float scale    = 1.0f;
    float max_bias = 0.0f;

    memcpy(&scale, (const float *) softmax->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) softmax->op_params + 1, sizeof(float));

    if (!ggml_is_contiguous(softmax->src[0]) || !ggml_is_contiguous(weights)) {
        return false;
    }

    if (scale != 1.0f || max_bias != 0.0f) {
        return false;
    }

    // don't fuse when masks or sinks are present
    if (softmax->src[1] || softmax->src[2]) {
        return false;
    }

    // n_expert must be a power of 2
    if ((n_expert & (n_expert - 1)) != 0 || n_expert > 512) {
        return false;
    }

    if (clamp) {
        if (clamp->op != GGML_OP_CLAMP) {
            return false;
        }
        float max_val = ggml_get_op_params_f32(clamp, 1);

        if (max_val != INFINITY) {
            return false;
        }
    }


    return true;
}

std::initializer_list<enum ggml_op> ggml_cuda_topk_moe_ops(bool norm, bool delayed_softmax) {
    static std::initializer_list<enum ggml_op> norm_ops = { GGML_OP_SOFT_MAX, GGML_OP_RESHAPE,  GGML_OP_ARGSORT,
                                                            GGML_OP_VIEW,     GGML_OP_GET_ROWS, GGML_OP_RESHAPE,
                                                            GGML_OP_SUM_ROWS, GGML_OP_CLAMP,    GGML_OP_DIV,
                                                            GGML_OP_RESHAPE };

    static std::initializer_list<enum ggml_op> no_norm_ops = { GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT,
                                                               GGML_OP_VIEW, GGML_OP_GET_ROWS };

    static std::initializer_list<enum ggml_op> delayed_softmax_ops = { GGML_OP_ARGSORT,  GGML_OP_VIEW,
                                                                       GGML_OP_GET_ROWS, GGML_OP_RESHAPE,
                                                                       GGML_OP_SOFT_MAX, GGML_OP_RESHAPE };

    GGML_ASSERT(!norm || !delayed_softmax);

    if (delayed_softmax) {
        return delayed_softmax_ops;
    }

    if (norm) {
        return norm_ops;
    }

    return no_norm_ops;
}
