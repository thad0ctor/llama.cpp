#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

//
// CUDA kernel declarations for MoE quantization
// Optimized for Blackwell RTX 5090 architecture
//

#ifdef __cplusplus
extern "C" {
#endif

// INT4 quantization kernels
void launch_moe_quantize_int4_kernel(
    const float* input,
    uint8_t* output,
    float* scales,
    float* zero_points,
    const int32_t* token_indices,
    int32_t num_tokens,
    int32_t hidden_size,
    int32_t block_size,
    cudaStream_t stream
);

void launch_moe_dequantize_int4_kernel(
    const uint8_t* input,
    float* output,
    const float* scales,
    const float* zero_points,
    const int32_t* token_indices,
    int32_t num_tokens,
    int32_t hidden_size,
    int32_t block_size,
    cudaStream_t stream
);

// Token permutation kernels
void launch_moe_token_permute_kernel(
    const float* input,
    float* output,
    const int32_t* indices,
    const int32_t* expert_offsets,
    int32_t num_tokens,
    int32_t hidden_size,
    cudaStream_t stream
);

// Expert selection kernels
void launch_moe_topk_selection_kernel(
    const float* gate_logits,
    int32_t* expert_indices,
    float* expert_weights,
    int32_t* token_expert_counts,
    int32_t num_tokens,
    int32_t num_experts,
    int32_t top_k,
    cudaStream_t stream
);

// Memory coalescing optimization
void launch_moe_coalesce_expert_data_kernel(
    const float* input,
    float* output,
    const int32_t* expert_assignments,
    const int32_t* expert_offsets,
    int32_t num_tokens,
    int32_t hidden_size,
    int32_t num_experts,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif