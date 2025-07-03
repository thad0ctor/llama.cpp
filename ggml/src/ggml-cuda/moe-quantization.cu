#include "moe-quantization.cuh"
#include "common.cuh"

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Blackwell RTX 5090 optimized MoE quantization kernels
// Based on vLLM's token permutation approach

//
// INT4 Quantization Kernels
//

__device__ __forceinline__ float4 load_float4(const float* ptr) {
    return *reinterpret_cast<const float4*>(ptr);
}

__device__ __forceinline__ void store_float4(float* ptr, float4 val) {
    *reinterpret_cast<float4*>(ptr) = val;
}

__device__ __forceinline__ uint32_t pack_int4_pairs(int8_t a, int8_t b) {
    // Pack two 4-bit values into lower 8 bits
    return ((b & 0xF) << 4) | (a & 0xF);
}

__device__ __forceinline__ void unpack_int4_pairs(uint8_t packed, int8_t& a, int8_t& b) {
    a = static_cast<int8_t>(packed & 0xF);
    b = static_cast<int8_t>((packed >> 4) & 0xF);
    
    // Sign extend from 4-bit to 8-bit
    if (a > 7) a -= 16;
    if (b > 7) b -= 16;
}

// Blackwell-optimized INT4 quantization kernel with block processing
__global__ void moe_quantize_int4_kernel(
    const float* __restrict__ input,
    uint8_t* __restrict__ output,
    float* __restrict__ scales,
    float* __restrict__ zero_points,
    const int32_t* __restrict__ token_indices,
    int32_t num_tokens,
    int32_t hidden_size,
    int32_t block_size
) {
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t token_id = blockIdx.y;
    
    if (token_id >= num_tokens) return;
    
    // Get actual token index from permutation
    const int32_t actual_token = token_indices[token_id];
    const int32_t block_id = tid / (block_size / 2); // Each thread handles 2 elements
    const int32_t block_offset = tid % (block_size / 2);
    
    if (block_id * block_size >= hidden_size) return;
    
    // Shared memory for block statistics
    __shared__ float block_min[32];  // Support up to 32 blocks per SM
    __shared__ float block_max[32];
    __shared__ float block_scale[32];
    __shared__ float block_zero[32];
    
    const int32_t warp_id = threadIdx.x / 32;
    const int32_t lane_id = threadIdx.x % 32;
    
    // Load input data for this block
    const int32_t base_idx = actual_token * hidden_size + block_id * block_size;
    float2 vals = {0.0f, 0.0f};
    
    if (base_idx + block_offset * 2 < actual_token * hidden_size + min(hidden_size, (block_id + 1) * block_size)) {
        const float* input_ptr = input + base_idx + block_offset * 2;
        vals.x = input_ptr[0];
        if (block_offset * 2 + 1 < block_size && base_idx + block_offset * 2 + 1 < (actual_token + 1) * hidden_size) {
            vals.y = input_ptr[1];
        }
    }
    
    // Compute block-wise min/max using warp reductions
    float local_min = fminf(vals.x, vals.y);
    float local_max = fmaxf(vals.x, vals.y);
    
    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_min = fminf(local_min, __shfl_down_sync(0xFFFFFFFF, local_min, offset));
        local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, offset));
    }
    
    // Store per-warp results
    if (lane_id == 0 && warp_id < 32) {
        block_min[warp_id] = local_min;
        block_max[warp_id] = local_max;
    }
    
    __syncthreads();
    
    // Final reduction across warps
    if (threadIdx.x < 32) {
        local_min = (threadIdx.x < blockDim.x / 32) ? block_min[threadIdx.x] : FLT_MAX;
        local_max = (threadIdx.x < blockDim.x / 32) ? block_max[threadIdx.x] : -FLT_MAX;
        
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            local_min = fminf(local_min, __shfl_down_sync(0xFFFFFFFF, local_min, offset));
            local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, offset));
        }
        
        if (threadIdx.x == 0) {
            // Compute quantization parameters for this block
            float range = local_max - local_min;
            float scale = range / 14.0f; // INT4 range: [-7, 7]
            float zero_point = local_min + 7.0f * scale;
            
            // Avoid division by zero
            if (scale == 0.0f) {
                scale = 1.0f;
                zero_point = 0.0f;
            }
            
            block_scale[0] = scale;
            block_zero[0] = zero_point;
            
            // Store global scale and zero point
            const int32_t global_block_id = token_id * ((hidden_size + block_size - 1) / block_size) + block_id;
            scales[global_block_id] = scale;
            zero_points[global_block_id] = zero_point;
        }
    }
    
    __syncthreads();
    
    // Quantize using computed parameters
    const float scale = block_scale[0];
    const float zero_point = block_zero[0];
    
    // Quantize the two values
    int8_t q1 = static_cast<int8_t>(roundf((vals.x - zero_point) / scale));
    int8_t q2 = static_cast<int8_t>(roundf((vals.y - zero_point) / scale));
    
    // Clamp to 4-bit range [-7, 7]
    q1 = max(-7, min(7, (int)q1));
    q2 = max(-7, min(7, (int)q2));
    
    // Pack and store
    const int32_t output_idx = actual_token * ((hidden_size + 1) / 2) + block_id * (block_size / 2) + block_offset;
    if (output_idx < num_tokens * ((hidden_size + 1) / 2)) {
        output[output_idx] = pack_int4_pairs(q1, q2);
    }
}

// Blackwell-optimized INT4 dequantization kernel
__global__ void moe_dequantize_int4_kernel(
    const uint8_t* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ scales,
    const float* __restrict__ zero_points,
    const int32_t* __restrict__ token_indices,
    int32_t num_tokens,
    int32_t hidden_size,
    int32_t block_size
) {
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t token_id = blockIdx.y;
    
    if (token_id >= num_tokens) return;
    
    const int32_t actual_token = token_indices[token_id];
    const int32_t block_id = tid / (block_size / 2);
    const int32_t block_offset = tid % (block_size / 2);
    
    if (block_id * block_size >= hidden_size) return;
    
    // Load quantization parameters
    const int32_t global_block_id = token_id * ((hidden_size + block_size - 1) / block_size) + block_id;
    const float scale = scales[global_block_id];
    const float zero_point = zero_points[global_block_id];
    
    // Load packed quantized values
    const int32_t input_idx = actual_token * ((hidden_size + 1) / 2) + block_id * (block_size / 2) + block_offset;
    if (input_idx >= num_tokens * ((hidden_size + 1) / 2)) return;
    
    const uint8_t packed = input[input_idx];
    
    // Unpack quantized values
    int8_t q1, q2;
    unpack_int4_pairs(packed, q1, q2);
    
    // Dequantize
    float val1 = static_cast<float>(q1) * scale + zero_point;
    float val2 = static_cast<float>(q2) * scale + zero_point;
    
    // Store dequantized values
    const int32_t output_base = actual_token * hidden_size + block_id * block_size + block_offset * 2;
    if (output_base < (actual_token + 1) * hidden_size) {
        output[output_base] = val1;
        if (output_base + 1 < (actual_token + 1) * hidden_size) {
            output[output_base + 1] = val2;
        }
    }
}

// Token permutation kernel optimized for MoE routing
__global__ void moe_token_permute_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ expert_offsets,
    int32_t num_tokens,
    int32_t hidden_size
) {
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t token_id = blockIdx.y;
    
    if (token_id >= num_tokens || tid >= hidden_size) return;
    
    // Get source and destination indices
    const int32_t src_token = indices[token_id];
    const int32_t dst_offset = expert_offsets[token_id];
    
    // Copy data using vectorized loads for better memory bandwidth
    if (tid % 4 == 0 && tid + 3 < hidden_size) {
        // Load 4 floats at once (float4)
        const float4 data = load_float4(input + src_token * hidden_size + tid);
        store_float4(output + dst_offset * hidden_size + tid, data);
    } else {
        // Handle remaining elements
        output[dst_offset * hidden_size + tid] = input[src_token * hidden_size + tid];
    }
}

// Expert-specific top-K selection kernel (following vLLM approach)
__global__ void moe_topk_selection_kernel(
    const float* __restrict__ gate_logits,
    int32_t* __restrict__ expert_indices,
    float* __restrict__ expert_weights,
    int32_t* __restrict__ token_expert_counts,
    int32_t num_tokens,
    int32_t num_experts,
    int32_t top_k
) {
    const int32_t token_id = blockIdx.x;
    
    if (token_id >= num_tokens) return;
    
    // Shared memory for this token's logits and indices
    extern __shared__ float shared_mem[];
    float* logits = shared_mem;
    int32_t* indices = reinterpret_cast<int32_t*>(shared_mem + num_experts);
    
    // Load logits for this token
    const float* token_logits = gate_logits + token_id * num_experts;
    
    for (int32_t i = threadIdx.x; i < num_experts; i += blockDim.x) {
        logits[i] = token_logits[i];
        indices[i] = i;
    }
    
    __syncthreads();
    
    // Parallel bitonic sort for top-K selection
    for (int32_t k = 2; k <= num_experts; k <<= 1) {
        for (int32_t j = k >> 1; j > 0; j >>= 1) {
            for (int32_t i = threadIdx.x; i < num_experts; i += blockDim.x) {
                int32_t ij = i ^ j;
                if (ij > i && ij < num_experts) {
                    bool ascending = ((i & k) == 0);
                    if ((logits[i] > logits[ij]) == ascending) {
                        // Swap logits and indices
                        float temp_logit = logits[i];
                        logits[i] = logits[ij];
                        logits[ij] = temp_logit;
                        
                        int32_t temp_idx = indices[i];
                        indices[i] = indices[ij];
                        indices[ij] = temp_idx;
                    }
                }
            }
            __syncthreads();
        }
    }
    
    // Compute softmax for top-K experts
    if (threadIdx.x == 0) {
        float max_logit = logits[num_experts - 1]; // Largest logit after sort
        float sum_exp = 0.0f;
        
        // Compute exp and sum for top-K
        for (int32_t k = 0; k < top_k; ++k) {
            int32_t idx = num_experts - 1 - k; // Top-K are at the end after sort
            float exp_val = expf(logits[idx] - max_logit);
            sum_exp += exp_val;
            logits[idx] = exp_val; // Store exp value temporarily
        }
        
        // Normalize and store results
        for (int32_t k = 0; k < top_k; ++k) {
            int32_t idx = num_experts - 1 - k;
            int32_t output_idx = token_id * top_k + k;
            
            expert_indices[output_idx] = indices[idx];
            expert_weights[output_idx] = logits[idx] / sum_exp;
        }
        
        // Update expert token counts (atomic)
        for (int32_t k = 0; k < top_k; ++k) {
            int32_t expert_id = expert_indices[token_id * top_k + k];
            atomicAdd(&token_expert_counts[expert_id], 1);
        }
    }
}

// Blackwell-specific memory coalescing optimization
__global__ void moe_coalesce_expert_data_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int32_t* __restrict__ expert_assignments,
    const int32_t* __restrict__ expert_offsets,
    int32_t num_tokens,
    int32_t hidden_size,
    int32_t num_experts
) {
    // Use Blackwell's advanced memory subsystem for optimal coalescing
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t expert_id = blockIdx.y;
    
    if (expert_id >= num_experts) return;
    
    // Process tokens assigned to this expert
    const int32_t expert_start = expert_offsets[expert_id];
    const int32_t expert_end = (expert_id + 1 < num_experts) ? 
                               expert_offsets[expert_id + 1] : num_tokens;
    const int32_t expert_tokens = expert_end - expert_start;
    
    // Use block-strided access pattern for optimal HBM3 bandwidth
    for (int32_t token_offset = 0; token_offset < expert_tokens; token_offset += blockDim.x) {
        const int32_t local_token = token_offset + threadIdx.x;
        if (local_token >= expert_tokens) break;
        
        const int32_t global_token = expert_start + local_token;
        const int32_t src_token = expert_assignments[global_token];
        
        // Vectorized copy using shared memory staging
        __shared__ float4 shared_buffer[256]; // 4KB shared memory per block
        
        // Load with coalesced access
        if (tid < hidden_size / 4) {
            shared_buffer[threadIdx.x] = load_float4(input + src_token * hidden_size + tid * 4);
        }
        
        __syncthreads();
        
        // Store with coalesced access
        if (tid < hidden_size / 4) {
            store_float4(output + global_token * hidden_size + tid * 4, shared_buffer[threadIdx.x]);
        }
        
        __syncthreads();
    }
}

//
// Host Functions (C interface)
//

extern "C" {

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
) {
    const int32_t threads_per_block = 256;
    const int32_t blocks_per_token = (hidden_size / 2 + threads_per_block - 1) / threads_per_block;
    
    dim3 grid(blocks_per_token, num_tokens);
    dim3 block(threads_per_block);
    
    moe_quantize_int4_kernel<<<grid, block, 0, stream>>>(
        input, output, scales, zero_points, token_indices,
        num_tokens, hidden_size, block_size
    );
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(error));
    }
}

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
) {
    const int32_t threads_per_block = 256;
    const int32_t blocks_per_token = (hidden_size / 2 + threads_per_block - 1) / threads_per_block;
    
    dim3 grid(blocks_per_token, num_tokens);
    dim3 block(threads_per_block);
    
    moe_dequantize_int4_kernel<<<grid, block, 0, stream>>>(
        input, output, scales, zero_points, token_indices,
        num_tokens, hidden_size, block_size
    );
}

void launch_moe_token_permute_kernel(
    const float* input,
    float* output,
    const int32_t* indices,
    const int32_t* expert_offsets,
    int32_t num_tokens,
    int32_t hidden_size,
    cudaStream_t stream
) {
    const int32_t threads_per_block = 256;
    const int32_t blocks_per_dim = (hidden_size + threads_per_block - 1) / threads_per_block;
    
    dim3 grid(blocks_per_dim, num_tokens);
    dim3 block(threads_per_block);
    
    moe_token_permute_kernel<<<grid, block, 0, stream>>>(
        input, output, indices, expert_offsets, num_tokens, hidden_size
    );
}

void launch_moe_topk_selection_kernel(
    const float* gate_logits,
    int32_t* expert_indices,
    float* expert_weights,
    int32_t* token_expert_counts,
    int32_t num_tokens,
    int32_t num_experts,
    int32_t top_k,
    cudaStream_t stream
) {
    const int32_t threads_per_block = min(num_experts, 1024);
    const size_t shared_mem_size = sizeof(float) * num_experts + sizeof(int32_t) * num_experts;
    
    dim3 grid(num_tokens);
    dim3 block(threads_per_block);
    
    moe_topk_selection_kernel<<<grid, block, shared_mem_size, stream>>>(
        gate_logits, expert_indices, expert_weights, token_expert_counts,
        num_tokens, num_experts, top_k
    );
}

} // extern "C"