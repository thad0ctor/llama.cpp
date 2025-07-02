#include "blackwell-attention.cuh"
#include "common.cuh"

// Blackwell L2 cache features require CUDA 11.8+ and compute capability 12.0+
#if CUDART_VERSION >= 11080 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200

#include <cooperative_groups.h>
#include <cuda/std/cmath>

namespace cg = cooperative_groups;

// L2 Cache Utilization for Attention Mechanisms (128MB on RTX 5090)
// Target: Flash Attention improvements for long sequences
// Benefits: Better performance for sequence lengths 2048+ on large models

// Attention configuration optimized for RTX 5090 L2 cache (128MB)
// Conservative tile sizes to guarantee shared memory fit (49KB max)
constexpr int ATTN_TILE_SIZE_Q = 16;           // Query tile size (very conservative)
constexpr int ATTN_TILE_SIZE_K = 16;           // Key tile size (very conservative)  
constexpr int ATTN_TILE_SIZE_V = 16;           // Value tile size (very conservative)
constexpr int ATTN_BLOCK_SIZE = 256;           // Threads per block
constexpr int BLACKWELL_BLACKWELL_ATTN_WARP_SIZE = 32;   // Avoid macro conflicts
constexpr int ATTN_WARPS_PER_BLOCK = 8;       // 256/32 = 8 warps

// L2 cache-aware Flash Attention kernel for long sequences
template<typename T, int HEAD_DIM>
__global__ void blackwell_l2_flash_attention_kernel(
    const T* __restrict__ Q,        // Query [seq_len, num_heads, head_dim]
    const T* __restrict__ K,        // Key [seq_len, num_heads, head_dim]  
    const T* __restrict__ V,        // Value [seq_len, num_heads, head_dim]
    T* __restrict__ O,              // Output [seq_len, num_heads, head_dim]
    float* __restrict__ L,          // Row sum normalization [seq_len, num_heads]
    float* __restrict__ M,          // Row max [seq_len, num_heads]
    const int seq_len,
    const int num_heads,
    const float scale) {
    
    const int batch_id = blockIdx.z;
    const int head_id = blockIdx.y;
    const int q_block_id = blockIdx.x;
    
    const int tid = threadIdx.x;
    const int warp_id = tid / BLACKWELL_BLACKWELL_ATTN_WARP_SIZE;
    const int lane_id = tid % BLACKWELL_BLACKWELL_ATTN_WARP_SIZE;
    
    // Bounds checking
    if (head_id >= num_heads) return;
    
    const int q_start = q_block_id * ATTN_TILE_SIZE_Q;
    const int q_end = min(q_start + ATTN_TILE_SIZE_Q, seq_len);
    
    // Shared memory for L2 cache optimization
    __shared__ T Q_tile[ATTN_TILE_SIZE_Q][HEAD_DIM];
    __shared__ T K_tile[ATTN_TILE_SIZE_K][HEAD_DIM];
    __shared__ T V_tile[ATTN_TILE_SIZE_V][HEAD_DIM];
    __shared__ float S_tile[ATTN_TILE_SIZE_Q][ATTN_TILE_SIZE_K];
    __shared__ float rowmax[ATTN_TILE_SIZE_Q];
    __shared__ float rowsum[ATTN_TILE_SIZE_Q];
    
    // Load Q tile into shared memory (coalesced access)
    if (warp_id == 0) {
        for (int i = lane_id; i < ATTN_TILE_SIZE_Q * HEAD_DIM; i += BLACKWELL_BLACKWELL_ATTN_WARP_SIZE) {
            const int q_row = i / HEAD_DIM;
            const int q_col = i % HEAD_DIM;
            const int global_q_row = q_start + q_row;
            
            if (global_q_row < seq_len && q_row < ATTN_TILE_SIZE_Q) {
                Q_tile[q_row][q_col] = Q[batch_id * seq_len * num_heads * HEAD_DIM + 
                                         global_q_row * num_heads * HEAD_DIM + 
                                         head_id * HEAD_DIM + q_col];
            }
        }
    }
    
    // Initialize output accumulators
    float O_acc[HEAD_DIM] = {0.0f};
    float l_acc = 0.0f;
    float m_acc = -INFINITY;
    
    // Process K,V tiles (outer loop optimized for L2 cache)
    for (int k_start = 0; k_start < seq_len; k_start += ATTN_TILE_SIZE_K) {
        const int k_end = min(k_start + ATTN_TILE_SIZE_K, seq_len);
        const int k_size = k_end - k_start;
        
        __syncthreads();
        
        // Load K tile into shared memory
        if (warp_id == 1) {
            for (int i = lane_id; i < k_size * HEAD_DIM; i += BLACKWELL_ATTN_WARP_SIZE) {
                const int k_row = i / HEAD_DIM;
                const int k_col = i % HEAD_DIM;
                const int global_k_row = k_start + k_row;
                
                if (global_k_row < seq_len && k_row < ATTN_TILE_SIZE_K) {
                    K_tile[k_row][k_col] = K[batch_id * seq_len * num_heads * HEAD_DIM + 
                                             global_k_row * num_heads * HEAD_DIM + 
                                             head_id * HEAD_DIM + k_col];
                }
            }
        }
        
        // Load V tile into shared memory
        if (warp_id == 2) {
            for (int i = lane_id; i < k_size * HEAD_DIM; i += BLACKWELL_ATTN_WARP_SIZE) {
                const int v_row = i / HEAD_DIM;
                const int v_col = i % HEAD_DIM;
                const int global_v_row = k_start + v_row;
                
                if (global_v_row < seq_len && v_row < ATTN_TILE_SIZE_V) {
                    V_tile[v_row][v_col] = V[batch_id * seq_len * num_heads * HEAD_DIM + 
                                             global_v_row * num_heads * HEAD_DIM + 
                                             head_id * HEAD_DIM + v_col];
                }
            }
        }
        
        __syncthreads();
        
        // Compute attention scores S = Q @ K^T (within shared memory)
        for (int q_idx = warp_id; q_idx < (q_end - q_start); q_idx += ATTN_WARPS_PER_BLOCK) {
            for (int k_idx = lane_id; k_idx < k_size; k_idx += BLACKWELL_ATTN_WARP_SIZE) {
                float score = 0.0f;
                
                // Dot product Q[q_idx] · K[k_idx]
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    score += float(Q_tile[q_idx][d]) * float(K_tile[k_idx][d]);
                }
                
                S_tile[q_idx][k_idx] = score * scale;
            }
        }
        
        __syncthreads();
        
        // Compute row-wise softmax with numerically stable implementation
        for (int q_idx = warp_id; q_idx < (q_end - q_start); q_idx += ATTN_WARPS_PER_BLOCK) {
            // Find row maximum
            float row_max = -INFINITY;
            for (int k_idx = lane_id; k_idx < k_size; k_idx += BLACKWELL_ATTN_WARP_SIZE) {
                row_max = fmaxf(row_max, S_tile[q_idx][k_idx]);
            }
            
            // Warp-level reduction for row maximum
            #pragma unroll
            for (int offset = BLACKWELL_ATTN_WARP_SIZE / 2; offset > 0; offset /= 2) {
                row_max = fmaxf(row_max, __shfl_down_sync(0xffffffff, row_max, offset));
            }
            
            if (lane_id == 0) {
                rowmax[q_idx] = row_max;
            }
            
            __syncthreads();
            
            // Compute row sum after subtracting max
            float row_sum = 0.0f;
            for (int k_idx = lane_id; k_idx < k_size; k_idx += BLACKWELL_ATTN_WARP_SIZE) {
                const float exp_val = expf(S_tile[q_idx][k_idx] - rowmax[q_idx]);
                S_tile[q_idx][k_idx] = exp_val;
                row_sum += exp_val;
            }
            
            // Warp-level reduction for row sum
            #pragma unroll
            for (int offset = BLACKWELL_ATTN_WARP_SIZE / 2; offset > 0; offset /= 2) {
                row_sum += __shfl_down_sync(0xffffffff, row_sum, offset);
            }
            
            if (lane_id == 0) {
                rowsum[q_idx] = row_sum;
            }
        }
        
        __syncthreads();
        
        // Compute output O += S @ V (accumulate across K tiles)
        for (int q_idx = warp_id; q_idx < (q_end - q_start); q_idx += ATTN_WARPS_PER_BLOCK) {
            for (int d = lane_id; d < HEAD_DIM; d += BLACKWELL_ATTN_WARP_SIZE) {
                float o_val = 0.0f;
                
                // Matrix multiplication S[q_idx, :] @ V[:, d]
                for (int k_idx = 0; k_idx < k_size; ++k_idx) {
                    const float s_val = S_tile[q_idx][k_idx] / rowsum[q_idx];
                    o_val += s_val * float(V_tile[k_idx][d]);
                }
                
                // Update running statistics for multi-tile attention
                const float new_max = fmaxf(m_acc, rowmax[q_idx]);
                const float exp_old = expf(m_acc - new_max);
                const float exp_new = expf(rowmax[q_idx] - new_max);
                
                // Update output with corrected values
                O_acc[d] = O_acc[d] * exp_old + o_val * exp_new;
                
                if (d == 0) {
                    l_acc = l_acc * exp_old + rowsum[q_idx] * exp_new;
                    m_acc = new_max;
                }
            }
        }
    }
    
    // Write final output
    for (int q_idx = warp_id; q_idx < (q_end - q_start); q_idx += ATTN_WARPS_PER_BLOCK) {
        const int global_q_idx = q_start + q_idx;
        
        for (int d = lane_id; d < HEAD_DIM; d += BLACKWELL_ATTN_WARP_SIZE) {
            O[batch_id * seq_len * num_heads * HEAD_DIM + 
              global_q_idx * num_heads * HEAD_DIM + 
              head_id * HEAD_DIM + d] = T(O_acc[d] / l_acc);
        }
        
        if (lane_id == 0) {
            L[batch_id * seq_len * num_heads + global_q_idx * num_heads + head_id] = l_acc;
            M[batch_id * seq_len * num_heads + global_q_idx * num_heads + head_id] = m_acc;
        }
    }
}

// Host function to launch L2-optimized Flash Attention
void launch_blackwell_l2_flash_attention(
    const void* Q, const void* K, const void* V,
    void* O, void* L, void* M,
    int batch_size, int seq_len, int num_heads, int head_dim,
    float scale, ggml_type type, cudaStream_t stream) {
    
    // Grid configuration optimized for L2 cache utilization
    const int q_blocks = (seq_len + ATTN_TILE_SIZE_Q - 1) / ATTN_TILE_SIZE_Q;
    
    dim3 grid_size(q_blocks, num_heads, batch_size);
    dim3 block_size(ATTN_BLOCK_SIZE);
    
    // Launch kernel based on head dimension
    if (type == GGML_TYPE_F16) {
        if (head_dim == 64) {
            blackwell_l2_flash_attention_kernel<half, 64><<<grid_size, block_size, 0, stream>>>(
                (const half*)Q, (const half*)K, (const half*)V,
                (half*)O, (float*)L, (float*)M,
                seq_len, num_heads, scale);
        } else if (head_dim == 128) {
            blackwell_l2_flash_attention_kernel<half, 128><<<grid_size, block_size, 0, stream>>>(
                (const half*)Q, (const half*)K, (const half*)V,
                (half*)O, (float*)L, (float*)M,
                seq_len, num_heads, scale);
        }
    } else if (type == GGML_TYPE_F32) {
        if (head_dim == 64) {
            blackwell_l2_flash_attention_kernel<float, 64><<<grid_size, block_size, 0, stream>>>(
                (const float*)Q, (const float*)K, (const float*)V,
                (float*)O, (float*)L, (float*)M,
                seq_len, num_heads, scale);
        } else if (head_dim == 128) {
            blackwell_l2_flash_attention_kernel<float, 128><<<grid_size, block_size, 0, stream>>>(
                (const float*)Q, (const float*)K, (const float*)V,
                (float*)O, (float*)L, (float*)M,
                seq_len, num_heads, scale);
        }
    }
    
    CUDA_CHECK(cudaGetLastError());
}

// High-level interface for Blackwell Flash Attention
void ggml_cuda_flash_attn_blackwell_optimized(
    ggml_backend_cuda_context & ctx,
    ggml_tensor * dst) {
    
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    
    const int64_t ne0 = Q->ne[0]; // head_dim
    const int64_t ne1 = Q->ne[1]; // seq_len
    const int64_t ne2 = Q->ne[2]; // num_heads
    const int64_t ne3 = Q->ne[3]; // batch_size
    
    const float scale = 1.0f / sqrtf((float)ne0);
    
    // Allocate temporary storage for L and M
    ggml_cuda_pool_alloc<float> L_tmp(ctx.pool(), ne1 * ne2 * ne3);
    ggml_cuda_pool_alloc<float> M_tmp(ctx.pool(), ne1 * ne2 * ne3);
    
    // Launch optimized Flash Attention
    launch_blackwell_l2_flash_attention(
        Q->data, K->data, V->data,
        dst->data, L_tmp.get(), M_tmp.get(),
        (int)ne3, (int)ne1, (int)ne2, (int)ne0,
        scale, Q->type, ctx.stream()
    );
}

// Performance threshold for using L2-optimized attention
bool should_use_l2_flash_attention(const ggml_tensor * src0, int device_id) {
    // Only use on Blackwell GPUs with large L2 cache
    if (!ggml_cuda_can_use_large_shared_memory(device_id)) {
        return false;
    }
    
    const int64_t seq_len = src0->ne[1];
    const int64_t head_dim = src0->ne[0];
    const int64_t num_heads = src0->ne[2];
    
    // Performance thresholds for L2 cache benefits
    const bool long_sequence = seq_len >= 2048;        // Long sequences benefit most
    const bool large_attention = (seq_len * num_heads >= 8192); // High memory pressure
    const bool standard_head_dim = (head_dim == 64 || head_dim == 128); // Optimized dimensions
    
    return long_sequence && large_attention && standard_head_dim;
}

#else // CUDART_VERSION < 11080 || __CUDA_ARCH__ < 1200

// Fallback implementations for older CUDA versions
void ggml_cuda_flash_attn_blackwell_optimized(
    ggml_backend_cuda_context & ctx,
    ggml_tensor * dst) {
    GGML_UNUSED(ctx); GGML_UNUSED(dst);
    GGML_ABORT("Blackwell Flash Attention not supported on this CUDA version or architecture");
}

void launch_blackwell_l2_flash_attention(
    const void* Q, const void* K, const void* V,
    void* O, void* L, void* M,
    int batch_size, int seq_len, int num_heads, int head_dim,
    float scale, ggml_type type, cudaStream_t stream) {
    GGML_UNUSED(Q); GGML_UNUSED(K); GGML_UNUSED(V); GGML_UNUSED(O); GGML_UNUSED(L); GGML_UNUSED(M);
    GGML_UNUSED(batch_size); GGML_UNUSED(seq_len); GGML_UNUSED(num_heads); GGML_UNUSED(head_dim);
    GGML_UNUSED(scale); GGML_UNUSED(type); GGML_UNUSED(stream);
    GGML_ABORT("Blackwell Flash Attention not supported on this CUDA version or architecture");
}

bool should_use_l2_flash_attention(const ggml_tensor * src0, int device_id) {
    GGML_UNUSED(src0); GGML_UNUSED(device_id);
    return false; // Never use on older systems
}

#endif // CUDART_VERSION >= 11080 && __CUDA_ARCH__ >= 1200 