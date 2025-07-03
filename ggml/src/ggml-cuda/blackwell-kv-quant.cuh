#pragma once

#include "common.cuh"

// Blackwell KV Cache Quantization Kernels
// Phase 3: Memory bandwidth optimization through quantized KV storage
// Target: RTX 5090 with HBM3e memory and 128MB L2 cache

// Quantization parameters for KV cache compression
struct blackwell_kv_quant_params {
    float scale_k;              // K tensor quantization scale
    float scale_v;              // V tensor quantization scale
    float zero_point_k;         // K tensor zero point
    float zero_point_v;         // V tensor zero point
    int quant_level;            // 0=none, 1=INT8, 2=INT4
    bool use_asymmetric;        // Use asymmetric quantization
    bool use_block_quant;       // Use block-wise quantization for better quality
};

// Statistics for monitoring quantization quality
struct blackwell_kv_quant_stats {
    float mse_k;                // Mean squared error for K tensor
    float mse_v;                // Mean squared error for V tensor
    float compression_ratio;    // Achieved compression ratio
    float bandwidth_improvement; // Memory bandwidth improvement
};

// KV Cache quantization functions optimized for Blackwell RTX 5090
namespace blackwell_kv_quant {

    // INT8 quantization kernels
    void quantize_kv_cache_int8(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * k_src, const ggml_tensor * v_src,
        ggml_tensor * k_dst, ggml_tensor * v_dst,
        blackwell_kv_quant_params & params,
        int layer_id, int cache_slot_start, int cache_slot_count);
    
    void dequantize_kv_cache_int8(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * k_src, const ggml_tensor * v_src,
        ggml_tensor * k_dst, ggml_tensor * v_dst,
        const blackwell_kv_quant_params & params,
        int layer_id, int cache_slot_start, int cache_slot_count);
    
    // INT4 quantization kernels  
    void quantize_kv_cache_int4(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * k_src, const ggml_tensor * v_src,
        ggml_tensor * k_dst, ggml_tensor * v_dst,
        blackwell_kv_quant_params & params,
        int layer_id, int cache_slot_start, int cache_slot_count);
    
    void dequantize_kv_cache_int4(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * k_src, const ggml_tensor * v_src,
        ggml_tensor * k_dst, ggml_tensor * v_dst,
        const blackwell_kv_quant_params & params,
        int layer_id, int cache_slot_start, int cache_slot_count);
    
    // Adaptive quantization with quality monitoring
    void adaptive_quantize_kv_cache(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * k_src, const ggml_tensor * v_src,
        ggml_tensor * k_dst, ggml_tensor * v_dst,
        blackwell_kv_quant_params & params,
        blackwell_kv_quant_stats & stats,
        int layer_id, int cache_slot_start, int cache_slot_count,
        float quality_threshold);
    
    // Streaming quantization for long contexts
    void stream_quantize_kv_cache(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * k_src, const ggml_tensor * v_src,
        ggml_tensor * k_dst, ggml_tensor * v_dst,
        blackwell_kv_quant_params & params,
        int layer_id, int stream_offset, int stream_length);
    
    // Importance-based compression for distant tokens
    void compress_distant_kv_tokens(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * k_cache, ggml_tensor * v_cache,
        const float * importance_scores,
        int layer_id, int current_pos, int distance_threshold);
    
    // HBM3e optimized batch quantization
    void batch_quantize_kv_layers(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor ** k_layers, const ggml_tensor ** v_layers,
        ggml_tensor ** k_quantized, ggml_tensor ** v_quantized,
        blackwell_kv_quant_params * layer_params,
        int num_layers, int cache_slot_start, int cache_slot_count);
    
    // Quality assessment and parameter tuning
    float assess_quantization_quality(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * original_k, const ggml_tensor * original_v,
        const ggml_tensor * quantized_k, const ggml_tensor * quantized_v,
        blackwell_kv_quant_stats & stats);
    
    // Performance benchmarking
    float benchmark_quantization_performance(
        ggml_backend_cuda_context & ctx,
        int n_layers, int cache_size, int head_dim,
        int quant_level, int num_iterations);

} // namespace blackwell_kv_quant

// Low-level kernel declarations (internal use)
namespace blackwell_kv_quant_kernels {

    // Core quantization kernels optimized for tensor cores
    template<typename T_src, typename T_dst, int BLOCK_SIZE>
    void launch_quantize_kernel(
        const T_src* src, T_dst* dst, 
        float scale, float zero_point,
        int rows, int cols, int ld_src, int ld_dst,
        cudaStream_t stream);
    
    template<typename T_src, typename T_dst, int BLOCK_SIZE>
    void launch_dequantize_kernel(
        const T_src* src, T_dst* dst,
        float scale, float zero_point,
        int rows, int cols, int ld_src, int ld_dst,
        cudaStream_t stream);
    
    // HBM3e memory bandwidth optimized transfers
    void launch_hbm3_coalesced_copy(
        const void* src, void* dst, size_t bytes,
        int coalescing_factor, cudaStream_t stream);
    
    // L2 cache utilization for frequently accessed KV data
    void launch_l2_cache_prefetch(
        const void* kv_data, size_t data_size,
        int prefetch_distance, cudaStream_t stream);

} // namespace blackwell_kv_quant_kernels 