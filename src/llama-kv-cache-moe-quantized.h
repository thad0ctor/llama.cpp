#pragma once

#include "llama-kv-cache-unified.h"
#include "llama-model.h"
#include "ggml.h"

#include <vector>
#include <unordered_map>
#include <memory>

#ifdef GGML_USE_CUDA
#include <cuda_runtime.h>
#endif

//
// MoE-Aware Quantized KV Cache Implementation
// Based on vLLM's token permutation approach for MoE models
//

// Forward declarations
struct llama_model;
struct llama_hparams;

// Expert routing information per token
struct moe_token_route {
    int32_t expert_id;          // Which expert this token is routed to
    float weight;               // Routing weight for this expert
    int32_t token_idx;          // Original token index
    int32_t expert_token_idx;   // Index within expert's token batch
};

// Per-expert quantization metadata
struct moe_expert_quant_meta {
    // K tensor quantization
    float k_scale;
    float k_zero_point;
    
    // V tensor quantization  
    float v_scale;
    float v_zero_point;
    
    // Memory layout info
    size_t k_offset;            // Offset in quantized K buffer
    size_t v_offset;            // Offset in quantized V buffer
    size_t tokens_count;        // Number of tokens assigned to this expert
    size_t tokens_capacity;     // Allocated capacity for tokens
};

// Block-aligned storage for expert tokens
struct moe_expert_block {
    // Quantized storage (INT4 packed)
    std::vector<uint8_t> k_quantized;
    std::vector<uint8_t> v_quantized;
    
    // Token permutation mapping
    std::vector<int32_t> token_indices;     // Original -> Expert order
    std::vector<int32_t> inverse_indices;   // Expert -> Original order
    
    // Expert metadata
    moe_expert_quant_meta metadata;
    
    // CUDA device pointers (if using GPU)
#ifdef GGML_USE_CUDA
    void* d_k_quantized = nullptr;
    void* d_v_quantized = nullptr;
    void* d_token_indices = nullptr;
#endif
};

// Main MoE quantized KV cache class
class llama_kv_cache_moe_quantized {
public:
    // Quantization parameters
    enum class quant_type_t {
        INT4,
        INT8,
        FP8
    };
    
    struct params {
        quant_type_t k_quant_type = quant_type_t::INT4;
        quant_type_t v_quant_type = quant_type_t::INT4;
        float quality_threshold = 0.02f;  // 2% quality degradation max
        size_t block_size = 64;           // Block size for quantization
        bool use_zero_point = true;       // Use zero-point quantization
        int32_t max_experts = 128;        // Maximum number of experts
        int32_t experts_per_token = 8;    // Top-K experts per token
    };

private:
    // Core data structures
    llama_kv_cache_unified* base_cache;
    const llama_hparams& hparams;
    params quant_params;
    
    // Expert management
    std::vector<moe_expert_block> expert_blocks;
    std::unordered_map<int32_t, size_t> expert_id_to_block;
    
    // Token routing state
    std::vector<moe_token_route> current_routes;
    std::vector<int32_t> sorted_token_indices;
    
    // Memory management
    size_t total_tokens;
    size_t max_context_length;
    bool is_cuda_enabled;

public:
    // Constructor
    llama_kv_cache_moe_quantized(
        llama_kv_cache_unified* base_cache,
        const params& params
    );
    
    // Destructor
    ~llama_kv_cache_moe_quantized();
    
    // Core operations
    bool initialize(const llama_model& model);
    void clear();
    
    // Token routing and expert assignment
    void route_tokens(
        const float* gate_logits,     // Expert gate outputs [seq_len, num_experts]
        int32_t seq_len,
        int32_t num_experts
    );
    
    // KV cache operations with MoE routing
    ggml_tensor* get_k_for_expert(
        ggml_context* ctx,
        int32_t expert_id,
        int32_t layer_id
    );
    
    ggml_tensor* get_v_for_expert(
        ggml_context* ctx, 
        int32_t expert_id,
        int32_t layer_id
    );
    
    // Copy operations with expert routing
    ggml_tensor* cpy_k_moe(
        ggml_context* ctx,
        ggml_tensor* k_cur,
        const float* gate_logits,
        int32_t layer_id,
        int32_t seq_len
    );
    
    ggml_tensor* cpy_v_moe(
        ggml_context* ctx,
        ggml_tensor* v_cur,
        const float* gate_logits,
        int32_t layer_id,
        int32_t seq_len
    );
    
    // Quantization operations
    void quantize_expert_k(
        int32_t expert_id,
        const float* k_data,
        size_t k_size
    );
    
    void quantize_expert_v(
        int32_t expert_id,
        const float* v_data,
        size_t v_size
    );
    
    // Dequantization operations
    void dequantize_expert_k(
        int32_t expert_id,
        float* k_data,
        size_t k_size
    );
    
    void dequantize_expert_v(
        int32_t expert_id,
        float* v_data,
        size_t v_size
    );
    
    // Memory management
    size_t get_memory_usage() const;
    void optimize_memory_layout();
    
    // Expert statistics
    void get_expert_stats(
        std::vector<size_t>& tokens_per_expert,
        std::vector<float>& utilization_per_expert
    ) const;
    
    // Debug and profiling
    void print_routing_stats() const;
    void validate_routing() const;

private:
    // Internal helper methods
    void setup_expert_blocks();
    void allocate_expert_memory(int32_t expert_id, size_t token_count);
    void resize_expert_block(int32_t expert_id, size_t new_capacity);
    
    // Token permutation
    void permute_tokens_by_expert(
        const std::vector<moe_token_route>& routes,
        std::vector<int32_t>& sorted_indices
    );
    
    void unpermute_tokens_from_expert(
        const std::vector<int32_t>& sorted_indices,
        float* output_data,
        size_t data_size
    );
    
    // Quantization helpers
    void compute_quantization_params(
        const float* data,
        size_t size,
        quant_type_t quant_type,
        float& scale,
        float& zero_point
    );
    
    // Top-K expert selection (following vLLM approach)
    void select_top_k_experts(
        const float* gate_logits,
        int32_t seq_len,
        int32_t num_experts,
        int32_t top_k,
        std::vector<moe_token_route>& routes
    );
    
    // Memory alignment for optimal performance
    size_t align_memory_size(size_t size, size_t alignment = 64) const;
    
#ifdef GGML_USE_CUDA
    // CUDA-specific operations
    void setup_cuda_memory();
    void cleanup_cuda_memory();
    void launch_quantization_kernel(
        int32_t expert_id,
        const float* input_data,
        uint8_t* output_data,
        size_t data_size,
        quant_type_t quant_type
    );
#endif
};

// Factory function for creating MoE quantized cache
std::unique_ptr<llama_kv_cache_moe_quantized> create_moe_quantized_cache(
    llama_kv_cache_unified* base_cache,
    const llama_kv_cache_moe_quantized::params& params
);

// Utility functions
bool is_moe_architecture(const llama_hparams& hparams);
int32_t get_num_experts(const llama_hparams& hparams);
int32_t get_experts_per_token(const llama_hparams& hparams);

// Performance benchmarking
struct moe_cache_benchmark_result {
    double routing_time_ms;
    double quantization_time_ms;
    double memory_transfer_time_ms;
    double total_time_ms;
    size_t memory_saved_bytes;
    float compression_ratio;
};

moe_cache_benchmark_result benchmark_moe_cache(
    llama_kv_cache_moe_quantized& cache,
    const float* test_data,
    size_t test_size,
    int32_t num_iterations = 100
);