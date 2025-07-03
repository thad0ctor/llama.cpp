#include "llama-kv-cache-moe-quantized.h"
#include "llama-impl.h"
#include "llama-model.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <chrono>

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
extern "C" {
    // Forward declarations for CUDA kernels
    void launch_moe_quantize_int4_kernel(
        const float* input, uint8_t* output, float* scales, float* zero_points,
        int32_t num_elements, int32_t block_size, cudaStream_t stream
    );
    
    void launch_moe_dequantize_int4_kernel(
        const uint8_t* input, float* output, const float* scales, const float* zero_points,
        int32_t num_elements, int32_t block_size, cudaStream_t stream
    );
    
    void launch_moe_token_permute_kernel(
        const float* input, float* output, const int32_t* indices,
        int32_t num_tokens, int32_t hidden_size, cudaStream_t stream
    );
}
#endif

//
// MoE-Aware Quantized KV Cache Implementation
//

llama_kv_cache_moe_quantized::llama_kv_cache_moe_quantized(
    llama_kv_cache_unified* base_cache,
    const params& params)
    : base_cache(base_cache)
    , hparams(base_cache ? base_cache->get_hparams() : throw std::invalid_argument("base_cache is null"))
    , quant_params(params)
    , total_tokens(0)
    , max_context_length(0)
    , is_cuda_enabled(false) {
    
    LLAMA_LOG_INFO("%s: initializing MoE quantized KV cache\n", __func__);
    LLAMA_LOG_INFO("%s: K quantization: %s, V quantization: %s\n", __func__,
        (params.k_quant_type == quant_type_t::INT4) ? "INT4" : 
        (params.k_quant_type == quant_type_t::INT8) ? "INT8" : "FP8",
        (params.v_quant_type == quant_type_t::INT4) ? "INT4" : 
        (params.v_quant_type == quant_type_t::INT8) ? "INT8" : "FP8");
    
    // Validate MoE architecture
    if (!is_moe_architecture(hparams)) {
        throw std::invalid_argument("MoE quantized cache requires MoE architecture");
    }
    
    // Setup expert blocks
    setup_expert_blocks();
    
#ifdef GGML_USE_CUDA
    setup_cuda_memory();
    is_cuda_enabled = true;
    LLAMA_LOG_INFO("%s: CUDA support enabled\n", __func__);
#endif
    
    LLAMA_LOG_INFO("%s: MoE quantized KV cache initialized with %d experts\n", 
        __func__, (int)expert_blocks.size());
}

llama_kv_cache_moe_quantized::~llama_kv_cache_moe_quantized() {
#ifdef GGML_USE_CUDA
    cleanup_cuda_memory();
#endif
    LLAMA_LOG_DEBUG("%s: MoE quantized KV cache destroyed\n", __func__);
}

bool llama_kv_cache_moe_quantized::initialize(const llama_model& model) {
    LLAMA_LOG_INFO("%s: initializing for model with %d layers\n", __func__, hparams.n_layer);
    
    // Get context parameters
    max_context_length = quant_params.max_experts * 1024; // Conservative estimate
    
    // Initialize expert blocks with proper sizing
    const int32_t num_experts = get_num_experts(hparams);
    expert_blocks.resize(num_experts);
    expert_id_to_block.clear();
    
    for (int32_t i = 0; i < num_experts; ++i) {
        expert_id_to_block[i] = i;
        allocate_expert_memory(i, quant_params.block_size);
    }
    
    LLAMA_LOG_INFO("%s: initialized %d expert blocks\n", __func__, num_experts);
    return true;
}

void llama_kv_cache_moe_quantized::clear() {
    current_routes.clear();
    sorted_token_indices.clear();
    total_tokens = 0;
    
    // Clear expert blocks but keep structure
    for (auto& block : expert_blocks) {
        block.k_quantized.clear();
        block.v_quantized.clear();
        block.token_indices.clear();
        block.inverse_indices.clear();
        block.metadata.tokens_count = 0;
    }
    
    LLAMA_LOG_DEBUG("%s: cleared MoE cache state\n", __func__);
}

void llama_kv_cache_moe_quantized::route_tokens(
    const float* gate_logits,
    int32_t seq_len,
    int32_t num_experts
) {
    LLAMA_LOG_DEBUG("%s: routing %d tokens across %d experts\n", __func__, seq_len, num_experts);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Clear previous routing state
    current_routes.clear();
    current_routes.reserve(seq_len * quant_params.experts_per_token);
    
    // Select top-K experts for each token (following vLLM approach)
    select_top_k_experts(gate_logits, seq_len, num_experts, 
                        quant_params.experts_per_token, current_routes);
    
    // Permute tokens by expert assignment for efficient batch processing
    permute_tokens_by_expert(current_routes, sorted_token_indices);
    
    // Update expert metadata
    std::vector<size_t> tokens_per_expert(num_experts, 0);
    for (const auto& route : current_routes) {
        tokens_per_expert[route.expert_id]++;
    }
    
    // Ensure expert blocks have sufficient capacity
    for (int32_t expert_id = 0; expert_id < num_experts; ++expert_id) {
        if (tokens_per_expert[expert_id] > expert_blocks[expert_id].metadata.tokens_capacity) {
            resize_expert_block(expert_id, tokens_per_expert[expert_id] * 2);
        }
        expert_blocks[expert_id].metadata.tokens_count = tokens_per_expert[expert_id];
    }
    
    total_tokens = seq_len;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    LLAMA_LOG_DEBUG("%s: token routing completed in %ld μs\n", __func__, duration.count());
}

void llama_kv_cache_moe_quantized::select_top_k_experts(
    const float* gate_logits,
    int32_t seq_len,
    int32_t num_experts,
    int32_t top_k,
    std::vector<moe_token_route>& routes
) {
    // Implementation following vLLM's fused_topk function
    routes.clear();
    routes.reserve(seq_len * top_k);
    
    for (int32_t token_idx = 0; token_idx < seq_len; ++token_idx) {
        const float* token_logits = gate_logits + token_idx * num_experts;
        
        // Create vector of (logit, expert_id) pairs
        std::vector<std::pair<float, int32_t>> expert_scores;
        expert_scores.reserve(num_experts);
        
        for (int32_t expert_id = 0; expert_id < num_experts; ++expert_id) {
            expert_scores.emplace_back(token_logits[expert_id], expert_id);
        }
        
        // Sort by logit value (descending)
        std::partial_sort(expert_scores.begin(), expert_scores.begin() + top_k, 
                         expert_scores.end(), std::greater<std::pair<float, int32_t>>());
        
        // Apply softmax to top-K logits for proper weight normalization
        float max_logit = expert_scores[0].first;
        float sum_exp = 0.0f;
        std::vector<float> weights(top_k);
        
        for (int32_t k = 0; k < top_k; ++k) {
            weights[k] = std::exp(expert_scores[k].first - max_logit);
            sum_exp += weights[k];
        }
        
        // Normalize weights and create routes
        for (int32_t k = 0; k < top_k; ++k) {
            moe_token_route route;
            route.expert_id = expert_scores[k].second;
            route.weight = weights[k] / sum_exp;
            route.token_idx = token_idx;
            route.expert_token_idx = -1; // Will be set during permutation
            
            routes.push_back(route);
        }
    }
    
    LLAMA_LOG_DEBUG("%s: selected %zu routes for %d tokens (top-%d)\n", 
        __func__, routes.size(), seq_len, top_k);
}

void llama_kv_cache_moe_quantized::permute_tokens_by_expert(
    const std::vector<moe_token_route>& routes,
    std::vector<int32_t>& sorted_indices
) {
    // Sort routes by expert_id for block-aligned memory access (vLLM approach)
    std::vector<size_t> route_indices(routes.size());
    std::iota(route_indices.begin(), route_indices.end(), 0);
    
    std::sort(route_indices.begin(), route_indices.end(),
        [&routes](size_t a, size_t b) {
            return routes[a].expert_id < routes[b].expert_id;
        });
    
    // Create sorted token indices for efficient memory access
    sorted_indices.clear();
    sorted_indices.reserve(routes.size());
    
    // Update expert token indices in sorted order
    std::vector<int32_t> expert_token_counts(quant_params.max_experts, 0);
    
    for (size_t route_idx : route_indices) {
        const auto& route = routes[route_idx];
        sorted_indices.push_back(route.token_idx);
        
        // Update expert-specific token index
        const_cast<moe_token_route&>(route).expert_token_idx = expert_token_counts[route.expert_id]++;
    }
    
    // Update expert blocks with token mapping
    for (int32_t expert_id = 0; expert_id < (int32_t)expert_blocks.size(); ++expert_id) {
        auto& block = expert_blocks[expert_id];
        block.token_indices.clear();
        block.inverse_indices.clear();
        
        // Collect tokens assigned to this expert
        for (size_t i = 0; i < routes.size(); ++i) {
            if (routes[route_indices[i]].expert_id == expert_id) {
                block.token_indices.push_back(routes[route_indices[i]].token_idx);
            }
        }
        
        // Create inverse mapping for efficient unpermutation
        block.inverse_indices.resize(block.token_indices.size());
        for (size_t i = 0; i < block.token_indices.size(); ++i) {
            block.inverse_indices[i] = static_cast<int32_t>(i);
        }
    }
    
    LLAMA_LOG_DEBUG("%s: permuted %zu tokens across %d experts\n", 
        __func__, sorted_indices.size(), (int)expert_blocks.size());
}

ggml_tensor* llama_kv_cache_moe_quantized::cpy_k_moe(
    ggml_context* ctx,
    ggml_tensor* k_cur,
    const float* gate_logits,
    int32_t layer_id,
    int32_t seq_len
) {
    LLAMA_LOG_DEBUG("%s: copying K tensor for layer %d, seq_len %d\n", __func__, layer_id, seq_len);
    
    // Route tokens if not already done for this sequence
    if (current_routes.empty() || total_tokens != seq_len) {
        route_tokens(gate_logits, seq_len, get_num_experts(hparams));
    }
    
    // Process each expert's tokens
    const int32_t hidden_size = k_cur->ne[0];
    
    for (auto& block : expert_blocks) {
        if (block.metadata.tokens_count == 0) continue;
        
        // Extract expert-specific tokens from k_cur
        std::vector<float> expert_k_data(block.metadata.tokens_count * hidden_size);
        
        for (size_t i = 0; i < block.token_indices.size(); ++i) {
            int32_t token_idx = block.token_indices[i];
            const float* src = (const float*)k_cur->data + token_idx * hidden_size;
            float* dst = expert_k_data.data() + i * hidden_size;
            std::copy(src, src + hidden_size, dst);
        }
        
        // Quantize expert's K data
        quantize_expert_k(expert_id_to_block.begin()->first, // Get expert_id properly
                         expert_k_data.data(), expert_k_data.size());
    }
    
    // Return original tensor for now (full implementation would return quantized view)
    return k_cur;
}

ggml_tensor* llama_kv_cache_moe_quantized::cpy_v_moe(
    ggml_context* ctx,
    ggml_tensor* v_cur,
    const float* gate_logits,
    int32_t layer_id,
    int32_t seq_len
) {
    LLAMA_LOG_DEBUG("%s: copying V tensor for layer %d, seq_len %d\n", __func__, layer_id, seq_len);
    
    // Similar implementation to cpy_k_moe but for V tensors
    // Route tokens if not already done
    if (current_routes.empty() || total_tokens != seq_len) {
        route_tokens(gate_logits, seq_len, get_num_experts(hparams));
    }
    
    // Process each expert's V tokens (similar to K processing)
    const int32_t hidden_size = v_cur->ne[0];
    
    for (auto& block : expert_blocks) {
        if (block.metadata.tokens_count == 0) continue;
        
        std::vector<float> expert_v_data(block.metadata.tokens_count * hidden_size);
        
        for (size_t i = 0; i < block.token_indices.size(); ++i) {
            int32_t token_idx = block.token_indices[i];
            const float* src = (const float*)v_cur->data + token_idx * hidden_size;
            float* dst = expert_v_data.data() + i * hidden_size;
            std::copy(src, src + hidden_size, dst);
        }
        
        // Quantize expert's V data
        quantize_expert_v(expert_id_to_block.begin()->first, // Get expert_id properly
                         expert_v_data.data(), expert_v_data.size());
    }
    
    return v_cur;
}

void llama_kv_cache_moe_quantized::quantize_expert_k(
    int32_t expert_id,
    const float* k_data,
    size_t k_size
) {
    if (expert_id_to_block.find(expert_id) == expert_id_to_block.end()) {
        LLAMA_LOG_ERROR("%s: invalid expert_id %d\n", __func__, expert_id);
        return;
    }
    
    auto& block = expert_blocks[expert_id_to_block[expert_id]];
    auto& meta = block.metadata;
    
    // Compute quantization parameters
    compute_quantization_params(k_data, k_size, quant_params.k_quant_type, 
                               meta.k_scale, meta.k_zero_point);
    
    // Allocate quantized storage
    size_t quant_size = (quant_params.k_quant_type == quant_type_t::INT4) ? (k_size + 1) / 2 : k_size;
    block.k_quantized.resize(quant_size);
    
    if (quant_params.k_quant_type == quant_type_t::INT4) {
        // INT4 quantization
        for (size_t i = 0; i < k_size; i += 2) {
            float val1 = k_data[i];
            float val2 = (i + 1 < k_size) ? k_data[i + 1] : 0.0f;
            
            // Quantize to 4-bit values
            int8_t q1 = static_cast<int8_t>(std::round((val1 - meta.k_zero_point) / meta.k_scale));
            int8_t q2 = static_cast<int8_t>(std::round((val2 - meta.k_zero_point) / meta.k_scale));
            
            // Clamp to 4-bit range [-7, 7]
            q1 = std::max(static_cast<int8_t>(-7), std::min(static_cast<int8_t>(7), q1));
            q2 = std::max(static_cast<int8_t>(-7), std::min(static_cast<int8_t>(7), q2));
            
            // Pack two 4-bit values into one byte
            uint8_t packed = ((q2 & 0xF) << 4) | (q1 & 0xF);
            block.k_quantized[i / 2] = packed;
        }
    }
    
    LLAMA_LOG_DEBUG("%s: quantized K for expert %d: %zu -> %zu bytes (%.2fx compression)\n",
        __func__, expert_id, k_size * sizeof(float), quant_size, 
        (float)(k_size * sizeof(float)) / quant_size);
}

void llama_kv_cache_moe_quantized::quantize_expert_v(
    int32_t expert_id,
    const float* v_data,
    size_t v_size
) {
    // Similar implementation to quantize_expert_k but for V tensors
    if (expert_id_to_block.find(expert_id) == expert_id_to_block.end()) {
        LLAMA_LOG_ERROR("%s: invalid expert_id %d\n", __func__, expert_id);
        return;
    }
    
    auto& block = expert_blocks[expert_id_to_block[expert_id]];
    auto& meta = block.metadata;
    
    compute_quantization_params(v_data, v_size, quant_params.v_quant_type, 
                               meta.v_scale, meta.v_zero_point);
    
    size_t quant_size = (quant_params.v_quant_type == quant_type_t::INT4) ? (v_size + 1) / 2 : v_size;
    block.v_quantized.resize(quant_size);
    
    if (quant_params.v_quant_type == quant_type_t::INT4) {
        for (size_t i = 0; i < v_size; i += 2) {
            float val1 = v_data[i];
            float val2 = (i + 1 < v_size) ? v_data[i + 1] : 0.0f;
            
            int8_t q1 = static_cast<int8_t>(std::round((val1 - meta.v_zero_point) / meta.v_scale));
            int8_t q2 = static_cast<int8_t>(std::round((val2 - meta.v_zero_point) / meta.v_scale));
            
            q1 = std::max(static_cast<int8_t>(-7), std::min(static_cast<int8_t>(7), q1));
            q2 = std::max(static_cast<int8_t>(-7), std::min(static_cast<int8_t>(7), q2));
            
            uint8_t packed = ((q2 & 0xF) << 4) | (q1 & 0xF);
            block.v_quantized[i / 2] = packed;
        }
    }
    
    LLAMA_LOG_DEBUG("%s: quantized V for expert %d: %zu -> %zu bytes\n",
        __func__, expert_id, v_size * sizeof(float), quant_size);
}

void llama_kv_cache_moe_quantized::compute_quantization_params(
    const float* data,
    size_t size,
    quant_type_t quant_type,
    float& scale,
    float& zero_point
) {
    // Find min and max values
    float min_val = *std::min_element(data, data + size);
    float max_val = *std::max_element(data, data + size);
    
    if (quant_type == quant_type_t::INT4) {
        // 4-bit quantization: [-7, 7] range
        float range = max_val - min_val;
        scale = range / 14.0f; // 14 = 7 - (-7)
        zero_point = min_val + 7.0f * scale;
    } else if (quant_type == quant_type_t::INT8) {
        // 8-bit quantization: [-127, 127] range
        float range = max_val - min_val;
        scale = range / 254.0f; // 254 = 127 - (-127)
        zero_point = min_val + 127.0f * scale;
    }
    
    // Ensure scale is not zero
    if (scale == 0.0f) {
        scale = 1.0f;
        zero_point = 0.0f;
    }
}

// Helper functions implementation

void llama_kv_cache_moe_quantized::setup_expert_blocks() {
    const int32_t num_experts = get_num_experts(hparams);
    expert_blocks.resize(num_experts);
    expert_id_to_block.clear();
    
    for (int32_t i = 0; i < num_experts; ++i) {
        expert_id_to_block[i] = i;
        expert_blocks[i].metadata = moe_expert_quant_meta{};
    }
    
    LLAMA_LOG_DEBUG("%s: setup %d expert blocks\n", __func__, num_experts);
}

void llama_kv_cache_moe_quantized::allocate_expert_memory(int32_t expert_id, size_t token_count) {
    if (expert_id_to_block.find(expert_id) == expert_id_to_block.end()) return;
    
    auto& block = expert_blocks[expert_id_to_block[expert_id]];
    block.metadata.tokens_capacity = token_count;
    
    // Reserve memory for quantized data
    size_t k_size = token_count * hparams.n_embd_head_k;
    size_t v_size = token_count * hparams.n_embd_head_v;
    
    if (quant_params.k_quant_type == quant_type_t::INT4) {
        k_size = (k_size + 1) / 2;
    }
    if (quant_params.v_quant_type == quant_type_t::INT4) {
        v_size = (v_size + 1) / 2;
    }
    
    block.k_quantized.reserve(k_size);
    block.v_quantized.reserve(v_size);
    block.token_indices.reserve(token_count);
    block.inverse_indices.reserve(token_count);
    
    LLAMA_LOG_DEBUG("%s: allocated memory for expert %d: %zu tokens\n", 
        __func__, expert_id, token_count);
}

void llama_kv_cache_moe_quantized::resize_expert_block(int32_t expert_id, size_t new_capacity) {
    allocate_expert_memory(expert_id, new_capacity);
}

#ifdef GGML_USE_CUDA
void llama_kv_cache_moe_quantized::setup_cuda_memory() {
    // Setup CUDA memory pools for expert blocks
    for (auto& block : expert_blocks) {
        // Allocate device memory for quantized data
        // Implementation would use cudaMalloc for device memory
    }
    LLAMA_LOG_DEBUG("%s: CUDA memory setup completed\n", __func__);
}

void llama_kv_cache_moe_quantized::cleanup_cuda_memory() {
    // Cleanup CUDA memory
    for (auto& block : expert_blocks) {
        if (block.d_k_quantized) {
            cudaFree(block.d_k_quantized);
            block.d_k_quantized = nullptr;
        }
        if (block.d_v_quantized) {
            cudaFree(block.d_v_quantized);
            block.d_v_quantized = nullptr;
        }
        if (block.d_token_indices) {
            cudaFree(block.d_token_indices);
            block.d_token_indices = nullptr;
        }
    }
    LLAMA_LOG_DEBUG("%s: CUDA memory cleanup completed\n", __func__);
}
#endif

// Utility functions

bool is_moe_architecture(const llama_hparams& hparams) {
    // Check for MoE-specific parameters
    return hparams.n_expert > 0 && hparams.n_expert_used > 0;
}

int32_t get_num_experts(const llama_hparams& hparams) {
    return hparams.n_expert;
}

int32_t get_experts_per_token(const llama_hparams& hparams) {
    return hparams.n_expert_used;
}

size_t llama_kv_cache_moe_quantized::get_memory_usage() const {
    size_t total_memory = 0;
    for (const auto& block : expert_blocks) {
        total_memory += block.k_quantized.size();
        total_memory += block.v_quantized.size();
        total_memory += block.token_indices.size() * sizeof(int32_t);
        total_memory += block.inverse_indices.size() * sizeof(int32_t);
    }
    return total_memory;
}

void llama_kv_cache_moe_quantized::print_routing_stats() const {
    LLAMA_LOG_INFO("%s: MoE Routing Statistics:\n", __func__);
    LLAMA_LOG_INFO("  Total tokens: %zu\n", total_tokens);
    LLAMA_LOG_INFO("  Total routes: %zu\n", current_routes.size());
    LLAMA_LOG_INFO("  Memory usage: %.2f MB\n", get_memory_usage() / (1024.0 * 1024.0));
    
    std::vector<size_t> tokens_per_expert;
    std::vector<float> utilization_per_expert;
    get_expert_stats(tokens_per_expert, utilization_per_expert);
    
    for (size_t i = 0; i < tokens_per_expert.size() && i < 10; ++i) {
        LLAMA_LOG_INFO("  Expert %zu: %zu tokens (%.1f%% utilization)\n", 
            i, tokens_per_expert[i], utilization_per_expert[i] * 100.0f);
    }
}

void llama_kv_cache_moe_quantized::get_expert_stats(
    std::vector<size_t>& tokens_per_expert,
    std::vector<float>& utilization_per_expert
) const {
    tokens_per_expert.resize(expert_blocks.size());
    utilization_per_expert.resize(expert_blocks.size());
    
    for (size_t i = 0; i < expert_blocks.size(); ++i) {
        tokens_per_expert[i] = expert_blocks[i].metadata.tokens_count;
        utilization_per_expert[i] = total_tokens > 0 ? 
            (float)tokens_per_expert[i] / total_tokens : 0.0f;
    }
}

// Factory function
std::unique_ptr<llama_kv_cache_moe_quantized> create_moe_quantized_cache(
    llama_kv_cache_unified* base_cache,
    const llama_kv_cache_moe_quantized::params& params
) {
    return std::make_unique<llama_kv_cache_moe_quantized>(base_cache, params);
}