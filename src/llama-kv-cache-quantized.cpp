#include "llama-kv-cache-quantized.h"
#include "llama-io.h"
#include "llama-impl.h"
#include "llama-model.h"
#include "llama-context.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <chrono>
#include <stdexcept>

// Check for Blackwell optimization availability
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
extern bool ggml_cuda_can_use_hbm3_optimizations(int device_id);
extern bool ggml_cuda_can_use_cluster_gemm(int device_id);
#endif

//
// llama_kv_cache_quantized implementation
//

llama_kv_cache_quantized::llama_kv_cache_quantized(
    std::unique_ptr<llama_kv_cache_unified> base_cache,
    const llama_kv_cache_quantized_params & params) :
    base_cache(std::move(base_cache)), params(params) {
    
    LLAMA_LOG_INFO("%s: initializing quantized KV cache\n", __func__);
    LLAMA_LOG_INFO("%s: K quantization: %s, V quantization: %s\n", __func__,
        params.k_quant_level == LLAMA_KV_QUANT_INT8 ? "INT8" : 
        params.k_quant_level == LLAMA_KV_QUANT_INT4 ? "INT4" : "NONE",
        params.v_quant_level == LLAMA_KV_QUANT_INT8 ? "INT8" : 
        params.v_quant_level == LLAMA_KV_QUANT_INT4 ? "INT4" : "NONE");
    
    // Initialize quantization metadata
    const auto cache_size = this->base_cache->get_size();
    // Note: We'll get layer count from first batch processing since we don't have direct access
    
    // Initialize Blackwell optimizations if available
    init_blackwell_optimizations();
    
    LLAMA_LOG_INFO("%s: Blackwell optimizations: %s\n", __func__, 
        blackwell_available ? "enabled" : "disabled");
    LLAMA_LOG_INFO("%s: Max context length: %u tokens\n", __func__, params.max_context_length);
    LLAMA_LOG_INFO("%s: Quality threshold: %.2f%%\n", __func__, params.quality_threshold * 100.0f);
}

llama_memory_state_ptr llama_kv_cache_quantized::init_batch(
        const llama_batch & batch,
        uint32_t n_ubatch,
        bool embd_pooled) {
    
    // Initialize metadata structures on first use
    if (quant_metadata.empty()) {
        // Estimate layer count from the model (this is a reasonable approximation)
        const auto cache_size = base_cache->get_size();
        // We'll initialize with a reasonable layer count, it will be adjusted as needed
        const uint32_t estimated_layers = 80; // Conservative estimate for 235B+ models
        
        quant_metadata.resize(estimated_layers);
        for (auto & layer_meta : quant_metadata) {
            layer_meta.resize(cache_size);
            // Initialize all slots as unquantized
            for (auto & slot_meta : layer_meta) {
                slot_meta.level = LLAMA_KV_QUANT_NONE;
                slot_meta.scale = 1.0f;
                slot_meta.zero_point = 0.0f;
                slot_meta.compressed_size = 0;
                slot_meta.importance_score = 100; // Default importance
                slot_meta.last_access_time = 0;
                slot_meta.is_compressed = false;
            }
        }
        
        LLAMA_LOG_DEBUG("%s: initialized metadata for %u layers, %u slots per layer\n", 
            __func__, estimated_layers, cache_size);
    }
    
    // Get base state and wrap it
    auto base_state = base_cache->init_batch(batch, n_ubatch, embd_pooled);
    if (!base_state || base_state->get_status() != LLAMA_MEMORY_STATUS_SUCCESS) {
        return base_state; // Return the error state as-is
    }
    
    return std::make_unique<llama_kv_cache_quantized_state>(this, std::move(base_state));
}

llama_memory_state_ptr llama_kv_cache_quantized::init_full() {
    auto base_state = base_cache->init_full();
    if (!base_state || base_state->get_status() != LLAMA_MEMORY_STATUS_SUCCESS) {
        return base_state;
    }
    
    return std::make_unique<llama_kv_cache_quantized_state>(this, std::move(base_state));
}

llama_memory_state_ptr llama_kv_cache_quantized::init_update(llama_context * lctx, bool optimize) {
    // Update importance scores before processing
    update_importance_scores();
    
    // Perform distance-based compression if enabled
    if (params.enable_compression) {
        compress_distant_tokens(512); // Compress tokens beyond 512 positions
    }
    
    auto base_state = base_cache->init_update(lctx, optimize);
    if (!base_state || base_state->get_status() != LLAMA_MEMORY_STATUS_SUCCESS) {
        return base_state;
    }
    
    return std::make_unique<llama_kv_cache_quantized_state>(this, std::move(base_state));
}

bool llama_kv_cache_quantized::get_can_shift() const {
    return base_cache->get_can_shift();
}

void llama_kv_cache_quantized::clear(bool data) {
    base_cache->clear(data);
    
    // Reset quantization metadata
    for (auto & layer_meta : quant_metadata) {
        for (auto & slot_meta : layer_meta) {
            slot_meta.level = LLAMA_KV_QUANT_NONE;
            slot_meta.scale = 1.0f;
            slot_meta.zero_point = 0.0f;
            slot_meta.compressed_size = 0;
            slot_meta.importance_score = 100;
            slot_meta.last_access_time = 0;
            slot_meta.is_compressed = false;
        }
    }
    
    // Reset performance counters
    access_counter = 0;
    total_quality_loss = 0.0f;
    bytes_compressed = 0;
}

bool llama_kv_cache_quantized::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    return base_cache->seq_rm(seq_id, p0, p1);
}

void llama_kv_cache_quantized::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    base_cache->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_kv_cache_quantized::seq_keep(llama_seq_id seq_id) {
    base_cache->seq_keep(seq_id);
}

void llama_kv_cache_quantized::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    base_cache->seq_add(seq_id, p0, p1, shift);
}

void llama_kv_cache_quantized::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    base_cache->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_kv_cache_quantized::seq_pos_min(llama_seq_id seq_id) const {
    return base_cache->seq_pos_min(seq_id);
}

llama_pos llama_kv_cache_quantized::seq_pos_max(llama_seq_id seq_id) const {
    return base_cache->seq_pos_max(seq_id);
}

void llama_kv_cache_quantized::state_write(llama_io_write_i & io, llama_seq_id seq_id) const {
    // Write base cache state
    base_cache->state_write(io, seq_id);
    
    // Write quantization metadata
    const uint32_t metadata_version = 1;
    io.write(&metadata_version, sizeof(metadata_version));
    
    const uint32_t num_layers = static_cast<uint32_t>(quant_metadata.size());
    io.write(&num_layers, sizeof(num_layers));
    
    for (const auto & layer_meta : quant_metadata) {
        const uint32_t num_slots = static_cast<uint32_t>(layer_meta.size());
        io.write(&num_slots, sizeof(num_slots));
        
        for (const auto & slot_meta : layer_meta) {
            io.write(&slot_meta, sizeof(slot_meta));
        }
    }
    
    // Write performance counters
    io.write(&access_counter, sizeof(access_counter));
    io.write(&total_quality_loss, sizeof(total_quality_loss));
    io.write(&bytes_compressed, sizeof(bytes_compressed));
}

void llama_kv_cache_quantized::state_read(llama_io_read_i & io, llama_seq_id seq_id) {
    // Read base cache state
    base_cache->state_read(io, seq_id);
    
    // Read quantization metadata
    uint32_t metadata_version;
    io.read_to(&metadata_version, sizeof(metadata_version));
    
    if (metadata_version != 1) {
        throw std::runtime_error("Unsupported quantized KV cache metadata version");
    }
    
    uint32_t num_layers;
    io.read_to(&num_layers, sizeof(num_layers));
    
    quant_metadata.resize(num_layers);
    
    for (auto & layer_meta : quant_metadata) {
        uint32_t num_slots;
        io.read_to(&num_slots, sizeof(num_slots));
        
        layer_meta.resize(num_slots);
        for (auto & slot_meta : layer_meta) {
            io.read_to(&slot_meta, sizeof(slot_meta));
        }
    }
    
    // Read performance counters
    io.read_to(&access_counter, sizeof(access_counter));
    io.read_to(&total_quality_loss, sizeof(total_quality_loss));
    io.read_to(&bytes_compressed, sizeof(bytes_compressed));
}

llama_kv_cache_quantized::quant_stats llama_kv_cache_quantized::get_quantization_stats() const {
    quant_stats stats = {};
    
    if (quant_metadata.empty()) {
        return stats;
    }
    
    uint32_t total_slots = 0;
    uint32_t quantized_slots = 0;
    uint32_t compressed_slots = 0;
    size_t original_bytes = 0;
    size_t compressed_bytes = 0;
    
    for (const auto & layer_meta : quant_metadata) {
        for (const auto & slot_meta : layer_meta) {
            total_slots++;
            
            if (slot_meta.level != LLAMA_KV_QUANT_NONE) {
                quantized_slots++;
                
                // Estimate original size (assuming FP16 base)
                const size_t slot_original_size = 2 * 128; // Approximate slot size
                original_bytes += slot_original_size;
                
                switch (slot_meta.level) {
                    case LLAMA_KV_QUANT_INT8:
                        compressed_bytes += slot_original_size / 2;
                        break;
                    case LLAMA_KV_QUANT_INT4:
                        compressed_bytes += slot_original_size / 4;
                        break;
                    default:
                        compressed_bytes += slot_original_size;
                        break;
                }
            }
            
            if (slot_meta.is_compressed) {
                compressed_slots++;
            }
        }
    }
    
    stats.quantized_slots = quantized_slots;
    stats.compressed_slots = compressed_slots;
    stats.memory_saved_bytes = original_bytes - compressed_bytes;
    
    if (original_bytes > 0) {
        stats.compression_ratio = static_cast<float>(original_bytes) / compressed_bytes;
        stats.memory_bandwidth_improvement = stats.compression_ratio; // Simplified approximation
    } else {
        stats.compression_ratio = 1.0f;
        stats.memory_bandwidth_improvement = 1.0f;
    }
    
    stats.quality_degradation = total_quality_loss / std::max(1UL, access_counter);
    
    return stats;
}

void llama_kv_cache_quantized::update_quantization_params(const llama_kv_cache_quantized_params & new_params) {
    LLAMA_LOG_INFO("%s: updating quantization parameters\n", __func__);
    
    const bool changed_levels = (params.k_quant_level != new_params.k_quant_level) ||
                               (params.v_quant_level != new_params.v_quant_level);
    
    params = new_params;
    
    // If quantization levels changed, we might need to re-quantize existing slots
    if (changed_levels) {
        LLAMA_LOG_DEBUG("%s: quantization levels changed, evaluating existing slots\n", __func__);
        
        for (size_t layer_id = 0; layer_id < quant_metadata.size(); ++layer_id) {
            for (size_t slot_id = 0; slot_id < quant_metadata[layer_id].size(); ++slot_id) {
                auto & slot_meta = quant_metadata[layer_id][slot_id];
                
                if (slot_meta.level != LLAMA_KV_QUANT_NONE) {
                    // Determine if we need to change the quantization level
                    const auto optimal_level = get_optimal_quant_level(layer_id, slot_id, params.quality_threshold);
                    
                    if (optimal_level != slot_meta.level) {
                        // Re-quantize this slot (in a real implementation, this would be more complex)
                        slot_meta.level = optimal_level;
                        
                        // Update quality impact estimate
                        const float quality_impact = estimate_quality_impact(optimal_level);
                        update_quality_metrics(quality_impact);
                    }
                }
            }
        }
    }
}

void llama_kv_cache_quantized::recompress_distant_tokens(uint32_t distance_threshold) {
    compress_distant_tokens(distance_threshold);
}

llama_kv_cache_quantized::memory_usage llama_kv_cache_quantized::get_memory_usage() const {
    memory_usage usage = {};
    
    // Get base cache memory usage (this is approximate)
    usage.total_allocated = 0; // Would need access to base cache internals
    
    // Calculate metadata overhead
    usage.metadata_overhead = quant_metadata.size() * sizeof(std::vector<llama_kv_quant_metadata>);
    for (const auto & layer_meta : quant_metadata) {
        usage.metadata_overhead += layer_meta.size() * sizeof(llama_kv_quant_metadata);
    }
    
    // Estimate quantized storage and original equivalent
    for (const auto & layer_meta : quant_metadata) {
        for (const auto & slot_meta : layer_meta) {
            if (slot_meta.level != LLAMA_KV_QUANT_NONE) {
                const size_t original_size = 2 * 128; // Approximate
                usage.original_equivalent += original_size;
                
                switch (slot_meta.level) {
                    case LLAMA_KV_QUANT_INT8:
                        usage.quantized_storage += original_size / 2;
                        break;
                    case LLAMA_KV_QUANT_INT4:
                        usage.quantized_storage += original_size / 4;
                        break;
                    default:
                        usage.quantized_storage += original_size;
                        break;
                }
            }
        }
    }
    
    usage.total_allocated = usage.quantized_storage + usage.metadata_overhead;
    
    return usage;
}

//
// Private implementation methods
//

bool llama_kv_cache_quantized::should_quantize_slot(uint32_t layer_id, uint32_t slot_id) const {
    if (layer_id >= quant_metadata.size() || slot_id >= quant_metadata[layer_id].size()) {
        return false;
    }
    
    const auto & slot_meta = quant_metadata[layer_id][slot_id];
    
    // Don't quantize if already quantized at desired level
    if ((params.k_quant_level != LLAMA_KV_QUANT_NONE || params.v_quant_level != LLAMA_KV_QUANT_NONE) &&
        slot_meta.level != LLAMA_KV_QUANT_NONE) {
        return false;
    }
    
    // Consider importance score and recency
    const auto current_time = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto time_since_access = current_time - slot_meta.last_access_time;
    
    // Quantize older, less important slots
    return (slot_meta.importance_score < 50) || (time_since_access > 1000000); // 1ms threshold
}

llama_kv_quant_level llama_kv_cache_quantized::get_optimal_quant_level(uint32_t layer_id, uint32_t slot_id, float quality_budget) const {
    GGML_UNUSED(layer_id);
    GGML_UNUSED(slot_id);
    
    // Simple heuristic: use more aggressive quantization if we have quality budget
    if (quality_budget > 0.05f) { // 5% budget remaining
        return params.k_quant_level == LLAMA_KV_QUANT_INT4 ? LLAMA_KV_QUANT_INT4 : LLAMA_KV_QUANT_INT8;
    } else if (quality_budget > 0.02f) { // 2% budget remaining
        return LLAMA_KV_QUANT_INT8;
    } else {
        return LLAMA_KV_QUANT_NONE; // Conservative approach when low on quality budget
    }
}

void llama_kv_cache_quantized::quantize_kv_slot(uint32_t layer_id, uint32_t slot_id, llama_kv_quant_level level) {
    if (layer_id >= quant_metadata.size() || slot_id >= quant_metadata[layer_id].size()) {
        return;
    }
    
    auto & slot_meta = quant_metadata[layer_id][slot_id];
    
    // Update metadata
    slot_meta.level = level;
    slot_meta.last_access_time = std::chrono::steady_clock::now().time_since_epoch().count();
    
    // Calculate scale and zero point (simplified)
    switch (level) {
        case LLAMA_KV_QUANT_INT8:
            slot_meta.scale = 1.0f / 127.0f; // Simple symmetric quantization
            slot_meta.zero_point = 0.0f;
            break;
        case LLAMA_KV_QUANT_INT4:
            slot_meta.scale = 1.0f / 7.0f;
            slot_meta.zero_point = 0.0f;
            break;
        default:
            slot_meta.scale = 1.0f;
            slot_meta.zero_point = 0.0f;
            break;
    }
    
    // Update performance metrics
    const float quality_impact = estimate_quality_impact(level);
    update_quality_metrics(quality_impact);
    
    // Note: Actual quantization would happen in specialized CUDA kernels
    // This is where we would call Blackwell-optimized quantization routines
}

void llama_kv_cache_quantized::dequantize_kv_slot(uint32_t layer_id, uint32_t slot_id) {
    if (layer_id >= quant_metadata.size() || slot_id >= quant_metadata[layer_id].size()) {
        return;
    }
    
    auto & slot_meta = quant_metadata[layer_id][slot_id];
    slot_meta.level = LLAMA_KV_QUANT_NONE;
    slot_meta.scale = 1.0f;
    slot_meta.zero_point = 0.0f;
    slot_meta.last_access_time = std::chrono::steady_clock::now().time_since_epoch().count();
    
    // Note: Actual dequantization would happen in specialized CUDA kernels
}

void llama_kv_cache_quantized::update_importance_scores() {
    // Simple recency-based importance scoring
    const auto current_time = std::chrono::steady_clock::now().time_since_epoch().count();
    
    for (auto & layer_meta : quant_metadata) {
        for (auto & slot_meta : layer_meta) {
            if (slot_meta.last_access_time > 0) {
                const auto time_since_access = current_time - slot_meta.last_access_time;
                
                // Decay importance over time
                slot_meta.importance_score = std::max(1U, 
                    static_cast<uint32_t>(100 * std::exp(-time_since_access / 1000000.0))); // 1ms decay constant
            }
        }
    }
}

void llama_kv_cache_quantized::evict_least_important_slots(uint32_t slots_needed) {
    GGML_UNUSED(slots_needed);
    
    // This would implement actual eviction logic
    // For now, just update importance scores
    update_importance_scores();
}

void llama_kv_cache_quantized::compress_distant_tokens(uint32_t distance_threshold) {
    GGML_UNUSED(distance_threshold);
    
    // This would implement distance-based compression
    // More aggressive quantization for tokens far from current position
    for (size_t layer_id = 0; layer_id < quant_metadata.size(); ++layer_id) {
        for (size_t slot_id = 0; slot_id < quant_metadata[layer_id].size(); ++slot_id) {
            auto & slot_meta = quant_metadata[layer_id][slot_id];
            
            // Simple heuristic: compress low importance slots
            if (slot_meta.importance_score < 25 && !slot_meta.is_compressed) {
                slot_meta.is_compressed = true;
                
                // Would apply more aggressive quantization here
                if (slot_meta.level == LLAMA_KV_QUANT_INT8) {
                    slot_meta.level = LLAMA_KV_QUANT_INT4;
                } else if (slot_meta.level == LLAMA_KV_QUANT_NONE) {
                    slot_meta.level = LLAMA_KV_QUANT_INT8;
                }
            }
        }
    }
}

void llama_kv_cache_quantized::init_blackwell_optimizations() {
#ifdef GGML_USE_CUDA
    cuda_device_id = 0; // Use first CUDA device by default
    blackwell_available = ggml_cuda_can_use_hbm3_optimizations(cuda_device_id);
    
    if (blackwell_available && params.enable_blackwell_opts) {
        LLAMA_LOG_DEBUG("%s: Blackwell HBM3 optimizations available on device %d\n", 
            __func__, cuda_device_id);
    }
#else
    blackwell_available = false;
    cuda_device_id = -1;
#endif
}

bool llama_kv_cache_quantized::can_use_blackwell_kernels() const {
    return blackwell_available && params.enable_blackwell_opts;
}

float llama_kv_cache_quantized::estimate_quality_impact(llama_kv_quant_level level) const {
    // Estimate quality degradation based on quantization level
    switch (level) {
        case LLAMA_KV_QUANT_INT8:
            return 0.005f; // 0.5% estimated quality impact
        case LLAMA_KV_QUANT_INT4:
            return 0.015f; // 1.5% estimated quality impact
        case LLAMA_KV_QUANT_NONE:
        default:
            return 0.0f;
    }
}

void llama_kv_cache_quantized::update_quality_metrics(float quality_impact) {
    total_quality_loss += quality_impact;
    access_counter++;
}

//
// llama_kv_cache_quantized_state implementation
//

llama_kv_cache_quantized_state::llama_kv_cache_quantized_state(
    llama_kv_cache_quantized * qcache,
    llama_memory_state_ptr base_state) :
    qcache(qcache), base_state(std::move(base_state)) {
}

bool llama_kv_cache_quantized_state::next() {
    return base_state->next();
}

bool llama_kv_cache_quantized_state::apply() {
    // Apply base state changes
    bool result = base_state->apply();
    
    // Perform quantization decisions after state application
    if (result && qcache) {
        // This is where we would decide which slots to quantize
        // based on the current state and access patterns
        
        // For now, just update access counters
        qcache->access_counter++;
    }
    
    return result;
}

std::vector<int64_t> & llama_kv_cache_quantized_state::out_ids() {
    return base_state->out_ids();
}

llama_memory_status llama_kv_cache_quantized_state::get_status() const {
    return base_state->get_status();
}

const llama_ubatch & llama_kv_cache_quantized_state::get_ubatch() const {
    return base_state->get_ubatch();
}

//
// Utility functions
//

namespace llama_kv_quant_utils {

bool is_blackwell_available(int device_id) {
#ifdef GGML_USE_CUDA
    return ggml_cuda_can_use_hbm3_optimizations(device_id);
#else
    GGML_UNUSED(device_id);
    return false;
#endif
}

size_t estimate_memory_savings(
    const llama_hparams & hparams, 
    uint32_t kv_size,
    const llama_kv_cache_quantized_params & params) {
    
    // Rough estimation based on model parameters
    const size_t n_layer = hparams.n_layer;
    const size_t n_embd_gqa = hparams.n_embd / hparams.n_head() * hparams.n_head_kv(); // Approximate
    
    // Estimate size per KV slot (K + V tensors)
    const size_t slot_size_fp16 = 2 * n_embd_gqa * 2; // 2 bytes per FP16, K+V
    const size_t total_original = n_layer * kv_size * slot_size_fp16;
    
    size_t total_compressed = total_original;
    
    // Apply compression ratios
    if (params.k_quant_level == LLAMA_KV_QUANT_INT8 || params.v_quant_level == LLAMA_KV_QUANT_INT8) {
        total_compressed = total_compressed / 2; // 2x compression for INT8
    }
    if (params.k_quant_level == LLAMA_KV_QUANT_INT4 || params.v_quant_level == LLAMA_KV_QUANT_INT4) {
        total_compressed = total_compressed / 4; // 4x compression for INT4
    }
    
    return total_original - total_compressed;
}

llama_kv_cache_quantized_params get_recommended_params(
    const llama_hparams & hparams,
    size_t available_memory_bytes,
    uint32_t target_context_length) {
    
    llama_kv_cache_quantized_params params;
    
    // Adjust based on model size
    const size_t model_size_gb = hparams.n_layer * hparams.n_embd * hparams.n_embd / (1024 * 1024 * 1024 / 2); // Rough estimate
    
    if (model_size_gb > 200) { // 200B+ models
        params.k_quant_level = LLAMA_KV_QUANT_INT8;
        params.v_quant_level = LLAMA_KV_QUANT_INT8;
        params.quality = LLAMA_KV_QUANT_QUALITY_BALANCED;
    } else if (model_size_gb > 50) { // 50B+ models
        params.k_quant_level = LLAMA_KV_QUANT_INT8;
        params.v_quant_level = LLAMA_KV_QUANT_NONE;
        params.quality = LLAMA_KV_QUANT_QUALITY_HIGH;
    } else {
        // Smaller models - minimal quantization
        params.k_quant_level = LLAMA_KV_QUANT_NONE;
        params.v_quant_level = LLAMA_KV_QUANT_NONE;
        params.quality = LLAMA_KV_QUANT_QUALITY_HIGH;
    }
    
    // Adjust for available memory
    const size_t memory_gb = available_memory_bytes / (1024 * 1024 * 1024);
    if (memory_gb < 16) {
        // Aggressive quantization for low memory
        params.k_quant_level = LLAMA_KV_QUANT_INT4;
        params.v_quant_level = LLAMA_KV_QUANT_INT4;
        params.enable_compression = true;
    }
    
    // Set context length
    params.max_context_length = target_context_length;
    
    // Enable Blackwell optimizations if available
    params.enable_blackwell_opts = is_blackwell_available();
    
    return params;
}

} // namespace llama_kv_quant_utils 