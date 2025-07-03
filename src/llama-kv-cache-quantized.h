#pragma once

#include "llama-kv-cache-unified.h"
#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-memory.h"

#include <unordered_map>
#include <vector>
#include <memory>

struct llama_cparams;
struct llama_hparams;
struct llama_model;
struct llama_context;

// Forward declaration for MoE quantization - temporarily disabled
// class llama_kv_cache_moe_quantized;

//
// llama_kv_cache_quantized - Blackwell RTX 5090 Optimized KV Cache
// 
// Phase 3: KV Cache Memory Optimization Implementation
// Target: 2-4x memory bandwidth improvement for 235B+ parameter models
//

// Quantization levels for KV cache compression
enum llama_kv_quant_level {
    LLAMA_KV_QUANT_NONE = 0,    // No quantization (fallback)
    LLAMA_KV_QUANT_INT8 = 1,    // INT8 quantization (2x compression)
    LLAMA_KV_QUANT_INT4 = 2,    // INT4 quantization (4x compression)
};

// Quality vs performance trade-off settings
enum llama_kv_quant_quality {
    LLAMA_KV_QUANT_QUALITY_FAST = 0,      // Maximum performance, minimal quality loss
    LLAMA_KV_QUANT_QUALITY_BALANCED = 1,  // Balanced performance and quality
    LLAMA_KV_QUANT_QUALITY_HIGH = 2,      // High quality, moderate performance impact
};

// Cache management strategy for long contexts
enum llama_kv_cache_strategy {
    LLAMA_KV_CACHE_STRATEGY_FIFO = 0,        // Simple FIFO eviction
    LLAMA_KV_CACHE_STRATEGY_IMPORTANCE = 1,  // Importance-based eviction
    LLAMA_KV_CACHE_STRATEGY_HYBRID = 2,      // Hybrid approach with recency and importance
};

// Configuration for quantized KV cache
struct llama_kv_cache_quantized_params {
    llama_kv_quant_level k_quant_level = LLAMA_KV_QUANT_INT8;     // K tensor quantization
    llama_kv_quant_level v_quant_level = LLAMA_KV_QUANT_INT8;     // V tensor quantization
    llama_kv_quant_quality quality = LLAMA_KV_QUANT_QUALITY_BALANCED;
    llama_kv_cache_strategy strategy = LLAMA_KV_CACHE_STRATEGY_HYBRID;
    
    float quality_threshold = 0.02f;        // Maximum allowed quality degradation (2%)
    uint32_t max_context_length = 128000;   // Maximum context length to support
    bool enable_blackwell_opts = true;      // Enable RTX 5090 optimizations
    bool enable_compression = true;         // Enable distance-based compression
    bool enable_streaming = true;           // Enable streaming for long contexts
    
    // HBM3e optimization parameters
    uint32_t coalescing_factor = 16;        // Memory coalescing factor
    uint32_t prefetch_distance = 8;        // Prefetch distance for sequential access
    bool use_l2_cache_hints = true;        // Use L2 cache hints for RTX 5090
};

// Quantization metadata for cache management
struct llama_kv_quant_metadata {
    llama_kv_quant_level level;
    float scale;                  // Quantization scale factor
    float zero_point;            // Zero point for quantization
    uint32_t compressed_size;    // Size after compression
    uint32_t importance_score;   // Importance score for eviction
    uint64_t last_access_time;   // Last access timestamp
    bool is_compressed;          // Whether this slot is compressed
};

class llama_kv_cache_quantized : public llama_memory_i {
    friend class llama_kv_cache_quantized_state;
public:
    // Constructor - wraps existing unified cache with quantization
    llama_kv_cache_quantized(
        llama_kv_cache_unified * base_cache,
        const llama_kv_cache_quantized_params & params);
    
    // Properly manage base cache lifetime
    ~llama_kv_cache_quantized();

    // Disable move and copy semantics to prevent issues
    llama_kv_cache_quantized(llama_kv_cache_quantized &&) = delete;
    llama_kv_cache_quantized & operator=(llama_kv_cache_quantized &&) = delete;
    llama_kv_cache_quantized(const llama_kv_cache_quantized &) = delete;
    llama_kv_cache_quantized & operator=(const llama_kv_cache_quantized &) = delete;

    //
    // llama_memory_i interface (delegates to base cache with quantization)
    //

    llama_memory_state_ptr init_batch(
            const llama_batch & batch,
            uint32_t n_ubatch,
            bool embd_pooled) override;

    llama_memory_state_ptr init_full() override;

    llama_memory_state_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1)       override;

    //
    // Quantized KV cache specific API
    //

    // Get quantization statistics
    struct quant_stats {
        float compression_ratio;
        float memory_bandwidth_improvement;
        float quality_degradation;
        uint32_t quantized_slots;
        uint32_t compressed_slots;
        size_t memory_saved_bytes;
    };
    
    quant_stats get_quantization_stats() const;
    
    // Update quantization parameters dynamically
    void update_quantization_params(const llama_kv_cache_quantized_params & new_params);
    
    // Force recompression of distant tokens
    void recompress_distant_tokens(uint32_t distance_threshold);
    
    // Get current memory usage breakdown
    struct memory_usage {
        size_t total_allocated;
        size_t quantized_storage;
        size_t metadata_overhead;
        size_t original_equivalent;
    };
    
    memory_usage get_memory_usage() const;

    // Access the underlying base cache
    llama_kv_cache_unified * get_base_cache() const { return base_cache; }

private:
    llama_kv_cache_unified * base_cache;
    llama_kv_cache_quantized_params params;
    
    // MoE-specific quantized cache (used for MoE models) - temporarily disabled
    // std::unique_ptr<llama_kv_cache_moe_quantized> moe_cache;
    
    // CRITICAL: Corruption detection and prevention
    uint32_t base_cache_size_cached = 0;     // Cached size for validation
    uintptr_t base_cache_address = 0;        // Cached address for validation
    
    // Quantization metadata per cache slot
    std::vector<std::vector<llama_kv_quant_metadata>> quant_metadata; // [layer][slot]
    
    // Performance monitoring
    mutable uint64_t access_counter = 0;
    mutable float total_quality_loss = 0.0f;
    mutable size_t bytes_compressed = 0;
    
    // Blackwell-specific optimization state
    bool blackwell_available = false;
    int cuda_device_id = -1;
    
    // Corruption detection method
    void validate_base_cache_integrity() const;
    
    // Internal quantization functions
    bool should_quantize_slot(uint32_t layer_id, uint32_t slot_id) const;
    llama_kv_quant_level get_optimal_quant_level(uint32_t layer_id, uint32_t slot_id, float quality_budget) const;
    void quantize_kv_slot(uint32_t layer_id, uint32_t slot_id, llama_kv_quant_level level);
    void dequantize_kv_slot(uint32_t layer_id, uint32_t slot_id);
    
    // Cache management functions
    void update_importance_scores();
    void evict_least_important_slots(uint32_t slots_needed);
    void compress_distant_tokens(uint32_t distance_threshold);
    
    // Blackwell optimization functions
    void init_blackwell_optimizations();
    bool can_use_blackwell_kernels() const;
    
    // Quality monitoring
    float estimate_quality_impact(llama_kv_quant_level level) const;
    void update_quality_metrics(float quality_impact);
    
    // Dynamic metadata allocation helper
    void ensure_metadata_for_layer(uint32_t layer_id);
};

// State wrapper for quantized cache
class llama_kv_cache_quantized_state : public llama_memory_state_i {
public:
    // Constructor for error states
    llama_kv_cache_quantized_state(llama_memory_status status);
    
    llama_kv_cache_quantized_state(
        llama_kv_cache_quantized * qcache,
        llama_memory_state_ptr base_state);
    
    virtual ~llama_kv_cache_quantized_state() = default;

    //
    // llama_memory_state_i interface
    //

    bool next() override;
    bool apply() override;

    std::vector<int64_t> & out_ids() override;
    
    llama_memory_status get_status() const override;
    const llama_ubatch & get_ubatch() const override;

private:
    llama_memory_status status = LLAMA_MEMORY_STATUS_SUCCESS;
    llama_kv_cache_quantized * qcache;
    llama_memory_state_ptr base_state;
};

// Utility functions for quantization
namespace llama_kv_quant_utils {
    // Check if Blackwell optimizations are available
    bool is_blackwell_available(int device_id = -1);
    
    // Estimate memory savings for given parameters
    size_t estimate_memory_savings(
        const llama_hparams & hparams, 
        uint32_t kv_size,
        const llama_kv_cache_quantized_params & params);
    
    // Get recommended quantization parameters for model size
    llama_kv_cache_quantized_params get_recommended_params(
        const llama_hparams & hparams,
        size_t available_memory_bytes,
        uint32_t target_context_length);
} 