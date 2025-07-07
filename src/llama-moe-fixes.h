#pragma once

#include "ggml.h"
#include <vector>

// ========================================
// UNIFIED BLACKWELL MOE SYSTEM
// ========================================

// This header provides compatibility declarations for the unified MoE system
// The actual implementation is now integrated directly into llama-graph.cpp

// Configuration is handled via blackwell_moe_config.h
#include "../blackwell_moe_config.h"

// ========================================
// COMPATIBILITY DECLARATIONS
// ========================================

// Backward compatibility structures (deprecated)
struct moe_expert_selection_fix {
    float temperature_scale = 1.0f;
    float diversity_penalty = 0.1f;
    bool enable_expert_dropout = true;
    float expert_dropout_rate = 0.1f;
    int max_consecutive_same_expert = 3;
    
    // Track expert usage to prevent loops
    std::vector<int> recent_expert_selections;
    std::vector<float> expert_usage_history;
    int history_window = 16;
};

// ========================================
// UNIFIED SYSTEM NOTES
// ========================================

// The unified MoE system is implemented directly in llama-graph.cpp:build_moe_ffn()
// This eliminates the dual implementation conflict and provides:
// - Single, consistent MoE processing path
// - Integrated Blackwell optimizations
// - No fallback mechanism complexity
// - Unified configuration system
// - Enhanced stability for large models

// ========================================
// MIGRATION NOTES
// ========================================

// Old system (deprecated):
// - llama_moe_apply_fixes() - replaced by unified build_moe_ffn()
// - Separate configuration files - replaced by blackwell_moe_config.h
// - Dual implementation paths - replaced by single unified path

// New system (current):
// - Unified processing in build_moe_ffn()
// - Single configuration in blackwell_moe_config.h
// - Integrated Blackwell optimizations
// - Consistent behavior across all models

// ========================================
// DEPRECATED FUNCTION DECLARATIONS
// ========================================

// The following functions are deprecated and replaced by the unified system:
// All functionality is now integrated into llama-graph.cpp:build_moe_ffn()

/*
 * DEPRECATED: Use unified build_moe_ffn() instead
 * 
 * Previous functions:
 * - apply_stable_softmax_moe() -> integrated into build_moe_ffn()
 * - enforce_expert_diversity() -> integrated into build_moe_ffn()
 * - fix_moe_weight_normalization() -> integrated into build_moe_ffn()
 * - validate_moe_output() -> integrated into build_moe_ffn()
 * - create_moe_emergency_fallback() -> integrated into build_moe_ffn()
 * - llama_moe_apply_fixes() -> replaced by unified build_moe_ffn()
 */ 