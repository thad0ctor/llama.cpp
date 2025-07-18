//
// Unified Blackwell MoE Configuration
// Single configuration system for all MoE enhancements
//

#ifndef BLACKWELL_MOE_CONFIG_H
#define BLACKWELL_MOE_CONFIG_H

// ========================================
// UNIFIED MOE CONFIGURATION
// ========================================

// Master switch for unified MoE processing - DISABLED to restore original MoE
#define BLACKWELL_UNIFIED_MOE_ENABLED 0

// Enhanced processing thresholds
#define MOE_ENHANCED_PROCESSING_THRESHOLD 64    // Enable enhanced processing for models with >64 experts
#define MOE_LARGE_MODEL_THRESHOLD 128          // Special handling for models with >128 experts

// ========================================
// NUMERICAL STABILITY SETTINGS (FIXED - LESS AGGRESSIVE)
// ========================================

// Temperature scaling for expert diversity - FIXED TO BE LESS AGGRESSIVE
#define MOE_TEMPERATURE_SCALE 0.98f            // Less aggressive scaling (was 0.90f)
#define MOE_STABILITY_EPSILON 1e-7f            // Reduced precision to prevent over-correction (was 1e-8f)
#define MOE_WEIGHT_EPSILON 1e-8f               // Reduced precision (was 1e-9f)

// ========================================
// OUTPUT STABILIZATION SETTINGS (FIXED - LESS AGGRESSIVE)
// ========================================

// Progressive clamping values - FIXED TO BE LESS AGGRESSIVE
#define MOE_BASE_CLAMP_VALUE 35.0f             // Less aggressive base clamping (was 25.0f)
#define MOE_MEDIUM_CLAMP_VALUE 30.0f           // Less aggressive for 64+ experts (was 20.0f)
#define MOE_LARGE_CLAMP_VALUE 25.0f            // Less aggressive for 128+ experts (was 15.0f)

// Architecture-specific scaling - FIXED TO BE LESS AGGRESSIVE
#define MOE_QWEN3_CLAMP_MULTIPLIER 0.9f        // Less aggressive for Qwen3 (was 0.7f)
#define MOE_QWEN3_STABILITY_SCALE 0.98f        // Less aggressive Qwen3 stability scaling (was 0.90f)
#define MOE_GENERAL_STABILITY_SCALE 0.99f      // Less aggressive general stability scaling (was 0.95f)

// Emergency measures - FIXED TO BE LESS AGGRESSIVE
#define MOE_EMERGENCY_NOISE_SCALE 1.0f + 1e-7f // Reduced noise injection scale (was 2e-6f)

// ========================================
// BLACKWELL INTEGRATION SETTINGS
// ========================================

// Unified integration with Blackwell optimizations
#define BLACKWELL_MOE_INTEGRATION_ENABLED 1    // Enable seamless integration
#define BLACKWELL_MOE_ATTENTION_COMPAT 1       // Maintain attention compatibility

// ========================================
// PERFORMANCE SETTINGS
// ========================================

// Logging and debugging - ENHANCED FOR TROUBLESHOOTING
#define MOE_VERBOSE_LOGGING 1                  // Enable detailed logging
#define MOE_DEBUG_VALIDATION 1                 // Enable debug validation for troubleshooting

// ========================================
// ARCHITECTURE-SPECIFIC SETTINGS
// ========================================

// Qwen3 MoE specific settings - ENHANCED
#ifdef LLM_ARCH_QWEN3MOE
    #define MOE_QWEN3_OPTIMIZATIONS 1
    #define MOE_QWEN3_ENHANCED_STABILITY 1
    #define MOE_QWEN3_AGGRESSIVE_FIXES 1       // NEW: Enable aggressive fixes for Qwen3
#endif

// DeepSeek2 specific settings - ENHANCED
#ifdef LLM_ARCH_DEEPSEEK2
    #define MOE_DEEPSEEK2_OPTIMIZATIONS 1
    #define MOE_DEEPSEEK2_ENHANCED_STABILITY 1
    #define MOE_DEEPSEEK2_AGGRESSIVE_FIXES 1   // NEW: Enable aggressive fixes for DeepSeek2
#endif

// ========================================
// EMERGENCY FALLBACK SETTINGS (NEW)
// ========================================

// Emergency measures for severe gibberish cases
#define MOE_ENABLE_EMERGENCY_FALLBACK 1        // Enable emergency fallback
#define MOE_QWEN3_CACHE_CLEAR_INTERVAL 2       // Clear KV cache more frequently
#define MOE_MAX_REPETITION_THRESHOLD 5         // Detect repetitive patterns
#define MOE_PATTERN_DETECTION_ENABLED 1        // Enable real-time pattern detection

// ========================================
// COMPATIBILITY SETTINGS
// ========================================

// Ensure backward compatibility
#define MOE_BACKWARD_COMPATIBILITY 1          // Maintain compatibility with existing code

// ========================================
// DEPRECATED SETTINGS (for reference)
// ========================================

// The following settings are deprecated and replaced by the unified system:
// - BLACKWELL_MOE_FIXES_ENABLED (replaced by BLACKWELL_UNIFIED_MOE_ENABLED)
// - Individual fix controls (consolidated into unified processing)
// - Separate config systems (unified into single system)

#endif // BLACKWELL_MOE_CONFIG_H 