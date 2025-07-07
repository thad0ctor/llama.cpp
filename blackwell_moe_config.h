//
// Unified Blackwell MoE Configuration
// Single configuration system for all MoE enhancements
//

#ifndef BLACKWELL_MOE_CONFIG_H
#define BLACKWELL_MOE_CONFIG_H

// ========================================
// UNIFIED MOE CONFIGURATION
// ========================================

// Master switch for unified MoE processing
#define BLACKWELL_UNIFIED_MOE_ENABLED 1

// Enhanced processing thresholds
#define MOE_ENHANCED_PROCESSING_THRESHOLD 64    // Enable enhanced processing for models with >64 experts
#define MOE_LARGE_MODEL_THRESHOLD 128          // Special handling for models with >128 experts

// ========================================
// NUMERICAL STABILITY SETTINGS (AGGRESSIVE FIX)
// ========================================

// Temperature scaling for expert diversity - MORE AGGRESSIVE
#define MOE_TEMPERATURE_SCALE 0.90f            // More aggressive scaling for better diversity
#define MOE_STABILITY_EPSILON 1e-8f            // Higher precision numerical stability
#define MOE_WEIGHT_EPSILON 1e-9f               // Higher precision weight normalization

// ========================================
// OUTPUT STABILIZATION SETTINGS (AGGRESSIVE FIX)
// ========================================

// Progressive clamping values - MORE AGGRESSIVE
#define MOE_BASE_CLAMP_VALUE 25.0f             // More aggressive base clamping
#define MOE_MEDIUM_CLAMP_VALUE 20.0f           // More aggressive for 64+ experts  
#define MOE_LARGE_CLAMP_VALUE 15.0f            // More aggressive for 128+ experts

// Architecture-specific scaling - MORE AGGRESSIVE
#define MOE_QWEN3_CLAMP_MULTIPLIER 0.7f        // Much more aggressive for Qwen3
#define MOE_QWEN3_STABILITY_SCALE 0.90f        // More aggressive Qwen3 stability scaling
#define MOE_GENERAL_STABILITY_SCALE 0.95f      // More aggressive general stability scaling

// Emergency measures - ENHANCED
#define MOE_EMERGENCY_NOISE_SCALE 1.0f + 2e-6f // Stronger noise injection scale

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