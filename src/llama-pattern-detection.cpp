#include "llama-pattern-detection.h"
#include <algorithm>
#include <cmath>

bool PatternDetector::detect_repetition(int token) {
    if (!MOE_PATTERN_DETECTION_ENABLED) {
        return false;
    }
    
    // Add token to recent history
    state.recent_tokens.push_back(token);
    
    // Keep only recent tokens for pattern detection
    const size_t max_history = 50;
    if (state.recent_tokens.size() > max_history) {
        state.recent_tokens.erase(state.recent_tokens.begin());
    }
    
    // Count token occurrences
    state.token_counts[token]++;
    
    // Check for repetitive patterns
    if (state.recent_tokens.size() < 6) {
        return false;
    }
    
    // Pattern detection: Check for sequences of length 2-4
    for (int pattern_len = 2; pattern_len <= 4; pattern_len++) {
        if (state.recent_tokens.size() < static_cast<size_t>(pattern_len * 3)) continue;
        
        // Check if the last pattern_len tokens repeat at least 3 times
        bool is_repetitive = true;
        for (int i = 0; i < pattern_len; i++) {
            size_t size = state.recent_tokens.size();
            int base_token = state.recent_tokens[size - pattern_len + i];
            int prev_token = state.recent_tokens[size - 2 * pattern_len + i];
            int prev_prev_token = state.recent_tokens[size - 3 * pattern_len + i];
            
            if (base_token != prev_token || base_token != prev_prev_token) {
                is_repetitive = false;
                break;
            }
        }
        
        if (is_repetitive) {
            state.repetition_count++;
            
            // Trigger emergency mode if repetition exceeds threshold
            if (state.repetition_count >= MOE_MAX_REPETITION_THRESHOLD) {
                state.emergency_mode = true;
                state.last_detection = std::chrono::steady_clock::now();
                return true;
            }
        }
    }
    
    // Check for single token repetition
    if (state.recent_tokens.size() >= 10) {
        int last_token = state.recent_tokens.back();
        int repeat_count = 0;
        
        for (int i = static_cast<int>(state.recent_tokens.size()) - 1; i >= 0 && state.recent_tokens[i] == last_token; i--) {
            repeat_count++;
        }
        
        if (repeat_count >= MOE_MAX_REPETITION_THRESHOLD) {
            state.repetition_count += repeat_count;
            state.emergency_mode = true;
            state.last_detection = std::chrono::steady_clock::now();
            return true;
        }
    }
    
    return false;
}

bool PatternDetector::should_clear_kv_cache() {
    if (!MOE_PATTERN_DETECTION_ENABLED) {
        return false;
    }
    
    // Check if enough time has passed since last detection
    auto now = std::chrono::steady_clock::now();
    auto time_diff = std::chrono::duration_cast<std::chrono::seconds>(now - state.last_detection).count();
    
    // For Qwen3, clear KV cache more frequently when in emergency mode
    if (state.emergency_mode && time_diff >= MOE_QWEN3_CACHE_CLEAR_INTERVAL) {
        return true;
    }
    
    return false;
}

void PatternDetector::reset_state() {
    state.reset();
}