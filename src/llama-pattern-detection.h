#pragma once

#include "../blackwell_moe_config.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>

struct PatternDetectionState {
    std::vector<int> recent_tokens;
    std::unordered_map<int, int> token_counts;
    std::chrono::steady_clock::time_point last_detection;
    int repetition_count = 0;
    bool emergency_mode = false;
    
    void reset() {
        recent_tokens.clear();
        token_counts.clear();
        repetition_count = 0;
        emergency_mode = false;
    }
};

class PatternDetector {
private:
    PatternDetectionState state;
    
public:
    // Check if current token creates a repetitive pattern
    bool detect_repetition(int token);
    
    // Check if we should clear KV cache for Qwen3
    bool should_clear_kv_cache();
    
    // Reset detection state
    void reset_state();
    
    // Get current repetition count
    int get_repetition_count() const { return state.repetition_count; }
    
    // Check if in emergency mode
    bool is_emergency_mode() const { return state.emergency_mode; }
};