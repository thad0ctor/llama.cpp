#pragma once

#include "common.cuh"

// Blackwell HBM3 Bandwidth Optimizations and L2 Cache Utilization
// Phase 2.2: Memory optimizations for model loading and tensor operations

// HBM3 optimized tensor copying
void ggml_cuda_cpy_hbm3_optimized(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src, ggml_tensor * dst);

// L2 cache-aware tensor transpose  
void ggml_cuda_transpose_l2_optimized(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src, ggml_tensor * dst);

// Memory bandwidth benchmark for performance validation
float benchmark_hbm3_bandwidth(ggml_backend_cuda_context & ctx, size_t test_size_mb);

// Performance validation for HBM3 optimizations
bool validate_hbm3_optimizations(int device_id); 