#pragma once

#include "common.cuh"

// Blackwell HBM3 Memory Optimization Declarations
// Phase 3: Memory bandwidth optimization for large models
// Target: RTX 5090 with HBM3e memory and optimized L2 cache usage

// Function declarations for HBM3 optimized memory operations
void ggml_cuda_cpy_hbm3_optimized(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src, ggml_tensor * dst);

void ggml_cuda_transpose_l2_optimized(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src, ggml_tensor * dst);

// Performance validation and benchmarking
float benchmark_hbm3_bandwidth(ggml_backend_cuda_context & ctx, size_t test_size_mb);
bool validate_hbm3_optimizations(int device_id); 