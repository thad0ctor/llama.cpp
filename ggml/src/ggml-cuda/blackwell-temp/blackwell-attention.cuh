#pragma once

#include "common.cuh"

// Blackwell Flash Attention Optimization Declarations
// Phase 3: Large context attention optimization for 235B+ models
// Target: RTX 5090 with L2 cache optimization and cluster support

// Flash attention with Blackwell L2 cache optimization
void ggml_cuda_flash_attn_blackwell_optimized(
    ggml_backend_cuda_context & ctx,
    ggml_tensor * dst);

// Performance validation for Blackwell attention optimizations
bool validate_blackwell_attention_optimizations(int device_id);

// Low-level kernel launcher for L2-optimized Flash Attention
void launch_blackwell_l2_flash_attention(
    const void* Q, const void* K, const void* V,
    void* O, void* L, void* M,
    int batch_size, int seq_len, int num_heads, int head_dim,
    float scale, ggml_type type, cudaStream_t stream);

// Performance threshold for using L2-optimized attention
bool should_use_l2_flash_attention(const ggml_tensor * src0, int device_id); 