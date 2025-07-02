#pragma once

#include "common.cuh"

// Blackwell Cluster-Based GEMM Interface
// Phase 2.1: Thread Block Cluster implementation for large matrix multiplications

// High-level interface for Blackwell cluster GEMM optimization
void ggml_cuda_mul_mat_cluster_gemm(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
    const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i,
    const int64_t row_low, const int64_t row_high,
    const int64_t src1_ncols, const int64_t src1_padded_row_size,
    cudaStream_t stream);

// Performance threshold checking for cluster GEMM usage
bool should_use_cluster_gemm(const ggml_tensor * src0, const ggml_tensor * src1, int device_id);

// Low-level kernel launcher
void launch_blackwell_cluster_gemm(
    const void* A, const void* B, void* C,
    int M, int N, int K,
    int lda, int ldb, int ldc,
    float alpha, float beta,
    ggml_type type, cudaStream_t stream); 