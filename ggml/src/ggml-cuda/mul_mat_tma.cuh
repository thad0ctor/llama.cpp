#pragma once

#include "common.cuh"

// Forward declarations for TMA matmul functions
// Implementation is in mul_mat_tma.cu

void ggml_cuda_mul_mat_tma(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    ggml_tensor * dst);

bool ggml_cuda_should_use_tma(
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    int cc);
