#pragma once

#include "common.cuh"
#include "fattn-common.cuh"

// Blackwell F16 Flash Attention entry points (compiled in fattn-blackwell-f16.cu)
void ggml_cuda_flash_attn_ext_blackwell_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_flash_attn_ext_blackwell_f16_supported(const ggml_tensor * dst);
