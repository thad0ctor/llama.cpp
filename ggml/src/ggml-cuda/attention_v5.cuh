// attention_v5.cuh - SM_120 (Blackwell) optimized Flash Attention kernel declarations
#pragma once

#include "common.cuh"

// Forward declarations for attention_v5 kernel wrapper
void ggml_cuda_flash_attn_ext_attention_v5(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_flash_attn_ext_attention_v5_supported(const ggml_tensor * dst);
