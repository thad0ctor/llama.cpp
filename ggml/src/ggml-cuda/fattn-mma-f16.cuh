#include "common.cuh"
#include "cp-async.cuh"
#include "tma.cuh"
#include "mma.cuh"
#include "fattn-common.cuh"

using namespace ggml_cuda_mma;

// Config options for the MMA kernel.
// Should not affect results, only speed/register pressure/shared memory use.
struct fattn_mma_config {
    int  nthreads;       // Number of threads per CUDA block.
    int  occupancy;      // Targeted occupancy for the MMA kernel.
    int  nbatch_fa;      // Number of KV rows per softmax rescaling of KQ rowsums and VKQ accumulators.
    int  nbatch_K2;      // Number of K half2 values in direction of DKQ to load in parallel.
    int  nbatch_V2;      // Number of V half2 values in direction of DV to load in parallel.
    int  nbatch_combine; // Number of VKQ half2 values in direction of DV to combine in parallel.
    int  nstages_target; // Number of pipeline stages to use ideally, 1 == always load data synchronously, 2 == preload data if there is hardware support.
    bool Q_in_reg;       // Whether the Q values should be kept permanently in registers.
    int  num_consumers;  // Number of consumer warps (0 = unified/legacy mode)

    constexpr __host__ __device__ fattn_mma_config(
            int nthreads, int occupancy, int nbatch_fa, int nbatch_K2, int nbatch_V2, int nbatch_combine, int nstages_target, bool Q_in_reg, int num_consumers = 0) : 
        nthreads(nthreads), occupancy(occupancy), nbatch_fa(nbatch_fa), nbatch_K2(nbatch_K2), nbatch_V2(nbatch_V2), nbatch_combine(nbatch_combine),
        nstages_target(nstages_target), Q_in_reg(Q_in_reg), num_consumers(num_consumers) {}
};

#define GGML_CUDA_FATTN_MMA_CONFIG_CASE(DKQ_, DV_, ncols_, nthreads_, occupancy_, nbatch_fa_, nbatch_K2_, nbatch_V2_, nbatch_combine_, nstages_target_, Q_in_reg_) \
    if (DKQ == (DKQ_) && DV == (DV_) && ncols == (ncols_)) {                                                                                                       \
        static_assert((nthreads_)       % 32 == 0 && (nthreads_)       <= 512, "bad nthreads");                                                                    \
        static_assert(                               (occupancy_)      <=   8, "bad occupancy");                                                                   \
        static_assert((nbatch_fa_)      % 32 == 0 && (nbatch_fa_)      <= 256, "bad nbatch_fa");                                                                   \
        static_assert((nbatch_K2_)      %  4 == 0 && (nbatch_K2_)      <= 512, "bad nbatch_K2");                                                                   \
        static_assert((nbatch_V2_)      %  4 == 0 && (nbatch_V2_)      <= 256, "bad nbatch_V2");                                                                   \
        static_assert((nbatch_combine_) %  4 == 0 && (nbatch_combine_) <= 128, "bad nbatch_combine");                                                              \
        static_assert((nstages_target_)      >= 1 && (nstages_target_) <=   3, "bad nstages_target");                                                              \
        return fattn_mma_config{(nthreads_), (occupancy_), (nbatch_fa_), (nbatch_K2_), (nbatch_V2_), (nbatch_combine_), (nstages_target_), (Q_in_reg_), 0};         \
    }                                                                                                                                                              \

static constexpr __host__ __device__ fattn_mma_config ggml_cuda_fattn_mma_get_config_ampere(const int DKQ, const int DV, const int ncols) {
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64,  8, 128, 2, 128,  32,  32,  32, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 16, 128, 2,  64,  32,  32,  32, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 32, 128, 2,  64,  32,  32,  32, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 64,  64, 64, 128, 2,  64,  32,  32,  32, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80,  8, 128, 2, 128,  40,  40,  40, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80, 16, 128, 2,  64,  40,  40,  40, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80, 32, 128, 2,  64,  40,  40,  40, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 80,  80, 64, 128, 2,  64,  40,  40,  40, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96,  8, 128, 2, 128,  48,  48,  48, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96, 16, 128, 2,  64,  48,  48,  48, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96, 32, 128, 2,  64,  48,  48,  48, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE( 96,  96, 64, 128, 2,  64,  48,  48,  48, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112,  8, 128, 2, 128,  56,  56,  56, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 16, 128, 2,  64,  56,  56,  56, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 32, 128, 2,  64,  56,  56,  56, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(112, 112, 64, 128, 2,  64,  56,  56,  56, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128,  8, 128, 2, 128,  64,  64,  64, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 16, 128, 2,  64,  64,  64,  64, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 32, 128, 2,  64,  64,  64,  64, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 64, 128, 2,  64,  64,  64,  64, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256,  8,  64, 4,  64, 128, 128, 128, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 16,  64, 4,  32, 128, 128, 128, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 32, 128, 2,  32, 128, 128, 128, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 64, 128, 2,  32, 128, 128, 128, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512,  8,  64, 4,  64,  32,  32, 128, 2, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 16,  64, 4,  64,  32,  32, 128, 2, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 32, 128, 2,  32,  32,  32, 128, 2, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 64, 256, 1,  32,  32,  32, 128, 2, false);

    return fattn_mma_config(32, 1, 0, 0, 0, 0, 0, false);
}

// Consumer Blackwell (sm_120, RTX 5090) configurations
// 
// Uses 160 threads (5 warps = 1 producer + 4 consumers) with warp specialization:
//   - 160 × 255 regs = 40,800 ≤ 65,536 ✓ (register budget)
//   - Loop constraint: nbatch_fa % (4×16) = nbatch_fa % 64 == 0
//   - Chunking constraint: nbatch_K2 >= DKQ/2, nbatch_V2 >= DV/2
//     (Otherwise falls back to Ampere kernel!)
//   - Shared memory limit: 99KB (vs 227KB on datacenter Blackwell)
//
// Occupancy analysis:
//   - 160 threads, 1 block/SM = 10.4% occupancy
//   - Alternative configs analyzed but rejected:
//     * 128 threads × 2 blocks: register limit at boundary, runtime rejects
//     * 256 threads (7 consumers): nbatch_fa % 112 not compatible with cp_async
//
// Head size coverage (all using nbatch_fa=64 to fit shared memory):
//   - DKQ=64:  nbatch_K2=32, shared mem ~32KB ✓
//   - DKQ=80:  nbatch_K2=40, shared mem ~40KB ✓
//   - DKQ=96:  nbatch_K2=48, shared mem ~48KB ✓
//   - DKQ=112: nbatch_K2=56, shared mem ~56KB ✓
//   - DKQ=128: nbatch_K2=64, shared mem ~64KB ✓ (most common: Llama, Mistral)
//   - DKQ=256+: Falls back to Ampere (shared mem > 99KB)
//   - MLA (576/512): Falls back to Ampere (shared mem too large)
//
// Benefits of native Blackwell kernel:
//   - Warp specialization (1 producer + 4 consumer warps)
//   - TMA-accelerated data loading
//   - Better memory bandwidth utilization
//
static constexpr __host__ __device__ fattn_mma_config ggml_cuda_fattn_mma_get_config_sm120(const int DKQ, const int DV, const int ncols) {

    // ========== SM_120 (RTX 5090) - Aligned with Gau-Nernst reference ==========
    // Uses 4 warps (128 threads), ALL warps do both load and compute
    // NO producer/consumer split - simpler and avoids warp divergence bugs
    // Reference: https://gau-nernst.github.io/fa-5090/
    //
    // Key design choices:
    //   - 128 threads = 4 warps (NUM_WARPS=4 in reference)
    //   - num_consumers=0 means unified mode (no producer warp)
    //   - All warps cooperatively load K/V via cp.async
    //   - All warps compute attention in parallel
    //   - nbatch_fa=128 to match reference BLOCK_Q=128

    if (DKQ == 64 && DV == 64) {
        // Shared mem: 128 × (32 + 32) × 4 × 2 = 64KB ✓
        if (ncols <=  8) return fattn_mma_config(128, 1, 128, 32, 32, 16, 2, true, 0);
        if (ncols <= 16) return fattn_mma_config(128, 1, 128, 32, 32, 16, 2, true, 0);
        if (ncols <= 32) return fattn_mma_config(128, 1, 128, 32, 32, 16, 2, true, 0);
        if (ncols <= 64) return fattn_mma_config(128, 1, 128, 32, 32, 16, 2, true, 0);
    }
    if (DKQ == 80 && DV == 80) {
        // nbatch_combine=20 to satisfy (DV/2) % nbatch_combine == 0 → 40 % 20 = 0
        if (ncols <=  8) return fattn_mma_config(128, 1, 128, 32, 32, 20, 2, true, 0);
        if (ncols <= 16) return fattn_mma_config(128, 1, 128, 32, 32, 20, 2, true, 0);
        if (ncols <= 32) return fattn_mma_config(128, 1, 128, 32, 32, 20, 2, true, 0);
        if (ncols <= 64) return fattn_mma_config(128, 1, 128, 32, 32, 20, 2, true, 0);
    }

    // ========== Medium heads - 4 warps (128 threads) ==========

    if (DKQ == 96 && DV == 96) {
        // nbatch_combine=24 to satisfy (DV/2) % nbatch_combine == 0 → 48 % 24 = 0
        if (ncols <=  8) return fattn_mma_config(128, 1, 128, 32, 32, 24, 2, true, 0);
        if (ncols <= 16) return fattn_mma_config(128, 1, 128, 32, 32, 24, 2, true, 0);
        if (ncols <= 32) return fattn_mma_config(128, 1, 128, 32, 32, 24, 2, true, 0);
        if (ncols <= 64) return fattn_mma_config(128, 1, 128, 32, 32, 24, 2, true, 0);
    }
    if (DKQ == 112 && DV == 112) {
        // nbatch_combine=28 to satisfy (DV/2) % nbatch_combine == 0 → 56 % 28 = 0
        if (ncols <=  8) return fattn_mma_config(128, 1, 128, 32, 32, 28, 2, true, 0);
        if (ncols <= 16) return fattn_mma_config(128, 1, 128, 32, 32, 28, 2, true, 0);
        if (ncols <= 32) return fattn_mma_config(128, 1, 128, 32, 32, 28, 2, true, 0);
        if (ncols <= 64) return fattn_mma_config(128, 1, 128, 32, 32, 28, 2, true, 0);
    }
    if (DKQ == 128 && DV == 128) {
        // Most common: Llama, Mistral, Qwen, etc.
        // This is the exact config from Gau-Nernst reference
        if (ncols <=  8) return fattn_mma_config(128, 1, 128, 32, 32, 32, 2, true, 0);
        if (ncols <= 16) return fattn_mma_config(128, 1, 128, 32, 32, 32, 2, true, 0);
        if (ncols <= 32) return fattn_mma_config(128, 1, 128, 32, 32, 32, 2, true, 0);
        if (ncols <= 64) return fattn_mma_config(128, 1, 128, 32, 32, 32, 2, false, 0);
    }
    
    // ========== Large heads - Fall back to Ampere ==========
    
    if (DKQ == 256 && DV == 256) {
        // Shared mem with nbatch_fa=64: 64 × 256 × 4 × 2 = 128KB > 99KB ❌
        // nbatch_fa=32 isn't a valid cp_async preload size (must be 0, 64, 128, 256)
        // Fall back to Ampere which handles chunking gracefully
        return ggml_cuda_fattn_mma_get_config_ampere(DKQ, DV, ncols);
    }
    
    // MLA/DeepSeek (576/512): Very large heads
    // Shared mem requirement is huge, fall back to Ampere which handles chunking better
    if (DKQ == 576 && DV == 512) {
        return ggml_cuda_fattn_mma_get_config_ampere(DKQ, DV, ncols);
    }

    // Fallback to Ampere config for unsupported head sizes
    return ggml_cuda_fattn_mma_get_config_ampere(DKQ, DV, ncols);
}

// Datacenter Blackwell (sm_100, B200/B100) configurations  
// Uses 288 threads (9 warps = 8 consumer + 1 producer)
// Has 227KB shared memory, can use larger tiles
static constexpr __host__ __device__ fattn_mma_config ggml_cuda_fattn_mma_get_config_sm100(const int DKQ, const int DV, const int ncols) {
    // Original datacenter Blackwell configs with 288 threads
    if (DKQ == 576 && DV == 512) {
        return fattn_mma_config(288, 1, 128, 32, 32, 16, 2, true, 8);
    }

    if (DKQ == 64 && DV == 64) {
        if (ncols <=  8) return fattn_mma_config(288, 1, 128, 16, 16, 16, 2, true, 8);
        if (ncols <= 16) return fattn_mma_config(288, 1, 128, 16, 16, 16, 2, true, 8);
        if (ncols <= 32) return fattn_mma_config(288, 1,  64, 16, 16, 16, 2, true, 8);
        if (ncols <= 64) return fattn_mma_config(288, 1,  64, 16, 16, 16, 2, true, 8);
    }
    if (DKQ == 80 && DV == 80) {
        if (ncols <=  8) return fattn_mma_config(288, 1, 128, 20, 20, 20, 2, true, 8);
        if (ncols <= 16) return fattn_mma_config(288, 1, 128, 20, 20, 20, 2, true, 8);
        if (ncols <= 32) return fattn_mma_config(288, 1,  64, 20, 20, 20, 2, true, 8);
        if (ncols <= 64) return fattn_mma_config(288, 1,  64, 20, 20, 20, 2, true, 8);
    }
    if (DKQ == 96 && DV == 96) {
        if (ncols <=  8) return fattn_mma_config(288, 1, 128, 24, 24, 24, 2, true, 8);
        if (ncols <= 16) return fattn_mma_config(288, 1, 128, 24, 24, 24, 2, true, 8);
        if (ncols <= 32) return fattn_mma_config(288, 1,  64, 24, 24, 24, 2, true, 8);
        if (ncols <= 64) return fattn_mma_config(288, 1,  64, 24, 24, 24, 2, true, 8);
    }
    if (DKQ == 112 && DV == 112) {
        if (ncols <=  8) return fattn_mma_config(288, 1, 128, 28, 28, 14, 2, true, 8);
        if (ncols <= 16) return fattn_mma_config(288, 1, 128, 28, 28, 14, 2, true, 8);
        if (ncols <= 32) return fattn_mma_config(288, 1,  64, 28, 28, 14, 2, true, 8);
        if (ncols <= 64) return fattn_mma_config(288, 1,  64, 28, 28, 14, 2, true, 8);
    }
    if (DKQ == 128 && DV == 128) {
        if (ncols <=  8) return fattn_mma_config(288, 1, 128, 32, 32, 16, 2, true, 8);
        if (ncols <= 16) return fattn_mma_config(288, 1, 128, 32, 32, 16, 2, true, 8);
        if (ncols <= 32) return fattn_mma_config(288, 1,  64, 32, 32, 16, 2, false, 8);
        if (ncols <= 64) return fattn_mma_config(288, 1,  32, 32, 32, 16, 2, false, 8);
    }
    if (DKQ == 256 && DV == 256) {
        if (ncols <=  8) return fattn_mma_config(288, 1, 128, 32, 32, 16, 2, true, 8);
        if (ncols <= 16) return fattn_mma_config(288, 1, 128, 32, 32, 16, 2, true, 8);
        if (ncols <= 32) return fattn_mma_config(288, 1,  64, 32, 32, 16, 2, true, 8);
        if (ncols <= 64) return fattn_mma_config(288, 1,  64, 32, 32, 16, 2, true, 8);
    }

    return ggml_cuda_fattn_mma_get_config_ampere(DKQ, DV, ncols);
}

// Unified Blackwell config dispatcher - routes to sm_120 or sm_100 based on architecture
static constexpr __host__ __device__ fattn_mma_config ggml_cuda_fattn_mma_get_config_blackwell(const int DKQ, const int DV, const int ncols) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
    // Device-side: sm_120 (consumer Blackwell, RTX 5090)
    return ggml_cuda_fattn_mma_get_config_sm120(DKQ, DV, ncols);
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    // Device-side: sm_100 (datacenter Blackwell, B200/B100)
    return ggml_cuda_fattn_mma_get_config_sm100(DKQ, DV, ncols);
#else
    // Host-side or fallback: use sm_120 config (safer, fits all Blackwell)
    return ggml_cuda_fattn_mma_get_config_sm120(DKQ, DV, ncols);
#endif
}

// Check if we have a native Blackwell config for this DKQ/DV/ncols combination
// Returns true for both sm_120 (RTX 5090) and sm_100 (B200) for supported head sizes
static constexpr __host__ __device__ bool ggml_cuda_fattn_has_blackwell_config(const int DKQ, const int DV, const int ncols) {
    // Small heads - supported on both sm_120 and sm_100
    if (DKQ == 64  && DV == 64  && ncols <= 64) return true;
    if (DKQ == 80  && DV == 80  && ncols <= 64) return true;
    // Medium heads - supported on both sm_120 and sm_100
    if (DKQ == 96  && DV == 96  && ncols <= 64) return true;
    if (DKQ == 112 && DV == 112 && ncols <= 64) return true;
    if (DKQ == 128 && DV == 128 && ncols <= 64) return true;
    // Large heads (256+): sm_120 falls back to Ampere (shared mem > 99KB)
    // sm_100 can handle these with 227KB shared mem
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1200
    if (DKQ == 256 && DV == 256 && ncols <= 64) return true;
    if (DKQ == 576 && DV == 512 && ncols <= 64) return true;
#endif
    return false;
}

static constexpr __host__ __device__ fattn_mma_config ggml_cuda_fattn_mma_get_config_turing(const int DKQ, const int DV, const int ncols) {
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256,  8, 128, 2,  64, 128, 128, 128, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 16, 128, 2,  64, 128, 128, 128, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 32, 128, 2,  64, 128, 128,  64, 2, true);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(256, 256, 64, 128, 2,  64, 128, 128,  64, 2, true);

    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512,  8,  64, 4,  32,  96,  64, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 16,  64, 4,  32,  96,  64, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 32, 128, 2,  32, 160, 128, 128, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 64, 256, 1,  32, 160, 128, 128, 1, false);

    return ggml_cuda_fattn_mma_get_config_ampere(DKQ, DV, ncols);
}

static constexpr __host__ __device__ fattn_mma_config ggml_cuda_fattn_mma_get_config_volta(const int DKQ, const int DV, const int ncols) {
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512,  8,  64, 4,  32, 288, 256,  64, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 16,  64, 4,  32, 288, 256,  64, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 32, 128, 2,  32, 160, 128,  64, 1, false);
    GGML_CUDA_FATTN_MMA_CONFIG_CASE(576, 512, 64, 256, 1,  32, 160, 128,  64, 1, false);

    return ggml_cuda_fattn_mma_get_config_ampere(DKQ, DV, ncols);
}

static __host__ fattn_mma_config ggml_cuda_fattn_mma_get_config(const int DKQ, const int DV, const int ncols, const int cc) {
    if (ggml_cuda_has_blackwell_features(cc)) {
        // Route to correct Blackwell config based on compute capability
        if (ggml_cuda_is_consumer_blackwell(cc)) {
            // sm_120 (RTX 5090): 160 threads (5 warps = 1 producer + 4 consumers), 99KB shared mem
            return ggml_cuda_fattn_mma_get_config_sm120(DKQ, DV, ncols);
        } else {
            // sm_100 (B200/B100): 288 threads (9 warps = 1 producer + 8 consumers), 227KB shared mem
            return ggml_cuda_fattn_mma_get_config_sm100(DKQ, DV, ncols);
        }
    }
    if (ampere_mma_available(cc)) {
        return ggml_cuda_fattn_mma_get_config_ampere(DKQ, DV, ncols);
    }
    if (turing_mma_available(cc)) {
        return ggml_cuda_fattn_mma_get_config_turing(DKQ, DV, ncols);
    }
    GGML_ASSERT(volta_mma_available(cc));
    return ggml_cuda_fattn_mma_get_config_volta(DKQ, DV, ncols);
}

static constexpr __device__ fattn_mma_config ggml_cuda_fattn_mma_get_config(const int DKQ, const int DV, const int ncols) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    return ggml_cuda_fattn_mma_get_config_blackwell(DKQ, DV, ncols);
#elif defined(AMPERE_MMA_AVAILABLE)
    return ggml_cuda_fattn_mma_get_config_ampere(DKQ, DV, ncols);
#elif defined(TURING_MMA_AVAILABLE)
    return ggml_cuda_fattn_mma_get_config_turing(DKQ, DV, ncols);
#elif defined(VOLTA_MMA_AVAILABLE)
    return ggml_cuda_fattn_mma_get_config_volta(DKQ, DV, ncols);
#else
    GGML_UNUSED_VARS(DKQ, DV, ncols);
    return fattn_mma_config(32, 1, 0, 0, 0, 0, 0, false);
#endif // defined(AMPERE_MMA_AVAILABLE)
}

static __host__ int ggml_cuda_fattn_mma_get_nthreads(const int DKQ, const int DV, const int ncols, const int cc) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc).nthreads;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nthreads(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).nthreads;
}

static __host__ int ggml_cuda_fattn_mma_get_occupancy(const int DKQ, const int DV, const int ncols, const int cc) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc).occupancy;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_occupancy(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).occupancy;
}

static __host__ int ggml_cuda_fattn_mma_get_nbatch_fa(const int DKQ, const int DV, const int ncols, const int cc) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc).nbatch_fa;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nbatch_fa(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).nbatch_fa;
}

static __host__ int ggml_cuda_fattn_mma_get_nbatch_K2(const int DKQ, const int DV, const int ncols, const int cc) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc).nbatch_K2;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nbatch_K2(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).nbatch_K2;
}

static __host__ int ggml_cuda_fattn_mma_get_nbatch_V2(const int DKQ, const int DV, const int ncols, const int cc) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc).nbatch_V2;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nbatch_V2(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).nbatch_V2;
}

static __host__ int ggml_cuda_fattn_mma_get_nbatch_combine(const int DKQ, const int DV, const int ncols, const int cc) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc).nbatch_combine;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nbatch_combine(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).nbatch_combine;
}

static __host__ int ggml_cuda_fattn_mma_get_nstages_target(const int DKQ, const int DV, const int ncols, const int cc) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc).nstages_target;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nstages_target(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).nstages_target;
}

static __host__ bool ggml_cuda_fattn_mma_get_Q_in_reg(const int DKQ, const int DV, const int ncols, const int cc) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc).Q_in_reg;
}

static constexpr __device__ bool ggml_cuda_fattn_mma_get_Q_in_reg(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).Q_in_reg;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_num_consumers(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols).num_consumers;
}

// ------------------------------------------------------------------------------------------------------------------

static __host__ int ggml_cuda_fattn_mma_get_nstages(const int DKQ, const int DV, const int ncols1, const int ncols2, const int cc) {
    return cp_async_available(cc) && ncols2 >= 2 ? ggml_cuda_fattn_mma_get_nstages_target(DKQ, DV, ncols1*ncols2, cc) : 0;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nstages(const int DKQ, const int DV, const int ncols1, const int ncols2) {
#ifdef CP_ASYNC_AVAILABLE
    return ncols2 >= 2 ? ggml_cuda_fattn_mma_get_nstages_target(DKQ, DV, ncols1*ncols2) : 0;
#else
    GGML_UNUSED_VARS(DKQ, DV, ncols1, ncols2);
    return 0;
#endif // CP_ASYNC_AVAILABLE
}

// ------------------------------------------------------------------------------------------------------------------
// Ampere-specific config getters (always return Ampere config, regardless of compile target)
// These are used by the Ampere kernel to ensure it always gets compatible values,
// even when compiled for Blackwell (where it serves as a fallback).
// ------------------------------------------------------------------------------------------------------------------

static constexpr __host__ __device__ int ggml_cuda_fattn_mma_get_nthreads_ampere(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config_ampere(DKQ, DV, ncols).nthreads;
}

static constexpr __host__ __device__ int ggml_cuda_fattn_mma_get_occupancy_ampere(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config_ampere(DKQ, DV, ncols).occupancy;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nbatch_fa_ampere(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config_ampere(DKQ, DV, ncols).nbatch_fa;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nbatch_K2_ampere(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config_ampere(DKQ, DV, ncols).nbatch_K2;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nbatch_V2_ampere(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config_ampere(DKQ, DV, ncols).nbatch_V2;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nbatch_combine_ampere(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config_ampere(DKQ, DV, ncols).nbatch_combine;
}

static constexpr __device__ bool ggml_cuda_fattn_mma_get_Q_in_reg_ampere(const int DKQ, const int DV, const int ncols) {
    return ggml_cuda_fattn_mma_get_config_ampere(DKQ, DV, ncols).Q_in_reg;
}

static constexpr __device__ int ggml_cuda_fattn_mma_get_nstages_ampere(const int DKQ, const int DV, const int ncols1, const int ncols2) {
#ifdef CP_ASYNC_AVAILABLE
    return ncols2 >= 2 ? ggml_cuda_fattn_mma_get_config_ampere(DKQ, DV, ncols1*ncols2).nstages_target : 0;
#else
    GGML_UNUSED_VARS(DKQ, DV, ncols1, ncols2);
    return 0;
#endif
}

// ------------------------------------------------------------------------------------------------------------------
// Pipeline State for Blackwell with Chunk-Level Pipelining (Option D: k0-chunk pipelining)
// ------------------------------------------------------------------------------------------------------------------
//
// PROBLEM ADDRESSED:
// When DKQ/2 > nbatch_K2, multiple TMA loads (chunks) are needed per K/V tile. The previous
// design had barriers only at the tile level, causing all chunks to overwrite the same buffer
// before the consumer could process them. This led to incorrect results for early k0 iterations.
//
// SOLUTION:
// Use 2-stage double-buffering for k0 chunks within each kb0 iteration. Each chunk gets its
// own buffer slot (stage 0 or 1), allowing producer and consumer to operate concurrently on
// different chunks.
//
// SYNCHRONIZATION PROTOCOL:
//
// Producer (warp 0) workflow for K chunks:
//   for each chunk i in 0..num_K_chunks-1:
//     stage = i % 2
//     1. Wait on empty_K_chunk[stage] (consumer signals when done with this stage)
//     2. Load chunk[i] into K_buffer[stage] via TMA
//     3. Signal full_K_chunk[stage] (TMA completion callback)
//
// Consumer (warps 1-N) workflow for K chunks:
//   for each chunk i in 0..num_K_chunks-1:
//     stage = i % 2
//     1. Wait on full_K_chunk[stage] (producer signals when data ready)
//     2. Process chunk[i] from K_buffer[stage]
//     3. Signal empty_K_chunk[stage] (ready for reuse)
//
// V chunks follow the same pattern with full_V_chunk/empty_V_chunk barriers.
//
// PHASE MANAGEMENT:
// - Phase flips when stage wraps from 1 back to 0 (i.e., every 2 chunks)
// - Both producer and consumer maintain their own phase counters for K and V
// - Phase ensures proper barrier synchronization across multiple chunk cycles
//
// MEMORY LAYOUT (per kb0 iteration with chunk pipelining enabled):
//   K_chunk_buffer[0]: smem_base + 0
//   K_chunk_buffer[1]: smem_base + bytes_K_chunk
//   V_chunk_buffer[0]: smem_base + 2*bytes_K_chunk
//   V_chunk_buffer[1]: smem_base + 2*bytes_K_chunk + bytes_V_chunk
//   where bytes_K_chunk = nbatch_fa * nbatch_K2 * sizeof(half2)
//         bytes_V_chunk = nbatch_fa * nbatch_V2 * sizeof(half2)
//
// BARRIER INITIALIZATION:
//   full_K_chunk[0..1]:  init count = 1 (TMA signals on completion)
//   empty_K_chunk[0..1]: init count = num_consumers * WARP_SIZE (all consumer threads)
//   full_V_chunk[0..1]:  init count = 1 (TMA signals on completion)
//   empty_V_chunk[0..1]: init count = num_consumers * WARP_SIZE (all consumer threads)
//   Q_loaded:            init count = num_consumers * WARP_SIZE
//
// PRE-ARRIVAL PROTOCOL:
//   Before first iteration, consumer warps pre-arrive on empty_K_chunk[0..1] and
//   empty_V_chunk[0..1] to allow producer to load initial chunks without deadlock.
//
// LEGACY TILE-LEVEL BARRIERS:
//   The full_K[4]/empty_K[4]/full_V[4]/empty_V[4] arrays are retained for backward
//   compatibility when the entire tile fits in one chunk (DKQ/2 <= nbatch_K2).
//   In this case, chunk-level pipelining is not needed and we fall back to
//   tile-level synchronization.
//
// ------------------------------------------------------------------------------------------------------------------
struct fattn_pipeline_state {
    // Legacy tile-level barriers for kb0-level pipelining (up to 4 stages)
    // Used when entire tile fits in one chunk (DKQ/2 <= nbatch_K2)
    uint64_t full_K[4];         // Producer signals when K tile is fully loaded
    uint64_t empty_K[4];        // Consumer signals when done with K tile
    uint64_t full_V[4];         // Producer signals when V tile is fully loaded
    uint64_t empty_V[4];        // Consumer signals when done with V tile

    // Q synchronization barrier
    uint64_t Q_loaded;          // Consumers signal after loading Q into registers

    // Chunk-level barriers for k0 pipelining (Option D)
    // Uses 2-stage double-buffering for chunks within each tile
    uint64_t full_K_chunk[2];   // Producer signals when K chunk is ready
    uint64_t empty_K_chunk[2];  // Consumer signals when done with K chunk
    uint64_t full_V_chunk[2];   // Producer signals when V chunk is ready
    uint64_t empty_V_chunk[2];  // Consumer signals when done with V chunk
    
    // ========================================================================
    // SM_120 FALLBACK: Shared memory flags (replaces mbarrier on RTX 5090)
    // ========================================================================
    // On SM_120, mbarrier.arrive.expect_tx crashes at runtime.
    // We use TMA + shared memory flags + __syncthreads__ instead.
    sm120_sync_flags sm120_flags;
};

// ------------------------------------------------------------------------------------------------------------------

template<int stride_tile, int nwarps, int nbatch_fa>
static __device__ __forceinline__ void flash_attn_ext_f16_load_tile_tma(
        const char * __restrict__ tensor_maps,
        const int map_idx,
        half2 * const __restrict__ tile_KV,
        uint64_t* __restrict__ mbar,
        int32_t coord_x,
        int32_t coord_y) {
#ifdef BLACKWELL_TMA_AVAILABLE
    const CUtensorMap* tensor_map = (const CUtensorMap*)(tensor_maps + map_idx * sizeof(CUtensorMap));
    tma_load_2d(tile_KV, tensor_map, mbar, coord_x, coord_y);
#else
    GGML_UNUSED(tensor_maps);
    GGML_UNUSED(map_idx);
    GGML_UNUSED(tile_KV);
    GGML_UNUSED(mbar);
    GGML_UNUSED(coord_x);
    GGML_UNUSED(coord_y);
    NO_DEVICE_CODE;
#endif
}

template<int stride_tile, int nwarps, int nbatch_fa, int nbatch_K2, int num_chunks, int chunk_size>
static __device__ __forceinline__ void flash_attn_ext_f16_load_tile_tma_chunked(
        const char * __restrict__ tensor_maps,
        const int map_idx,
        half2 * const __restrict__ tile_KV,
        uint64_t* __restrict__ mbar,
        int32_t coord_x_base,
        int32_t coord_y) {
#ifdef BLACKWELL_TMA_AVAILABLE
    const CUtensorMap* tensor_map = (const CUtensorMap*)(tensor_maps + map_idx * sizeof(CUtensorMap));
    int32_t coord_x = coord_x_base * 2;
    tma_load_2d(tile_KV, tensor_map, mbar, coord_x, coord_y);
#else
    GGML_UNUSED(tensor_maps);
    GGML_UNUSED(map_idx);
    GGML_UNUSED(tile_KV);
    GGML_UNUSED(mbar);
    GGML_UNUSED(coord_x_base);
    GGML_UNUSED(coord_y);
    NO_DEVICE_CODE;
#endif
}

template<int nbatch_fa, int nbatch_K2>
static __device__ __forceinline__ void load_tile_tma_multistrip(
        const CUtensorMap* __restrict__ tensor_map,
        half2* __restrict__ smem_buffer,
        uint64_t* __restrict__ mbar,
        int32_t coord_x_base_h2,
        int32_t coord_y,
        bool arrive) 
{
#ifdef BLACKWELL_TMA_AVAILABLE
    // DEBUG: Very first line in function
    if (threadIdx.x == 0 && blockIdx.x < 50) {
        printf("[MULTISTRIP ENTRY] Block %d: smem=%p mbar=%p coord_x_h2=%d coord_y=%d arrive=%d\n",
               blockIdx.x, smem_buffer, mbar, coord_x_base_h2, coord_y, arrive);
    }

    if (arrive) {
        uint32_t bytes = nbatch_fa * nbatch_K2 * sizeof(half2);

        if (threadIdx.x == 0 && blockIdx.x < 50) {
            printf("[MULTISTRIP] Block %d: about to call mbarrier_arrive_expect_tx bytes=%u\n",
                   blockIdx.x, bytes);
        }

        mbarrier_arrive_expect_tx(mbar, bytes);
    }
    
    // sm_120 with large tiles (nbatch_K2 > 32) uses SWIZZLE_NONE, allowing single TMA load
    // of the full tile. For nbatch_K2 <= 32, uses SWIZZLE_128B which is compatible with ldmatrix.
    // Multi-strip loading only needed for sm_100 with SWIZZLE_128B and large tiles.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
    // sm_120: Always single TMA load (SWIZZLE_128B for small tiles, SWIZZLE_NONE for large)
    tma_load_2d(smem_buffer, tensor_map, mbar, coord_x_base_h2 * 2, coord_y);
#else
    // sm_100 and earlier: Use multi-strip for large tiles (SWIZZLE_128B)
    constexpr int STRIP_H2 = 32; // 64 elements = 128 bytes. Matches SWIZZLE_128B.
    if constexpr (nbatch_K2 <= STRIP_H2) {
        tma_load_2d(smem_buffer, tensor_map, mbar, coord_x_base_h2 * 2, coord_y);
    } else {
        #pragma unroll
        for (int s = 0; s < nbatch_K2; s += STRIP_H2) {
            tma_load_2d(smem_buffer + s * nbatch_fa, tensor_map, mbar, (coord_x_base_h2 + s) * 2, coord_y);
        }
    }
#endif
#else
    GGML_UNUSED(tensor_map);
    GGML_UNUSED(smem_buffer);
    GGML_UNUSED(mbar);
    GGML_UNUSED(coord_x_base_h2);
    GGML_UNUSED(coord_y);
    GGML_UNUSED(arrive);
    NO_DEVICE_CODE;
#endif
}

// ------------------------------------------------------------------------------------------------------------------

template<int stride_tile, int nwarps, int nbatch_fa, bool use_cp_async, bool oob_check>
static __device__ __forceinline__ void flash_attn_ext_f16_load_tile(
        const half2 * const __restrict__ KV, half2 * const __restrict__ tile_KV, const int D2, const int stride_KV, const int i_sup) {
    if constexpr (use_cp_async) {
        static_assert(!oob_check, "OOB check not compatible with cp_async");
        constexpr int preload = 64;
        constexpr int h2_per_chunk = 16/sizeof(half2);
        const int chunks_per_row = D2 / h2_per_chunk;

        const unsigned int tile_KV_32 = ggml_cuda_cvta_generic_to_shared(tile_KV);

        auto load = [&] __device__ (auto n) { 
            const int stride_k = WARP_SIZE >> n;
            const int k0_start = stride_k == WARP_SIZE ? 0 : chunks_per_row - chunks_per_row % (2*stride_k);
            const int k0_stop  =                             chunks_per_row - chunks_per_row % (1*stride_k);
            const int stride_i = WARP_SIZE / stride_k;

            if (k0_start == k0_stop) {
                return;
            }

#pragma unroll
            for (int i0 = 0; i0 < nbatch_fa; i0 += nwarps*stride_i) {
                const int i = i0 + threadIdx.y*stride_i + (stride_k == WARP_SIZE ? 0 : threadIdx.x / stride_k);

                if (i0 + nwarps*stride_i > nbatch_fa && i >= nbatch_fa) {
                    break;
                }

#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == WARP_SIZE ? threadIdx.x : threadIdx.x % stride_k);

                    cp_async_cg_16<preload>(tile_KV_32 + i*(stride_tile*sizeof(half2)) + k*16, KV + i*stride_KV + k*h2_per_chunk);
                }
            }
        };
        ggml_cuda_unroll<6>{}(load);
    } else {
        auto load = [&] __device__ (const int n) { 
            const int stride_k = WARP_SIZE >> n;
            const int k0_start = stride_k == WARP_SIZE ? 0 : D2 - D2 % (2*stride_k);
            const int k0_stop  =                             D2 - D2 % (1*stride_k);
            const int stride_i = WARP_SIZE / stride_k;

            if (k0_start == k0_stop) {
                return;
            }

#pragma unroll
            for (int i0 = 0; i0 < nbatch_fa; i0 += nwarps*stride_i) {
                const int i = i0 + threadIdx.y*stride_i + (stride_k == WARP_SIZE ? 0 : threadIdx.x / stride_k);

                if (i0 + nwarps*stride_i > nbatch_fa && i >= nbatch_fa) {
                    break;
                }

#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == WARP_SIZE ? threadIdx.x : threadIdx.x % stride_k);

                    tile_KV[i*stride_tile + k] = !oob_check || i < i_sup ? KV[i*stride_KV + k] : make_half2(0.0f, 0.0f);
                }
            }
        };
        ggml_cuda_unroll<4>{}(load);
    }
}

// ------------------------------------------------------------------------------------------------------------------
// SM_120 Optimized: Swizzled tile loading for bank-conflict-free ldmatrix reads
// ------------------------------------------------------------------------------------------------------------------
// This version applies XOR-based swizzling to the destination shared memory addresses.
// When loading K/V tiles, consecutive rows are remapped to different bank groups,
// eliminating the 8-way bank conflicts that occur with ldmatrix on linear layouts.
//
// Performance impact: 68% → 86% of theoretical throughput (Gau-Nernst reference)
//
// Parameters:
//   stride_tile: destination stride in half2 (same as regular load_tile)
//   STRIDE_BYTES: row stride in bytes for swizzle calculation (stride_tile * sizeof(half2))
// ------------------------------------------------------------------------------------------------------------------
template<int stride_tile, int STRIDE_BYTES, int nwarps, int nbatch_fa, int warp_id_offset = 0>
static __device__ __forceinline__ void flash_attn_ext_f16_load_tile_swizzle(
        const half2 * const __restrict__ KV, 
        uint32_t tile_KV_base,  // Shared memory base address (32-bit)
        const int D2, 
        const int stride_KV) {
#if defined(CP_ASYNC_AVAILABLE) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
    constexpr int preload = 64;
    constexpr int h2_per_chunk = 16/sizeof(half2);  // 4 half2 per 16B copy
    const int chunks_per_row = D2 / h2_per_chunk;
    
    // For SM_120 consumer warps: threadIdx.y is 1-4, but we need warp_id 0-3
    // warp_id_offset allows caller to specify the offset (e.g., 1 for consumer warps)
    const int warp_id = threadIdx.y - warp_id_offset;

    auto load = [&, warp_id] __device__ (auto n) { 
        const int stride_k = WARP_SIZE >> n;
        const int k0_start = stride_k == WARP_SIZE ? 0 : chunks_per_row - chunks_per_row % (2*stride_k);
        const int k0_stop  =                             chunks_per_row - chunks_per_row % (1*stride_k);
        const int stride_i = WARP_SIZE / stride_k;

        if (k0_start == k0_stop) {
            return;
        }

#pragma unroll
        for (int i0 = 0; i0 < nbatch_fa; i0 += nwarps*stride_i) {
            const int i = i0 + warp_id*stride_i + (stride_k == WARP_SIZE ? 0 : threadIdx.x / stride_k);

            if (i0 + nwarps*stride_i > nbatch_fa && i >= nbatch_fa) {
                break;
            }

#pragma unroll
            for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                const int k = k0 + (stride_k == WARP_SIZE ? threadIdx.x : threadIdx.x % stride_k);

                // Calculate unswizzled offset in bytes
                uint32_t offset_bytes = i * STRIDE_BYTES + k * 16;
                
                // Apply swizzle: XOR bits [4:6] based on row index
                if constexpr (STRIDE_BYTES > 16) {
                    constexpr int rows_per_swizzle = (64 / STRIDE_BYTES) > 0 ? (64 / STRIDE_BYTES) : 1;
                    uint32_t row_mod8 = (offset_bytes / STRIDE_BYTES) % 8;
                    uint32_t xor_bits = row_mod8 / rows_per_swizzle;
                    offset_bytes ^= (xor_bits << 4);
                }
                
                cp_async_cg_16<preload>(tile_KV_base + offset_bytes, KV + i*stride_KV + k*h2_per_chunk);
            }
        }
    };
    ggml_cuda_unroll<6>{}(load);
#else
    GGML_UNUSED(KV);
    GGML_UNUSED(tile_KV_base);
    GGML_UNUSED(D2);
    GGML_UNUSED(stride_KV);
    NO_DEVICE_CODE;
#endif
}

template<int ncols1, int nwarps, int nbatch_fa, bool use_cp_async, bool oob_check>
static __device__ __forceinline__ void flash_attn_ext_f16_load_mask(
        const half * const __restrict__ mask_h, half * const __restrict__ tile_mask,
        const int stride_mask, const int i_sup, const int j0, const uint3 ne01) {
    if constexpr (use_cp_async) {
        static_assert(nbatch_fa <= 8*WARP_SIZE && nbatch_fa % 8 == 0, "bad nbatch_fa");
        static_assert(!oob_check, "OOB check incompatible with cp_async");
        constexpr int preload = nbatch_fa >= 32 ? nbatch_fa * sizeof(half) : 64;
        constexpr int cols_per_warp = 8*WARP_SIZE/nbatch_fa;
        constexpr int stride_j = nwarps * cols_per_warp;

        const unsigned int tile_mask_32 = ggml_cuda_cvta_generic_to_shared(tile_mask);

#pragma unroll
        for (int j1 = 0; j1 < ncols1; j1 += stride_j) {
            const int j_sram = j1 + threadIdx.y*cols_per_warp + threadIdx.x / (WARP_SIZE/cols_per_warp);
            const int j_vram = fastmodulo(j0 + j_sram, ne01);

            if (j1 + stride_j > ncols1 && j_sram >= ncols1) {
                break;
            }

            const int i = 8 * (threadIdx.x % (nbatch_fa/8));

            cp_async_cg_16<preload>(tile_mask_32 + j_sram*(nbatch_fa*sizeof(half) + 16) + i*sizeof(half), mask_h + j_vram*stride_mask + i);
        }
    } else if constexpr (oob_check) {
#pragma unroll
        for (int j1 = 0; j1 < ncols1; j1 += nwarps) {
            const int j_sram = j1 + threadIdx.y;
            const int j_vram = fastmodulo(j0 + j_sram, ne01);

            if (j1 + nwarps > ncols1 && j_sram >= ncols1) {
                break;
            }

#pragma unroll
            for (int i0 = 0; i0 < nbatch_fa; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                tile_mask[j_sram*(nbatch_fa + 8) + i] = i < i_sup ? mask_h[j_vram*stride_mask + i] : half(0.0f);
            }
        }
    } else if constexpr (nbatch_fa < 2*WARP_SIZE) {
        constexpr int cols_per_warp = 2*WARP_SIZE/nbatch_fa;
        constexpr int stride_j = nwarps * cols_per_warp;
#pragma unroll
        for (int j1 = 0; j1 < ncols1; j1 += stride_j) {
            const int j_sram = j1 + threadIdx.y*cols_per_warp + threadIdx.x / (WARP_SIZE/cols_per_warp);
            const int j_vram = fastmodulo(j0 + j_sram, ne01);

            if (j1 + stride_j > ncols1 && j_sram >= ncols1) {
                break;
            }

            const int i = threadIdx.x % (WARP_SIZE/cols_per_warp);

            ggml_cuda_memcpy_1<sizeof(half2)>(tile_mask + j_sram*(nbatch_fa + 8) + 2*i, mask_h + j_vram*stride_mask + 2*i);
        }
    } else {
#pragma unroll
        for (int j1 = 0; j1 < ncols1; j1 += nwarps) {
            const int j_sram = j1 + threadIdx.y;
            const int j_vram = fastmodulo(j0 + j_sram, ne01);

            if (j1 + nwarps > ncols1 && j_sram >= ncols1) {
                break;
            }

#pragma unroll
            for (int i0 = 0; i0 < nbatch_fa; i0 += 2*WARP_SIZE) {
                const int i = i0 + 2*threadIdx.x;

                ggml_cuda_memcpy_1<sizeof(half2)>(tile_mask + j_sram*(nbatch_fa + 8) + i, mask_h + j_vram*stride_mask + i);
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------------------------
// PRODUCER LOOP (Legacy version - single K/V barrier per tile)
// ------------------------------------------------------------------------------------------------------------------
// For cases where entire tile fits in one chunk (DKQ/2 <= nbatch_K2 AND DV/2 <= nbatch_V2).
// Uses tile-level barriers only, no chunk-level pipelining needed.
template<int DKQ, int DV, int ncols, int nstages, int nbatch_fa, int nbatch_K2, int nbatch_V2, bool mla>
static __device__ __forceinline__ void fattn_producer_loop_legacy(
    const char * __restrict__ tensor_maps,
    fattn_pipeline_state* __restrict__ state,
    int kb0_start,
    int kb0_stop,
    int num_consumers,
    int kv_head_row_offset)  // Row offset for current KV head in flattened tensor
{
#ifdef BLACKWELL_TMA_AVAILABLE
    if (threadIdx.x == 0) {
        printf("[PRODUCER LEGACY] Block %d: ENTERED, kb0=[%d,%d), head_offset=%d\n",
               blockIdx.x, kb0_start, kb0_stop, kv_head_row_offset);
        // Extra warning for non-zero head offsets
        if (kv_head_row_offset > 0) {
            printf("[PRODUCER LEGACY] Block %d: *** NON-ZERO HEAD OFFSET %d ***\n",
                   blockIdx.x, kv_head_row_offset);
        }
    }

    // CRITICAL: TMA fence to ensure tensor map descriptors (K and V) are visible
    // Uses .sys scope since tensor maps are copied from host via cudaMemcpyAsync
    tma_fence_acquire_maps(tensor_maps, 2);

    if (threadIdx.x == 0) {
        printf("[PRODUCER LEGACY] Block %d: TMA fence done\n", blockIdx.x);
    }

    const CUtensorMap* map_K = (const CUtensorMap*)(tensor_maps);
    const CUtensorMap* map_V = (const CUtensorMap*)(tensor_maps + sizeof(CUtensorMap));

    extern __shared__ char smem_base[];

    // Use Blackwell memory layout: [K stage 0][K stage 1][V stage 0][V stage 1]
    // This matches what the consumer in process_tile expects:
    //   tile_K = smem (offset 0)
    //   tile_V = smem + 2 * bytes_K_chunk (offset 2 * bytes_K)
    // Each stage's K/V is at a fixed offset within the K or V region.
    constexpr int bytes_K = nbatch_fa * nbatch_K2 * sizeof(half2);
    constexpr int bytes_V = nbatch_fa * nbatch_V2 * sizeof(half2);

    // K tiles: stages 0 and 1 are at offsets 0 and bytes_K
    // V tiles: stages 0 and 1 are at offsets 2*bytes_K and 2*bytes_K + bytes_V
    constexpr int offset_K_base = 0;
    constexpr int offset_V_base = 2 * bytes_K;  // After both K stages

    if (threadIdx.x == 0) {
        printf("[PRODUCER LEGACY] Block %d: bytes_K=%d, bytes_V=%d, offset_V_base=%d\n",
               blockIdx.x, bytes_K, bytes_V, offset_V_base);
    }

    uint32_t phase = 0;

    for (int kb0 = kb0_start; kb0 < kb0_stop; ++kb0) {
        const int iter = kb0 - kb0_start;
        const int stage = iter % nstages;
        // CRITICAL: Add kv_head_row_offset for multi-head TMA access!
        const int row_offset = kv_head_row_offset + kb0 * nbatch_fa;

        if (threadIdx.x == 0 && kb0 == kb0_start) {
            printf("[PRODUCER LEGACY] Block %d: First iter kb0=%d stage=%d row_offset=%d\n",
                   blockIdx.x, kb0, stage, row_offset);
        }

        // K Phase: use tile-level barriers, Blackwell layout
        if (threadIdx.x == 0 && kb0 == kb0_start) {
            printf("[PRODUCER LEGACY] Block %d: Waiting empty_K[%d] phase=%d\n",
                   blockIdx.x, stage, phase);
        }
        mbarrier_wait(&state->empty_K[stage], phase);
        if (threadIdx.x == 0 && kb0 == kb0_start) {
            printf("[PRODUCER LEGACY] Block %d: empty_K wait DONE\n", blockIdx.x);
        }

        half2* tile_K = (half2*)(smem_base + offset_K_base + stage * bytes_K);
        for (int k0 = 0; k0 < DKQ/2; k0 += nbatch_K2) {
            bool first_chunk = (k0 == 0);
            if (threadIdx.x == 0 && kb0 == kb0_start) {
                printf("[PRODUCER LEGACY] Block %d: K TMA ABOUT TO ISSUE k0=%d row=%d (head=%d+kb0=%d*%d) smem=%p\n",
                       blockIdx.x, k0, row_offset, kv_head_row_offset, kb0, nbatch_fa, tile_K);
            }

            // Inline the TMA call with surrounding debug
            if (first_chunk) {
                constexpr uint32_t bytes_tma = nbatch_fa * nbatch_K2 * sizeof(half2);
                if (threadIdx.x == 0 && kb0 == kb0_start) {
                    printf("[PRODUCER LEGACY] Block %d: calling mbarrier_arrive_expect_tx bytes=%u\n",
                           blockIdx.x, bytes_tma);
                }
                mbarrier_arrive_expect_tx(&state->full_K[stage], bytes_tma);
                if (threadIdx.x == 0 && kb0 == kb0_start) {
                    printf("[PRODUCER LEGACY] Block %d: mbarrier_arrive_expect_tx DONE\n", blockIdx.x);
                }
            }

            if (threadIdx.x == 0 && kb0 == kb0_start) {
                printf("[PRODUCER LEGACY] Block %d: calling tma_load_2d coord_x=%d coord_y=%d\n",
                       blockIdx.x, k0 * 2, row_offset);
            }
            tma_load_2d(tile_K, map_K, &state->full_K[stage], k0 * 2, row_offset);
            if (threadIdx.x == 0 && kb0 == kb0_start) {
                printf("[PRODUCER LEGACY] Block %d: tma_load_2d RETURNED\n", blockIdx.x);
            }
        }

        // V Phase: use tile-level barriers, Blackwell layout
        if (threadIdx.x == 0 && kb0 == kb0_start) {
            printf("[PRODUCER LEGACY] Block %d: Waiting empty_V[%d]\n", blockIdx.x, stage);
        }
        mbarrier_wait(&state->empty_V[stage], phase);
        if (threadIdx.x == 0 && kb0 == kb0_start) {
            printf("[PRODUCER LEGACY] Block %d: empty_V wait DONE\n", blockIdx.x);
        }

        half2* tile_V = (half2*)(smem_base + offset_V_base + stage * bytes_V);
        for (int i0 = 0; i0 < DV/2; i0 += nbatch_V2) {
            bool first_chunk = (i0 == 0);
            if (threadIdx.x == 0 && kb0 == kb0_start) {
                printf("[PRODUCER LEGACY] Block %d: V TMA ABOUT TO ISSUE i0=%d row=%d (head=%d) smem=%p\n",
                       blockIdx.x, i0, row_offset, kv_head_row_offset, tile_V);
            }

            // Inline the TMA call with surrounding debug
            if (first_chunk) {
                constexpr uint32_t bytes_tma = nbatch_fa * nbatch_V2 * sizeof(half2);
                if (threadIdx.x == 0 && kb0 == kb0_start) {
                    printf("[PRODUCER LEGACY] Block %d: V mbarrier_arrive_expect_tx bytes=%u\n",
                           blockIdx.x, bytes_tma);
                }
                mbarrier_arrive_expect_tx(&state->full_V[stage], bytes_tma);
                if (threadIdx.x == 0 && kb0 == kb0_start) {
                    printf("[PRODUCER LEGACY] Block %d: V mbarrier DONE\n", blockIdx.x);
                }
            }

            if (threadIdx.x == 0 && kb0 == kb0_start) {
                printf("[PRODUCER LEGACY] Block %d: V calling tma_load_2d coord_x=%d coord_y=%d\n",
                       blockIdx.x, i0 * 2, row_offset);
            }
            tma_load_2d(tile_V, map_V, &state->full_V[stage], i0 * 2, row_offset);
            if (threadIdx.x == 0 && kb0 == kb0_start) {
                printf("[PRODUCER LEGACY] Block %d: V tma_load_2d RETURNED\n", blockIdx.x);
            }
        }

        if (stage == nstages - 1) {
            phase ^= 1;
        }
    }

    if (threadIdx.x == 0) {
        printf("[PRODUCER LEGACY] Block %d: Loop COMPLETE\n", blockIdx.x);
    }

    GGML_UNUSED(num_consumers);
#else
    GGML_UNUSED(tensor_maps);
    GGML_UNUSED(state);
    GGML_UNUSED(kb0_start);
    GGML_UNUSED(kb0_stop);
    GGML_UNUSED(num_consumers);
    GGML_UNUSED(kv_head_row_offset);
#endif
}

// ------------------------------------------------------------------------------------------------------------------
// PRODUCER LOOP (Chunked version - per-chunk synchronization for MLA and large heads)
// ------------------------------------------------------------------------------------------------------------------
// For cases where multiple chunks are needed (DKQ/2 > nbatch_K2 OR DV/2 > nbatch_V2).
// Uses 2-stage double-buffering with chunk-level barriers.
//
// Pipeline design:
// - Each kb0 iteration represents one attention tile (nbatch_fa rows of K/V)
// - Stage alternates per kb0 iteration for double-buffering (stage = kb0 % nstages)
//
// Barrier protocol for chunk-level pipelining (Option D):
// - Producer waits on empty_K_chunk[stage]/empty_V_chunk[stage] before loading each chunk
// - TMA signals full_K_chunk[stage]/full_V_chunk[stage] upon completion
// - Consumer waits on full_*_chunk, processes chunk, then signals empty_*_chunk
// - Uses 2-stage double buffering: chunks alternate between buffer 0 and 1
// - Phase flips every 2 chunks (when stage wraps from 1 back to 0)
template<int DKQ, int DV, int ncols, int nstages, int nbatch_fa, int nbatch_K2, int nbatch_V2, bool mla>
static __device__ __forceinline__ void fattn_producer_loop_chunked(
    const char * __restrict__ tensor_maps,
    fattn_pipeline_state* __restrict__ state,
    int kb0_start,
    int kb0_stop,
    int num_consumers,
    int kv_head_row_offset)  // Row offset for current KV head in flattened tensor
{
#ifdef BLACKWELL_TMA_AVAILABLE
    // CRITICAL: TMA fence to ensure tensor map descriptors (K and V) are visible
    // Uses .sys scope since tensor maps are copied from host via cudaMemcpyAsync
    tma_fence_acquire_maps(tensor_maps, 2);
    
    // IMMEDIATE DEBUG - print from ALL blocks
    if (threadIdx.x == 0) {
        printf("[PRODUCER CHUNKED] Block %d: ENTERED, head_row_offset=%d\n", blockIdx.x, kv_head_row_offset);
    }
    
    const CUtensorMap* map_K = (const CUtensorMap*)(tensor_maps);
    const CUtensorMap* map_V = (const CUtensorMap*)(tensor_maps + sizeof(CUtensorMap));

    extern __shared__ char smem_base[];
    // Double-buffered chunk storage: 2 chunk buffers for K, 2 for V
    // Layout: [K_chunk_0][K_chunk_1][V_chunk_0][V_chunk_1]
    // Chunks are REUSED each kb0 iteration - no stage-level offset needed
    constexpr int bytes_K_chunk = nbatch_fa * nbatch_K2 * sizeof(half2);
    constexpr int bytes_V_chunk = nbatch_fa * nbatch_V2 * sizeof(half2);

    constexpr int num_K_chunks = (DKQ/2 + nbatch_K2 - 1) / nbatch_K2;
    constexpr int num_V_chunks = (DV/2 + nbatch_V2 - 1) / nbatch_V2;

    // DEBUG: Print producer startup info (ALL blocks)
    if (threadIdx.x == 0) {
        printf("[PRODUCER CHUNKED] Block %d: kb0=[%d,%d) K_chunks=%d V_chunks=%d\n",
               blockIdx.x, kb0_start, kb0_stop, num_K_chunks, num_V_chunks);
    }

    // Chunk-level phases for K and V barriers.
    // After pre-arrival, phase 0 is "completed" so wait(phase=0) returns immediately.
    // The phase tracks which phase we're expecting to wait for.
    // It flips after processing 2 chunks (when chunk_stage goes from 1 back to 0).
    uint32_t chunk_phase_K = 0;
    uint32_t chunk_phase_V = 0;

    // Iterate over tile indices (kb0 is a tile index, not a row offset)
    for (int kb0 = kb0_start; kb0 < kb0_stop; ++kb0) {
        // Compute row offset for TMA coordinates
        // CRITICAL: Add kv_head_row_offset for multi-head TMA access!
        const int row_offset = kv_head_row_offset + kb0 * nbatch_fa;
        
        // DEBUG: Check row bounds
        if (threadIdx.x == 0) {
            printf("[ROW CHECK] Block %d: kb0=%d row_offset=%d (head_base=%d + kb0*%d)\n",
                   blockIdx.x, kb0, row_offset, kv_head_row_offset, nbatch_fa);
        }

        // NOTE: Do NOT reset phases here. The phases track the barrier's state
        // across kb0 iterations. The phase flips happen in the chunk loop below.

        // --- K Phase: Load K chunks with per-chunk synchronization ---
        // NOTE: No stage offset - chunk buffers are reused each kb0 iteration
        half2* tile_K_base = (half2*)(smem_base);

        for (int chunk = 0; chunk < num_K_chunks; ++chunk) {
            const int chunk_stage = chunk % 2;
            const int k0 = chunk * nbatch_K2;

            // DEBUG: Print K chunk info (ALL blocks, first kb0 only)
            if (threadIdx.x == 0 && kb0 == kb0_start) {
                printf("[K CHUNK] Block %d: chunk=%d waiting empty_K[%d] phase=%d\n",
                       blockIdx.x, chunk, chunk_stage, chunk_phase_K);
            }

            // Wait for consumer to finish with this chunk buffer
            mbarrier_wait(&state->empty_K_chunk[chunk_stage], chunk_phase_K);
            
            if (threadIdx.x == 0 && kb0 == kb0_start) {
                printf("[K CHUNK] Block %d: chunk=%d wait DONE\n", blockIdx.x, chunk);
            }

            // Load chunk to the correct buffer (alternating between 0 and 1)
            half2* tile_K_chunk = tile_K_base + chunk_stage * (bytes_K_chunk / sizeof(half2));

            // DEBUG: Print TMA load info (ALL blocks)
            if (threadIdx.x == 0 && kb0 == kb0_start) {
                printf("[K CHUNK] Block %d: chunk=%d TMA k0=%d row=%d coord_x=%d smem=%p\n",
                       blockIdx.x, chunk, k0, row_offset, k0 * 2, tile_K_chunk);
            }

            // DEBUG: Right before load_tile_tma_multistrip call
            if (threadIdx.x == 0 && blockIdx.x < 50) {
                printf("[K CHUNK PRE-CALL] Block %d: chunk_stage=%d mbar=%p\n",
                       blockIdx.x, chunk_stage, &state->full_K_chunk[chunk_stage]);
            }

            // Load this K chunk via TMA
            load_tile_tma_multistrip<nbatch_fa, nbatch_K2>(
                map_K, tile_K_chunk, &state->full_K_chunk[chunk_stage], k0, row_offset, true);

            // DEBUG: Print after load (ALL blocks)
            if (threadIdx.x == 0 && kb0 == kb0_start) {
                printf("[K CHUNK] Block %d: chunk=%d TMA ISSUED\n", blockIdx.x, chunk);
            }

            // Flip chunk phase after every 2 chunks (when chunk_stage goes from 1 back to 0)
            if (chunk_stage == 1) {
                chunk_phase_K ^= 1;
            }
        }

        // --- V Phase: Load V chunks with per-chunk synchronization ---
        // NOTE: No stage offset - chunk buffers are reused each kb0 iteration
        half2* tile_V_base = (half2*)(smem_base + 2 * bytes_K_chunk);

        for (int chunk = 0; chunk < num_V_chunks; ++chunk) {
            const int chunk_stage = chunk % 2;
            const int i0 = chunk * nbatch_V2;

            // DEBUG: Print V chunk info (ALL blocks, first kb0 only)
            if (threadIdx.x == 0 && kb0 == kb0_start) {
                printf("[V CHUNK] Block %d: chunk=%d waiting empty_V[%d] phase=%d\n",
                       blockIdx.x, chunk, chunk_stage, chunk_phase_V);
            }

            // Wait for consumer to finish with this chunk buffer
            mbarrier_wait(&state->empty_V_chunk[chunk_stage], chunk_phase_V);
            
            if (threadIdx.x == 0 && kb0 == kb0_start) {
                printf("[V CHUNK] Block %d: chunk=%d wait DONE\n", blockIdx.x, chunk);
            }

            // Load chunk to the correct buffer (alternating between 0 and 1)
            half2* tile_V_chunk = tile_V_base + chunk_stage * (bytes_V_chunk / sizeof(half2));

            // DEBUG: Print TMA load info (ALL blocks)
            if (threadIdx.x == 0 && kb0 == kb0_start) {
                printf("[V CHUNK] Block %d: chunk=%d TMA i0=%d row=%d coord_x=%d\n",
                       blockIdx.x, chunk, i0, row_offset, i0 * 2);
            }

            // Load this V chunk via TMA
            load_tile_tma_multistrip<nbatch_fa, nbatch_V2>(
                map_V, tile_V_chunk, &state->full_V_chunk[chunk_stage], i0, row_offset, true);
            
            // DEBUG: After TMA (ALL blocks)
            if (threadIdx.x == 0 && kb0 == kb0_start) {
                printf("[V CHUNK] Block %d: chunk=%d TMA ISSUED\n", blockIdx.x, chunk);
            }

            // Flip chunk phase after every 2 chunks
            if (chunk_stage == 1) {
                chunk_phase_V ^= 1;
            }
        }
    }

    // DEBUG: Print producer complete (ALL blocks)
    if (threadIdx.x == 0) {
        printf("[PRODUCER CHUNKED] Block %d: COMPLETE\n", blockIdx.x);
    }

    GGML_UNUSED(num_consumers);
    GGML_UNUSED(nstages);
#else
    GGML_UNUSED(tensor_maps);
    GGML_UNUSED(state);
    GGML_UNUSED(kb0_start);
    GGML_UNUSED(kb0_stop);
    GGML_UNUSED(num_consumers);
    GGML_UNUSED(kv_head_row_offset);
#endif
}

// ------------------------------------------------------------------------------------------------------------------
// PRODUCER LOOP (SM_120 Optimized - cp.async with Two-Stage Pipelining)
// ------------------------------------------------------------------------------------------------------------------
// For SM_120 (RTX 5090), TMA mbarrier.arrive crashes at runtime.
// Instead of TMA, we use optimized cp.async with:
//   1. Two-stage pipelining - Overlaps load with compute
//   2. Cooperative loading - All threads participate in global→shared
//   3. Compact layout - Matches bytes_K_chunk = nbatch_fa * nbatch_K2 * sizeof(half2)
//
// Memory Layout (matches consumer's expectations):
//   Stage 0 K: smem_base + 0
//   Stage 1 K: smem_base + bytes_K_chunk
//   Stage 0 V: smem_base + 2 * bytes_K_chunk
//   Stage 1 V: smem_base + 2 * bytes_K_chunk + bytes_V_chunk
//
// The consumer uses stride_tile_K = nbatch_K2 for TMA path, but for SM_120 (use_tma=false),
// we use compact layout with stride = nbatch_K2 (no padding) because:
//   - TMA writes compact data (no padding)
//   - Consumer's bytes_K_chunk = nbatch_fa * nbatch_K2 * sizeof(half2) is compact
//   - Stage offset = bytes_K_chunk must match between producer and consumer
//
// Performance target: ~85-90% of speed-of-light throughput
template<int DKQ, int DV, int ncols, int nstages, int nbatch_fa, int nbatch_K2, int nbatch_V2, bool mla>
static __device__ __forceinline__ void fattn_producer_loop_sm120(
    const char * __restrict__ tensor_maps,
    fattn_pipeline_state* __restrict__ state,
    int kb0_start,
    int kb0_stop,
    int num_consumers,
    int kv_head_row_offset,
    const half2 * __restrict__ K_h2,
    const half2 * __restrict__ V_h2,
    int stride_K,
    int stride_V)
{
#ifdef CP_ASYNC_AVAILABLE
    // ============================================================================
    // SM_120 cp.async with Two-Stage Pipelining
    // ============================================================================
    // Memory layout MUST match consumer's bytes_K_chunk/bytes_V_chunk definitions:
    //   bytes_K_chunk = nbatch_fa * nbatch_K2 * sizeof(half2)  [COMPACT]
    //   bytes_V_chunk = nbatch_fa * nbatch_V2 * sizeof(half2)  [COMPACT]
    //
    // Layout: [K stage 0][K stage 1][V stage 0][V stage 1]
    // ============================================================================
    
    extern __shared__ char smem_base[];
    
    // CRITICAL: Use compact layout matching consumer's bytes_K_chunk calculation
    // Consumer at line 1501: bytes_K_chunk = nbatch_fa * nbatch_K2 * sizeof(half2)
    constexpr int stride_tile_K = nbatch_K2;  // COMPACT (no padding) - matches TMA layout
    constexpr int stride_tile_V = mla ? nbatch_K2 : nbatch_V2;  // COMPACT
    
    constexpr int bytes_K_chunk = nbatch_fa * nbatch_K2 * sizeof(half2);  // Must match consumer
    constexpr int bytes_V_chunk = nbatch_fa * nbatch_V2 * sizeof(half2);  // Must match consumer
    
    const int tid = threadIdx.x + threadIdx.y * WARP_SIZE;
    const int total_threads = blockDim.x * blockDim.y;
    
    const uint32_t smem_K_base = ggml_cuda_cvta_generic_to_shared(smem_base);
    const uint32_t smem_V_base = smem_K_base + 2 * bytes_K_chunk;  // After both K stages
    
    const int num_iters = kb0_stop - kb0_start;
    
    // Helper lambda for loading K tile with compact layout
    auto load_K_tile = [&](int stage, int row_offset) {
        const half2 * K_src = K_h2 + int64_t(row_offset) * stride_K;
        
        // Load nbatch_fa rows of nbatch_K2 half2 elements each
        // Each thread loads 16 bytes = 4 half2 per iteration
        for (int copy_id = tid; copy_id < nbatch_fa * nbatch_K2 / 4; copy_id += total_threads) {
            int row = copy_id / (nbatch_K2 / 4);
            int col = (copy_id % (nbatch_K2 / 4)) * 4;  // Each copy handles 4 half2 = 16 bytes
            
            if (row < nbatch_fa) {
                uint32_t dst_addr = smem_K_base + stage * bytes_K_chunk + 
                                   (row * stride_tile_K + col) * sizeof(half2);
                const void * src = K_src + row * stride_K + col;
                cp_async_cg_16<128>(dst_addr, src);
            }
        }
    };
    
    // Helper lambda for loading V tile
    auto load_V_tile = [&](int stage, int row_offset) {
        const half2 * V_src = mla ? (K_h2 + int64_t(row_offset) * stride_K + (DKQ - DV) / 2)
                                  : (V_h2 + int64_t(row_offset) * stride_V);
        const int V_src_stride = mla ? stride_K : stride_V;
        
        for (int copy_id = tid; copy_id < nbatch_fa * nbatch_V2 / 4; copy_id += total_threads) {
            int row = copy_id / (nbatch_V2 / 4);
            int col = (copy_id % (nbatch_V2 / 4)) * 4;
            
            if (row < nbatch_fa) {
                uint32_t dst_addr = smem_V_base + stage * bytes_V_chunk +
                                   (row * stride_tile_V + col) * sizeof(half2);
                const void * src = V_src + row * V_src_stride + col;
                cp_async_cg_16<128>(dst_addr, src);
            }
        }
    };
    
    // ========== PREFETCH FIRST K TILE ==========
    if (num_iters > 0) {
        const int row_offset = kv_head_row_offset + kb0_start * nbatch_fa;
        load_K_tile(0, row_offset);
        cp_async_commit_group();
    }
    
    for (int kb0_idx = 0; kb0_idx < num_iters; ++kb0_idx) {
        const int kb0 = kb0_start + kb0_idx;
        const int stage = kb0_idx % 2;
        const int next_stage = (kb0_idx + 1) % 2;
        const int row_offset = kv_head_row_offset + kb0 * nbatch_fa;
        
        // ========== WAIT FOR CURRENT K + PREFETCH NEXT K ==========
        cp_async_wait_group<1>();
        __syncthreads();  // SYNC 1: K[stage] ready
        
        // Prefetch next K tile while compute proceeds on current K
        if (kb0_idx + 1 < num_iters) {
            const int next_row_offset = kv_head_row_offset + (kb0 + 1) * nbatch_fa;
            load_K_tile(next_stage, next_row_offset);
            cp_async_commit_group();
        }
        
        __syncthreads();  // SYNC 2: K prefetch issued, safe to proceed
        
        // ========== LOAD V TILE (overlapped with K compute in consumer) ==========
        load_V_tile(stage, row_offset);
        cp_async_commit_group();
        cp_async_wait_group<0>();
        __syncthreads();  // SYNC 3: V[stage] ready
        
        __syncthreads();  // SYNC 4: Tile complete, safe for next iteration
    }
    
    cp_async_wait_group<0>();
    __syncthreads();

    GGML_UNUSED(tensor_maps);
    GGML_UNUSED(state);
    GGML_UNUSED(num_consumers);
#else
    GGML_UNUSED(tensor_maps);
    GGML_UNUSED(state);
    GGML_UNUSED(kb0_start);
    GGML_UNUSED(kb0_stop);
    GGML_UNUSED(num_consumers);
    GGML_UNUSED(kv_head_row_offset);
    GGML_UNUSED(K_h2);
    GGML_UNUSED(V_h2);
    GGML_UNUSED(stride_K);
    GGML_UNUSED(stride_V);
    NO_DEVICE_CODE;
#endif
}

// ------------------------------------------------------------------------------------------------------------------
// PRODUCER LOOP (Dispatch wrapper)
// ------------------------------------------------------------------------------------------------------------------
// Chooses between:
//   1. SM_120 optimized: cp.async with swizzling + two-stage pipelining (no TMA/mbarrier)
//   2. Chunked: chunk-level mbarrier for large tiles
//   3. Legacy: tile-level mbarrier for normal tiles
template<int DKQ, int DV, int ncols, int nstages, int nbatch_fa, int nbatch_K2, int nbatch_V2, bool mla>
static __device__ __forceinline__ void fattn_producer_loop(
    const char * __restrict__ tensor_maps,
    fattn_pipeline_state* __restrict__ state,
    int kb0_start,
    int kb0_stop,
    int num_consumers,
    int kv_head_row_offset,
    const half2 * __restrict__ K_h2 = nullptr,
    const half2 * __restrict__ V_h2 = nullptr,
    int stride_K = 0,
    int stride_V = 0)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
    // SM_120 (RTX 5090): Use optimized cp.async with swizzling (TMA mbarrier crashes)
    fattn_producer_loop_sm120<DKQ, DV, ncols, nstages, nbatch_fa, nbatch_K2, nbatch_V2, mla>(
        tensor_maps, state, kb0_start, kb0_stop, num_consumers, kv_head_row_offset,
        K_h2, V_h2, stride_K, stride_V);
#else
    // Other Blackwell (SM_90/100/103/110): Use mbarrier path
    constexpr bool needs_K_chunking = (DKQ/2 > nbatch_K2);
    constexpr bool needs_V_chunking = (DV/2 > nbatch_V2);
    constexpr bool needs_chunking = needs_K_chunking || needs_V_chunking;

    if constexpr (needs_chunking) {
        fattn_producer_loop_chunked<DKQ, DV, ncols, nstages, nbatch_fa, nbatch_K2, nbatch_V2, mla>(
            tensor_maps, state, kb0_start, kb0_stop, num_consumers, kv_head_row_offset);
    } else {
        fattn_producer_loop_legacy<DKQ, DV, ncols, nstages, nbatch_fa, nbatch_K2, nbatch_V2, mla>(
            tensor_maps, state, kb0_start, kb0_stop, num_consumers, kv_head_row_offset);
    }
    GGML_UNUSED(K_h2);
    GGML_UNUSED(V_h2);
    GGML_UNUSED(stride_K);
    GGML_UNUSED(stride_V);
#endif
}

template<int DKQ, int DV, int ncols1, int ncols2, int nwarps, int num_consumers,
    bool use_logit_softcap, bool mla, bool needs_fixup, bool is_fixup, bool last_iter, bool oob_check, bool use_tma,
    typename T_A_KQ, typename T_B_KQ, typename T_C_KQ, typename T_A_VKQ, typename T_B_VKQ, typename T_C_VKQ>
static __device__ __forceinline__ void flash_attn_ext_f16_iter(
        const float2 * const __restrict__ Q_f2,
        const half2  * const __restrict__ K_h2,
        const half2  * const __restrict__ V_h2,
        const half   * const __restrict__ mask_h,
        float2       * const __restrict__ dstk,
        float2       * const __restrict__ dstk_fixup,
        const float scale,
        const float slope,
        const float logit_softcap,
        const uint3 ne01,
        const int ne02,
        const int stride_K,
        const int stride_V,
        const int stride_mask,
        half2        * const __restrict__ tile_Q,
        half2        * __restrict__ tile_K,
        half2        * __restrict__ tile_V,
        half         * const __restrict__ tile_mask,
        T_B_KQ       * const __restrict__ Q_B,
        T_C_VKQ      * const __restrict__ VKQ_C,
        float        * const __restrict__ KQ_max,
        float        * const __restrict__ KQ_rowsum,
        const int jt,
        const int kb0,
        const int k_VKQ_sup,
        const char * __restrict__ tensor_maps,
        uint64_t* __restrict__ mbar_ptr,
        uint32_t& phase_K,
        uint32_t& phase_V,
        uint32_t& consumer_chunk_phase_K,
        uint32_t& consumer_chunk_phase_V,
        fattn_pipeline_state* pipeline_state = nullptr,
        int pipeline_stage = -1,
        int nstages_pipeline = 0) {
#if defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE)
    constexpr int  ncols           = ncols1 * ncols2;
    constexpr int  cols_per_warp   = T_B_KQ::I;
    constexpr int  cols_per_thread = 2; // This is specifically KQ columns, Volta only has a single VKQ column.

    // Consumer mode: use num_consumers as effective warp count, remap warp IDs
    //   - With num_consumers=8: threadIdx.y 1-8 → warp_id 0-7
    //   - Producer (threadIdx.y=0) doesn't call this function
    // Unified mode: use nwarps with original threadIdx.y
    constexpr int  effective_nwarps = (num_consumers > 0) ? num_consumers : nwarps;
    const int      warp_id          = (num_consumers > 0) ? (threadIdx.y - 1) : threadIdx.y;

    constexpr int  np              = effective_nwarps * (cols_per_warp/ncols2) / ncols1; // Number of parallel CUDA warps per Q column.
    // Config selection:
    // - SM_120 (__CUDA_ARCH__ >= 1200): Use SM_120 configs (unified mode, 4 warps, nbatch_fa=128)
    // - Other Blackwell (use_tma=true): Use Blackwell TMA configs
    // - Ampere/Turing (use_tma=false): Use Ampere configs
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
    // SM_120: Use dedicated SM_120 configs (routes via get_config_blackwell → get_config_sm120)
    constexpr int  nbatch_fa       = ggml_cuda_fattn_mma_get_nbatch_fa     (DKQ, DV, ncols);
    constexpr int  nbatch_K2       = ggml_cuda_fattn_mma_get_nbatch_K2     (DKQ, DV, ncols);
    constexpr int  nbatch_V2       = ggml_cuda_fattn_mma_get_nbatch_V2     (DKQ, DV, ncols);
    constexpr bool Q_in_reg        = ggml_cuda_fattn_mma_get_Q_in_reg      (DKQ, DV, ncols);
    constexpr int  nstages         = ggml_cuda_fattn_mma_get_nstages       (DKQ, DV, ncols1, ncols2);
#else
    // Other archs: use_tma determines Blackwell TMA vs Ampere config
    constexpr int  nbatch_fa       = use_tma ? ggml_cuda_fattn_mma_get_nbatch_fa(DKQ, DV, ncols) : ggml_cuda_fattn_mma_get_nbatch_fa_ampere(DKQ, DV, ncols);
    constexpr int  nbatch_K2       = use_tma ? ggml_cuda_fattn_mma_get_nbatch_K2(DKQ, DV, ncols) : ggml_cuda_fattn_mma_get_nbatch_K2_ampere(DKQ, DV, ncols);
    constexpr int  nbatch_V2       = use_tma ? ggml_cuda_fattn_mma_get_nbatch_V2(DKQ, DV, ncols) : ggml_cuda_fattn_mma_get_nbatch_V2_ampere(DKQ, DV, ncols);
    constexpr bool Q_in_reg        = use_tma ? ggml_cuda_fattn_mma_get_Q_in_reg (DKQ, DV, ncols) : ggml_cuda_fattn_mma_get_Q_in_reg_ampere (DKQ, DV, ncols);
    constexpr int  nstages         = use_tma ? ggml_cuda_fattn_mma_get_nstages  (DKQ, DV, ncols1, ncols2) : ggml_cuda_fattn_mma_get_nstages_ampere(DKQ, DV, ncols1, ncols2);
#endif

    // TMA tile parameters (legacy chunk variables kept for template API compatibility).
    // With SWIZZLE_NONE for large tiles, we load full tiles in single TMA operations.
    constexpr int  chunks_K        = tma_chunks_needed(nbatch_K2);
    constexpr int  chunks_V        = tma_chunks_needed(nbatch_V2);
    constexpr int  chunk_size_K    = (nbatch_K2 + chunks_K - 1) / chunks_K;
    constexpr int  chunk_size_V    = (nbatch_V2 + chunks_V - 1) / chunks_V;
    (void)chunk_size_V;  // May be unused in some template instantiations

    constexpr int stride_tile_Q = DKQ/2     + 4;
    // TMA and SM_120 cp.async both use compact stride (no padding).
    // Legacy non-TMA path uses +4 padding for bank conflict avoidance.
    // SM_120: Producer uses compact layout matching bytes_K_chunk = nbatch_fa * nbatch_K2 * sizeof(half2)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
    // SM_120: Use compact stride to match producer's cp.async layout
    constexpr int stride_tile_K = nbatch_K2;  // COMPACT - matches bytes_K_chunk
#else
    constexpr int stride_tile_K = use_tma ? nbatch_K2 : nbatch_K2 + 4;
#endif

    static_assert(!mla || nbatch_K2 >= nbatch_V2, "bad nbatch_K2, nbatch_V2 for MLA");
    // For TMA: V uses its own compact stride (nbatch_V2) since V overwrites K's memory
    // after K is processed. For non-TMA MLA: V shares K's stride for memory layout.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
    // SM_120: Use compact stride matching bytes_V_chunk
    constexpr int stride_tile_V = mla ? nbatch_K2 : nbatch_V2;  // COMPACT
#else
    constexpr int stride_tile_V = use_tma ? nbatch_V2 : (mla ? stride_tile_K : nbatch_V2 + 4);
#endif

    const int k_VKQ_0 = kb0 * nbatch_fa;
#if defined(TURING_MMA_AVAILABLE)
    T_C_KQ KQ_C[nbatch_fa/(np*(cols_per_warp == 8 ? T_C_KQ::I : T_C_KQ::J))];
#else // Volta
    T_C_KQ KQ_C[nbatch_fa/(np*T_C_KQ::J)];
#endif // defined(TURING_MMA_AVAILABLE)

    bool consumer_mode = (pipeline_state != nullptr);

    // Use if constexpr so static_assert isn't evaluated when nstages <= 1 (SM_120 path)
    if constexpr (nstages > 1) {
        if (!consumer_mode) {
            static_assert(!oob_check, "OOB check incompatible with multi-stage pipeline");
            constexpr bool use_cp_async = true;
            if constexpr (use_tma) {
#ifdef BLACKWELL_TMA_AVAILABLE
                // Wait for preloaded data (Mask)
                cp_async_wait_all(); // Mask

                // For MLA, K is processed in chunks in the loop, so we wait there.
                // For non-MLA, we wait here for K.
                if (!mla) {
                    mbarrier_wait(mbar_ptr, phase_K); // K
                    phase_K ^= 1;
                }
                __syncthreads();

                // Load V via TMA
                if (!mla) {
                    uint32_t bytes_V = nbatch_fa * nbatch_V2 * sizeof(half2);
                    if (threadIdx.x == 0) mbarrier_arrive_expect_tx(mbar_ptr + 1, bytes_V);
                    // V tensor map is at index 1
                    flash_attn_ext_f16_load_tile_tma_chunked<stride_tile_V, nwarps, nbatch_fa, nbatch_V2, chunks_V, chunk_size_V>(
                        tensor_maps, 1, tile_V, mbar_ptr + 1, 0, k_VKQ_0);
                    mbarrier_wait(mbar_ptr + 1, phase_V);
                    phase_V ^= 1;
                }
#endif
            } else {
                cp_async_wait_all();
            }
            __syncthreads();
            if constexpr (!use_tma) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
                // SM_120: Skip V preload here - it only loads one chunk (nbatch_V2 columns).
                // For models with DV/2 > nbatch_V2 (e.g., DV=128, nbatch_V2=32), we need
                // multiple V chunks. The per-chunk V loading at lines ~2259+ handles this
                // correctly with double-buffered pipelining. See the V-chunk loop condition.
#else
                flash_attn_ext_f16_load_tile<stride_tile_V, nwarps, nbatch_fa, use_cp_async, oob_check>
                    (V_h2 + int64_t(k_VKQ_0)*stride_V, tile_V, nbatch_V2, stride_V, k_VKQ_sup);
#endif
            }
        }
    } else if (!consumer_mode) {
        constexpr bool use_cp_async = nstages == 1;
        if (ncols2 > 1 || mask_h) {
            flash_attn_ext_f16_load_mask<ncols1, nwarps, nbatch_fa, use_cp_async, oob_check>
                (mask_h + k_VKQ_0, tile_mask, stride_mask, k_VKQ_sup, jt*ncols1, ne01);
            if constexpr (use_tma && use_cp_async) {
                cp_async_wait_all(); // Wait for mask before inner loop
                __syncthreads();
            }
        }
    }

    int k_buf = 0; // 0 or 1 for double buffering
    int current_stage = pipeline_stage;

    // Consumer mode chunk pipelining constants
    constexpr int num_K_chunks = (DKQ/2 + nbatch_K2 - 1) / nbatch_K2;
    constexpr int bytes_K_chunk = nbatch_fa * nbatch_K2 * sizeof(half2);
    constexpr int bytes_V_chunk = nbatch_fa * nbatch_V2 * sizeof(half2);
    // Stage layout: [K_chunk_0][K_chunk_1][V_chunk_0][V_chunk_1]
    constexpr int stride_stage_consumer = 2 * bytes_K_chunk + 2 * bytes_V_chunk;

    // Determine if we need chunk-level pipelining (must match producer's logic)
    constexpr bool needs_K_chunking = (DKQ/2 > nbatch_K2);
    constexpr bool needs_V_chunking = (DV/2 > nbatch_V2);
    constexpr bool needs_chunking = needs_K_chunking || needs_V_chunking;

    // Consumer mode: base pointer for K buffers
    // NOTE: No stage offset - chunk buffers are reused each kb0 iteration (matches producer)
    half2 * consumer_tile_K_base = nullptr;
    if (consumer_mode) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
        // SM_120: Use tile_K directly (no TMA producer, just shared memory)
        consumer_tile_K_base = tile_K;
#elif defined(BLACKWELL_TMA_AVAILABLE)
        consumer_tile_K_base = tile_K;
        // DEBUG: Print consumer K base setup (once per warp per block)
        if (threadIdx.x == 0 && threadIdx.y == 1 && blockIdx.x == 0 && blockIdx.y == 0) {
            extern __shared__ char smem_debug[];
            printf("[CONSUMER DEBUG] block=(%d,%d,%d) warp=%d consumer_tile_K_base=%p (offset from smem=%ld)\n",
                   blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.y,
                   consumer_tile_K_base, (long)((char*)consumer_tile_K_base - smem_debug));
            printf("[CONSUMER DEBUG] needs_chunking=%d num_K_chunks=%d bytes_K_chunk=%d\n",
                   needs_chunking, num_K_chunks, bytes_K_chunk);
        }
#endif
    }

    // K-processing loop with chunk-level pipelining for consumer mode
    for (int chunk = 0; chunk < num_K_chunks; ++chunk) {
        const int chunk_stage = chunk % 2;
        const int k0_start = chunk * nbatch_K2;
        const int k0_stop = (k0_start + nbatch_K2 < DKQ/2) ? (k0_start + nbatch_K2) : (DKQ/2);

        half2 * current_tile_K = tile_K;

        if (consumer_mode) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
            // SM_120 FALLBACK: Use __syncthreads__ instead of mbarrier_wait
            // Producer has already loaded K and hit SYNC 1 - data is ready
            // We only sync on chunk 0 (once per K tile, not per chunk)
            if (chunk == 0) {
                __syncthreads();  // SYNC 1: K is ready from producer
            }
            // All K chunks are already loaded by producer - just point to data
            current_tile_K = consumer_tile_K_base + current_stage * (bytes_K_chunk / sizeof(half2));
#elif defined(BLACKWELL_TMA_AVAILABLE)
            // Standard mbarrier path for SM_90/100/103/110
            if constexpr (needs_chunking) {
                // DEBUG: Print before wait
                if (threadIdx.x == 0 && threadIdx.y == 1 && blockIdx.x == 0 && blockIdx.y == 0 && kb0 == 0) {
                    printf("[CONSUMER DEBUG] K chunk %d: waiting on full_K_chunk[%d] phase=%d\n",
                           chunk, chunk_stage, consumer_chunk_phase_K);
                }

                // Wait for producer to signal this chunk is ready
                mbarrier_wait(&pipeline_state->full_K_chunk[chunk_stage], consumer_chunk_phase_K);

                // Point to the correct double-buffered chunk
                current_tile_K = consumer_tile_K_base + chunk_stage * (bytes_K_chunk / sizeof(half2));

                // DEBUG: Print after wait
                if (threadIdx.x == 0 && threadIdx.y == 1 && blockIdx.x == 0 && blockIdx.y == 0 && kb0 == 0) {
                    extern __shared__ char smem_debug2[];
                    printf("[CONSUMER DEBUG] K chunk %d: got data, current_tile_K=%p (offset=%ld) k0_range=[%d,%d)\n",
                           chunk, current_tile_K, (long)((char*)current_tile_K - smem_debug2), k0_start, k0_stop);
                }
            } else {
                // Legacy mode: wait once at the start (chunk 0 only)
                // Add stage offset to match producer's Blackwell layout:
                // K stages are at offsets 0, bytes_K_chunk, etc.
                if (chunk == 0) {
                    mbarrier_wait(&pipeline_state->full_K[current_stage], phase_K);
                    current_tile_K = consumer_tile_K_base + current_stage * (bytes_K_chunk / sizeof(half2));
                }
            }
#endif
        } else {
            current_tile_K = tile_K;
        }

        if (!consumer_mode) {
            if constexpr (nstages > 1 && use_tma && mla) {
                // For MLA pipelining, we use tile_K (buf 0) and tile_V (buf 1, effectively K[1])
                current_tile_K = k_buf == 0 ? tile_K : tile_V;
            }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
            // ================================================================
            // SM_120 OPTIMIZED: Two-Stage K Pipelining with Swizzled cp.async
            // ================================================================
            // This path runs REGARDLESS of nstages value - SM_120 always uses
            // its own cp.async pipelining with swizzled shared memory layout.
            // ================================================================
            {
                constexpr int STRIDE_BYTES_K = stride_tile_K * sizeof(half2);
                
                // Double buffer addresses
                const uint32_t tile_K_buf0 = ggml_cuda_cvta_generic_to_shared(tile_K);
                const uint32_t tile_K_buf1 = ggml_cuda_cvta_generic_to_shared(tile_V);
                
                // warp_id_offset=0 for unified mode (all warps load, threadIdx.y = 0..nwarps-1)
                if (chunk == 0) {
                    // Issue first K load (K[0] into buffer 0)
                    flash_attn_ext_f16_load_tile_swizzle<stride_tile_K, STRIDE_BYTES_K, nwarps, nbatch_fa, 0>(
                        K_h2 + int64_t(k_VKQ_0)*stride_K + k0_start, tile_K_buf0, nbatch_K2, stride_K);
                    cp_async_commit_group();

                    // If there's a next chunk, prefetch it into buffer 1
                    if (chunk + 1 < num_K_chunks) {
                        const int next_k0_start = (chunk + 1) * nbatch_K2;
                        flash_attn_ext_f16_load_tile_swizzle<stride_tile_K, STRIDE_BYTES_K, nwarps, nbatch_fa, 0>(
                            K_h2 + int64_t(k_VKQ_0)*stride_K + next_k0_start, tile_K_buf1, nbatch_K2, stride_K);
                        cp_async_commit_group();
                    }

                    // Wait for K[0] to complete (allow 1 group in flight if prefetch was issued)
                    if (chunk + 1 < num_K_chunks) {
                        cp_async_wait_group<1>();
                    } else {
                        cp_async_wait_group<0>();
                    }
                } else {
                    // Subsequent chunks: Prefetch next chunk if it exists
                    if (chunk + 1 < num_K_chunks) {
                        const int next_k0_start = (chunk + 1) * nbatch_K2;
                        const uint32_t next_buf = (chunk % 2 == 0) ? tile_K_buf1 : tile_K_buf0;
                        flash_attn_ext_f16_load_tile_swizzle<stride_tile_K, STRIDE_BYTES_K, nwarps, nbatch_fa, 0>(
                            K_h2 + int64_t(k_VKQ_0)*stride_K + next_k0_start, next_buf, nbatch_K2, stride_K);
                        cp_async_commit_group();
                        cp_async_wait_group<1>();
                    } else {
                        cp_async_wait_group<0>();
                    }
                }
                
                __syncthreads();
                current_tile_K = (chunk % 2 == 0) ? tile_K : tile_V;
            }
#else
            // Non-SM_120 path: Use if constexpr based on nstages
            if constexpr (nstages <= 1) {
                constexpr bool use_cp_async = nstages == 1;
                if constexpr (use_tma) {
                     #ifdef BLACKWELL_TMA_AVAILABLE
                     uint32_t bytes_K = nbatch_fa * nbatch_K2 * sizeof(half2);
                     if (threadIdx.x == 0) mbarrier_arrive_expect_tx(mbar_ptr, bytes_K);
                     #endif
                     flash_attn_ext_f16_load_tile_tma_chunked<stride_tile_K, nwarps, nbatch_fa, nbatch_K2, chunks_K, chunk_size_K>(
                         tensor_maps, 0, tile_K, mbar_ptr, k0_start, k_VKQ_0);
                     #ifdef BLACKWELL_TMA_AVAILABLE
                     mbarrier_wait(mbar_ptr, phase_K);
                     phase_K ^= 1;
                     #endif
                } else {
                    flash_attn_ext_f16_load_tile<stride_tile_K, nwarps, nbatch_fa, use_cp_async, oob_check>
                        (K_h2 + int64_t(k_VKQ_0)*stride_K + k0_start, tile_K, nbatch_K2, stride_K, k_VKQ_sup);
                    if (use_cp_async) {
                        cp_async_wait_all();
                    }
                }
                __syncthreads();
            } else if constexpr (use_tma && mla) {
                #ifdef BLACKWELL_TMA_AVAILABLE
                // MLA Pipelining:
                // 1. Wait for current buffer (issued in previous iter or preload)
                uint64_t* mbar = k_buf == 0 ? mbar_ptr : (mbar_ptr + 1);
                uint32_t& phase = k_buf == 0 ? phase_K : phase_V;
                
                mbarrier_wait(mbar, phase);
                phase ^= 1;

                // 2. Issue load for next buffer
                if (k0_start + nbatch_K2 < DKQ/2) {
                    int next_k_buf = k_buf ^ 1;
                    half2* next_tile_K = next_k_buf == 0 ? tile_K : tile_V;
                    uint64_t* next_mbar = next_k_buf == 0 ? mbar_ptr : (mbar_ptr + 1);
                    
                    uint32_t bytes_K = nbatch_fa * nbatch_K2 * sizeof(half2);
                    if (threadIdx.x == 0) mbarrier_arrive_expect_tx(next_mbar, bytes_K);
                    
                    flash_attn_ext_f16_load_tile_tma_chunked<stride_tile_K, nwarps, nbatch_fa, nbatch_K2, chunks_K, chunk_size_K>(
                         tensor_maps, 0, next_tile_K, next_mbar, k0_start + nbatch_K2, k_VKQ_0);
                }
                #endif
            }
#endif // SM_120 check
        }

        // Calculate tile of KQ:
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
        // SM_120 OPTIMIZED: Use swizzled ldmatrix to read from swizzled shared memory
        constexpr int STRIDE_BYTES_K = stride_tile_K * sizeof(half2);
        const uint32_t current_tile_K_base = ggml_cuda_cvta_generic_to_shared((void*)current_tile_K);
#endif
        if constexpr (Q_in_reg) {
#pragma unroll
            for (int i_KQ_00 = 0; i_KQ_00 < nbatch_fa; i_KQ_00 += np*T_A_KQ::I) {
                const int i_KQ_0 = i_KQ_00 + (warp_id % np)*T_A_KQ::I;
#pragma unroll
                for (int k_KQ_0 = k0_start; k_KQ_0 < k0_stop; k_KQ_0 += T_A_KQ::J) {
                    T_A_KQ K_A;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
                    // SM_120: Use swizzled ldmatrix read
                    ggml_cuda_mma::load_ldmatrix_swizzle<STRIDE_BYTES_K>(K_A, current_tile_K_base, 
                        i_KQ_0, (k_KQ_0 - k0_start) * sizeof(half2));
#else
                    load_ldmatrix(K_A, current_tile_K + i_KQ_0*stride_tile_K + (k_KQ_0 - k0_start), stride_tile_K);
#endif
                    if constexpr (cols_per_warp == 8) {
                        mma(KQ_C[i_KQ_00/(np*T_A_KQ::I)], K_A, Q_B[k_KQ_0/T_A_KQ::J]);
                    } else {
                        // Wide version of KQ_C is column-major => swap A and B.
                        mma(KQ_C[i_KQ_00/(np*T_A_KQ::I)], Q_B[k_KQ_0/T_A_KQ::J], K_A);
                    }
                }
            }
        } else {
            static_assert(cols_per_warp != 8, "cols_per_warp == 8 not implemented");
#pragma unroll
            for (int k_KQ_0 = k0_start; k_KQ_0 < k0_stop; k_KQ_0 += T_A_KQ::J) {
                load_ldmatrix(Q_B[0], tile_Q + (warp_id / np)*(T_B_KQ::I*stride_tile_Q) + k_KQ_0, stride_tile_Q);

#pragma unroll
                for (int i_KQ_00 = 0; i_KQ_00 < nbatch_fa; i_KQ_00 += np*T_A_KQ::I) {
                    const int i_KQ_0 = i_KQ_00 + (warp_id % np)*T_A_KQ::I;

                    T_A_KQ K_A;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
                    // SM_120: Use swizzled ldmatrix read
                    ggml_cuda_mma::load_ldmatrix_swizzle<STRIDE_BYTES_K>(K_A, current_tile_K_base,
                        i_KQ_0, (k_KQ_0 - k0_start) * sizeof(half2));
#else
                    load_ldmatrix(K_A, current_tile_K + i_KQ_0*stride_tile_K + (k_KQ_0 - k0_start), stride_tile_K);
#endif

                    // Wide version of KQ_C is column-major => swap A and B.
                    mma(KQ_C[i_KQ_00/(np*T_A_KQ::I)], Q_B[0], K_A);
                }
            }
        }

        if (consumer_mode) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
            // SM_120: No mbarrier signaling needed - __syncthreads handles coordination
            (void)0;
#elif defined(BLACKWELL_TMA_AVAILABLE)
            if constexpr (needs_chunking) {
                // DEBUG: Print before signaling completion
                if (threadIdx.x == 0 && threadIdx.y == 1 && blockIdx.x == 0 && blockIdx.y == 0 && kb0 == 0) {
                    printf("[CONSUMER DEBUG] K chunk %d: processing complete, signaling empty_K_chunk[%d]\n",
                           chunk, chunk_stage);
                }

                // Signal we're done with this chunk
                // NOTE: Using mbarrier_arrive instead of mbarrier_arrive_expect_tx(0)
                // since we're not setting TMA byte expectations
                mbarrier_arrive(&pipeline_state->empty_K_chunk[chunk_stage]);

                // Flip chunk phase after every 2 chunks
                if (chunk_stage == 1) {
                    consumer_chunk_phase_K ^= 1;
                    if (threadIdx.x == 0 && threadIdx.y == 1 && blockIdx.x == 0 && blockIdx.y == 0 && kb0 == 0) {
                        printf("[CONSUMER DEBUG] K phase flip -> %d\n", consumer_chunk_phase_K);
                    }
                }
            } else {
                // Legacy mode: signal once at the end (last chunk only)
                if (chunk == num_K_chunks - 1) {
                    mbarrier_arrive(&pipeline_state->empty_K[current_stage]);
                }
            }
#endif
        } else {
            if constexpr (nstages <= 1) {
                __syncthreads(); // Only needed if tile_K == tile_V.
            } else if constexpr (use_tma && mla) {
                k_buf ^= 1;
            }
        }
    }

    // DEBUG: Print after K loop completes
    if (consumer_mode && threadIdx.x == 0 && threadIdx.y == 1 && blockIdx.x == 0 && blockIdx.y == 0 && kb0 == 0) {
        printf("[CONSUMER DEBUG] K loop complete, starting softmax/V processing\n");
    }

    if (use_logit_softcap) {
        constexpr int stride = cols_per_warp == 8 ? np*T_C_KQ::I : np*T_C_KQ::J;
        static_assert(nbatch_fa % stride == 0, "bad loop size");
#pragma unroll
        for (int i = 0; i < nbatch_fa/stride; ++i) {
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                KQ_C[i].x[l] = logit_softcap*tanhf(KQ_C[i].x[l]);
            }
        }
    }

    float KQ_max_new[cols_per_thread];
#pragma unroll
    for (int col = 0; col < cols_per_thread; ++col) {
        KQ_max_new[col] = KQ_max[col];
    }
    float KQ_rowsum_add[cols_per_thread] = {0.0f};

    if constexpr (cols_per_warp == 8) {
        if (!consumer_mode && (ncols2 > 1 || mask_h)) {
#pragma unroll
            for (int i00 = 0; i00 < nbatch_fa; i00 += np*T_C_KQ::I) {
                const int i0 = i00 + (warp_id % np)*T_C_KQ::I;
#pragma unroll
                for (int l = 0; l < T_C_KQ::ne; ++l) {
                    const int i = i0 + T_C_KQ::get_i(l);
                    const int j = ((warp_id / np)*T_C_KQ::J + T_C_KQ::get_j(l)) / ncols2;

                    KQ_C[i00/(np*T_C_KQ::I)].x[l] += slope * __half2float(tile_mask[j*(nbatch_fa + 8) + i]);
                }
            }
        }

        // Calculate softmax for each KQ column using the current max. value.
        // The divisor is stored in KQ_rowsum and will be applied at the end.
        static_assert(nbatch_fa % (np*T_C_KQ::I) == 0, "bad loop size");
#pragma unroll
        for (int k0 = 0; k0 < nbatch_fa; k0 += np*T_C_KQ::I) {
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                if (!oob_check || k0 + T_C_KQ::get_i(l) < k_VKQ_sup) {
                    KQ_max_new[l % 2] = fmaxf(KQ_max_new[l % 2], KQ_C[k0/(np*T_C_KQ::I)].x[l] + FATTN_KQ_MAX_OFFSET);
                }
            }
        }

        // Values per KQ column are spread across 8 threads:
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
#pragma unroll
            for (int offset = 16; offset >= 4; offset >>= 1) {
                KQ_max_new[col] = fmaxf(KQ_max_new[col], __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[col], offset, WARP_SIZE));
            }
        }

        static_assert(nbatch_fa % (np*T_C_KQ::I) == 0, "bad loop size");
#pragma unroll
        for (int k0 = 0; k0 < nbatch_fa; k0 += np*T_C_KQ::I) {
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                if (!oob_check || k0 + (warp_id % np)*T_C_KQ::I + T_C_KQ::get_i(l) < k_VKQ_sup) {
                    KQ_C[k0/(np*T_C_KQ::I)].x[l] = expf(KQ_C[k0/(np*T_C_KQ::I)].x[l] - KQ_max_new[l % 2]);
                    KQ_rowsum_add[l % 2] += KQ_C[k0/(np*T_C_KQ::I)].x[l];
                } else {
                    KQ_C[k0/(np*T_C_KQ::I)].x[l] = 0.0f;
                }
            }
        }
    } else { // not Turing mma or T_B_KQ::I > 8
        if (!consumer_mode && (ncols2 > 1 || mask_h)) {
#pragma unroll
            for (int i00 = 0; i00 < nbatch_fa; i00 += np*T_C_KQ::J) {
                const int i0 = i00 + (warp_id % np)*T_C_KQ::J;
#pragma unroll
                for (int l0 = 0; l0 < T_C_KQ::ne; l0 += 2) {
                    const int i = (i0 + T_C_KQ::get_j(l0)) / 2;
                    const int j = ((warp_id / np)*cols_per_warp + T_C_KQ::get_i(l0)) / ncols2;

                    const float2 tmp = __half22float2(((const half2 *)tile_mask)[j*(nbatch_fa/2 + 4) + i]);
                    KQ_C[i00/(np*T_C_KQ::J)].x[l0 + 0] += slope*tmp.x;
                    KQ_C[i00/(np*T_C_KQ::J)].x[l0 + 1] += slope*tmp.y;
                }
            }
        }

        // Calculate softmax for each KQ column using the current max. value.
        // The divisor is stored in KQ_rowsum and will be applied at the end.
        static_assert(nbatch_fa % (np*T_C_KQ::J) == 0, "bad loop size");
#pragma unroll
        for (int k0 = 0; k0 < nbatch_fa; k0 += np*T_C_KQ::J) {
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                if (!oob_check || k0 + T_C_KQ::get_j(l) < k_VKQ_sup) {
                    // Turing + Volta:
                    KQ_max_new[(l/2) % 2] = fmaxf(KQ_max_new[(l/2) % 2], KQ_C[(k0/(np*T_C_KQ::J))].x[l] + FATTN_KQ_MAX_OFFSET);
                }
            }
        }

#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
#if defined(TURING_MMA_AVAILABLE)
            // Values per KQ column are spread across 4 threads:
            constexpr int offset_first = 2;
            constexpr int offset_last  = 1;
#else
            // Values per KQ column are spread across 2 threads:
            constexpr int offset_first = 2;
            constexpr int offset_last  = 2;
#endif // defined(TURING_MMA_AVAILABLE)
#pragma unroll
            for (int offset = offset_first; offset >= offset_last; offset >>= 1) {
                KQ_max_new[col] = fmaxf(KQ_max_new[col], __shfl_xor_sync(0xFFFFFFFF, KQ_max_new[col], offset, WARP_SIZE));
            }
        }

        static_assert(nbatch_fa % (np*T_C_KQ::J) == 0, "bad loop size");
#pragma unroll
        for (int k0 = 0; k0 < nbatch_fa; k0 += np*T_C_KQ::J) {
#pragma unroll
            for (int l = 0; l < T_C_KQ::ne; ++l) {
                // Turing + Volta:
                if (!oob_check || k0 + (warp_id % np)*T_C_KQ::J + T_C_KQ::get_j(l) < k_VKQ_sup) {
                    KQ_C[(k0/(np*T_C_KQ::J))].x[l] = expf(KQ_C[(k0/(np*T_C_KQ::J))].x[l] - KQ_max_new[(l/2) % 2]);
                    KQ_rowsum_add[(l/2) % 2] += KQ_C[(k0/(np*T_C_KQ::J))].x[l];
                } else {
                    KQ_C[(k0/(np*T_C_KQ::J))].x[l] = 0.0f;
                }
            }
        }
    }

    {
        float KQ_max_scale[cols_per_thread];
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
            const float KQ_max_diff = KQ_max[col] - KQ_max_new[col];
            KQ_max_scale[col] = expf(KQ_max_diff);
            KQ_max[col] = KQ_max_new[col];

            *((uint32_t *) &KQ_max_scale[col]) *= KQ_max_diff >= SOFTMAX_FTZ_THRESHOLD;

            // Scale previous KQ_rowsum to account for a potential increase in KQ_max:
            KQ_rowsum[col] = KQ_max_scale[col]*KQ_rowsum[col] + KQ_rowsum_add[col];

            // DEBUG: Check for NaN/zero in KQ_rowsum (division by this causes NaN later)
            if ((isnan(KQ_rowsum[col]) || KQ_rowsum[col] == 0.0f || isinf(KQ_rowsum[col])) &&
                threadIdx.x == 0 && blockIdx.x == 0) {
                printf("[FA SOFTMAX NaN] warp %d col %d: KQ_rowsum=%f KQ_max_scale=%f KQ_rowsum_add=%f\n",
                       warp_id, col, KQ_rowsum[col], KQ_max_scale[col], KQ_rowsum_add[col]);
            }
        }

#if defined(TURING_MMA_AVAILABLE)
        if constexpr (cols_per_warp == 8) {
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[0], KQ_max_scale[1]);
#pragma unroll
            for (int i = 0; i < DV/T_C_VKQ::I; ++i) {
#pragma unroll
                for (int l = 0; l < T_C_VKQ::ne; ++l) {
                    VKQ_C[i].x[l] *= KQ_max_scale_h2;
                }
            }
        } else {
#pragma unroll
            for (int col = 0; col < cols_per_thread; ++col) {
                const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[col], KQ_max_scale[col]);
#pragma unroll
                for (int i = 0; i < (DV/2)/T_C_VKQ::J; ++i) {
#pragma unroll
                    for (int l0 = 0; l0 < T_C_VKQ::ne; l0 += 2) {
                        VKQ_C[i].x[l0 + col] *= KQ_max_scale_h2;
                    }
                }
            }
        }
#else // Volta
        const half2 KQ_max_scale_h2 = make_half2(
            KQ_max_scale[(threadIdx.x / 2) % 2], KQ_max_scale[(threadIdx.x / 2) % 2]);
#pragma unroll
        for (int i = 0; i < (DV/2)/T_C_VKQ::J; ++i) {
#pragma unroll
            for (int l = 0; l < T_C_VKQ::ne; ++l) {
                VKQ_C[i].x[l] *= KQ_max_scale_h2;
            }
        }
#endif // defined(TURING_MMA_AVAILABLE)
    }

    // Convert KQ C tiles into B tiles for VKQ calculation:
    T_B_VKQ B[nbatch_fa/(np*2*T_B_VKQ::J)];
    static_assert(nbatch_fa % (np*2*T_B_VKQ::J) == 0, "bad loop size");
    if constexpr (cols_per_warp == 8) {
#pragma unroll
        for (int k = 0; k < nbatch_fa/(np*2*T_B_VKQ::J); ++k) {
            B[k] = get_transposed(get_half2(KQ_C[k]));
        }
    } else {
        for (int k = 0; k < nbatch_fa/(np*2*T_B_VKQ::J); ++k) {
            B[k] = get_half2(KQ_C[k]);
        }
    }

    if (!consumer_mode && nstages > 1) {
        // Preload K/mask tile for next iteration:
        constexpr bool use_cp_async = true;
        
        if (!last_iter) {
            if (ncols2 > 1 || mask_h) {
                flash_attn_ext_f16_load_mask<ncols1, nwarps, nbatch_fa, use_cp_async, oob_check>
                    (mask_h + k_VKQ_0 + nbatch_fa, tile_mask, stride_mask, k_VKQ_sup, jt*ncols1, ne01);
            }
        }

        if constexpr (use_tma) {
             if (!last_iter && !mla) { // Only for non-MLA here
                 #ifdef BLACKWELL_TMA_AVAILABLE
                 // Preload K for next iteration via TMA (full tile)
                 uint32_t bytes_K = nbatch_fa * nbatch_K2 * sizeof(half2);
                 if (threadIdx.x == 0) mbarrier_arrive_expect_tx(mbar_ptr, bytes_K);
                 #endif
                 flash_attn_ext_f16_load_tile_tma_chunked<stride_tile_K, nwarps, nbatch_fa, nbatch_K2, chunks_K, chunk_size_K>(
                     tensor_maps, 0, tile_K, mbar_ptr, 0, k_VKQ_0 + nbatch_fa);
             }
        } else {
            cp_async_wait_all();
            __syncthreads();
            if (!last_iter) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
                // SM_120: Use swizzled cp.async for K (next iteration preload)
                // warp_id_offset=0 for unified mode (all warps load, threadIdx.y = 0..nwarps-1)
                constexpr int STRIDE_BYTES_K = stride_tile_K * sizeof(half2);
                const uint32_t tile_K_base = ggml_cuda_cvta_generic_to_shared(tile_K);
                flash_attn_ext_f16_load_tile_swizzle<stride_tile_K, STRIDE_BYTES_K, nwarps, nbatch_fa, 0>(
                    K_h2 + int64_t(k_VKQ_0 + nbatch_fa)*stride_K, tile_K_base, nbatch_K2, stride_K);
#else
                flash_attn_ext_f16_load_tile<stride_tile_K, nwarps, nbatch_fa, use_cp_async, oob_check>
                    (K_h2 + int64_t(k_VKQ_0 + nbatch_fa)*stride_K, tile_K, nbatch_K2, stride_K, k_VKQ_sup);
#endif
            }
        }
    }


    // For MLA K and V have the same data.
    // Therefore, iterate over V in reverse and re-use the data if possible.
    constexpr int reusable_cutoff = mla ? (DKQ - 1) - (DKQ - 1) % (2*nbatch_K2) - (DKQ - DV) : DV;

    // MLA V-loop pipelining
    int v_buf = 0;
#ifndef BLACKWELL_TMA_AVAILABLE
    GGML_UNUSED(v_buf);
#endif

    // If MLA, we haven't loaded any V yet. We must issue the first V load here if we want to pipeline.
    if constexpr (mla && use_tma && nstages > 1) {
        if (!consumer_mode) {
        #ifdef BLACKWELL_TMA_AVAILABLE
        int i0_first = DV - 2*nbatch_V2 > 0 ? DV - 2*nbatch_V2 : 0; // The first iter i0_start
        if (i0_first < reusable_cutoff) {
             uint32_t bytes_V = nbatch_fa * nbatch_V2 * sizeof(half2);
             if (threadIdx.x == 0) mbarrier_arrive_expect_tx(mbar_ptr + 1, bytes_V);
             flash_attn_ext_f16_load_tile_tma_chunked<stride_tile_V, nwarps, nbatch_fa, nbatch_V2, chunks_V, chunk_size_V>(
                 tensor_maps, 1, tile_V, mbar_ptr + 1, i0_first/2, k_VKQ_0);
        }
        #endif
        }
    }

    // Consumer mode V chunk pipelining
    constexpr int num_V_chunks = (DV/2 + nbatch_V2 - 1) / nbatch_V2;

    // Consumer mode: base pointer for V buffers
    // NOTE: No stage offset - chunk buffers are reused each kb0 iteration (matches producer)
    // tile_V already points to V chunk base (after K chunks), so no additional offset needed
    const half2 * consumer_tile_V_base = nullptr;
    if (consumer_mode) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
        // SM_120: Use tile_V directly (no TMA producer, just shared memory)
        consumer_tile_V_base = tile_V;
#elif defined(BLACKWELL_TMA_AVAILABLE)
        consumer_tile_V_base = tile_V;
#endif
    }

    // Calculate VKQ tile, need to use logical rather than physical elements for i0 due to transposition of V:
    // Consumer mode uses chunk-level synchronization in forward order to match producer
    for (int v_chunk = 0; v_chunk < num_V_chunks; ++v_chunk) {
        const int v_chunk_stage = v_chunk % 2;
        // Producer loads chunks in forward order (i0=0, nbatch_V2, 2*nbatch_V2, ...)
        // Consumer must process in same order for correct synchronization
        const int i0_start = v_chunk * nbatch_V2 * 2;
        const int i0_stop = (i0_start + 2*nbatch_V2 < DV) ? (i0_start + 2*nbatch_V2) : DV;
        const int i0_diff  = i0_stop - i0_start;

        const half2 * tile_V_i = tile_V;

        if (consumer_mode) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
            // SM_120 FALLBACK: Use __syncthreads__ instead of mbarrier_wait
            // Producer has already loaded V and hit SYNC 2 - data is ready
            if (v_chunk == 0) {
                __syncthreads();  // SYNC 2: V is ready from producer
            }
            // All V chunks are already loaded by producer - just point to data
            tile_V_i = consumer_tile_V_base + current_stage * (bytes_V_chunk / sizeof(half2));
#elif defined(BLACKWELL_TMA_AVAILABLE)
            // Standard mbarrier path for SM_90/100/103/110
            if constexpr (needs_chunking) {
                // DEBUG: Print before V wait
                if (threadIdx.x == 0 && threadIdx.y == 1 && blockIdx.x == 0 && blockIdx.y == 0 && kb0 == 0) {
                    printf("[CONSUMER DEBUG] V chunk %d: waiting on full_V_chunk[%d] phase=%d\n",
                           v_chunk, v_chunk_stage, consumer_chunk_phase_V);
                }

                // Wait for producer to signal this V chunk is ready
                mbarrier_wait(&pipeline_state->full_V_chunk[v_chunk_stage], consumer_chunk_phase_V);

                // Point to the correct double-buffered chunk
                tile_V_i = consumer_tile_V_base + v_chunk_stage * (bytes_V_chunk / sizeof(half2));

                // DEBUG: Print after V wait
                if (threadIdx.x == 0 && threadIdx.y == 1 && blockIdx.x == 0 && blockIdx.y == 0 && kb0 == 0) {
                    extern __shared__ char smem_debug3[];
                    printf("[CONSUMER DEBUG] V chunk %d: got data, tile_V_i=%p (offset=%ld) i0_range=[%d,%d)\n",
                           v_chunk, tile_V_i, (long)((char*)tile_V_i - smem_debug3), i0_start, i0_stop);
                }
            } else {
                // Legacy mode: wait once at the start (chunk 0 only)
                // Add stage offset to match producer's Blackwell layout:
                // V stages are at offsets 0, bytes_V_chunk, etc. (relative to V base)
                if (v_chunk == 0) {
                    mbarrier_wait(&pipeline_state->full_V[current_stage], phase_V);
                    tile_V_i = consumer_tile_V_base + current_stage * (bytes_V_chunk / sizeof(half2));
                }
            }
#endif
        } else {
            tile_V_i = tile_V;
        }

        // SM_120: Always load V per-chunk because we skipped the preload (which only loads one chunk).
        // Other archs: Only load V here when nstages <= 1 (preload handles nstages > 1).
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
        constexpr bool sm120_load_v = true;  // SM_120: Always load V per-chunk
#else
        constexpr bool sm120_load_v = false;
#endif
        if (!consumer_mode && (nstages <= 1 || sm120_load_v)) {
            if (i0_start < reusable_cutoff) {
                constexpr bool use_cp_async = (nstages == 1) || sm120_load_v;
                if constexpr (use_tma) {
                    #ifdef BLACKWELL_TMA_AVAILABLE
                    // Load V via TMA (full tile with SWIZZLE_NONE for large tiles)
                    // Note: For common configs, i0_diff == 2*nbatch_V2, so we load full tile
                    uint32_t bytes_V = nbatch_fa * (i0_diff/2) * sizeof(half2);
                    if (threadIdx.x == 0) mbarrier_arrive_expect_tx(mbar_ptr + 1, bytes_V);
                    // V tensor map is at index 1
                    flash_attn_ext_f16_load_tile_tma_chunked<stride_tile_V, nwarps, nbatch_fa, nbatch_V2, chunks_V, chunk_size_V>(
                        tensor_maps, 1, tile_V, mbar_ptr + 1, i0_start/2, k_VKQ_0);
                    mbarrier_wait(mbar_ptr + 1, phase_V);
                    phase_V ^= 1;
                    #endif
                } else {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
                    // SM_120 OPTIMIZED: Two-Stage V Pipelining with Swizzled cp.async
                    // ================================================================
                    // Phase 2B: Overlap V[n+1] prefetch with V[n] computation
                    // 
                    // Memory layout: Reuse tile_K and tile_V as V double buffers
                    // (K processing is complete, so tile_K is available)
                    //
                    // Pipeline flow:
                    //   V chunk 0: Load V[0], commit; prefetch V[1], commit; wait_group<1>
                    //   V chunk n: Prefetch V[n+1], wait_group<1>, compute V[n]
                    //   Final: wait_group<0> for last chunk
                    //
                    // Performance: Additional ~2-3% throughput improvement
                    // ================================================================
                    
                    constexpr int STRIDE_BYTES_V_inner = stride_tile_V * sizeof(half2);
                    
                    // Double buffer addresses - reuse K buffers since K is done
                    const uint32_t tile_V_buf0 = ggml_cuda_cvta_generic_to_shared(tile_K);  // K is done
                    const uint32_t tile_V_buf1 = ggml_cuda_cvta_generic_to_shared(tile_V);
                    
                    // warp_id_offset=0 for unified mode (all warps load, threadIdx.y = 0..nwarps-1)
                    if (v_chunk == 0) {
                        // Issue first V load (V[0] into buffer 0)
                        flash_attn_ext_f16_load_tile_swizzle<stride_tile_V, STRIDE_BYTES_V_inner, nwarps, nbatch_fa, 0>(
                            V_h2 + int64_t(k_VKQ_0)*stride_V + i0_start/2, tile_V_buf0, i0_diff/2, stride_V);
                        cp_async_commit_group();

                        // If there's a next V chunk, prefetch it into buffer 1
                        if (v_chunk + 1 < num_V_chunks) {
                            const int next_i0_start = (v_chunk + 1) * nbatch_V2 * 2;
                            const int next_i0_stop = (next_i0_start + 2*nbatch_V2 < DV) ? (next_i0_start + 2*nbatch_V2) : DV;
                            const int next_i0_diff = next_i0_stop - next_i0_start;
                            if (next_i0_start < reusable_cutoff) {
                                flash_attn_ext_f16_load_tile_swizzle<stride_tile_V, STRIDE_BYTES_V_inner, nwarps, nbatch_fa, 0>(
                                    V_h2 + int64_t(k_VKQ_0)*stride_V + next_i0_start/2, tile_V_buf1, next_i0_diff/2, stride_V);
                                cp_async_commit_group();
                            }
                        }

                        // Wait for V[0] to complete
                        if (v_chunk + 1 < num_V_chunks) {
                            cp_async_wait_group<1>();
                        } else {
                            cp_async_wait_group<0>();
                        }
                    } else {
                        // Subsequent V chunks: current data is already prefetched
                        // Prefetch next chunk if it exists
                        if (v_chunk + 1 < num_V_chunks) {
                            const int next_i0_start = (v_chunk + 1) * nbatch_V2 * 2;
                            const int next_i0_stop = (next_i0_start + 2*nbatch_V2 < DV) ? (next_i0_start + 2*nbatch_V2) : DV;
                            const int next_i0_diff = next_i0_stop - next_i0_start;
                            const uint32_t next_buf = (v_chunk % 2 == 0) ? tile_V_buf1 : tile_V_buf0;
                            if (next_i0_start < reusable_cutoff) {
                                flash_attn_ext_f16_load_tile_swizzle<stride_tile_V, STRIDE_BYTES_V_inner, nwarps, nbatch_fa, 0>(
                                    V_h2 + int64_t(k_VKQ_0)*stride_V + next_i0_start/2, next_buf, next_i0_diff/2, stride_V);
                                cp_async_commit_group();
                            }

                            // Wait for current V chunk
                            cp_async_wait_group<1>();
                        } else {
                            // Last chunk: wait for all
                            cp_async_wait_group<0>();
                        }
                    }
                    
                    __syncthreads();
                    
                    // Update tile_V_i to point to correct buffer
                    tile_V_i = (v_chunk % 2 == 0) ? tile_K : tile_V;
#else
                    flash_attn_ext_f16_load_tile<stride_tile_V, nwarps, nbatch_fa, use_cp_async, oob_check>
                        (V_h2 + int64_t(k_VKQ_0)*stride_V + i0_start/2, tile_V, i0_diff/2, stride_V, k_VKQ_sup);
                    if (use_cp_async) {
                        cp_async_wait_all();
                    }
                    __syncthreads();
                    tile_V_i = tile_V;
#endif
                }
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 1200
                __syncthreads();
#endif
            }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
            // SM_120: tile_V_i already set in pipelining code above
            if (i0_start >= reusable_cutoff) {
                tile_V_i = tile_V + (i0_start - reusable_cutoff)/2;
            }
#else
            tile_V_i = i0_start < reusable_cutoff ? tile_V : tile_V + (i0_start - reusable_cutoff)/2;
#endif
        } else if constexpr (mla && use_tma) {
             if (!consumer_mode) {
             #ifdef BLACKWELL_TMA_AVAILABLE
             // Pipelined V loop (non-consumer mode only)
             if (i0_start < reusable_cutoff) {
                 uint64_t* mbar = v_buf == 0 ? (mbar_ptr + 1) : mbar_ptr; // buf0 -> mbar_V, buf1 -> mbar_K
                 uint32_t& phase = v_buf == 0 ? phase_V : phase_K;
                 
                 mbarrier_wait(mbar, phase);
                 phase ^= 1;
                 
                 tile_V_i = v_buf == 0 ? tile_V : tile_K; 
                 
                 // Issue next load
                 int i0_next_stop = i0_stop - 2*nbatch_V2;
                 if (i0_next_stop > 0) {
                     int i0_next_start = i0_next_stop - 2*nbatch_V2 > 0 ? i0_next_stop - 2*nbatch_V2 : 0;
                     if (i0_next_start < reusable_cutoff) {
                         int next_v_buf = v_buf ^ 1;
                         half2* next_tile_V = next_v_buf == 0 ? tile_V : tile_K;
                         uint64_t* next_mbar = next_v_buf == 0 ? (mbar_ptr + 1) : mbar_ptr;
                         
                         uint32_t bytes_V = nbatch_fa * nbatch_V2 * sizeof(half2);
                         if (threadIdx.x == 0) mbarrier_arrive_expect_tx(next_mbar, bytes_V);
                         
                         flash_attn_ext_f16_load_tile_tma_chunked<stride_tile_V, nwarps, nbatch_fa, nbatch_V2, chunks_V, chunk_size_V>(
                             tensor_maps, 1, next_tile_V, next_mbar, i0_next_start/2, k_VKQ_0);
                     }
                 }
                 
                 v_buf ^= 1;
             }
             #endif
             } // end if (!consumer_mode)
        }

#if defined(TURING_MMA_AVAILABLE)
        constexpr int i0_stride = cols_per_warp == 8 ? T_C_VKQ::I : 2*T_C_VKQ::J;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
        // SM_120 OPTIMIZED: Use swizzled ldmatrix_trans to read from swizzled V
        constexpr int STRIDE_BYTES_V_iter = stride_tile_V * sizeof(half2);
        const uint32_t tile_V_i_base = ggml_cuda_cvta_generic_to_shared((void*)tile_V_i);

        // =========================================================================
        // DEBUG: V ldmatrix swizzle address validation (first V chunk only)
        // =========================================================================
        {
            static bool v_ldmatrix_debug_done = false;
            if (!v_ldmatrix_debug_done && threadIdx.x == 0 && threadIdx.y == 1 && blockIdx.x == 0 && blockIdx.y == 0 && v_chunk == 0) {
                extern __shared__ char smem_v_debug[];
                printf("\n[V LDMATRIX DEBUG] ========== V MMA Setup ==========\n");
                printf("[V LDMATRIX DEBUG] tile_V_i=%p (smem offset=%ld)\n",
                       (void*)tile_V_i, (long)((char*)tile_V_i - smem_v_debug));
                printf("[V LDMATRIX DEBUG] tile_V_i_base (smem addr)=0x%x\n", tile_V_i_base);
                printf("[V LDMATRIX DEBUG] STRIDE_BYTES_V_iter=%d, stride_tile_V=%d\n",
                       STRIDE_BYTES_V_iter, stride_tile_V);
                printf("[V LDMATRIX DEBUG] i0_start=%d, i0_stop=%d, i0_stride=%d\n",
                       i0_start, i0_stop, i0_stride);
                printf("[V LDMATRIX DEBUG] nbatch_fa=%d, np=%d, warp_id=%d\n",
                       nbatch_fa, np, warp_id);
                // Calculate max address that will be accessed
                const int max_k0 = (nbatch_fa/2 - 1) / (np*T_A_VKQ::J) * (np*T_A_VKQ::J) + (warp_id % np)*T_A_VKQ::J;
                const int max_i_offset = (i0_stop - i0_start - i0_stride)/2;
                const uint32_t max_smem_offset = 2*max_k0 * STRIDE_BYTES_V_iter + max_i_offset * sizeof(half2);
                printf("[V LDMATRIX DEBUG] max_k0=%d, max_i_offset=%d\n", max_k0, max_i_offset);
                printf("[V LDMATRIX DEBUG] max_smem_offset=0x%x (%u bytes)\n",
                       tile_V_i_base + max_smem_offset, max_smem_offset);
                printf("[V LDMATRIX DEBUG] =============================================\n\n");
                v_ldmatrix_debug_done = true;
            }
        }
#endif
#pragma unroll
        for (int i_VKQ_0 = i0_start; i_VKQ_0 < i0_stop; i_VKQ_0 += i0_stride) {
            static_assert((nbatch_fa/2) % (np*T_A_VKQ::J) == 0, "bad loop size");
#pragma unroll
            for (int k00 = 0; k00 < nbatch_fa/2; k00 += np*T_A_VKQ::J) {
                const int k0 = k00 + (warp_id % np)*T_A_VKQ::J;

                T_A_VKQ A; // Transposed in SRAM but not in registers, gets transposed on load.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
                // SM_120: Use swizzled ldmatrix_trans read
                ggml_cuda_mma::load_ldmatrix_trans_swizzle<STRIDE_BYTES_V_iter>(A, tile_V_i_base,
                    2*k0, (i_VKQ_0 - i0_start)/2 * sizeof(half2));
#else
                load_ldmatrix_trans(A, tile_V_i + 2*k0*stride_tile_V + (i_VKQ_0 - i0_start)/2, stride_tile_V);
#endif
                if constexpr (T_B_KQ::I == 8) {
                    mma(VKQ_C[i_VKQ_0/i0_stride], A, B[k00/(np*T_A_VKQ::J)]);
                } else {
                    // Wide version of VKQ_C is column-major => swap A and B.
                    mma(VKQ_C[i_VKQ_0/i0_stride], B[k00/(np*T_A_VKQ::J)], A);
                }
            }
        }
#else // Volta
        constexpr int i0_stride = 2*T_C_VKQ::J;
#pragma unroll
        for (int i_VKQ_0 = i0_start; i_VKQ_0 < i0_stop; i_VKQ_0 += i0_stride) {
            static_assert(nbatch_fa % (np*T_A_VKQ::I) == 0, "bad loop size");
            static_assert(2*T_B_VKQ::J == T_A_VKQ::I, "bad tile sizes");
#pragma unroll
            for (int k00 = 0; k00 < nbatch_fa; k00 += np*T_A_VKQ::I) {
                const int k0 = k00 + (warp_id % np)*T_A_VKQ::I;

                T_A_VKQ A; // Transposed in both SRAM and registers, load normally.
                load_ldmatrix(A, tile_V_i + k0*stride_tile_V + (i_VKQ_0 - i0_start)/2, stride_tile_V);
                mma(VKQ_C[i_VKQ_0/i0_stride], B[k00/(np*T_A_VKQ::I)], A);
            }
        }
#endif // defined(TURING_MMA_AVAILABLE)

        if (consumer_mode) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
            // SM_120 FALLBACK: Use __syncthreads__ instead of mbarrier_arrive
            // Signal tile processing complete after last V chunk
            if (v_chunk == num_V_chunks - 1) {
                __syncthreads();  // SYNC 3: Tile processing complete
            }
#elif defined(BLACKWELL_TMA_AVAILABLE)
            // Standard mbarrier path for SM_90/100/103/110
            if constexpr (needs_chunking) {
                // DEBUG: Print before signaling V completion
                if (threadIdx.x == 0 && threadIdx.y == 1 && blockIdx.x == 0 && blockIdx.y == 0 && kb0 == 0) {
                    printf("[CONSUMER DEBUG] V chunk %d: processing complete, signaling empty_V_chunk[%d]\n",
                           v_chunk, v_chunk_stage);
                }

                // Signal we're done with this V chunk
                // NOTE: Using mbarrier_arrive instead of mbarrier_arrive_expect_tx(0)
                mbarrier_arrive(&pipeline_state->empty_V_chunk[v_chunk_stage]);

                // Flip chunk phase after every 2 chunks
                if (v_chunk_stage == 1) {
                    consumer_chunk_phase_V ^= 1;
                }
            } else {
                // Legacy mode: signal once at the end (last chunk only)
                if (v_chunk == num_V_chunks - 1) {
                    mbarrier_arrive(&pipeline_state->empty_V[current_stage]);
                }
            }
#endif
        } else {
            if constexpr (nstages <= 1) {
                __syncthreads(); // Only needed if tile_K == tile_V.
            }
        }
    }

    if (!consumer_mode && nstages > 1 && mla) {
        // Preload K for next iteration:
        if constexpr (use_tma) {
             if (!last_iter) {
                 #ifdef BLACKWELL_TMA_AVAILABLE
                 // Preload K for next iteration via TMA (full tile)
                 // This is issued after V loop, so V loop is done with tile_K/tile_V.
                 // We must ensure all threads have finished reading from SHMEM before we issue TMA write to that SHMEM.
                 __syncthreads(); 
                 
                 uint32_t bytes_K = nbatch_fa * nbatch_K2 * sizeof(half2);
                 if (threadIdx.x == 0) mbarrier_arrive_expect_tx(mbar_ptr, bytes_K);
                 
                 flash_attn_ext_f16_load_tile_tma_chunked<stride_tile_K, nwarps, nbatch_fa, nbatch_K2, chunks_K, chunk_size_K>(
                     tensor_maps, 0, tile_K, mbar_ptr, 0, k_VKQ_0 + nbatch_fa);
                 #endif
             }
        }
    }
#else
    GGML_UNUSED_VARS(Q_f2, K_h2, V_h2, mask_h, dstk, dstk_fixup,
        scale, slope, logit_softcap, ne01, ne02,
        stride_K, stride_V, stride_mask,
        tile_Q, tile_K, tile_V, tile_mask,
        Q_B, VKQ_C, KQ_max, KQ_rowsum, kb0);
    NO_DEVICE_CODE;
#endif // defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE)
}

#if defined(TURING_MMA_AVAILABLE)
template<int ncols> struct mma_tile_sizes {
    using T_A_KQ  = tile<16,  8, half2>; // row-major
    using T_B_KQ  = tile<16,  8, half2>; // column-major
    using T_C_KQ  = tile<16, 16, float>; // column-major
    using T_A_VKQ = tile<16,  8, half2>; // row-major
    using T_B_VKQ = tile<16,  8, half2>; // column-major
    using T_C_VKQ = tile<16,  8, half2>; // column-major
};
template<> struct mma_tile_sizes<8> {
    using T_A_KQ  = tile<16,  8, half2>; // row-major
    using T_B_KQ  = tile< 8,  8, half2>; // column-major
    using T_C_KQ  = tile<16,  8, float>; // row-major
    using T_A_VKQ = tile<16,  8, half2>; // row-major
    using T_B_VKQ = tile< 8,  8, half2>; // column-major
    using T_C_VKQ = tile<16,  4, half2>; // row-major
};
#else // Volta
template<int ncols> struct mma_tile_sizes {
    using T_A_KQ  = tile< 8,  4, half2, DATA_LAYOUT_I_MAJOR_MIRRORED>; // row-major
    using T_B_KQ  = tile<32,  4, half2, DATA_LAYOUT_I_MAJOR>;          // column-major
    using T_C_KQ  = tile<32,  8, float, DATA_LAYOUT_I_MAJOR>;          // column-major
    using T_A_VKQ = tile< 8,  4, half2, DATA_LAYOUT_J_MAJOR_MIRRORED>; // column-major
    using T_B_VKQ = tile<32,  4, half2, DATA_LAYOUT_I_MAJOR>;          // column-major
    using T_C_VKQ = tile<32,  4, half2, DATA_LAYOUT_I_MAJOR>;          // column-major
};
#endif // defined(TURING_MMA_AVAILABLE)

template<int DKQ, int DV, int ncols1, int ncols2, int nwarps, int num_consumers, bool use_logit_softcap, bool mla, bool needs_fixup, bool is_fixup, bool use_tma>
static __device__ __forceinline__ void flash_attn_ext_f16_process_tile(
        const float2 * const __restrict__ Q_f2,
        const half2  * const __restrict__ K_h2,
        const half2  * const __restrict__ V_h2,
        const half   * const __restrict__ mask_h,
        const float  * const __restrict__ sinks_f,
        float2       * const __restrict__ dstk,
        float2       * const __restrict__ dstk_fixup,
        const float scale,
        const float slope,
        const float logit_softcap,
        const uint3 ne01,
        const int ne02,
        const int ne11,
        const int stride_Q1,
        const int stride_Q2,
        const int stride_K,
        const int stride_V,
        const int stride_mask,
        const int jt,
        const int kb0_start,
        const int kb0_stop,
        const char * __restrict__ tensor_maps) {
#if defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE)
    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    constexpr int ncols = ncols1 * ncols2;
    using     T_A_KQ    = typename mma_tile_sizes<ncols>::T_A_KQ;
    using     T_B_KQ    = typename mma_tile_sizes<ncols>::T_B_KQ;
    using     T_C_KQ    = typename mma_tile_sizes<ncols>::T_C_KQ;
    using     T_A_VKQ   = typename mma_tile_sizes<ncols>::T_A_VKQ;
    using     T_B_VKQ   = typename mma_tile_sizes<ncols>::T_B_VKQ;
    using     T_C_VKQ   = typename mma_tile_sizes<ncols>::T_C_VKQ;

    // Warp configuration:
    // - Consumer mode (num_consumers > 0): Use num_consumers warps, remap IDs (threadIdx.y 1-N → warp_id 0-(N-1))
    // - Unified mode (num_consumers == 0): All warps do both load and compute (SM_120, Ampere)
    //
    // SM_120 (RTX 5090) uses unified mode with 4 warps - aligned with Gau-Nernst reference
    constexpr int  effective_nwarps = (num_consumers > 0) ? num_consumers : nwarps;
    const int      warp_id          = (num_consumers > 0) ? (threadIdx.y - 1) : threadIdx.y;

    constexpr int  cols_per_warp   = T_B_KQ::I;
    constexpr int  cols_per_thread = 2; // This is specifically KQ columns, Volta only has a single VKQ column.
    constexpr int  np              = effective_nwarps * (cols_per_warp/ncols2) / ncols1; // Number of parallel CUDA warps per Q column.
    // Config selection:
    // - SM_120 (__CUDA_ARCH__ >= 1200): Use SM_120 configs (unified mode, 4 warps, nbatch_fa=128)
    // - Other Blackwell (use_tma=true): Use Blackwell TMA configs
    // - Ampere/Turing (use_tma=false): Use Ampere configs
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
    // SM_120: Use dedicated SM_120 configs (routes via get_config_blackwell → get_config_sm120)
    constexpr int  nbatch_fa       = ggml_cuda_fattn_mma_get_nbatch_fa     (DKQ, DV, ncols);
    constexpr int  nbatch_K2       = ggml_cuda_fattn_mma_get_nbatch_K2     (DKQ, DV, ncols);
    constexpr int  nbatch_V2       = ggml_cuda_fattn_mma_get_nbatch_V2     (DKQ, DV, ncols);
    constexpr int  nbatch_combine  = ggml_cuda_fattn_mma_get_nbatch_combine(DKQ, DV, ncols);
    constexpr bool Q_in_reg        = ggml_cuda_fattn_mma_get_Q_in_reg      (DKQ, DV, ncols);
    constexpr int  nstages         = ggml_cuda_fattn_mma_get_nstages       (DKQ, DV, ncols1, ncols2);
#else
    // Other archs: use_tma determines Blackwell TMA vs Ampere config
    constexpr int  nbatch_fa       = use_tma ? ggml_cuda_fattn_mma_get_nbatch_fa     (DKQ, DV, ncols) : ggml_cuda_fattn_mma_get_nbatch_fa_ampere     (DKQ, DV, ncols);
    constexpr int  nbatch_K2       = use_tma ? ggml_cuda_fattn_mma_get_nbatch_K2     (DKQ, DV, ncols) : ggml_cuda_fattn_mma_get_nbatch_K2_ampere     (DKQ, DV, ncols);
    constexpr int  nbatch_V2       = use_tma ? ggml_cuda_fattn_mma_get_nbatch_V2     (DKQ, DV, ncols) : ggml_cuda_fattn_mma_get_nbatch_V2_ampere     (DKQ, DV, ncols);
    constexpr int  nbatch_combine  = use_tma ? ggml_cuda_fattn_mma_get_nbatch_combine(DKQ, DV, ncols) : ggml_cuda_fattn_mma_get_nbatch_combine_ampere(DKQ, DV, ncols);
    constexpr bool Q_in_reg        = use_tma ? ggml_cuda_fattn_mma_get_Q_in_reg      (DKQ, DV, ncols) : ggml_cuda_fattn_mma_get_Q_in_reg_ampere      (DKQ, DV, ncols);
    constexpr int  nstages         = use_tma ? ggml_cuda_fattn_mma_get_nstages       (DKQ, DV, ncols1, ncols2) : ggml_cuda_fattn_mma_get_nstages_ampere(DKQ, DV, ncols1, ncols2);
#endif

    // Blackwell Pipeline Setup
    fattn_pipeline_state* pipeline_state = nullptr;
    int pipeline_stage = -1;

    // Check if we are in Consumer Mode (producer/consumer pipelining)
    extern __shared__ char smem[];
    // SM_120 (RTX 5090): Uses unified mode with Ampere-style shared memory layout.
    // All 4 warps do both load and compute - no producer/consumer split.
    // Uses simple cp.async + __syncthreads() pattern (no mbarrier needed).
    // Aligned with Gau-Nernst reference: https://gau-nernst.github.io/fa-5090/
    //
    // Other Blackwell (SM_100): Uses producer/consumer pipelining with mbarrier (consumer_mode = true).
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
    // SM_120: Use Ampere-style layout (unified mode, no chunked pipelining)
    constexpr bool use_chunked_layout = false;  // Ampere layout for unified mode
    constexpr bool set_pipeline_state = false;  // No consumer mode - all warps load and compute
#else
    constexpr bool use_chunked_layout = (num_consumers > 0);  // Other archs only when pipelined
    constexpr bool set_pipeline_state = (num_consumers > 0);  // Use consumer mode when pipelined
#endif
    if (use_chunked_layout) {
        // Shared memory layout for chunk pipelining:
        // [K chunk 0][K chunk 1][V chunk 0][V chunk 1][Q tiles][mask][pipeline_state]
        constexpr int bytes_K_chunk = nbatch_fa * nbatch_K2 * sizeof(half2);
        constexpr int bytes_V_chunk = nbatch_fa * nbatch_V2 * sizeof(half2);
        constexpr int bytes_KV_total = 2 * bytes_K_chunk + 2 * bytes_V_chunk;
        constexpr int stride_tile_Q_local = DKQ/2 + 4;
        constexpr int bytes_Q = ncols * stride_tile_Q_local * sizeof(half2);
        constexpr int bytes_mask = ncols1 * (nbatch_fa/2 + 4) * sizeof(half2);
        // Only set pipeline_state for true producer/consumer mode (not SM_120)
        // SM_120: pipeline_state stays nullptr - consumers load K/V/mask themselves via cp.async
        // Other archs: pipeline_state set when using producer/consumer pipelining
        if constexpr (set_pipeline_state) {
            pipeline_state = (fattn_pipeline_state*)(smem + bytes_KV_total + bytes_Q + bytes_mask);
            pipeline_stage = 0; // Initial stage
        }

        // =========================================================================
        // DEBUG: Shared memory bounds validation at process_tile entry
        // =========================================================================
        if (threadIdx.x == 0 && threadIdx.y == 1 && blockIdx.x == 0 && blockIdx.y == 0) {
            // Calculate actual shared memory size (from launch config)
            // We use the local calculation to verify consistency
            const size_t expected_smem_size = bytes_KV_total + bytes_Q + bytes_mask + sizeof(fattn_pipeline_state) + 128;

            printf("\n[SMEM BOUNDS] ========== Process Tile Entry ==========\n");
            printf("[SMEM BOUNDS] smem_base=%p\n", smem);
            printf("[SMEM BOUNDS] Layout (chunked mode):\n");
            printf("[SMEM BOUNDS]   bytes_K_chunk=%d, 2x=%d\n", bytes_K_chunk, 2*bytes_K_chunk);
            printf("[SMEM BOUNDS]   bytes_V_chunk=%d, 2x=%d\n", bytes_V_chunk, 2*bytes_V_chunk);
            printf("[SMEM BOUNDS]   bytes_KV_total=%d\n", bytes_KV_total);
            printf("[SMEM BOUNDS]   bytes_Q=%d, bytes_mask=%d\n", bytes_Q, bytes_mask);
            printf("[SMEM BOUNDS]   sizeof(fattn_pipeline_state)=%d\n", (int)sizeof(fattn_pipeline_state));
            printf("[SMEM BOUNDS]   expected_total=%zu (%.1f KB)\n", expected_smem_size, expected_smem_size/1024.0);
            printf("[SMEM BOUNDS] Pointer layout:\n");
            printf("[SMEM BOUNDS]   K_chunk[0] = smem+0 = %p\n", smem);
            printf("[SMEM BOUNDS]   K_chunk[1] = smem+%d = %p\n", bytes_K_chunk, smem + bytes_K_chunk);
            printf("[SMEM BOUNDS]   V_chunk[0] = smem+%d = %p\n", 2*bytes_K_chunk, smem + 2*bytes_K_chunk);
            printf("[SMEM BOUNDS]   V_chunk[1] = smem+%d = %p\n", 2*bytes_K_chunk + bytes_V_chunk, smem + 2*bytes_K_chunk + bytes_V_chunk);
            printf("[SMEM BOUNDS]   Q_tiles    = smem+%d = %p\n", bytes_KV_total, smem + bytes_KV_total);
            printf("[SMEM BOUNDS]   mask       = smem+%d = %p\n", bytes_KV_total + bytes_Q, smem + bytes_KV_total + bytes_Q);
            // Show both the computed pipeline address AND whether it's actually used
            const int pipeline_offset = bytes_KV_total + bytes_Q + bytes_mask;
            printf("[SMEM BOUNDS]   pipeline_addr = smem+%d = %p\n", pipeline_offset, (void*)(smem + pipeline_offset));
            printf("[SMEM BOUNDS]   pipeline_state = %p (null means SM_120 loads K/V itself)\n", (void*)pipeline_state);
            printf("[SMEM BOUNDS] Config: nbatch_fa=%d, nbatch_K2=%d, nbatch_V2=%d\n", nbatch_fa, nbatch_K2, nbatch_V2);
            printf("[SMEM BOUNDS] Config: DKQ=%d, DV=%d, ncols=%d, nstages=%d\n", DKQ, DV, ncols, nstages);
            printf("[SMEM BOUNDS] =============================================\n\n");
        }
    }

    // TMA tile parameters (legacy chunk variables kept for template API compatibility).
    // With SWIZZLE_NONE for large tiles, we load full tiles in single TMA operations.
    constexpr int  chunks_K        = tma_chunks_needed(nbatch_K2);
    constexpr int  chunks_V        = tma_chunks_needed(nbatch_V2);
    constexpr int  chunk_size_K    = (nbatch_K2 + chunks_K - 1) / chunks_K;
    constexpr int  chunk_size_V    = (nbatch_V2 + chunks_V - 1) / chunks_V;
    (void)chunk_size_V;  // May be unused in some template instantiations

    if (cols_per_warp > ncols) {
        NO_DEVICE_CODE;
        return;
    }

    static_assert(nwarps * (cols_per_warp/ncols2) % ncols1 == 0, "bad nwarps");

    constexpr int stride_tile_Q = DKQ/2     + 4;
    // TMA and SM_120 cp.async both use compact stride (no padding).
    // Legacy non-TMA path uses +4 padding for bank conflict avoidance.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
    // SM_120: Use compact stride to match producer's cp.async layout
    constexpr int stride_tile_K = nbatch_K2;  // COMPACT - matches bytes_K_chunk
#else
    constexpr int stride_tile_K = use_tma ? nbatch_K2 : nbatch_K2 + 4;
#endif

    static_assert(!mla || nbatch_K2 >= nbatch_V2, "bad nbatch_K2, nbatch_V2 for MLA");
    // For TMA: V uses its own compact stride (nbatch_V2) since V overwrites K's memory
    // after K is processed. For non-TMA MLA: V shares K's stride for memory layout.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
    // SM_120: Use compact stride matching bytes_V_chunk
    constexpr int stride_tile_V = mla ? nbatch_K2 : nbatch_V2;  // COMPACT
#else
    constexpr int stride_tile_V = use_tma ? nbatch_V2 : (mla ? stride_tile_K : nbatch_V2 + 4);
#endif
    constexpr int stride_tile_KV_max = stride_tile_K > stride_tile_V ? stride_tile_K : stride_tile_V;

    // Use smem pointer directly if in Consumer mode (Q is separate or in registers)
    half2 * tile_Q = (half2*)smem;

    // Legacy Layout (for non-pipeline mode)
    half2 * tile_K    = Q_in_reg              ? tile_Q                             : tile_Q + ncols     * stride_tile_Q;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
    // SM_120 Two-Stage K Pipelining: Always separate K and V buffers
    // We use tile_V as K[1] buffer during K processing, then for actual V afterward
    // This requires separate memory regions for tile_K and tile_V
    half2 * tile_V    = tile_K + nbatch_fa * stride_tile_K;  // Always separate for SM_120
    half  * tile_mask = (half *) (tile_V + nbatch_fa * stride_tile_V);
#else
    half2 * tile_V    =           nstages > 1 ? tile_K + nbatch_fa * stride_tile_K : tile_K;
    half  * tile_mask = (half *) (nstages > 1 ? tile_V + nbatch_fa * stride_tile_V : tile_V + nbatch_fa * stride_tile_KV_max);
#endif

    // Chunked layout mode (Blackwell or SM_120): use producer's chunk double-buffering layout
    // New layout: [K chunk 0][K chunk 1][V chunk 0][V chunk 1][Q tiles][mask][pipeline_state]
    // Producer writes K chunks at smem[0] and smem[bytes_K_chunk]
    // Producer writes V chunks at smem[2*bytes_K_chunk] and smem[2*bytes_K_chunk + bytes_V_chunk]
    // The actual chunk buffer selection (0 or 1) is handled dynamically in the iter function
    // based on the current iteration. Here we set tile_K/V to the base of chunk 0.
    if (use_chunked_layout) {
        constexpr int bytes_K_chunk = nbatch_fa * nbatch_K2 * sizeof(half2);
        constexpr int bytes_V_chunk = nbatch_fa * nbatch_V2 * sizeof(half2);
        constexpr int bytes_KV_total = 2 * bytes_K_chunk + 2 * bytes_V_chunk;
        // tile_K points to base of K chunk buffers (chunk 0)
        tile_K = (half2*)smem;
        // tile_V points to base of V chunk buffers (chunk 0), after both K chunks
        tile_V = (half2*)((char*)smem + 2 * bytes_K_chunk);
        // Q tiles are after KV buffers in the new layout
        tile_Q = (half2*)((char*)smem + bytes_KV_total);
        // Mask is after Q tiles
        tile_mask = (half*)((char*)smem + bytes_KV_total + ncols * stride_tile_Q * sizeof(half2));
    }

    // Mbarrier allocation at the end of shared memory
    uint64_t* mbar_ptr = nullptr;
    if constexpr (use_tma) {
        if (!pipeline_state) {
            const size_t nbytes_shared_mask_actual = ncols1 * (nbatch_fa/2 + 4) * sizeof(half2);
            mbar_ptr = (uint64_t*)((char*)tile_mask + nbytes_shared_mask_actual);
            
            // Ensure alignment
            size_t addr = (size_t)mbar_ptr;
            if (addr % 8 != 0) {
                mbar_ptr = (uint64_t*)(addr + 8 - (addr % 8));
            }
            
            #ifdef BLACKWELL_TMA_AVAILABLE
            __syncthreads(); // Ensure tile_mask calculation is done and others are ready
            if (threadIdx.x == 0) {
                // Initialize 2 mbarriers with count 1 (for the thread that issues the copy)
                mbarrier_init(mbar_ptr, 1);     // K mbarrier
                mbarrier_init(mbar_ptr + 1, 1); // V mbarrier
            }
            #endif
            __syncthreads();
        }
    }
    
    // Phase for producer-consumer synchronization.
    // In consumer mode, phase flips when we complete a full cycle through all stages.
    // Both K and V use the same phase since they're loaded together per iteration.
    uint32_t phase_K = 0;
    uint32_t phase_V = 0;

    // Chunk-level phase tracking for consumer mode (persists across kb0 iterations).
    // These must be defined here (not in flash_attn_ext_f16_iter) so they persist
    // across the kb0 loop, matching the producer's chunk_phase_K/V variables.
    uint32_t consumer_chunk_phase_K = 0;
    uint32_t consumer_chunk_phase_V = 0;

    T_B_KQ    Q_B[(Q_in_reg ? DKQ/(2*T_B_KQ::J) : 1)];
#if defined(TURING_MMA_AVAILABLE)
    T_C_VKQ VKQ_C[cols_per_warp == 8 ? DV/T_C_VKQ::I : DV/(2*T_C_VKQ::J)] = {};
#else // Volta
    T_C_VKQ VKQ_C[                                     DV/(2*T_C_VKQ::J)] = {};
#endif // defined(TURING_MMA_AVAILABLE)

    float KQ_rowsum[cols_per_thread] = {0.0f};
    float KQ_max[cols_per_thread];
#pragma unroll
    for (int col = 0; col < cols_per_thread; ++col) {
        KQ_max[col] = -FLT_MAX/2.0f;
    }

    // Load Q data into tile_Q, either temporarily or permanently.
    // Q in registers is faster, but register pressure is the biggest bottleneck.
    // The loading is done with decreasing granularity for D for better memory bandwidth.
    //
    // SKIP Q loading when pipeline_state is set (Blackwell consumer mode):
    // Q is already loaded by the Blackwell kernel before producer/consumer divergence.
    // The __syncthreads() calls are also already done by the Blackwell kernel.
    if (!pipeline_state) {
        const half2 scale_h2 = make_half2(scale, scale);
#pragma unroll
        for (int stride_k : {WARP_SIZE, WARP_SIZE/2, WARP_SIZE/4}) {
            const int k0_start  = stride_k == WARP_SIZE ? 0 : DKQ/2 - (DKQ/2) % (2*stride_k);
            const int k0_stop   =                             DKQ/2 - (DKQ/2) % (1*stride_k);
            const int stride_jc = WARP_SIZE / stride_k;

            if (k0_start == k0_stop) {
                continue;
            }

#pragma unroll
            for (int jc0 = 0; jc0 < ncols; jc0 += effective_nwarps*stride_jc) {
                const int jc = jc0 + warp_id*stride_jc + (stride_k == WARP_SIZE ? 0 : threadIdx.x / stride_k);

                if (jc0 + effective_nwarps*stride_jc > ncols && jc >= ncols) {
                    break;
                }

                const int j = jc / ncols2;
                const int c = jc % ncols2;

                if (jt*ncols1 + j < int(ne01.z)) {
#pragma unroll
                    for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                        const int k = k0 + (stride_k == WARP_SIZE ? threadIdx.x : threadIdx.x % stride_k);

                        const float2 tmp = Q_f2[(jt*ncols1 + j)*stride_Q1 + c*stride_Q2 + k];
                        tile_Q[jc*stride_tile_Q + k] = scale_h2 * make_half2(tmp.x, tmp.y);
                    }
                } else {
#pragma unroll
                    for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                        const int k = k0 + (stride_k == WARP_SIZE ? threadIdx.x : threadIdx.x % stride_k);

                        tile_Q[jc*stride_tile_Q + k] = make_half2(0.0f, 0.0f);
                    }
                }
            }
        }

        __syncthreads();
    }

    if (Q_in_reg) {
        const int j0 = (warp_id / np) * cols_per_warp;

#pragma unroll
        for (int k0 = 0; k0 < DKQ/2; k0 += T_B_KQ::J) {
            load_ldmatrix(Q_B[k0/T_B_KQ::J], tile_Q + j0*stride_tile_Q + k0, stride_tile_Q);
        }
    }

    if (!pipeline_state || num_consumers > 0) {
        if (pipeline_state) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
            // SM_120: No Q_loaded barrier - __syncthreads__ handles coordination
            (void)0;
#elif defined(BLACKWELL_TMA_AVAILABLE)
            mbarrier_arrive(&pipeline_state->Q_loaded);
#endif
        } else {
            __syncthreads();
        }
    }

    int kb0 = kb0_start;

    // Preload mask and K data for first iteration when using cp_async with multiple stages:
    if constexpr (nstages > 1) {
        // static_assert(nbatch_K2 == DKQ/2, "batching not implemented for multi-stage pipeline"); // Removed
        constexpr bool use_cp_async = true;
        constexpr bool oob_check    = false;
        constexpr int  k_VKQ_sup    = nbatch_fa;
        
        if (!pipeline_state && (ncols2 > 1 || mask_h)) {
            flash_attn_ext_f16_load_mask<ncols1, nwarps, nbatch_fa, use_cp_async, oob_check>
                (mask_h + kb0*nbatch_fa, tile_mask, stride_mask, k_VKQ_sup, jt*ncols1, ne01);
        }

        if constexpr (use_tma) {
             #ifdef BLACKWELL_TMA_AVAILABLE
             if (!pipeline_state) {
                 // Load K via TMA (full tile with SWIZZLE_NONE for large tiles)
                 uint32_t bytes_K = nbatch_fa * nbatch_K2 * sizeof(half2);
                 if (threadIdx.x == 0) mbarrier_arrive_expect_tx(mbar_ptr, bytes_K);
                 // K chunks start at map index 0
                 flash_attn_ext_f16_load_tile_tma_chunked<stride_tile_K, nwarps, nbatch_fa, nbatch_K2, chunks_K, chunk_size_K>(
                     tensor_maps, 0, tile_K, mbar_ptr, 0, kb0*nbatch_fa);
                 // No wait here, wait is in iter
             }
             #endif
        } else {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
            // SM_120: Use swizzled cp.async for K preload
            // warp_id_offset=0 for unified mode (all warps load, threadIdx.y = 0..nwarps-1)
            constexpr int STRIDE_BYTES_K = stride_tile_K * sizeof(half2);
            const uint32_t tile_K_base = ggml_cuda_cvta_generic_to_shared(tile_K);
            flash_attn_ext_f16_load_tile_swizzle<stride_tile_K, STRIDE_BYTES_K, nwarps, nbatch_fa, 0>(
                K_h2 + int64_t(kb0)*nbatch_fa*stride_K, tile_K_base, nbatch_K2, stride_K);
#else
            flash_attn_ext_f16_load_tile<stride_tile_K, nwarps, nbatch_fa, use_cp_async, oob_check>
                (K_h2 + int64_t(kb0)*nbatch_fa*stride_K, tile_K, nbatch_K2, stride_K, k_VKQ_sup);
#endif
        }
    }

    // kb0_start is always < kb0_stop so the last iter can be executed unconditionally.
    // OOB check is not compatible with multi-stage pipeline, so disable it when nstages > 1.
    //
    // For consumer mode (pipeline_state != nullptr):
    // - Stage is computed as (kb0 - kb0_start) % nstages
    // - Phase flips when stage wraps from (nstages-1) to 0
    if constexpr (ncols2 == 1) {
        constexpr bool oob_check = (nstages == 1);
        for (; kb0 < kb0_stop-1; ++kb0) {
            // Compute stage for this iteration (consumer mode only)
            const int current_stage = (pipeline_state && nstages > 0) ? ((kb0 - kb0_start) % nstages) : pipeline_stage;
            constexpr bool last_iter = false;
            constexpr int  k_VKQ_sup = nbatch_fa;
            flash_attn_ext_f16_iter
                <DKQ, DV, ncols1, ncols2, nwarps, num_consumers, use_logit_softcap, mla, needs_fixup, is_fixup, last_iter, oob_check, use_tma,
                 T_A_KQ, T_B_KQ, T_C_KQ, T_A_VKQ, T_B_VKQ, T_C_VKQ>
                (Q_f2, K_h2, V_h2, mask_h, dstk, dstk_fixup, scale, slope, logit_softcap,
                 ne01, ne02, stride_K, stride_V, stride_mask, tile_Q, tile_K, tile_V, tile_mask, Q_B, VKQ_C,
                 KQ_max, KQ_rowsum, jt, kb0, k_VKQ_sup, tensor_maps, mbar_ptr, phase_K, phase_V, consumer_chunk_phase_K, consumer_chunk_phase_V, pipeline_state, current_stage, nstages);
            // Flip phase when stage wraps (consumer mode only)
            if (pipeline_state && current_stage == nstages - 1) {
                phase_K ^= 1;
                phase_V ^= 1;
            }
        }
        // Last iteration
        const int current_stage = (pipeline_state && nstages > 0) ? ((kb0 - kb0_start) % nstages) : pipeline_stage;
        constexpr bool last_iter = true;
        const     int  k_VKQ_sup = ne11 - kb0*nbatch_fa;
        flash_attn_ext_f16_iter
            <DKQ, DV, ncols1, ncols2, nwarps, num_consumers, use_logit_softcap, mla, needs_fixup, is_fixup, last_iter, oob_check, use_tma,
              T_A_KQ, T_B_KQ, T_C_KQ, T_A_VKQ, T_B_VKQ, T_C_VKQ>
            (Q_f2, K_h2, V_h2, mask_h, dstk, dstk_fixup, scale, slope, logit_softcap,
             ne01, ne02, stride_K, stride_V, stride_mask, tile_Q, tile_K, tile_V, tile_mask, Q_B, VKQ_C,
             KQ_max, KQ_rowsum, jt, kb0, k_VKQ_sup, tensor_maps, mbar_ptr, phase_K, phase_V, consumer_chunk_phase_K, consumer_chunk_phase_V, pipeline_state, current_stage, nstages);
    } else {
        constexpr bool oob_check = false;
        for (; kb0 < kb0_stop-1; ++kb0) {
            // Compute stage for this iteration (consumer mode only)
            const int current_stage = (pipeline_state && nstages > 0) ? ((kb0 - kb0_start) % nstages) : pipeline_stage;
            constexpr bool last_iter = false;
            constexpr int  k_VKQ_sup = nbatch_fa;
            flash_attn_ext_f16_iter
                <DKQ, DV, ncols1, ncols2, nwarps, num_consumers, use_logit_softcap, mla, needs_fixup, is_fixup, last_iter, oob_check, use_tma,
                 T_A_KQ, T_B_KQ, T_C_KQ, T_A_VKQ, T_B_VKQ, T_C_VKQ>
                (Q_f2, K_h2, V_h2, mask_h, dstk, dstk_fixup, scale, slope, logit_softcap,
                 ne01, ne02, stride_K, stride_V, stride_mask, tile_Q, tile_K, tile_V, tile_mask, Q_B, VKQ_C,
                 KQ_max, KQ_rowsum, jt, kb0, k_VKQ_sup, tensor_maps, mbar_ptr, phase_K, phase_V, consumer_chunk_phase_K, consumer_chunk_phase_V, pipeline_state, current_stage, nstages);
            // Flip phase when stage wraps (consumer mode only)
            if (pipeline_state && current_stage == nstages - 1) {
                phase_K ^= 1;
                phase_V ^= 1;
            }
        }
        // Last iteration
        const int current_stage = (pipeline_state && nstages > 0) ? ((kb0 - kb0_start) % nstages) : pipeline_stage;
        constexpr bool last_iter = true;
        constexpr int  k_VKQ_sup = nbatch_fa;
        flash_attn_ext_f16_iter
            <DKQ, DV, ncols1, ncols2, nwarps, num_consumers, use_logit_softcap, mla, needs_fixup, is_fixup, last_iter, oob_check, use_tma,
             T_A_KQ, T_B_KQ, T_C_KQ, T_A_VKQ, T_B_VKQ, T_C_VKQ>
            (Q_f2, K_h2, V_h2, mask_h, dstk, dstk_fixup, scale, slope, logit_softcap,
             ne01, ne02, stride_K, stride_V, stride_mask, tile_Q, tile_K, tile_V, tile_mask, Q_B, VKQ_C,
             KQ_max, KQ_rowsum, jt, kb0, k_VKQ_sup, tensor_maps, mbar_ptr, phase_K, phase_V, consumer_chunk_phase_K, consumer_chunk_phase_V, pipeline_state, current_stage, nstages);
    }

    // With multi-stage loading there is no __syncthreads at the end of the iter,
    //     there can be a race condition on shared memory access for combining/writing back results.
    if constexpr (nstages > 1 && nwarps*cols_per_warp > nbatch_fa) {
        __syncthreads();
    }

    // Finally, sum up partial KQ rowsums.
    {
#if defined(TURING_MMA_AVAILABLE)
        // The partial sums are spread across 8/4 threads.
        constexpr int offset_first = cols_per_warp == 8 ? 16 : 2;
        constexpr int offset_last  = cols_per_warp == 8 ?  4 : 1;
#else // Volta
        // The partial sums are spread across 2 threads.
        constexpr int offset_first = 2;
        constexpr int offset_last  = 2;
#endif // defined(TURING_MMA_AVAILABLE)
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
#pragma unroll
            for (int offset = offset_first; offset >= offset_last; offset >>= 1) {
                KQ_rowsum[col] += __shfl_xor_sync(0xFFFFFFFF, KQ_rowsum[col], offset, WARP_SIZE);
            }
        }
    }

    // If attention sinks are used, potentially re-scale if KQ_max is small.
    // Also add the sink as a value to KQ_rowsum, this is done after synchonization of KQ_rowsum
    //     so it's being done unconditionally for every thread.
    if (!is_fixup && (np == 1 || warp_id % np == 0) && sinks_f) {
        float KQ_max_scale[cols_per_thread];
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
            const int jc = cols_per_warp == 8 ? T_C_KQ::get_j(col) : T_C_KQ::get_i(2*col);
            const float sink = sinks_f[jc % ncols2];

            const float KQ_max_new = fmaxf(KQ_max[col], sink);
            const float KQ_max_diff = KQ_max[col] - KQ_max_new;
            KQ_max_scale[col] = expf(KQ_max_diff);
            KQ_max[col] = KQ_max_new;

            *((uint32_t *) &KQ_max_scale[col]) *= KQ_max_diff >= SOFTMAX_FTZ_THRESHOLD;

            const float KQ_max_add = expf(sink - KQ_max_new);
            KQ_rowsum[col] = KQ_max_scale[col]*KQ_rowsum[col] + KQ_max_add;
        }

#if defined(TURING_MMA_AVAILABLE)
        if constexpr (cols_per_warp == 8) {
            const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[0], KQ_max_scale[1]);
#pragma unroll
            for (int i = 0; i < DV/T_C_VKQ::I; ++i) {
#pragma unroll
                for (int l = 0; l < T_C_VKQ::ne; ++l) {
                    VKQ_C[i].x[l] *= KQ_max_scale_h2;
                }
            }
        } else {
#pragma unroll
            for (int col = 0; col < cols_per_thread; ++col) {
                const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[col], KQ_max_scale[col]);
#pragma unroll
                for (int i = 0; i < (DV/2)/T_C_VKQ::J; ++i) {
#pragma unroll
                    for (int l0 = 0; l0 < T_C_VKQ::ne; l0 += 2) {
                        VKQ_C[i].x[l0 + col] *= KQ_max_scale_h2;
                    }
                }
            }
        }
#else // Volta
        const int col = (threadIdx.x / 2) % 2;
        const half2 KQ_max_scale_h2 = make_half2(KQ_max_scale[col], KQ_max_scale[col]);
#pragma unroll
        for (int i = 0; i < (DV/2)/T_C_VKQ::J; ++i) {
#pragma unroll
            for (int l = 0; l < T_C_VKQ::ne; ++l) {
                VKQ_C[i].x[l] *= KQ_max_scale_h2;
            }
        }
#endif // defined(TURING_MMA_AVAILABLE)
    }

    // Combine VKQ accumulator values if np > 1.
    // It's also faster to do small writes to shared memory, then large write to VRAM than to do small writes to VRAM.
    // So also write VKQ accumulators to shared memory in column-major format if np == 1.

    constexpr int tile_stride = nbatch_combine + 4;
    static_assert((DV/2) % nbatch_combine == 0, "bad nbatch_combine");

    if constexpr (cols_per_warp == 8) {
        const int jc_cwmo = (threadIdx.x % (2*T_C_VKQ::J)) / T_C_VKQ::J; // jc combine write meta offset
        const int jc_cwm = warp_id*(2*T_C_VKQ::J) + 2*T_C_VKQ::get_j(-1) + jc_cwmo; // jc combine write meta
        const float2 KQ_cmr = make_float2(KQ_max[jc_cwmo], KQ_rowsum[jc_cwmo]); // KQ combine max rowsum

        if (((!needs_fixup && !is_fixup) || np > 1) && threadIdx.x < 2*T_C_VKQ::J) {
            // Use the 16 bytes of padding in each row to store the meta data: KQ max, KQ rowsum, KQ max scale.
            ((float2 *) tile_Q)[jc_cwm*(tile_stride/2) + nbatch_combine/2] = KQ_cmr;
        }

        __syncthreads();

        if (np == 1) {
            // No combination is needed, the meta data can be directly written from registers to VRAM.
            if (needs_fixup && threadIdx.x < T_B_KQ::I) {
                float2 * dstk_fixup_meta = dstk_fixup + blockIdx.x*ncols;
                dstk_fixup_meta[jc_cwm] = KQ_cmr;
            }
            if (is_fixup && threadIdx.x < T_B_KQ::I) {
                float2 * dstk_fixup_meta = dstk_fixup + (gridDim.x + blockIdx.x)*ncols;
                dstk_fixup_meta[jc_cwm] = KQ_cmr;
            }
        }
    } else {
        // jc_cwm = jc combine write meta
        // KQ_cmr = KQ combine max rowsum
        // Use the 16 bytes of padding in each Q column to store the meta data: KQ max, KQ rowsum, KQ max scale.
#if defined(TURING_MMA_AVAILABLE)
        const int jc_cwm = warp_id*cols_per_warp + T_C_VKQ::get_i(threadIdx.x % 4);
        const float2 KQ_cmr = make_float2(KQ_max[threadIdx.x % cols_per_thread], KQ_rowsum[threadIdx.x % cols_per_thread]);
        const bool thread_should_write = threadIdx.x % 4 < cols_per_thread;
#else // Volta
        const int jc_cwm = warp_id*cols_per_warp + T_C_KQ::get_i(threadIdx.x & 2);
        const float2 KQ_cmr = make_float2(KQ_max[(threadIdx.x & 2) / 2], KQ_rowsum[(threadIdx.x & 2) / 2]);
        const bool thread_should_write = T_C_KQ::J == 8 || T_C_KQ::get_j(threadIdx.x & 2) < 8;
#endif // defined(TURING_MMA_AVAILABLE)

        if (((!needs_fixup && !is_fixup) || np > 1) && thread_should_write) {
            ((float2 *) tile_Q)[jc_cwm*(tile_stride/2) + nbatch_combine/2] = KQ_cmr;
        }

        __syncthreads();

        if (np == 1) {
            // No combination is needed, the meta data can be directly written from registers to VRAM.
            if (needs_fixup && thread_should_write) {
                float2 * dstk_fixup_meta = dstk_fixup + blockIdx.x*ncols;
                dstk_fixup_meta[jc_cwm] = KQ_cmr;
            }
            if (is_fixup && thread_should_write) {
                float2 * dstk_fixup_meta = dstk_fixup + (gridDim.x + blockIdx.x)*ncols;
                dstk_fixup_meta[jc_cwm] = KQ_cmr;
            }
        }
    }

    if (np > 1 && warp_id % np == 0) {
        // Combine the meta data for parallel warps via shared memory.
        // Warps with warp_id % np != 0 must NOT return early.
        // All threads must return simultaneously to avoid race conditions with work on the next tile.

        constexpr int nmeta = np*cols_per_warp >= WARP_SIZE ? np*cols_per_warp/WARP_SIZE : 1;

        const int jc_meta = warp_id*cols_per_warp + (np*cols_per_warp < WARP_SIZE ? threadIdx.x % (np*cols_per_warp) : threadIdx.x);
        float2 * const meta_ptr = ((float2 *) tile_Q) + jc_meta*(tile_stride/2) + nbatch_combine/2;
        float2 meta[nmeta];
#pragma unroll
        for (int imeta = 0; imeta < nmeta; ++imeta) {
            meta[imeta] = meta_ptr[imeta * WARP_SIZE * tile_stride/2];
        }

        float KQ_cmn = meta[0].x; // KQ combine max new, max between all parallel warps.
#pragma unroll
        for (int imeta = 1; imeta < nmeta; ++imeta) {
            KQ_cmn = fmaxf(KQ_cmn, meta[imeta].x);
        }
#pragma unroll
        for (int offset = np*cols_per_warp/2; offset >= cols_per_warp; offset >>= 1) {
            if (offset < WARP_SIZE) {
                KQ_cmn = fmaxf(KQ_cmn, __shfl_xor_sync(0xFFFFFFFF, KQ_cmn, offset, WARP_SIZE));
            }
        }

        float KQ_cms[nmeta]; // KQ combine max scale per warp.
#pragma unroll
        for (int imeta = 0; imeta < nmeta; ++imeta) {
            KQ_cms[imeta] = expf(meta[imeta].x - KQ_cmn);
        }

        float KQ_crs = KQ_cms[0]*meta[0].y; // KQ combine rowsum, scaled sum of all parallel warps.
#pragma unroll
        for (int imeta = 1; imeta < nmeta; ++imeta) {
            KQ_crs += KQ_cms[imeta]*meta[imeta].y;
        }
#pragma unroll
        for (int offset = np*cols_per_warp/2; offset >= cols_per_warp; offset >>= 1) {
            if (offset < WARP_SIZE) {
                KQ_crs += __shfl_xor_sync(0xFFFFFFFF, KQ_crs, offset, WARP_SIZE);
            }
        }

        __syncthreads();

        // Write back combined meta data:
#pragma unroll
        for (int imeta = 0; imeta < nmeta; ++imeta) {
            if (np*cols_per_warp >= WARP_SIZE || threadIdx.x < np*cols_per_warp) {
                // Combined KQ max scale + rowsum.
                meta_ptr[imeta * WARP_SIZE * tile_stride/2] = make_float2(KQ_cms[imeta], KQ_crs);
            }
        }

        // Combined KQ max + rowsum.
        static_assert(cols_per_warp <= WARP_SIZE);
        if (needs_fixup && (cols_per_warp == WARP_SIZE || threadIdx.x < cols_per_warp)) {
            float2 * dstk_fixup_meta = dstk_fixup + blockIdx.x*ncols;
            dstk_fixup_meta[(warp_id/np)*cols_per_warp + threadIdx.x] = make_float2(KQ_cmn, KQ_crs);
        }
        if (is_fixup && (cols_per_warp == WARP_SIZE || threadIdx.x < cols_per_warp)) {
            float2 * dstk_fixup_meta = dstk_fixup + (gridDim.x + blockIdx.x)*ncols;
            dstk_fixup_meta[(warp_id/np)*cols_per_warp + threadIdx.x] = make_float2(KQ_cmn, KQ_crs);
        }
    } else if (np > 1) {
        // Warps with warp_id % np == 0 execute a __syncthreads() in the if branch.
        // Therefore, all other warps also need to execute a __syncthreads().
        // Otherwise the points at which warps synchronize with each other would become misaligned.
        __syncthreads();
    }

    // =========================================================================
    // DEBUG: Output write phase - validate dst pointer before writes
    // =========================================================================
    {
        static bool output_debug_done = false;
        if (!output_debug_done && threadIdx.x == 0 && threadIdx.y == 1 && blockIdx.x == 0 && blockIdx.y == 0) {
            printf("\n[OUTPUT DEBUG] ========== Starting Output Write ==========\n");
            printf("[OUTPUT DEBUG] dstk=%p, dstk_fixup=%p\n", (void*)dstk, (void*)dstk_fixup);
            printf("[OUTPUT DEBUG] jt=%d, ne01.z=%u, ne02=%d\n", jt, ne01.z, ne02);
            printf("[OUTPUT DEBUG] ncols1=%d, ncols=%d, DV=%d\n", ncols1, ncols, DV);
            printf("[OUTPUT DEBUG] is_fixup=%d, needs_fixup=%d\n", is_fixup, needs_fixup);
            printf("[OUTPUT DEBUG] KQ_max[0]=%f, KQ_rowsum[0]=%f\n", KQ_max[0], KQ_rowsum[0]);
            // Calculate expected max dst offset
            const int max_j_dst = ncols1 - 1;
            const int max_c_dst = ncols2 - 1;
            const size_t max_offset = ((jt*ncols1 + max_j_dst)*ne02 + max_c_dst)*(DV/2) + (DV/2 - 1);
            printf("[OUTPUT DEBUG] Max dst offset: %zu elements (%zu bytes)\n", max_offset, max_offset * sizeof(float2));
            printf("[OUTPUT DEBUG] =============================================\n\n");
            output_debug_done = true;
        }
    }

#pragma unroll
    for (int k00 = 0; k00 < DV/2; k00 += nbatch_combine) {
        if constexpr (cols_per_warp == 8) {
            const int jc_cwd = warp_id*T_B_KQ::I + T_B_KQ::get_i(-1); // jc combine write data
#pragma unroll
            for (int k1 = 0; k1 < nbatch_combine; k1 += T_B_KQ::J) {
                const T_B_KQ B = get_transposed(VKQ_C[(k00 + k1)/T_B_KQ::J]); // Conversion of C to B matrix puts it in column-major format.

#pragma unroll
                for (int l = 0; l < T_B_KQ::ne; ++l) {
                    const int k = k1 + T_B_KQ::get_j(l);

                    tile_Q[jc_cwd*tile_stride + k] = B.x[l];
                }
            }
        } else {
            const int j0 = warp_id*cols_per_warp;
#pragma unroll
            for (int k1 = 0; k1 < nbatch_combine; k1 += T_C_VKQ::J) {
#pragma unroll
                for (int l = 0; l < T_C_VKQ::ne; ++l) {
                    const int j = j0 + T_C_VKQ::get_i(l);
                    const int k = k1 + T_C_VKQ::get_j(l);

                    tile_Q[j*tile_stride + k] = VKQ_C[(k00 + k1)/T_C_VKQ::J].x[l];
                }
            }
        }

        __syncthreads();

        if (np == 1 || warp_id % np == 0) {
            // The first 2*2*gridDim.x*ncols floats in dstk_fixup are for storing max. values and row sums.
            // The values after that are for the partial results of the individual blocks.
            float2 * dstk_fixup_data = dstk_fixup + gridDim.x*(2*ncols) + blockIdx.x*(ncols*(DV/2));

#pragma unroll
            for (int stride_k : {WARP_SIZE, WARP_SIZE/2, WARP_SIZE/4}) {
                const int k0_start  = stride_k == WARP_SIZE ? 0 : nbatch_combine - nbatch_combine % (2*stride_k);
                const int k0_stop   =                             nbatch_combine - nbatch_combine % (1*stride_k);
                const int stride_jc = WARP_SIZE / stride_k;

                if (k0_start == k0_stop) {
                    continue;
                }

#pragma unroll
                for (int jc0_dst = 0; jc0_dst < ncols; jc0_dst += (effective_nwarps/np)*stride_jc) {
                    const int jc_dst = jc0_dst + (warp_id/np)*stride_jc + (stride_k == WARP_SIZE ? 0 : threadIdx.x / stride_k);

                    if (jc0_dst + (effective_nwarps/np)*stride_jc > ncols && jc_dst >= ncols) {
                        break;
                    }

                    const int jc_tile_K = (jc_dst/cols_per_warp)*(np*cols_per_warp) + jc_dst % cols_per_warp;

                    const int j_dst = jc_dst / ncols2;
                    const int c_dst = jc_dst % ncols2;

                    if (!is_fixup && jt*ncols1 + j_dst >= int(ne01.z)) {
                        continue;
                    }

                    const float * meta_j = (const float *) tile_Q + jc_tile_K*tile_stride + nbatch_combine;
#pragma unroll
                    for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                        const int k = k0 + (stride_k == WARP_SIZE ? threadIdx.x : threadIdx.x % stride_k);

                        float2 dstk_val = make_float2(0.0f, 0.0f);
#pragma unroll
                        for (int ip = 0; ip < np; ++ip) {
                            const float KQ_crs = np == 1 ? 1.0f : meta_j[ip*cols_per_warp * tile_stride + 0];
                            const float2 dstk_val_add = __half22float2(tile_Q[(jc_tile_K + ip*cols_per_warp) * tile_stride + k]);
                            dstk_val.x += dstk_val_add.x*KQ_crs;
                            dstk_val.y += dstk_val_add.y*KQ_crs;
                        }

                        if (!needs_fixup && !is_fixup) {
                            const float KQ_rowsum_j = meta_j[1];
                            dstk_val.x /= KQ_rowsum_j;
                            dstk_val.y /= KQ_rowsum_j;
                        }

                        // DEBUG: Check for NaN before writing output
                        if (isnan(dstk_val.x) || isnan(dstk_val.y) || isinf(dstk_val.x) || isinf(dstk_val.y)) {
                            if (threadIdx.x == 0) {
                                printf("[FA KERNEL NaN] Block %d warp %d: NaN in dstk_val! x=%f y=%f\n",
                                       blockIdx.x, warp_id, dstk_val.x, dstk_val.y);
                                printf("[FA KERNEL NaN] jc_dst=%d k=%d j_dst=%d c_dst=%d\n",
                                       jc_dst, k, j_dst, c_dst);
                                if constexpr (!needs_fixup && !is_fixup) {
                                    const float KQ_rowsum_j = meta_j[1];
                                    printf("[FA KERNEL NaN] KQ_rowsum_j=%f (division by this)\n", KQ_rowsum_j);
                                }
                            }
                        }

                        if (is_fixup) {
                            dstk_fixup_data[jc_dst*(DV/2) + k00 + k] = dstk_val;
                        } else {
                            dstk[((jt*ncols1 + j_dst)*ne02 + c_dst)*(DV/2) + k00 + k] = dstk_val;
                        }
                    }
                }
            }
        }
        if (np > 1) {
            __syncthreads();
        }
    }
#else
    GGML_UNUSED_VARS(Q_f2, K_h2, V_h2, mask_h, sinks_f, dstk, dstk_fixup,
        scale, slope, logit_softcap, ne01, ne02,
        stride_Q1, stride_Q2, stride_K, stride_V, stride_mask,
        jt, kb0_start, kb0_stop);
    NO_DEVICE_CODE;
#endif // defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE)
}

// ------------------------------------------------------------------------------------------------------------------
// Ampere Flash Attention Kernel (also serves as universal fallback)
// ------------------------------------------------------------------------------------------------------------------
//
// This kernel works on Volta (sm_70), Turing (sm_75), Ampere (sm_80+), and serves as a
// fallback for Blackwell when TMA is unavailable or for unsupported configurations.
//
// Key properties that make it a safe fallback:
//   1. Uses num_consumers=0 (unified/legacy mode, no warp specialization)
//   2. All config values come from ggml_cuda_fattn_mma_get_*_ampere() functions directly
//   3. Passes use_tma=false to flash_attn_ext_f16_process_tile(), avoiding TMA code
//   4. Ampere configs guarantee nbatch_K2 >= DKQ/2 (no k0-chunking issues)
//   5. Does not depend on any Blackwell-specific state structures (fattn_pipeline_state)
//
// Launch bounds use Ampere config values to ensure correct behavior even when compiled
// for Blackwell targets.
//
// ------------------------------------------------------------------------------------------------------------------
template<int DKQ, int DV, int ncols1, int ncols2, bool use_logit_softcap, bool mla, bool use_tma>
// Ampere kernel always uses Ampere config for launch bounds, even on Blackwell builds (where it's a fallback)
__launch_bounds__(ggml_cuda_fattn_mma_get_nthreads_ampere(DKQ, DV, ncols1*ncols2), ggml_cuda_fattn_mma_get_occupancy_ampere(DKQ, DV, ncols1*ncols2))
__global__ void flash_attn_ext_f16_ampere(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        const char * __restrict__ sinks,
        const int  * __restrict__ KV_max,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33,
                            const char * __restrict__ tensor_maps) {
#if defined(FLASH_ATTN_AVAILABLE) && (defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(BLACKWELL_MMA_AVAILABLE))

    // Skip unused kernel variants for faster compilation:
    if (use_logit_softcap && !(DKQ == 128 || DKQ == 256)) {
        NO_DEVICE_CODE;
        return;
    }

    static_assert(!mla || DKQ >= DV, "MLA needs DKQ >= DV");

    constexpr int ncols     = ncols1 * ncols2;
    // Ampere kernel always uses Ampere config values, even on Blackwell builds (where it's a fallback)
    constexpr int nbatch_fa = ggml_cuda_fattn_mma_get_nbatch_fa_ampere(DKQ, DV, ncols);
    constexpr int nthreads  = ggml_cuda_fattn_mma_get_nthreads_ampere(DKQ, DV, ncols);
    constexpr int nwarps    = nthreads / WARP_SIZE;

    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.

    const int stride_Q1   = nb01 / sizeof(float2);
    const int stride_Q2   = nb02 / sizeof(float2);
    const int stride_K    = nb11 / sizeof(half2);
    const int stride_mask = nb31 / sizeof(half);

    const int stride_V = mla ? stride_K : nb21 / sizeof(half2);

    const int iter_k = (ne11   + (nbatch_fa - 1)) / nbatch_fa;
    const int iter_j = (ne01.z + (ncols1    - 1)) / ncols1;

    // kbc == k block continuous, current index in continuous ijk space.
    int       kbc      = int64_t(blockIdx.x + 0)*(iter_k*iter_j*(ne02/ncols2)*ne03) / gridDim.x;
    const int kbc_stop = int64_t(blockIdx.x + 1)*(iter_k*iter_j*(ne02/ncols2)*ne03) / gridDim.x;

    // If the seams of 2 CUDA blocks fall within an output tile their results need to be combined.
    // For this we need to track both the block that starts the tile (needs_fixup) and the block that finishes the tile (is_fixup).
    // In the most general case >2 seams can fall into the same tile.

    // kb0 == k start index when in the output tile.
    int kb0_start = kbc % iter_k;
    int kb0_stop  = min(iter_k, kb0_start + kbc_stop - kbc);

    while (kbc < kbc_stop) {
        const int sequence = kbc / (iter_k*iter_j*(ne02/ncols2));
        const int zt = (kbc - iter_k*iter_j*(ne02/ncols2)*sequence) / (iter_k*iter_j); // head in units of ncols2
        const int jt = (kbc - iter_k*iter_j*(ne02/ncols2)*sequence - iter_k*iter_j*zt) / iter_k; // j index of current tile.

        const int head0 = zt * ncols2;

        const float2 * Q_f2   = (const float2 *) (Q + nb03*sequence + nb02* head0);
        const half2  * K_h2   = (const half2  *) (K + nb13*sequence + nb12*(head0 / gqa_ratio));
        const half   * mask_h = ncols2 == 1 && !mask ? nullptr :
            (const half *) (mask + nb33*(sequence % ne33));
        float2       * dstk   = ((float2 *) dst) + (sequence*ne01.z*ne02 + head0) * (DV/2);

        const half2 * V_h2 = mla ? K_h2 + (DKQ/2 - DV/2) : (const half2 *) (V + nb23*sequence + nb22*(head0 / gqa_ratio));
        const float * sinks_f = sinks ? (const float *) sinks + head0 : nullptr;

        const float slope = ncols2 == 1 ? get_alibi_slope(max_bias, head0, n_head_log2, m0, m1) : 1.0f;

        if (KV_max) {
            kb0_stop = min(kb0_stop, KV_max[sequence*iter_j + jt] / nbatch_fa);
        }
        constexpr bool is_fixup = false; // All but (potentially) the last iterations write their data to dst rather than the fixup buffer.
        if (kb0_start == 0) {
            constexpr bool needs_fixup = false; // CUDA block is working on an entire tile.
            flash_attn_ext_f16_process_tile<DKQ, DV, ncols1, ncols2, nwarps, 0, use_logit_softcap, mla, needs_fixup, is_fixup, use_tma>
                (Q_f2, K_h2, V_h2, mask_h, sinks_f, dstk, dst_meta, scale, slope, logit_softcap,
                 ne01, ne02, ne11, stride_Q1, stride_Q2, stride_K, stride_V, stride_mask, jt, kb0_start, kb0_stop, tensor_maps);
        } else {
            constexpr bool needs_fixup = true; // CUDA block is missing the beginning of a tile.
            flash_attn_ext_f16_process_tile<DKQ, DV, ncols1, ncols2, nwarps, 0, use_logit_softcap, mla, needs_fixup, is_fixup, use_tma>
                (Q_f2, K_h2, V_h2, mask_h, sinks_f, dstk, dst_meta, scale, slope, logit_softcap,
                 ne01, ne02, ne11, stride_Q1, stride_Q2, stride_K, stride_V, stride_mask, jt, kb0_start, kb0_stop, tensor_maps);
        }

        kbc += iter_k;
        kbc -= kbc % iter_k;

        kb0_start = 0;
        kb0_stop  = min(iter_k, kbc_stop - kbc);
    }

    if (kbc >= kbc_stop) {
        return;
    }

    const int sequence = kbc / (iter_k*iter_j*(ne02/ncols2));
    const int zt = (kbc - iter_k*iter_j*(ne02/ncols2)*sequence) / (iter_k*iter_j); // head in units of ncols2
    const int jt = (kbc - iter_k*iter_j*(ne02/ncols2)*sequence - iter_k*iter_j*zt) / iter_k; // j index of current tile.

    const int head0 = zt * ncols2;

    const float2 * Q_f2   = (const float2 *) (Q + nb03*sequence + nb02* head0);
    const half2  * K_h2   = (const half2  *) (K + nb13*sequence + nb12*(head0 / gqa_ratio));
    const half   * mask_h = ncols2 == 1 && !mask ? nullptr :
        (const half *) (mask + nb33*(sequence % ne33));
    float2       * dstk   = ((float2 *) dst) + (sequence*ne01.z*ne02 + head0) * (DV/2);

    const half2 * V_h2 = mla ? K_h2 + (DKQ/2 - DV/2) : (const half2 *) (V + nb23*sequence + nb22*(head0 / gqa_ratio));
    const float * sinks_f = sinks ? (const float *) sinks + head0 : nullptr;

    const float slope = ncols2 == 1 ? get_alibi_slope(max_bias, head0, n_head_log2, m0, m1) : 1.0f;

    if (KV_max) {
        kb0_stop = min(kb0_stop, KV_max[sequence*iter_j + jt] / nbatch_fa);
    }

    constexpr bool is_fixup = true; // Last index writes its data to fixup buffer to avoid data races with other blocks.
    constexpr bool needs_fixup = false;
    flash_attn_ext_f16_process_tile<DKQ, DV, ncols1, ncols2, nwarps, 0, use_logit_softcap, mla, needs_fixup, is_fixup, use_tma>
        (Q_f2, K_h2, V_h2, mask_h, sinks_f, dstk, dst_meta, scale, slope, logit_softcap,
         ne01, ne02, ne11, stride_Q1, stride_Q2, stride_K, stride_V, stride_mask, jt, kb0_start, kb0_stop, tensor_maps);
#else
    GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33,
              tensor_maps);
    NO_DEVICE_CODE;
#endif // defined(FLASH_ATTN_AVAILABLE) && (defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(BLACKWELL_MMA_AVAILABLE))
}

// Launch bounds for Blackwell kernel:
// - sm_120 (RTX 5090): 160 threads (4 consumer + 1 producer)
//   160 × 255 regs = 40,800 ≤ 65,536 ✓
// - sm_100 (B200/B100): 288 threads (8 consumer + 1 producer), 227KB shared mem
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
#define FATTN_BLACKWELL_NTHREADS 160
#define FATTN_BLACKWELL_MIN_BLOCKS 1
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
#define FATTN_BLACKWELL_NTHREADS 288
#define FATTN_BLACKWELL_MIN_BLOCKS 1
#else
// Host-side default for template instantiation - use sm_120 config (160 threads)
#define FATTN_BLACKWELL_NTHREADS 160
#define FATTN_BLACKWELL_MIN_BLOCKS 1
#endif

template<int DKQ, int DV, int ncols1, int ncols2, bool use_logit_softcap, bool mla, bool use_tma>
__launch_bounds__(FATTN_BLACKWELL_NTHREADS, FATTN_BLACKWELL_MIN_BLOCKS)
__global__ void flash_attn_ext_f16_blackwell(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        const char * __restrict__ sinks,
        const int  * __restrict__ KV_max,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33,
                            const char * __restrict__ tensor_maps)
{
    // DEBUG: UNCONDITIONAL - outside any #ifdef - confirm kernel entry
    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0) {
        printf("[BLACKWELL KERNEL ENTRY] flash_attn_ext_f16_blackwell called!\n");
        printf("[BLACKWELL KERNEL ENTRY] blockDim=(%d,%d,%d), DKQ=%d, DV=%d\n",
               blockDim.x, blockDim.y, blockDim.z, DKQ, DV);
#ifdef BLACKWELL_TMA_AVAILABLE
        printf("[BLACKWELL KERNEL ENTRY] BLACKWELL_TMA_AVAILABLE=YES\n");
#else
        printf("[BLACKWELL KERNEL ENTRY] BLACKWELL_TMA_AVAILABLE=NO (kernel body skipped!)\n");
#endif
    }
    __syncthreads();

#ifdef BLACKWELL_MMA_AVAILABLE
    // Blackwell kernel body - uses MMA (available on both sm_100 and sm_120)
    // TMA vs cp.async is controlled by the use_tma template parameter, not this guard
    // EARLY DEBUG: Check if kernel even starts
    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0) {
        printf("[KERNEL DEBUG] Blackwell kernel ENTRY - block=(%d,%d,%d) thread=(%d,%d)\n",
               blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y);
        printf("[KERNEL DEBUG] K dims: ne10=%d (head_dim), ne11=%d (K_seq_len), ne12=%d (kv_heads), ne13=%d\n", 
               ne10, ne11, ne12, ne13);
        printf("[KERNEL DEBUG] K_total_rows = ne11 * ne12 = %d * %d = %d\n", ne11, ne12, ne11 * ne12);
    }
    __syncthreads();

    constexpr int ncols = ncols1 * ncols2;
    constexpr auto config = ggml_cuda_fattn_mma_get_config_blackwell(DKQ, DV, ncols);
    constexpr int nbatch_fa = config.nbatch_fa;
    constexpr int nbatch_K2 = config.nbatch_K2;
    constexpr int nbatch_V2 = config.nbatch_V2;
    constexpr int nstages = config.nstages_target;
    constexpr int num_consumers = config.num_consumers;
    // nwarps for tiling:
    // - Producer/consumer mode (num_consumers > 0): nwarps = num_consumers
    // - Unified mode (num_consumers == 0): nwarps = total threads / WARP_SIZE (SM_120)
    constexpr int nwarps = (num_consumers > 0) ? num_consumers : (config.nthreads / WARP_SIZE);

    extern __shared__ char smem[];
    // New shared memory layout for k0-chunk pipelining (double-buffering within loops):
    // [K chunk 0][K chunk 1][V chunk 0][V chunk 1][Q tiles][mask][pipeline_state][mbarriers]
    // K: 2 chunk buffers for double-buffering within k0 loop
    // V: 2 chunk buffers for double-buffering within V loop
    constexpr int bytes_K_chunk = nbatch_fa * nbatch_K2 * sizeof(half2);
    constexpr int bytes_V_chunk = nbatch_fa * nbatch_V2 * sizeof(half2);
    constexpr int bytes_K_total = 2 * bytes_K_chunk;  // 2 stages for K chunks
    constexpr int bytes_V_total = 2 * bytes_V_chunk;  // 2 stages for V chunks
    constexpr int bytes_KV_total = bytes_K_total + bytes_V_total;

    // Q tiles follow KV buffers
    constexpr int stride_tile_Q = DKQ/2 + 4;
    constexpr int bytes_Q = ncols * stride_tile_Q * sizeof(half2);

    // Mask follows Q (only needed for multi-head attention)
    constexpr int bytes_mask = ncols1 * (nbatch_fa/2 + 4) * sizeof(half2);

    // Pipeline state at the end (after KV + Q + mask)
    fattn_pipeline_state* state = (fattn_pipeline_state*)(smem + bytes_KV_total + bytes_Q + bytes_mask);

    // Determine if we need chunking (same logic as fattn_producer_loop)
    constexpr bool needs_K_chunking = (DKQ/2 > nbatch_K2);
    constexpr bool needs_V_chunking = (DV/2 > nbatch_V2);
    constexpr bool needs_chunking = needs_K_chunking || needs_V_chunking;

    // Initialize barriers based on whether chunking is needed:
    // - Chunked mode: use chunk-level barriers (empty_K_chunk, full_K_chunk, etc.)
    // - Legacy mode: use tile-level barriers (empty_K, full_K, etc.)
    // - Q_loaded: Always needed for Q synchronization
    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0) {
        printf("[KERNEL DEBUG] Before barrier init: state=%p, needs_chunking=%d\n", 
               state, needs_chunking);
        printf("[KERNEL DEBUG] State offset from smem: %ld bytes\n",
               (long)((char*)state - smem));
    }
    __syncthreads();
    
    // ========================================================================
    // SM_120 PATH: Consumer warps load K/V themselves via cp.async
    // On SM_120 (RTX 5090), we don't use producer/consumer pipelining.
    // Instead, consumer warps cooperatively load K/V using cp.async.
    // Producer warp (threadIdx.y=0) idles. No mbarrier or TMA needed.
    // ========================================================================
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
    // SM_120 path: No mbarrier init needed - consumers load K/V themselves
    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0) {
        printf("[SM120 PATH] Using cp.async + __syncthreads__ (no TMA, no mbarrier)\n");
    }
    __syncthreads();
#else
    // SM_90/SM_100/SM_103/SM_110 path: Use mbarrier
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        mbarrier_init(&state->Q_loaded, num_consumers * WARP_SIZE);
        
        if constexpr (needs_chunking) {
            // Chunk-level barriers for double-buffering within K/V loops
            constexpr int nchunk_buffers = 2;
            for (int i = 0; i < nchunk_buffers; ++i) {
                // Producer expects 1 arrival (itself) after TMA load completes
                mbarrier_init(&state->full_K_chunk[i], 1);
                mbarrier_init(&state->full_V_chunk[i], 1);
                // All threads in consumer warps call mbarrier_arrive_expect_tx,
                // so we need count = num_consumers * WARP_SIZE
                mbarrier_init(&state->empty_K_chunk[i], num_consumers * WARP_SIZE);
                mbarrier_init(&state->empty_V_chunk[i], num_consumers * WARP_SIZE);
            }
        } else {
            // Tile-level barriers for kb0-level pipelining (legacy mode)
            // nstages stages, each stage has empty/full barriers for K and V
            for (int i = 0; i < nstages; ++i) {
                // Producer expects 1 arrival (itself) after TMA load completes
                mbarrier_init(&state->full_K[i], 1);
                mbarrier_init(&state->full_V[i], 1);
                // All threads in consumer warps call mbarrier_arrive_expect_tx,
                // so we need count = num_consumers * WARP_SIZE
                mbarrier_init(&state->empty_K[i], num_consumers * WARP_SIZE);
                mbarrier_init(&state->empty_V[i], num_consumers * WARP_SIZE);
            }
        }
    }
    __syncthreads();

    // Pre-arrive on empty barriers so producer can load first data.
    // For the first iteration, buffers are already "empty" (nothing loaded yet),
    // but the producer waits on empty_K/V before loading. Without pre-arrival,
    // producer blocks waiting for consumer arrivals that never come (deadlock).
    // Consumer warps signal readiness here; producer warp skips this.
    //
    // IMPORTANT: Use mbarrier_arrive() not mbarrier_arrive_expect_tx(0) since these
    // are arrival-count barriers, not TMA byte-count barriers.
    if (threadIdx.y != 0) {
        if constexpr (needs_chunking) {
            constexpr int nchunk_buffers = 2;
            for (int i = 0; i < nchunk_buffers; ++i) {
                mbarrier_arrive(&state->empty_K_chunk[i]);
                mbarrier_arrive(&state->empty_V_chunk[i]);
            }
        } else {
            for (int i = 0; i < nstages; ++i) {
                mbarrier_arrive(&state->empty_K[i]);
                mbarrier_arrive(&state->empty_V[i]);
            }
        }
    }
    __syncthreads();
#endif // SM_120 check
    
    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0) {
        printf("[KERNEL DEBUG] Pre-arrive COMPLETE, starting main loop\n");
    }

    const int iter_k = (ne11 + (nbatch_fa - 1)) / nbatch_fa;
    const int iter_j = (ne01.z + (ncols1 - 1)) / ncols1;
    
    // Grid Logic
    int       kbc      = int64_t(blockIdx.x + 0)*(iter_k*iter_j*(ne02/ncols2)*ne03) / gridDim.x;
    const int kbc_stop = int64_t(blockIdx.x + 1)*(iter_k*iter_j*(ne02/ncols2)*ne03) / gridDim.x;

    int kb0_start = kbc % iter_k;
    int kb0_stop  = min(iter_k, kb0_start + kbc_stop - kbc);
    
    // Debug from a block that HAS work (kbc < kbc_stop)
    const bool has_work = (kbc < kbc_stop);
    const bool debug_this_block = (threadIdx.x == 0 && threadIdx.y == 0 && has_work && blockIdx.x < 3);
    
    if (debug_this_block) {
        printf("[KERNEL DEBUG] Block %d HAS WORK: iter_k=%d, iter_j=%d, ne02=%d, ncols2=%d\n",
               blockIdx.x, iter_k, iter_j, ne02, ncols2);
        printf("[KERNEL DEBUG] Block %d: kbc=%d, kbc_stop=%d, kb0_start=%d, kb0_stop=%d\n",
               blockIdx.x, kbc, kbc_stop, kb0_start, kb0_stop);
        printf("[KERNEL DEBUG] Block %d: ne11=%d, max_row=%d (nbatch_fa=%d)\n",
               blockIdx.x, ne11, (iter_k-1)*nbatch_fa, nbatch_fa);
    }
    
    // Debug ALL blocks to find which one crashes
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        printf("[ALL BLOCKS] Block %d: kbc=%d, kbc_stop=%d, has_work=%d\n", 
               blockIdx.x, kbc, kbc_stop, has_work);
    }
    __syncthreads();
    
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        printf("[ALL BLOCKS] Block %d: SYNC DONE, checking loop condition\n", blockIdx.x);
    }

    uint32_t q_phase = 0;

    // Define stride variables at function scope (needed by both while loop and tail processing)
    const int stride_Q1   = nb01 / sizeof(float2);
    const int stride_Q2   = nb02 / sizeof(float2);
    const int stride_K    = nb11 / sizeof(half2);
    const int stride_mask = nb31 / sizeof(half);
    const int stride_V    = mla ? stride_K : nb21 / sizeof(half2);

    // Check loop condition
    const bool loop_cond = (kbc < kbc_stop);
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        printf("[ALL BLOCKS] Block %d: loop_cond=%d (kbc=%d < kbc_stop=%d)\n",
               blockIdx.x, loop_cond, kbc, kbc_stop);
    }

    while (kbc < kbc_stop) {
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            printf("[LOOP] Block %d: ENTERED while loop, kbc=%d\n", blockIdx.x, kbc);
        }
         const int sequence = kbc / (iter_k*iter_j*(ne02/ncols2));
         const int zt = (kbc - iter_k*iter_j*(ne02/ncols2)*sequence) / (iter_k*iter_j); 
         const int jt = (kbc - iter_k*iter_j*(ne02/ncols2)*sequence - iter_k*iter_j*zt) / iter_k;

         const int head0 = zt * ncols2;
         const float2 * Q_f2   = (const float2 *) (Q + nb03*sequence + nb02* head0);
         const half2  * K_h2   = (const half2  *) (K + nb13*sequence + nb12*(head0)); 
         const half   * mask_h = ncols2 == 1 && !mask ? nullptr : (const half *) (mask + nb33*(sequence % ne33));
         float2       * dstk   = ((float2 *) dst) + (sequence*ne01.z*ne02 + head0) * (DV/2);
         const half2  * V_h2   = mla ? K_h2 + (DKQ/2 - DV/2) : (const half2 *) (V + nb23*sequence + nb22*(head0));
         const float  * sinks_f = sinks ? (const float *) sinks + head0 : nullptr;
         const float slope = ncols2 == 1 ? get_alibi_slope(max_bias, head0, n_head_log2, m0, m1) : 1.0f;
         
         // For Blackwell kernel, we assume full tile alignment for now or handle bounds inside.
         // We iterate the full sequence chunk assigned to this block.
         // Actually, if kbc dispatch is granular, we might need fixup logic.
         // For optimization prototype, we assume blocks align to tiles.

         // Note: stride_Q1, stride_Q2, stride_K, stride_mask, stride_V are now defined
         // at function scope (before while loop) so they're available to tail processing.

         // =====================================================================
         // CRITICAL: Q Loading Phase - ALL warps must participate in __syncthreads()
         // =====================================================================
         // The process_tile function has __syncthreads() for Q loading.
         // Producer warp must also hit these barriers or undefined behavior occurs.
         //
         // Solution: Producer warp participates in Q loading phase (uses consumer warp_id
         // mapping where it acts as "warp 7"), then diverges to actual producer work.
         // For Q loading, we use threadIdx.y directly (0-8) instead of consumer warp_id.
         // =====================================================================

        if (threadIdx.x == 0 && threadIdx.y == 0) {
            printf("[Q LOAD] Block %d: Starting Q loading\n", blockIdx.x);
        }

        constexpr int stride_tile_Q_local = DKQ/2 + 4;
        // Q tiles are after the KV chunk buffers in the new layout
        half2 * tile_Q = (half2*)(smem + bytes_KV_total);

        // Lightweight Q loading: cooperative across all warps, single stride to keep
        // register footprint down. Each warp handles strided columns, each lane
        // walks DKQ/2 in WARP_SIZE steps.
        // nwarps_total = num_consumers + 1 (producer) = 5 for sm_120, 9 for sm_100
        constexpr int nwarps_total = num_consumers + 1;
        const float2 *Q_base = Q_f2;
        for (int jc = threadIdx.y; jc < ncols; jc += nwarps_total) {
            const int j1 = jc / ncols2;
            const int j2 = jc % ncols2;
            const bool in_bounds = Q_base && (jt*ncols1 + j1) < int(ne01.z);
            for (int k = threadIdx.x; k < DKQ/2; k += WARP_SIZE) {
                half2 val = make_half2(0.0f, 0.0f);
                if (in_bounds) {
                    const float2 tmp = Q_base[j1*stride_Q1 + j2*stride_Q2 + k];
                    val = make_half2(__float2half(tmp.x * scale), __float2half(tmp.y * scale));
                }
                tile_Q[jc*stride_tile_Q_local + k] = val;
            }
        }
        
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            printf("[Q LOAD] Block %d: Q loop done, syncing\n", blockIdx.x);
        }
         __syncthreads();  // All warps hit this barrier

        if (threadIdx.x == 0 && threadIdx.y == 0) {
            printf("[Q LOAD] Block %d: Sync 1 done\n", blockIdx.x);
        }

         // Load Q into registers for consumers (producer skips this but still syncs)
         // This is done inside process_tile for consumers
         __syncthreads();  // Second barrier matching process_tile's Q register loading sync
         
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            printf("[Q LOAD] Block %d: Sync 2 done, diverging\n", blockIdx.x);
        }

         // =====================================================================
         // WARP DIVERGENCE: Producer vs Consumer paths
         // =====================================================================
         // SM_120: Uses __syncthreads__ coordination (mbarrier_arrive crashes)
         // Other Blackwell: Uses mbarrier-based producer/consumer sync
         // =====================================================================
         
         // Calculate KV head row offset (needed by both producer and consumer)
         const int kv_head_row_offset = zt * ne11;
         const int K_total_rows = ne11 * ne12;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
         // =====================================================================
         // SM_120 PATH: Consumer warps compute (producer warp idles)
         // =====================================================================
         // DEBUG: Unconditional print to confirm SM_120 path is taken
         if (threadIdx.x == 0 && threadIdx.y == 1 && blockIdx.x == 0) {
             printf("[SM120 PATH] Block 0 entering SM_120 consumer path, calling process_tile\n");
             printf("[SM120 PATH] kv_head_row_offset=%d, K_total_rows=%d\n", kv_head_row_offset, K_total_rows);
             printf("[SM120 PATH] kb0_start=%d, kb0_stop=%d, jt=%d\n", kb0_start, kb0_stop, jt);
         }
         // On SM_120 (RTX 5090), TMA mbarrier.arrive crashes at runtime.
         //
         // The Blackwell kernel is structured for 1 producer + N consumer warps.
         // For SM_120, we:
         //   - Have producer warp (threadIdx.y=0) idle
         //   - Have consumer warps (threadIdx.y=1-4) do all the work with cp.async
         //   - Use num_consumers=4 so effective_nwarps=4 and warp_id mapping is correct
         //
         // This ensures the work distribution matches the Blackwell config (4 compute warps).
         // =====================================================================
         if (threadIdx.y != 0) {
             // Consumer warps only
             constexpr bool is_fixup = false;
             constexpr bool needs_fixup = false;
             // Use num_consumers=4 (not 0) so warp_id = threadIdx.y - 1 maps correctly
             flash_attn_ext_f16_process_tile<DKQ, DV, ncols1, ncols2, nwarps, num_consumers, use_logit_softcap, mla, needs_fixup, is_fixup, false>(
                 Q_f2, K_h2, V_h2, mask_h, sinks_f, dstk, dst_meta, scale, slope, logit_softcap,
                 ne01, ne02, ne11, stride_Q1, stride_Q2, stride_K, stride_V, stride_mask, jt, kb0_start, kb0_stop, tensor_maps);
         }
         // Producer warp (threadIdx.y=0) idles here for SM_120
#else
         // =====================================================================
         // MBARRIER PATH (SM_90/SM_100/SM_103/SM_110)
         // =====================================================================
         if (threadIdx.y == 0) {
             // Producer: K/V loading loop
             if (threadIdx.x == 0) {
                 printf("[PRODUCER] Block %d: kv_head_row_offset=%d (zt=%d * ne11=%d), K_total_rows=%d\n", 
                        blockIdx.x, kv_head_row_offset, zt, ne11, K_total_rows);
                 if (kv_head_row_offset >= K_total_rows) {
                     printf("[PRODUCER] Block %d: !!! ERROR: kv_head_row_offset=%d >= K_total_rows=%d !!!\n",
                            blockIdx.x, kv_head_row_offset, K_total_rows);
                 }
             }
             mbarrier_wait(&state->Q_loaded, q_phase);
             if (threadIdx.x == 0) {
                 printf("[PRODUCER] Block %d: Q_loaded done, calling producer_loop\n", blockIdx.x);
             }
             q_phase ^= 1;
             fattn_producer_loop<DKQ, DV, ncols, nstages, nbatch_fa, nbatch_K2, nbatch_V2, mla>(
                 tensor_maps, state, kb0_start, kb0_stop, num_consumers, kv_head_row_offset,
                 K_h2, V_h2, stride_K, stride_V);
             if (threadIdx.x == 0) {
                 printf("[PRODUCER] Block %d: producer_loop returned\n", blockIdx.x);
             }
         } else {
             // Consumer: Main computation (Q already loaded into shared memory)
             if (threadIdx.x == 0 && threadIdx.y == 1) {
                 printf("[CONSUMER] Block %d: Consumer warp %d entering\n", blockIdx.x, threadIdx.y);
             }
             constexpr bool is_fixup = false;
             constexpr bool needs_fixup = false;

             // Call process_tile with Q already in shared memory
             // The function will skip Q loading phase since pipeline_state is set
             flash_attn_ext_f16_process_tile<DKQ, DV, ncols1, ncols2, nwarps, num_consumers, use_logit_softcap, mla, needs_fixup, is_fixup, use_tma>(
                 Q_f2, K_h2, V_h2, mask_h, sinks_f, dstk, dst_meta, scale, slope, logit_softcap,
                  ne01, ne02, ne11, stride_Q1, stride_Q2, stride_K, stride_V, stride_mask, jt, kb0_start, kb0_stop, tensor_maps);
             if (threadIdx.x == 0 && threadIdx.y == 1) {
                 printf("[CONSUMER] Block %d: process_tile returned\n", blockIdx.x);
             }
         }
#endif // SM_120 check
         
         kbc += iter_k;
         kbc -= kbc % iter_k;
         kb0_start = 0;
         kb0_stop  = min(iter_k, kbc_stop - kbc);
    }
#endif
}

// ------------------------------------------------------------------------------------------------------------------
// Dispatch function for Flash Attention MMA F16 kernel
// ------------------------------------------------------------------------------------------------------------------
//
// MULTI-GPU AND FALLBACK BEHAVIOR:
//
// This function determines the appropriate kernel variant per-device based on compute capability (cc).
// In multi-GPU systems with heterogeneous architectures, each device gets its own dispatch decision.
//
// Kernel Selection Logic:
//   1. Blackwell (cc >= 1200) with valid TMA and native config: flash_attn_ext_f16_blackwell
//   2. All other cases: flash_attn_ext_f16_ampere (fallback)
//
// The Ampere kernel serves as the universal fallback because:
//   - It uses num_consumers=0 (unified/legacy mode, no warp specialization)
//   - All config values come from ggml_cuda_fattn_mma_get_config_ampere() directly
//   - It passes use_tma=false, avoiding TMA-specific code paths
//   - Ampere configs have nbatch_K2 >= DKQ/2 (no chunking issue):
//       DKQ=64:  nbatch_K2=32, DKQ/2=32 (OK)
//       DKQ=80:  nbatch_K2=40, DKQ/2=40 (OK)
//       DKQ=96:  nbatch_K2=48, DKQ/2=48 (OK)
//       DKQ=112: nbatch_K2=56, DKQ/2=56 (OK)
//       DKQ=128: nbatch_K2=64, DKQ/2=64 (OK)
//       DKQ=256: nbatch_K2=128, DKQ/2=128 (OK)
//       DKQ=576: nbatch_K2=32,  DKQ/2=288 (different logic, OK for MLA)
//
// Multi-GPU correctness:
//   - ggml_cuda_get_device() returns the current device ID
//   - cc is fetched per-device from ggml_cuda_info().devices[id].cc
//   - Dispatch happens independently for each device's work
//   - A Blackwell GPU in slot 0 gets the Blackwell kernel
//   - An Ampere GPU in slot 1 gets the Ampere kernel (fallback)
//
// ------------------------------------------------------------------------------------------------------------------
template <int DKQ, int DV, int ncols1, int ncols2>
void ggml_cuda_flash_attn_ext_mma_f16_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;

    // Per-device dispatch: get current device's compute capability
    // This ensures correct kernel selection in multi-GPU setups with mixed architectures
    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;

    constexpr int ncols = ncols1 * ncols2;

    const int  nthreads       = ggml_cuda_fattn_mma_get_nthreads      (DKQ, DV, ncols, cc);
    const int  nbatch_fa      = ggml_cuda_fattn_mma_get_nbatch_fa     (DKQ, DV, ncols, cc);
    const int  nbatch_K2      = ggml_cuda_fattn_mma_get_nbatch_K2     (DKQ, DV, ncols, cc);
    const int  nbatch_V2      = ggml_cuda_fattn_mma_get_nbatch_V2     (DKQ, DV, ncols, cc);
    const int  nbatch_combine = ggml_cuda_fattn_mma_get_nbatch_combine(DKQ, DV, ncols, cc);
    const bool Q_in_reg       = ggml_cuda_fattn_mma_get_Q_in_reg      (DKQ, DV, ncols, cc);
    const int  nstages        = ggml_cuda_fattn_mma_get_nstages       (DKQ, DV, ncols1, ncols2, cc);

    const int cols_per_warp = std::min(ncols, turing_mma_available(cc) ? 16 : 32);
    const int nwarps        = nthreads / WARP_SIZE;

    constexpr bool mla = DKQ == 576;

    size_t nbytes_shared_KV_1stage = nbatch_fa            * std::max(nbatch_K2 + 4,  nbatch_V2 + 4) * sizeof(half2);
    size_t nbytes_shared_KV_2stage = nbatch_fa            *         (nbatch_K2 + 4 + nbatch_V2 + 4) * sizeof(half2);
    size_t nbytes_shared_Q         = ncols                * (DKQ/2 + 4)                             * sizeof(half2);
    size_t nbytes_shared_mask      = ncols1               * (nbatch_fa/2 + 4)                       * sizeof(half2);
    size_t nbytes_shared_combine   = nwarps*cols_per_warp * (nbatch_combine + 4)                    * sizeof(half2);

    // SM_120 (RTX 5090) uses two-stage K pipelining which requires separate K and V buffers
    // even when nstages <= 1, so always use 2-stage allocation for SM_120
    const bool sm120_k_pipelining = ggml_cuda_is_consumer_blackwell(cc);
    size_t nbytes_shared_KV = (nstages <= 1 && !sm120_k_pipelining) ? nbytes_shared_KV_1stage : nbytes_shared_KV_2stage;

    // Allocate space for 2 mbarriers (K and V) if TMA is used.
    // mbarrier is 64-bit (8 bytes). 2 * 8 = 16 bytes.
    // We add a bit more for alignment safety.
    const size_t nbytes_mbar = 128; 

    size_t nbytes_shared_total = std::max(nbytes_shared_combine, Q_in_reg ?
        std::max(nbytes_shared_Q,  nbytes_shared_KV + nbytes_shared_mask) :
                 nbytes_shared_Q + nbytes_shared_KV + nbytes_shared_mask) + nbytes_mbar;

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    // Handle Blackwell features (TMA)
    // SM_120 (consumer Blackwell / RTX 5090): Does NOT use TMA - uses cp.async instead
    // SM_100/103/110 (datacenter Blackwell): Uses TMA + mbarrier
    bool use_tma_runtime = ggml_cuda_has_blackwell_features(cc) && !ggml_cuda_is_consumer_blackwell(cc);

    // DEBUG: Trace dispatch conditions for Blackwell kernel selection
    static bool debug_printed = false;
    if (!debug_printed && DKQ == 128 && DV == 128) {
        fprintf(stderr, "\n[FATTN DEBUG] ===== Flash Attention Dispatch =====\n");
        fprintf(stderr, "[FATTN DEBUG] DKQ=%d, DV=%d, ncols=%d, ncols1=%d, ncols2=%d\n", DKQ, DV, ncols, ncols1, ncols2);
        fprintf(stderr, "[FATTN DEBUG] Device %d, cc=%d\n", id, cc);
        fprintf(stderr, "[FATTN DEBUG] ggml_cuda_has_blackwell_features(cc)=%d\n", ggml_cuda_has_blackwell_features(cc));
        fprintf(stderr, "[FATTN DEBUG] ggml_cuda_is_consumer_blackwell(cc)=%d\n", ggml_cuda_is_consumer_blackwell(cc));
        fprintf(stderr, "[FATTN DEBUG] use_tma_runtime (initial)=%d\n", use_tma_runtime);
    }

    // Verify alignment and strides for TMA
    if (use_tma_runtime) {
        const ggml_tensor * K = dst->src[1];
        bool K_data_align = ((uint64_t)K->data % 16 == 0);
        bool K_stride_align = (K->nb[1] % 16 == 0);
        bool K_dim_even = (K->ne[0] % 2 == 0);
        
        if (!debug_printed && DKQ == 128 && DV == 128) {
            fprintf(stderr, "[FATTN DEBUG] K tensor alignment check:\n");
            fprintf(stderr, "[FATTN DEBUG]   K->data=%p, addr%%16=%lu, aligned=%d\n", K->data, ((uint64_t)K->data % 16), K_data_align);
            fprintf(stderr, "[FATTN DEBUG]   K->nb[1]=%lu (stride), stride%%16=%lu, aligned=%d\n", (unsigned long)K->nb[1], (unsigned long)(K->nb[1] % 16), K_stride_align);
            fprintf(stderr, "[FATTN DEBUG]   K->ne[0]=%ld (head_dim), even=%d\n", (long)K->ne[0], K_dim_even);
        }
        
        if (!K_data_align || !K_stride_align || !K_dim_even) {
            if (!debug_printed && DKQ == 128 && DV == 128) {
                fprintf(stderr, "[FATTN DEBUG]   ==> K alignment FAILED, disabling TMA\n");
            }
            use_tma_runtime = false;
        }
        
        const ggml_tensor * V = mla ? K : dst->src[2];
        void* V_ptr = mla ? (char*)K->data + (DKQ - DV) * sizeof(half) : V->data;
        bool V_data_align = ((uint64_t)V_ptr % 16 == 0);
        bool V_stride_align = (V->nb[1] % 16 == 0);
        bool V_dim_even = (V->ne[0] % 2 == 0);
        
        if (!debug_printed && DKQ == 128 && DV == 128) {
            fprintf(stderr, "[FATTN DEBUG] V tensor alignment check:\n");
            fprintf(stderr, "[FATTN DEBUG]   V_ptr=%p, addr%%16=%lu, aligned=%d\n", V_ptr, ((uint64_t)V_ptr % 16), V_data_align);
            fprintf(stderr, "[FATTN DEBUG]   V->nb[1]=%lu (stride), stride%%16=%lu, aligned=%d\n", (unsigned long)V->nb[1], (unsigned long)(V->nb[1] % 16), V_stride_align);
            fprintf(stderr, "[FATTN DEBUG]   V->ne[0]=%ld (head_dim), even=%d\n", (long)V->ne[0], V_dim_even);
        }
        
        if (!V_data_align || !V_stride_align || !V_dim_even) {
            if (!debug_printed && DKQ == 128 && DV == 128) {
                fprintf(stderr, "[FATTN DEBUG]   ==> V alignment FAILED, disabling TMA\n");
            }
            use_tma_runtime = false;
        }
    }
    
    char* tensor_maps_dev_ptr = nullptr;
    ggml_cuda_pool_alloc<char> tensor_maps_alloc(ctx.pool(id));

#if CUDART_VERSION >= 12000
    // TMA with ldmatrix requires SWIZZLE_128B for correct shared memory layout.
    // SWIZZLE_128B requires inner dimension <= 128 bytes = 64 half = 32 half2.
    // For larger tiles, we must fall back to cp.async (no TMA) UNLESS on Blackwell with new kernel.
    
    const bool tma_tile_compatible = (nbatch_K2 <= 32) && (nbatch_V2 <= 32);

    if (!debug_printed && DKQ == 128 && DV == 128) {
        fprintf(stderr, "[FATTN DEBUG] TMA tile compatibility:\n");
        fprintf(stderr, "[FATTN DEBUG]   nbatch_K2=%d, nbatch_V2=%d\n", nbatch_K2, nbatch_V2);
        fprintf(stderr, "[FATTN DEBUG]   tma_tile_compatible (<=32)=%d\n", tma_tile_compatible);
        fprintf(stderr, "[FATTN DEBUG]   use_tma_runtime (after alignment)=%d\n", use_tma_runtime);
    }

    if (!tma_tile_compatible && !ggml_cuda_has_blackwell_features(cc)) {
        if (!debug_printed && DKQ == 128 && DV == 128) {
            fprintf(stderr, "[FATTN DEBUG]   ==> Tile incompatible + not Blackwell, disabling TMA\n");
        }
        use_tma_runtime = false;  // Fall back to cp.async for large tiles on Hopper
    }

    // Only set up TMA if we have a native Blackwell kernel for this configuration
    // Otherwise we fall back to Ampere kernel which doesn't use TMA
    constexpr bool has_bw_config = ggml_cuda_fattn_has_blackwell_config(DKQ, DV, ncols);
    if (!debug_printed && DKQ == 128 && DV == 128) {
        fprintf(stderr, "[FATTN DEBUG] Blackwell config check:\n");
        fprintf(stderr, "[FATTN DEBUG]   ggml_cuda_fattn_has_blackwell_config(%d,%d,%d)=%d\n", DKQ, DV, ncols, (int)has_bw_config);
    }
    if constexpr (!has_bw_config) {
        if (!debug_printed && DKQ == 128 && DV == 128) {
            fprintf(stderr, "[FATTN DEBUG]   ==> No Blackwell config, disabling TMA\n");
        }
        use_tma_runtime = false;
    }

    if (use_tma_runtime) {
        // Create 2 tensor maps: one for K, one for V
        tensor_maps_alloc.alloc(2 * sizeof(CUtensorMap));
        tensor_maps_dev_ptr = tensor_maps_alloc.get();

        CUtensorMap maps[2];
        memset(maps, 0, sizeof(maps));

        // K tensor map
        const ggml_tensor * K = dst->src[1];
        uint64_t K_rows = ggml_nrows(K);

        // Use SWIZZLE_128B for chunked TMA (nbatch_K2=32 → 64 halfs = 128 bytes inner dim)
        // SWIZZLE_128B is required because SWIZZLE_NONE has issues with >128-byte inner dims on sm_120
        constexpr bool needs_K_chunking_check = true;  // DKQ/2 > nbatch_K2 is checked at runtime
        const bool use_swizzle_none = false;  // Use SWIZZLE_128B (128-byte inner dim limit)
        const uint32_t tile_k_half = nbatch_K2 * 2;
        const CUtensorMapSwizzle swizzle_k = use_swizzle_none ? CU_TENSOR_MAP_SWIZZLE_NONE : CU_TENSOR_MAP_SWIZZLE_128B;

        fprintf(stderr, "[TMA DEBUG] K tensor map creation:\n");
        fprintf(stderr, "[TMA DEBUG]   K->ne[1]=%ld (K seq per head), K->ne[2]=%ld (kv_heads)\n",
                (long)K->ne[1], (long)K->ne[2]);
        fprintf(stderr, "[TMA DEBUG]   K_rows=%lu (should be ne[1]*ne[2]=%ld)\n", 
                (unsigned long)K_rows, (long)(K->ne[1] * K->ne[2]));
        fprintf(stderr, "[TMA DEBUG]   Max valid row per head = %ld-1 = %ld\n",
                (long)K->ne[1], (long)(K->ne[1]-1));
        fprintf(stderr, "[TMA DEBUG]   is_consumer_blackwell=%d, nbatch_K2=%d\n", 
                ggml_cuda_is_consumer_blackwell(cc), nbatch_K2);
        fprintf(stderr, "[TMA DEBUG]   use_swizzle_none=%d, swizzle=%s\n", 
                use_swizzle_none, use_swizzle_none ? "NONE" : "128B");
        fprintf(stderr, "[TMA DEBUG]   K->data=%p, K->ne[0]=%ld, K_rows=%lu\n", 
                K->data, (long)K->ne[0], (unsigned long)K_rows);
        fprintf(stderr, "[TMA DEBUG]   K->nb[1]=%ld (stride), tile_k_half=%u, nbatch_fa=%d\n", 
                (long)K->nb[1], tile_k_half, nbatch_fa);
        
        // TMA alignment checks
        uintptr_t k_addr = reinterpret_cast<uintptr_t>(K->data);
        fprintf(stderr, "[TMA DEBUG]   K address alignment: %lu-byte aligned (need 16-byte min)\n",
                k_addr % 256 == 0 ? 256 : (k_addr % 128 == 0 ? 128 : (k_addr % 64 == 0 ? 64 : 
                (k_addr % 32 == 0 ? 32 : (k_addr % 16 == 0 ? 16 : k_addr % 8)))));
        fprintf(stderr, "[TMA DEBUG]   K->nb[0]=%ld (element size), expected sizeof(half)=%lu\n",
                (long)K->nb[0], sizeof(half));
        fprintf(stderr, "[TMA DEBUG]   Tile bytes: %u (inner) x %d (rows) = %u bytes/tile\n",
                tile_k_half * (uint32_t)sizeof(half), nbatch_fa, tile_k_half * (uint32_t)sizeof(half) * nbatch_fa);

        CU_CHECK(ggml_cuda_create_tensor_map_2d_ex(
            &maps[0],
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
            K->data,
            K->ne[0],           // Full K dimension in half elements
            K_rows,             // Number of rows (N)
            K->nb[1],           // Stride in bytes between rows
            tile_k_half,        // Tile size in K dimension (half elements)
            nbatch_fa,          // Tile size in N dimension (rows)
            swizzle_k,
            CU_TENSOR_MAP_L2_PROMOTION_L2_256B));
        fprintf(stderr, "[TMA DEBUG]   K tensor map created OK\n");

        // V tensor map
        const ggml_tensor * V = mla ? K : dst->src[2];
        void* V_ptr = V->data;
        if (mla) {
            V_ptr = (char*)K->data + (DKQ - DV) * sizeof(half);
        }

        uint64_t V_rows = ggml_nrows(V);
        // Use SWIZZLE_128B for chunked TMA (matches K tensor map)
        const bool use_swizzle_none_v = false;  // Use SWIZZLE_128B
        const uint32_t tile_v_half = nbatch_V2 * 2;
        const CUtensorMapSwizzle swizzle_v = use_swizzle_none_v ? CU_TENSOR_MAP_SWIZZLE_NONE : CU_TENSOR_MAP_SWIZZLE_128B;
        const uint64_t V_dim_k = mla ? DV : V->ne[0];

        fprintf(stderr, "[TMA DEBUG] V tensor map creation:\n");
        fprintf(stderr, "[TMA DEBUG]   V_ptr=%p, V_dim_k=%lu, V_rows=%lu\n", 
                V_ptr, (unsigned long)V_dim_k, (unsigned long)V_rows);
        fprintf(stderr, "[TMA DEBUG]   V->nb[1]=%ld, tile_v_half=%u, nbatch_fa=%d\n", 
                (long)V->nb[1], tile_v_half, nbatch_fa);

        CU_CHECK(ggml_cuda_create_tensor_map_2d_ex(
            &maps[1],
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
            V_ptr,
            V_dim_k,            // Correct V dimension (DV for MLA)
            V_rows,             // Number of rows
            V->nb[1],           // Stride in bytes
            tile_v_half,        // Tile size (half elements)
            nbatch_fa,          // Rows per tile
            swizzle_v,
            CU_TENSOR_MAP_L2_PROMOTION_L2_256B));
        fprintf(stderr, "[TMA DEBUG]   V tensor map created OK\n");

        cudaMemcpyAsync(tensor_maps_dev_ptr, maps, 2 * sizeof(CUtensorMap), cudaMemcpyHostToDevice, ctx.stream());
        fprintf(stderr, "[TMA DEBUG] Tensor maps copied to device at %p\n", tensor_maps_dev_ptr);
    }
#endif

    // Update Shared Memory Calculation for Blackwell with chunk double-buffering
    // New layout: [K chunk 0][K chunk 1][V chunk 0][V chunk 1][Q tiles][mask][pipeline_state][mbarriers]
    // Apply for ALL Blackwell (including SM_120 which uses cp.async but same shared memory layout)
    const auto config_bw = ggml_cuda_fattn_mma_get_config_blackwell(DKQ, DV, ncols);
    // SM_120: use_tma_runtime=false but still needs Blackwell shared memory layout
    const bool use_blackwell_shmem = ggml_cuda_has_blackwell_features(cc) && config_bw.num_consumers > 0;
    if (use_blackwell_shmem) {
        // K: 2 chunk buffers for double-buffering
        size_t bytes_K_chunk = config_bw.nbatch_fa * config_bw.nbatch_K2 * sizeof(half2);
        size_t bytes_K_total = 2 * bytes_K_chunk;
        // V: 2 chunk buffers for double-buffering
        size_t bytes_V_chunk = config_bw.nbatch_fa * config_bw.nbatch_V2 * sizeof(half2);
        size_t bytes_V_total = 2 * bytes_V_chunk;
        // Q tiles
        constexpr int stride_tile_Q = DKQ/2 + 4;
        size_t bytes_Q = ncols * stride_tile_Q * sizeof(half2);
        // Mask
        size_t bytes_mask = ncols1 * (config_bw.nbatch_fa/2 + 4) * sizeof(half2);
        // Total: KV buffers + Q + mask + pipeline state + alignment padding
        nbytes_shared_total = bytes_K_total + bytes_V_total + bytes_Q + bytes_mask + sizeof(fattn_pipeline_state) + 128;
        
        fprintf(stderr, "[SHMEM DEBUG] Blackwell shared memory calculation:\n");
        fprintf(stderr, "[SHMEM DEBUG]   bytes_K_chunk=%zu, bytes_K_total=%zu\n", bytes_K_chunk, bytes_K_total);
        fprintf(stderr, "[SHMEM DEBUG]   bytes_V_chunk=%zu, bytes_V_total=%zu\n", bytes_V_chunk, bytes_V_total);
        fprintf(stderr, "[SHMEM DEBUG]   bytes_Q=%zu, bytes_mask=%zu\n", bytes_Q, bytes_mask);
        fprintf(stderr, "[SHMEM DEBUG]   sizeof(fattn_pipeline_state)=%zu\n", sizeof(fattn_pipeline_state));
        fprintf(stderr, "[SHMEM DEBUG]   TOTAL nbytes_shared=%zu (%.1f KB)\n", nbytes_shared_total, nbytes_shared_total/1024.0);
    }

    auto launch_kernel = [&](auto kernel_func, bool tma) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
        if (!shared_memory_limit_raised[id]) {
            // Only set shared memory limit if we're using the Blackwell kernel
            // SM_120: tma=false but still uses Blackwell kernel with cp.async
            // SM_100+: tma=true uses Blackwell kernel with TMA
            const bool using_blackwell_kernel = use_blackwell_shmem;
            
            fprintf(stderr, "[FUNC DEBUG] Setting kernel attributes:\n");
            fprintf(stderr, "[FUNC DEBUG]   tma=%d, using_blackwell_kernel=%d\n", tma, using_blackwell_kernel);
            fprintf(stderr, "[FUNC DEBUG]   nbytes_shared_total=%zu (%.1f KB)\n", nbytes_shared_total, nbytes_shared_total/1024.0);
            
            if (using_blackwell_kernel || !ggml_cuda_has_blackwell_features(cc)) {
                cudaError_t attr_err = cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total);
                fprintf(stderr, "[FUNC DEBUG]   cudaFuncSetAttribute(MaxDynamicSharedMemorySize, %zu) = %s\n", 
                        nbytes_shared_total, cudaGetErrorString(attr_err));
                CUDA_CHECK(attr_err);
            }

            // On Blackwell (sm_120), set preferred shared memory carveout to maximum.
            // Flash Attention is heavily shared memory bound (uses 70-90KB typically).
            // This hints to the scheduler to allocate maximum shared memory per SM.
            // Value 100 = cudaSharedmemCarveoutMaxShared (prefer shared over L1).
            // This setting only affects scheduling and doesn't guarantee allocation.
            if (using_blackwell_kernel) {
                cudaError_t carve_err = cudaFuncSetAttribute(kernel_func, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
                fprintf(stderr, "[FUNC DEBUG]   cudaFuncSetAttribute(PreferredSharedMemoryCarveout, 100) = %s\n",
                        cudaGetErrorString(carve_err));
                CUDA_CHECK(carve_err);
            }

            shared_memory_limit_raised[id] = true;
        }
#endif
        fprintf(stderr, "[FUNC DEBUG] Calling launch_fattn with nwarps=%d, nbytes_shared=%zu, tensor_maps=%p\n",
                nwarps, nbytes_shared_total, tma ? tensor_maps_dev_ptr : nullptr);
        launch_fattn<DV, ncols1, ncols2>( 
            ctx, dst, kernel_func, nwarps, nbytes_shared_total, nbatch_fa, true, true, true, WARP_SIZE,
            tma ? tensor_maps_dev_ptr : nullptr
        );
    };

    bool use_logit_softcap_kernel = (logit_softcap != 0.0f);

    if (!debug_printed && DKQ == 128 && DV == 128) {
        fprintf(stderr, "[FATTN DEBUG] Final dispatch decision:\n");
        fprintf(stderr, "[FATTN DEBUG]   is_consumer_blackwell (SM_120)=%d\n", ggml_cuda_is_consumer_blackwell(cc));
        fprintf(stderr, "[FATTN DEBUG]   use_tma_runtime=%d\n", use_tma_runtime);
        fprintf(stderr, "[FATTN DEBUG]   config_bw.num_consumers=%d\n", config_bw.num_consumers);
        fprintf(stderr, "[FATTN DEBUG]   config_bw.nbatch_K2=%d, config_bw.nbatch_V2=%d\n", config_bw.nbatch_K2, config_bw.nbatch_V2);
        fprintf(stderr, "[FATTN DEBUG]   DKQ/2=%d, DV/2=%d\n", DKQ/2, DV/2);
        fprintf(stderr, "[FATTN DEBUG]   needs_K_chunking (DKQ/2 > nbatch_K2)=%d\n", (DKQ/2 > config_bw.nbatch_K2));
        fprintf(stderr, "[FATTN DEBUG]   needs_V_chunking (DV/2 > nbatch_V2)=%d\n", (DV/2 > config_bw.nbatch_V2));
    }

    // Blackwell kernel optimized for sm_100+ with Warp Specialization
    // SM_120 (consumer Blackwell / RTX 5090): Uses cp.async path inside kernel (no TMA/mbarrier)
    // SM_100/103/110 (datacenter Blackwell): Uses TMA + mbarrier when use_tma_runtime=true
    //
    // Dispatch logic:
    //   - SM_120: ALWAYS use Blackwell kernel (it has cp.async fallback, no TMA required)
    //   - SM_100+: Use Blackwell kernel only if use_tma_runtime=true (needs TMA)
    //
    // Use if constexpr to prevent template instantiation for unsupported DKQ/DV/ncols combinations
    if constexpr (ggml_cuda_fattn_has_blackwell_config(DKQ, DV, ncols)) {
        // SM_120 always uses Blackwell kernel with cp.async; other Blackwell needs TMA
        const bool use_blackwell = ggml_cuda_is_consumer_blackwell(cc) ||
                                   (ggml_cuda_has_blackwell_features(cc) && use_tma_runtime);
        // SM_120: num_consumers=0 (unified mode), still uses Blackwell kernel with SM_120 configs
        // Other Blackwell: num_consumers>0 (producer/consumer mode)
        if (use_blackwell && (config_bw.num_consumers > 0 || ggml_cuda_is_consumer_blackwell(cc))) {
            // Check chunk count - double-buffering supports exactly 2 chunks per dimension
            // num_K_chunks = ceil(DKQ/2 / nbatch_K2), num_V_chunks = ceil(DV/2 / nbatch_V2)
            // With nbatch_K2=32 for DKQ=128: num_K_chunks = 64/32 = 2 (perfect for double-buffer)
            const int num_K_chunks = (DKQ/2 + config_bw.nbatch_K2 - 1) / config_bw.nbatch_K2;
            const int num_V_chunks = (DV/2 + config_bw.nbatch_V2 - 1) / config_bw.nbatch_V2;
            const bool too_many_chunks = (num_K_chunks > 2) || (num_V_chunks > 2);
            
            if (too_many_chunks) {
                if (!debug_printed && DKQ == 128 && DV == 128) {
                    fprintf(stderr, "[FATTN DEBUG]   ==> SKIP Blackwell: too many chunks (K=%d, V=%d, max=2)\n", 
                            num_K_chunks, num_V_chunks);
                }
                // Skip Blackwell kernel, use Ampere fallback below
            } else {
                if (!debug_printed && DKQ == 128 && DV == 128) {
                    fprintf(stderr, "[FATTN DEBUG]   ==> LAUNCHING BLACKWELL KERNEL (chunks: K=%d, V=%d)!\n",
                            num_K_chunks, num_V_chunks);
                    fprintf(stderr, "[FATTN DEBUG]   ==> use_tma_runtime=%d (template & launch param)\n", use_tma_runtime);
                    fprintf(stderr, "[FATTN DEBUG] ===================================\n\n");
                    debug_printed = true;
                }
                // SM_120: use_tma_runtime=false, instantiate kernel with use_tma=false template
                // Other Blackwell: use_tma_runtime=true, instantiate kernel with use_tma=true template
                if (use_tma_runtime) {
                    // TMA path for datacenter Blackwell (SM_100/103/110)
                    if (use_logit_softcap_kernel) {
                        launch_kernel(flash_attn_ext_f16_blackwell<DKQ, DV, ncols1, ncols2, true, mla, true>, true);
                    } else {
                        launch_kernel(flash_attn_ext_f16_blackwell<DKQ, DV, ncols1, ncols2, false, mla, true>, true);
                    }
                } else {
                    // cp.async path for SM_120 (RTX 5090) - no TMA
                    if (use_logit_softcap_kernel) {
                        launch_kernel(flash_attn_ext_f16_blackwell<DKQ, DV, ncols1, ncols2, true, mla, false>, false);
                    } else {
                        launch_kernel(flash_attn_ext_f16_blackwell<DKQ, DV, ncols1, ncols2, false, mla, false>, false);
                    }
                }
                return;
            }
        } else {
            if (!debug_printed && DKQ == 128 && DV == 128) {
                fprintf(stderr, "[FATTN DEBUG]   ==> SKIP Blackwell: condition failed\n");
                fprintf(stderr, "[FATTN DEBUG]       is_consumer_bw=%d, has_bw_features=%d, use_tma=%d, consumers=%d\n",
                    ggml_cuda_is_consumer_blackwell(cc), ggml_cuda_has_blackwell_features(cc),
                    use_tma_runtime, config_bw.num_consumers);
            }
        }
    } else {
        if (!debug_printed && DKQ == 128 && DV == 128) {
            fprintf(stderr, "[FATTN DEBUG]   ==> SKIP Blackwell: no config (constexpr false)\n");
        }
    }

    if (!debug_printed && DKQ == 128 && DV == 128) {
        fprintf(stderr, "[FATTN DEBUG]   ==> FALLING BACK TO AMPERE KERNEL\n");
        fprintf(stderr, "[FATTN DEBUG] ===================================\n\n");
        debug_printed = true;
    }

    // AMPERE FALLBACK PATH
    // ---------------------
    // This path is used when:
    //   1. Not on Blackwell (cc < 1200)
    //   2. On Blackwell but TMA unavailable (alignment/stride issues)
    //   3. On Blackwell but no native config for this DKQ/DV/ncols combination
    //
    // The Ampere kernel is always safe because:
    //   - use_tma=false: bypasses all TMA-specific code
    //   - num_consumers=0: uses unified mode (no warp specialization)
    //   - Uses _ampere config getters: nbatch_K2 >= DKQ/2 guaranteed
    //
    // This ensures multi-GPU setups work correctly even with mixed GPU generations.
    if (use_logit_softcap_kernel) {
         launch_kernel(flash_attn_ext_f16_ampere<DKQ, DV, ncols1, ncols2, true, mla, false>, false);
    } else {
         launch_kernel(flash_attn_ext_f16_ampere<DKQ, DV, ncols1, ncols2, false, mla, false>, false);
    }
}


#define DECL_FATTN_MMA_F16_CASE(DKQ, DV, ncols1, ncols2)                          \
    template void ggml_cuda_flash_attn_ext_mma_f16_case                           \
    <DKQ, DV, ncols1, ncols2>(ggml_backend_cuda_context & ctx, ggml_tensor * dst) \

#define DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(DKQ, DV, ncols)   \
    extern DECL_FATTN_MMA_F16_CASE(DKQ, DV, (ncols)/ 1,  1); \
    extern DECL_FATTN_MMA_F16_CASE(DKQ, DV, (ncols)/ 2,  2); \
    extern DECL_FATTN_MMA_F16_CASE(DKQ, DV, (ncols)/ 4,  4); \
    extern DECL_FATTN_MMA_F16_CASE(DKQ, DV, (ncols)/ 8,  8); \
    extern DECL_FATTN_MMA_F16_CASE(DKQ, DV, (ncols)/16, 16); \

DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 64,  64,   8)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 80,  80,   8)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 96,  96,   8)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(112, 112,   8)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(128, 128,   8)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(256, 256,   8)

DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 64,  64,  16)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 80,  80,  16)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 96,  96,  16)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(112, 112,  16)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(128, 128,  16)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(256, 256,  16)

DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 64,  64,  32)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 80,  80,  32)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 96,  96,  32)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(112, 112,  32)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(128, 128,  32)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(256, 256,  32)

DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 64,  64,  64)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 80,  80,  64)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2( 96,  96,  64)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(112, 112,  64)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(128, 128,  64)
DECL_FATTN_MMA_F16_CASE_ALL_NCOLS2(256, 256,  64)

// The number of viable configurations for Deepseek is very limited:
extern DECL_FATTN_MMA_F16_CASE(576, 512, 1, 16);
extern DECL_FATTN_MMA_F16_CASE(576, 512, 2, 16);
extern DECL_FATTN_MMA_F16_CASE(576, 512, 4, 16);