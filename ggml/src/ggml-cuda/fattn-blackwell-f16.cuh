#pragma once

#include "common.cuh"
#include "fattn-common.cuh"
#include "tma.cuh"
#include "mma.cuh"

// Blackwell-specific Flash Attention Kernel (sm_120+)
// Architecture:
//   - Warp Specialization: 1 Producer Warp (TMA), 4 Consumer Warps (WGMMA)
//   - Circular Buffering: 2-stage pipeline for K/V tiles
//   - Q-in-Registers: Persistent Q for consumers
//   - Large Tiles: 64x128 (M x N)

#ifdef BLACKWELL_TMA_AVAILABLE

using namespace ggml_cuda_mma;
using namespace ggml_cuda_wgmma;

// ----------------------------------------------------------------------------
// WGMMA Primitives for Registers
// ----------------------------------------------------------------------------

// WGMMA m64n64k16 with A in registers (distributed) and B in Shared Memory (descriptor)
__device__ __forceinline__ void wgmma_m64n64k16_f16_f32_reg(
    float* accum,           // 64 floats in registers per thread (accumulators)
    uint32_t* A_regs,       // A matrix fragment in registers (f16x2)
    uint64_t B_desc,        // B matrix descriptor (Shared Memory)
    int scale_D = 1)
{
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31}, "
        "{%32, %33, %34, %35}, " // A regs (4x b32 = 8x f16)
        "%36, %37, %38, 0;"      // B desc, scale, zero
        : "+f"(accum[0]),  "+f"(accum[1]),  "+f"(accum[2]),  "+f"(accum[3]),
          "+f"(accum[4]),  "+f"(accum[5]),  "+f"(accum[6]),  "+f"(accum[7]),
          "+f"(accum[8]),  "+f"(accum[9]),  "+f"(accum[10]), "+f"(accum[11]),
          "+f"(accum[12]), "+f"(accum[13]), "+f"(accum[14]), "+f"(accum[15]),
          "+f"(accum[16]), "+f"(accum[17]), "+f"(accum[18]), "+f"(accum[19]),
          "+f"(accum[20]), "+f"(accum[21]), "+f"(accum[22]), "+f"(accum[23]),
          "+f"(accum[24]), "+f"(accum[25]), "+f"(accum[26]), "+f"(accum[27]),
          "+f"(accum[28]), "+f"(accum[29]), "+f"(accum[30]), "+f"(accum[31])
        : "r"(A_regs[0]), "r"(A_regs[1]), "r"(A_regs[2]), "r"(A_regs[3]),
          "l"(B_desc), "r"(scale_D), "r"(scale_D) // scale_D is used for D and C
        : "memory"
    );
}

// ----------------------------------------------------------------------------
// Configuration
// ----------------------------------------------------------------------------

template<int DKQ, int DV>
struct blackwell_config {
    // Tile sizes
    static constexpr int BLOCK_M = 64;  // Q rows
    static constexpr int BLOCK_N = 128; // K/V block size
    static constexpr int STAGES  = 2;
    
    // Dimensions
    static constexpr int HEAD_SIZE_Q = DKQ;
    static constexpr int HEAD_SIZE_V = DV;
    
    // Thread configuration
    static constexpr int WARPS_CONSUMER = 7;
    static constexpr int WARPS_PRODUCER = 1;
    static constexpr int NUM_WARPS      = WARPS_CONSUMER + WARPS_PRODUCER;
    static constexpr int NUM_THREADS    = NUM_WARPS * WARP_SIZE;
};

// ----------------------------------------------------------------------------
// Shared Memory Layout
// ----------------------------------------------------------------------------

template<typename Config>
struct SharedStorage {
    using half_t = half;
    
    // Q tile for loading (Global -> Shared -> Regs)
    // Size: BLOCK_M * DKQ
    half_t Q[Config::BLOCK_M * Config::HEAD_SIZE_Q];
    
    struct Stage {
        // K and V tiles.
        // We use a flat buffer. For MLA, they might be the same.
        // Size: BLOCK_N * Max(DKQ, DV)
        half_t K[Config::BLOCK_N * Config::HEAD_SIZE_Q]; 
        half_t V[Config::BLOCK_N * Config::HEAD_SIZE_V];
    };
    
    Stage stages[Config::STAGES];
    
    // mbarriers for synchronization
    uint64_t mbar_full[Config::STAGES];  // Producer signals arrival (full), Consumer waits
    uint64_t mbar_empty[Config::STAGES]; // Consumer signals done (empty), Producer waits
};

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------

// Make WGMMA descriptor for K (N x D)
// K is stored in shared memory as [BLOCK_N, D] row-major?
// Or [D, BLOCK_N] col-major?
// TMA loads K in natural layout (usually [N, D] row-major from global).
// WGMMA expects B to be in specific layouts.
// "Matrix B data can be laid out in row-major or column-major."
// "Standard layouts: ... with 128-byte swizzling."
// If we use SWIZZLE_NONE for TMA, we might need to match WGMMA expectations.
// For now, assume simple row-major layout descriptor.
__device__ __forceinline__ uint64_t make_wgmma_desc(
    const void* smem_ptr,
    int dim_h, int dim_w, int stride_bytes,
    int swizzle_mode = 0 // 0 = none, 1 = 32B, 2 = 64B, 3 = 128B
) {
    uint32_t addr = __cvta_generic_to_shared(smem_ptr);
    uint64_t desc = 0;
    
    // PTX 8.0+ Descriptor
    // [0-13]: Address (16B aligned) >> 4
    desc |= ((uint64_t)addr >> 4);
    
    // [16-29]: Leading Dimension (stride) >> 4
    desc |= ((uint64_t)(stride_bytes >> 4) << 16);
    
    // [32-45]: Stride dimension (usually same as LD for simple layout?) 
    // Actually for WGMMA B: 
    // "Stride is the stride in bytes between two leading dimension lines."
    desc |= ((uint64_t)(stride_bytes >> 4) << 32); 

    // Swizzle: [62-63]
    desc |= ((uint64_t)swizzle_mode << 62);
    
    return desc;
}

// ----------------------------------------------------------------------------
// Kernel
// ----------------------------------------------------------------------------

template<int DKQ, int DV, int ncols1, int ncols2, bool mla>
__launch_bounds__(256, 2)
__global__ void flash_attn_blackwell_f16(
    const char * __restrict__ Q,
    const char * __restrict__ K,
    const char * __restrict__ V,
    const char * __restrict__ mask,
    float      * __restrict__ dst,
    const float scale,
    const float logit_softcap,
    const uint3 ne01, const int32_t ne02,
    const int32_t nb01, const int32_t nb02, const int32_t nb03,
    const int32_t nb11, const int32_t nb12,
    const int32_t nb21, const int32_t nb22,
    // Group all int64_t strides together at end for proper 8-byte alignment
    const int64_t nb13, const int64_t nb23,
    const char * __restrict__ tensor_maps
) {
    using Config = blackwell_config<DKQ, DV>;
    extern __shared__ char smem_raw[];
    SharedStorage<Config>* smem = reinterpret_cast<SharedStorage<Config>*>(smem_raw);

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const bool is_producer = (warp_id == Config::WARPS_CONSUMER); // Warp 4

    // Initialize mbarriers
    if (tid == 0) {
        for (int i = 0; i < Config::STAGES; ++i) {
            mbarrier_init(&smem->mbar_full[i], 1); // Producer arrives
            mbarrier_init(&smem->mbar_empty[i], Config::WARPS_CONSUMER * WARP_SIZE); // Consumers arrive
        }
    }
    __syncthreads();
    
    // Base pointers
    const int q_head = blockIdx.y;
    const int q_batch = blockIdx.z; // Assumes grid (N, head, batch)
    
    // ------------------------------------------------------------------------
    // Consumer Role (WGMMA)
    // ------------------------------------------------------------------------
    if (!is_producer) {
        // 1. Load Q into Registers (Persistent)
        // -------------------------------------
        // Strategy: 
        // a) Load global Q -> SMEM Q (coalesced)
        // b) ldmatrix SMEM Q -> Regs Q (fragment layout)
        
        // Q Fragment storage: 
        // M=64. WGMMA uses 128 threads.
        // Q is MxK (64xDKQ).
        // Each WGMMA `m64n64k16` consumes 16 columns of K.
        // We need D/16 chunks of Q registers.
        // Each chunk: 64 rows x 16 cols (f16) = 1024 elems.
        // Distributed across 128 threads -> 8 elems/thread -> 4x half2 -> 4x 32-bit regs.
        
        constexpr int K_CHUNKS = DKQ / 16;
        uint32_t Q_regs[K_CHUNKS][4]; // [chunk][reg]
        
        // Load Q to SMEM
        // Simple cooperative load: 128 threads load 64 x DKQ elements.
        // Each thread loads (64*DKQ)/128 = DKQ/2 elements.
        // Only doing this once, so efficiency is secondary to correctness.
        const char* Q_ptr = Q + q_batch * nb03 + q_head * nb02;
        // Map blockIdx.x to Q block? 
        // Usually flash attn grid is (num_tiles_m, num_heads, batch).
        // Let's assume blockIdx.x is tile index M.
        int m_block = blockIdx.x;
        int m_base = m_block * Config::BLOCK_M;
        
        // Load loop... (implementation omitted for brevity, assume Q in SMEM)
        // ...
        __syncthreads(); // Q loaded in SMEM
        
        // Load Q to Regs using ldmatrix
        // Q in SMEM is [64, DKQ]. 
        // We iterate over K_CHUNKS (16 cols).
        for (int k = 0; k < K_CHUNKS; ++k) {
            int k_offset = k * 16;
            // ldmatrix.sync.aligned.m8n8.x4.trans.b16 ...
            // We need 4 regs per thread.
            // Address: smem->Q + ...
            // Use ldmatrix primitive from mma.cuh
            // ...
        }
        
        // 2. Main Loop
        // ------------
        float acc_s[2][32]; // 2x 64x64 tiles (N=128), 32 floats per thread
        float acc_o[32];    // Output accumulator
        // Clear O
        #pragma unroll
        for(int i=0; i<32; ++i) acc_o[i] = 0.0f;
        
        float l_i = 0.0f; // Softmax sum
        float m_i = -1e20f; // Softmax max
        
        const int num_k_blocks = (ne01.y + Config::BLOCK_N - 1) / Config::BLOCK_N;

        for (int k_block = 0; k_block < num_k_blocks; ++k_block) {
            int stage = k_block % Config::STAGES;
            
            // Wait for Producer
            // mbarrier_wait(&smem->mbar_full[stage], k_block % 2); // Phase
            // Wait logic needs to track phase manually or use count.
            // Using `mbarrier_wait` helper.
            
            // GEMM 1: S = Q * K^T
            // K is in smem->stages[stage].K
            // Dimensions: N=128, D=DKQ.
            // We split N into two 64-chunks: K0 (cols 0-63), K1 (cols 64-127).
            
            // Clear S accums
            #pragma unroll
            for(int i=0; i<32; ++i) { acc_s[0][i] = 0.0f; acc_s[1][i] = 0.0f; }
            
            // Iterate over D in chunks of 16
            for (int k = 0; k < K_CHUNKS; ++k) {
                // K0 Descriptor
                // Ptr: &smem->stages[stage].K[... k*16 ...]
                // This assumes K is stored such that we can address 16-col chunks?
                // Or K is [N, D] row-major?
                // If row-major, taking a 16-col slice is non-contiguous in memory.
                // WGMMA B-matrix needs contiguous layout or simple stride?
                // WGMMA `m64n64k16` expects B to be 64x16 (NxK).
                // If B is [N, D], and we want N=64, K=16 block.
                // With row-major, stride is D*sizeof(half).
                // So we can point to (row=0, col=k*16) and set stride=D*2.
                
                uint64_t desc_k0 = make_wgmma_desc(&smem->stages[stage].K[k*16], Config::BLOCK_N, DKQ*2, DKQ*2); 
                // Offset for K1 (next 64 rows of K? Or cols?)
                // K is [N, D]. We split N.
                // K0: Rows 0-63. K1: Rows 64-127.
                uint64_t desc_k1 = make_wgmma_desc(&smem->stages[stage].K[64*DKQ + k*16], Config::BLOCK_N, DKQ*2, DKQ*2);
                
                wgmma_m64n64k16_f16_f32_reg(acc_s[0], Q_regs[k], desc_k0);
                wgmma_m64n64k16_f16_f32_reg(acc_s[1], Q_regs[k], desc_k1);
            }
            wgmma_commit_group();
            wgmma_wait_group(0);
            
            // Softmax & Rescaling
            // ... (Online Softmax logic on acc_s) ...
            
            // GEMM 2: O = S * V
            // S is in acc_s (regs). V is in smem.
            // S (64x128) * V (128xD).
            // WGMMA `m64n64k16`.
            // Tile M=64, N=D_chunk?
            // K_dim of this GEMM is 128 (N from previous).
            // We loop k from 0..128 in steps of 16.
            // V is [N, D].
            // We need V in SMEM to be accessed as K=16 chunks.
            // Row-major V [128, D].
            // Chunk k (16 rows) of V: V[k*16 : (k+1)*16, :]
            // Target: O column chunk (D_chunk).
            // This is "Vector-Matrix" style WGMMA? Or just normal GEMM.
            // M=64 (Q rows), N=D (V cols), K=128 (V rows).
            // We want output O (64xD).
            // We iterate over K=128 (8 steps of 16).
            // But we need to calculate all N columns of O.
            // If D=128, N=128.
            // We can split O into O0 (cols 0-63), O1 (cols 64-127).
            
            // Loop k_inner (0..128, step 16):
            //   Load S_frag (from acc_s).
            //   S needs to be converted f32->f16 for WGMMA input?
            //   Or use f32 accumulators as input? Only supported on some paths.
            //   Assuming conversion needed.
            
            // Signal Empty
            uint64_t* mbar = &smem->mbar_empty[stage];
            asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "l"(mbar));
        }
        
        // Write Output O to Global
    }
    
    // ------------------------------------------------------------------------
    // Producer Role
    // ------------------------------------------------------------------------
    else {
        // Producer Logic
        const int num_k_blocks = (ne01.y + Config::BLOCK_N - 1) / Config::BLOCK_N;
        
        for (int k_block = 0; k_block < num_k_blocks; ++k_block) {
            int stage = k_block % Config::STAGES;
            bool wait_empty = (k_block >= Config::STAGES);
            
            if (wait_empty) {
                // Wait for Consumers to finish reading
                mbarrier_wait(&smem->mbar_empty[stage], (k_block - Config::STAGES) % 2); 
            }
            
            // Issue TMA Loads
            int32_t coord_x = k_block * Config::BLOCK_N;
            
            // Load K
            tma_load_2d(&smem->stages[stage].K[0], (const CUtensorMap*)(tensor_maps), 
                        &smem->mbar_full[stage], coord_x, 0);
            
            // Load V
            if (!mla || (DKQ != DV)) {
                tma_load_2d(&smem->stages[stage].V[0], (const CUtensorMap*)(tensor_maps + sizeof(CUtensorMap)), 
                            &smem->mbar_full[stage], coord_x, 0);
            }

            uint32_t bytes_expected = sizeof(half) * (Config::BLOCK_N * DKQ + (!mla ? Config::BLOCK_N * DV : 0));
            mbarrier_arrive_expect_tx(&smem->mbar_full[stage], bytes_expected);
        }
    }
}
