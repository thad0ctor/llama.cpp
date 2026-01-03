// Simplified API for asynchronous data loading.

#include "common.cuh"


static __device__ __forceinline__ unsigned int ggml_cuda_cvta_generic_to_shared(void * generic_ptr) {
#ifdef CP_ASYNC_AVAILABLE
    return __cvta_generic_to_shared(generic_ptr);
#else
    GGML_UNUSED(generic_ptr);
    NO_DEVICE_CODE;
    return 0;
#endif // CP_ASYNC_AVAILABLE
}

// Copies data from global to shared memory, cg == cache global.
// Both the src and dst pointers must be aligned to 16 bit.
// Shared memory uses 32 bit addressing, the pointer is passed as unsigned int.
// Generic pointers can be converted to 32 bit shared memory pointers using __cvta_generic_to_shared.
// Only the 16 bit copy is exposed because 4 and 8 bit copies did not yield performance improvements.
template <int preload>
static __device__ __forceinline__ void cp_async_cg_16(const unsigned int dst, const void * src) {
    static_assert(preload == 0 || preload == 64 || preload == 128 || preload == 256, "bad preload");
#ifdef CP_ASYNC_AVAILABLE
#if CUDART_VERSION >= 11040
    if (preload == 256) {
        asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16;"
            : : "r"(dst), "l"(src));
    } else if (preload == 128) {
        asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;"
            : : "r"(dst), "l"(src));
    } else if (preload == 64) {
        asm volatile("cp.async.cg.shared.global.L2::64B [%0], [%1], 16;"
            : : "r"(dst), "l"(src));
    } else
#endif // CUDART_VERSION >= 11040
    {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
            : : "r"(dst), "l"(src));
    }
#else
    GGML_UNUSED(dst);
    GGML_UNUSED(src);
    NO_DEVICE_CODE;
#endif // CP_ASYNC_AVAILABLE
}

// Commits all prior executed cp.async instructions into a new group.
static __device__ __forceinline__ void cp_async_commit_group() {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.commit_group;");
#else
    NO_DEVICE_CODE;
#endif // CP_ASYNC_AVAILABLE
}

// Waits until N or fewer groups are pending.
template <int N>
static __device__ __forceinline__ void cp_async_wait_group() {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.wait_group %0;" : : "n"(N));
#else
    NO_DEVICE_CODE;
#endif // CP_ASYNC_AVAILABLE
}

// Makes each thread wait until its asynchronous data copies are done.
// This does NOT provide any additional synchronization.
// In particular, when copying data with multiple warps a call to __syncthreads will be needed.
static __device__ __forceinline__ void cp_async_wait_all() {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.wait_all;");
#else
    NO_DEVICE_CODE;
#endif // CP_ASYNC_AVAILABLE
}

// ============================================================================
// SM_120 Optimized cp.async Primitives (RTX 5090)
// ============================================================================
// 
// These primitives implement Gau-Nernst style optimizations for achieving
// 90-95% of theoretical throughput on SM_120 without TMA.
//
// Key optimizations:
//   1. Shared memory swizzling - Eliminates bank conflicts in ldmatrix
//   2. Two-stage pipelining - Overlaps global→shared with compute
//   3. ldmatrix.x4 - Fewer instructions in hot loop
//
// Reference: https://gau-nern.st/blog/flash-attention-from-scratch.html
// ============================================================================

// ----------------------------------------------------------------------------
// Shared Memory Swizzling
// ----------------------------------------------------------------------------
// Problem: When loading 8x8 tiles from shared memory with ldmatrix, consecutive
// threads hit the same 4 banks, causing 8-way bank conflicts.
//
// Solution: XOR-based swizzling remaps addresses to distribute across all banks.
// The swizzle pattern depends on the stride (row width in bytes).
//
// For a row with stride S bytes:
//   - If S <= 16B: no swizzling needed (row fits in one bank group)
//   - If S > 16B: swizzle based on row index to avoid conflicts
//
// The formula XORs bits [4:6] of the address with bits derived from row index.
// This maps consecutive rows to different bank groups.
// ----------------------------------------------------------------------------

// Swizzle function for shared memory addressing
// STRIDE = stride in bytes (e.g., DIM * sizeof(half) for K/V tiles)
template <int STRIDE>
static __device__ __forceinline__ uint32_t sm120_swizzle(uint32_t addr) {
#ifdef CP_ASYNC_AVAILABLE
    if constexpr (STRIDE <= 16) {
        return addr;  // No swizzling needed for small strides
    } else {
        // Calculate row index from address and stride
        // XOR pattern: bits_to_xor = (row_idx / max(64/STRIDE, 1)) & 0x7
        // Then XOR with bits [4:6] of address
        constexpr int rows_per_swizzle_group = (64 / STRIDE) > 0 ? (64 / STRIDE) : 1;
        uint32_t row_idx = (addr / STRIDE) % 8;
        uint32_t bits_to_xor = row_idx / rows_per_swizzle_group;
        return addr ^ (bits_to_xor << 4);  // XOR with bits [4:6]
    }
#else
    return addr;
#endif
}

// Swizzle for 128-byte stride (DIM=64, sizeof(half)=2)
static __device__ __forceinline__ uint32_t sm120_swizzle_128(uint32_t addr) {
    return sm120_swizzle<128>(addr);
}

// Swizzle for 256-byte stride (DIM=128, sizeof(half)=2)  
static __device__ __forceinline__ uint32_t sm120_swizzle_256(uint32_t addr) {
    return sm120_swizzle<256>(addr);
}

// ----------------------------------------------------------------------------
// Swizzled cp.async Copy
// ----------------------------------------------------------------------------
// Copies 16 bytes from global to shared memory with swizzling applied to dst.
// This is the core primitive for bank-conflict-free global→shared transfers.
// ----------------------------------------------------------------------------

template <int STRIDE, int preload = 128>
static __device__ __forceinline__ void cp_async_cg_16_swizzle(
    uint32_t dst_base, 
    uint32_t offset, 
    const void * __restrict__ src
) {
#ifdef CP_ASYNC_AVAILABLE
    uint32_t dst = dst_base + sm120_swizzle<STRIDE>(offset);
    cp_async_cg_16<preload>(dst, src);
#else
    GGML_UNUSED(dst_base);
    GGML_UNUSED(offset);
    GGML_UNUSED(src);
    NO_DEVICE_CODE;
#endif
}

// ----------------------------------------------------------------------------
// Cooperative Global→Shared Transfer with Swizzling
// ----------------------------------------------------------------------------
// All threads in thread block cooperatively load a 2D tile from global memory
// to shared memory with swizzling applied.
//
// Parameters:
//   BLOCK_ROWS: number of rows in the tile
//   BLOCK_COLS: number of half elements per row (stride in halfs)
//   TB_SIZE: total threads in thread block
//   dst_smem: shared memory destination (32-bit address)
//   src_gmem: global memory source
//   gmem_stride: stride in halfs for global memory
//   tid: thread ID within thread block
// ----------------------------------------------------------------------------

template <int BLOCK_ROWS, int BLOCK_COLS, int TB_SIZE, int preload = 128>
static __device__ __forceinline__ void global_to_shared_swizzle(
    uint32_t dst_smem,
    const half * __restrict__ src_gmem,
    int gmem_stride,
    int tid
) {
#ifdef CP_ASYNC_AVAILABLE
    constexpr int STRIDE_BYTES = BLOCK_COLS * sizeof(half);
    constexpr int ELEMS_PER_COPY = 16 / sizeof(half);  // 8 halfs per 16B copy
    constexpr int COPIES_PER_ROW = BLOCK_COLS / ELEMS_PER_COPY;
    constexpr int TOTAL_COPIES = BLOCK_ROWS * COPIES_PER_ROW;
    
    // Each thread handles TOTAL_COPIES / TB_SIZE copies
    #pragma unroll
    for (int copy_id = tid; copy_id < TOTAL_COPIES; copy_id += TB_SIZE) {
        int row = copy_id / COPIES_PER_ROW;
        int col = (copy_id % COPIES_PER_ROW) * ELEMS_PER_COPY;
        
        uint32_t offset = row * STRIDE_BYTES + col * sizeof(half);
        const half * src = src_gmem + row * gmem_stride + col;
        
        cp_async_cg_16_swizzle<STRIDE_BYTES, preload>(dst_smem, offset, src);
    }
#else
    GGML_UNUSED(dst_smem);
    GGML_UNUSED(src_gmem);
    GGML_UNUSED(gmem_stride);
    GGML_UNUSED(tid);
    NO_DEVICE_CODE;
#endif
}

// ----------------------------------------------------------------------------
// ldmatrix Wrappers with Swizzle Support
// ----------------------------------------------------------------------------
// ldmatrix loads 8x8 matrix tiles from shared memory into registers.
// These wrappers apply swizzle to the shared memory address before loading.
// ----------------------------------------------------------------------------

// ldmatrix.x2: Load 2 8x8 tiles (for m16n8k16 MMA)
static __device__ __forceinline__ void ldmatrix_x2_swizzle(
    uint32_t & r0, uint32_t & r1,
    uint32_t smem_base,
    uint32_t offset,
    int stride_bytes
) {
#ifdef CP_ASYNC_AVAILABLE
    uint32_t addr = smem_base + sm120_swizzle<256>(offset);  // Assume 256B stride for common case
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
        : "=r"(r0), "=r"(r1)
        : "r"(addr)
    );
#else
    GGML_UNUSED(r0);
    GGML_UNUSED(r1);
    GGML_UNUSED(smem_base);
    GGML_UNUSED(offset);
    GGML_UNUSED(stride_bytes);
    NO_DEVICE_CODE;
#endif
}

// ldmatrix.x4: Load 4 8x8 tiles (for n8k32 MMA patterns) - fewer instructions in hot loop
// This is a key optimization: Gau-Nernst saw 91% → 93% from using ldmatrix.x4
static __device__ __forceinline__ void ldmatrix_x4_swizzle(
    uint32_t & r0, uint32_t & r1, uint32_t & r2, uint32_t & r3,
    uint32_t smem_base,
    uint32_t offset,
    int stride_bytes
) {
#ifdef CP_ASYNC_AVAILABLE
    uint32_t addr;
    if (stride_bytes == 256) {
        addr = smem_base + sm120_swizzle<256>(offset);
    } else if (stride_bytes == 128) {
        addr = smem_base + sm120_swizzle<128>(offset);
    } else {
        addr = smem_base + offset;  // No swizzle for unusual strides
    }
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(addr)
    );
#else
    GGML_UNUSED(r0);
    GGML_UNUSED(r1);
    GGML_UNUSED(r2);
    GGML_UNUSED(r3);
    GGML_UNUSED(smem_base);
    GGML_UNUSED(offset);
    GGML_UNUSED(stride_bytes);
    NO_DEVICE_CODE;
#endif
}

// ldmatrix.x4 with transposed layout (for B matrix in MMA)
static __device__ __forceinline__ void ldmatrix_x4_trans_swizzle(
    uint32_t & r0, uint32_t & r1, uint32_t & r2, uint32_t & r3,
    uint32_t smem_base,
    uint32_t offset,
    int stride_bytes
) {
#ifdef CP_ASYNC_AVAILABLE
    uint32_t addr;
    if (stride_bytes == 256) {
        addr = smem_base + sm120_swizzle<256>(offset);
    } else if (stride_bytes == 128) {
        addr = smem_base + sm120_swizzle<128>(offset);
    } else {
        addr = smem_base + offset;
    }
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(addr)
    );
#else
    GGML_UNUSED(r0);
    GGML_UNUSED(r1);
    GGML_UNUSED(r2);
    GGML_UNUSED(r3);
    GGML_UNUSED(smem_base);
    GGML_UNUSED(offset);
    GGML_UNUSED(stride_bytes);
    NO_DEVICE_CODE;
#endif
}

// ----------------------------------------------------------------------------
// Two-Stage Pipeline State for SM_120
// ----------------------------------------------------------------------------
// Manages double-buffered shared memory for K/V tiles.
// Stage 0 is being computed while Stage 1 is being loaded (and vice versa).
// ----------------------------------------------------------------------------

struct sm120_pipeline_state {
    int current_stage;      // 0 or 1
    int prefetch_stage;     // Opposite of current_stage
    
    __device__ __forceinline__ void init() {
        current_stage = 0;
        prefetch_stage = 1;
    }
    
    __device__ __forceinline__ void advance() {
        current_stage ^= 1;
        prefetch_stage ^= 1;
    }
    
    __device__ __forceinline__ int get_K_offset(int stage, int nbatch_fa, int nbatch_K2) const {
        return stage * nbatch_fa * nbatch_K2 * sizeof(half2);
    }
    
    __device__ __forceinline__ int get_V_offset(int stage, int nbatch_fa, int nbatch_K2, int nbatch_V2) const {
        // V follows K in shared memory: [K stage 0][K stage 1][V stage 0][V stage 1]
        int K_total = 2 * nbatch_fa * nbatch_K2 * sizeof(half2);
        return K_total + stage * nbatch_fa * nbatch_V2 * sizeof(half2);
    }
};
