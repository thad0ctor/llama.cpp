// TMA-based matmul for Blackwell (sm_90+)
// Uses PTX intrinsics instead of cuda/barrier for portability

#include "mul_mat_tma.cuh"
#include "common.cuh"

#include <cuda_fp16.h>
#include <cudaTypedefs.h>

// Only enable TMA on sm_90+ (Hopper/Blackwell)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define TMA_AVAILABLE 1
#else
#define TMA_AVAILABLE 0
#endif

// Host-side check for TMA support
static bool tma_supported_on_device(int cc) {
    return cc >= GGML_CUDA_CC_HOPPER;  // sm_90+
}

//
// PTX intrinsic wrappers (replaces cuda/barrier)
//

#if TMA_AVAILABLE

// Initialize mbarrier
__device__ __forceinline__ void mbarrier_init(uint64_t* mbar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "l"(mbar), "r"(count));
}

// Fence for async proxy
__device__ __forceinline__ void fence_proxy_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;");
}

// Arrive on mbarrier (non-tx thread)
__device__ __forceinline__ uint64_t mbarrier_arrive(uint64_t* mbar) {
    uint64_t state;
    asm volatile("mbarrier.arrive.shared.b64 %0, [%1];" : "=l"(state) : "l"(mbar));
    return state;
}

// Arrive with expected TX bytes (thread 0)
__device__ __forceinline__ uint64_t mbarrier_arrive_expect_tx(uint64_t* mbar, uint32_t tx_bytes) {
    uint64_t state;
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 %0, [%1], %2;"
                 : "=l"(state) : "l"(mbar), "r"(tx_bytes));
    return state;
}

// Wait on mbarrier phase
__device__ __forceinline__ void mbarrier_wait(uint64_t* mbar, uint64_t phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "WAIT_LOOP%=:\n\t"
        "mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n\t"
        "@!p bra WAIT_LOOP%=;\n\t"
        "}\n\t"
        :: "l"(mbar), "l"(phase & 1));
}

// TMA 2D copy global->shared
__device__ __forceinline__ void tma_copy_2d_g2s(
    void* smem_ptr,
    const CUtensorMap* tensor_map,
    int coord_x,
    int coord_y,
    uint64_t* mbar
) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(smem_addr), "l"(tensor_map), "r"(coord_x), "r"(coord_y), "l"(mbar)
        : "memory");
}

#endif // TMA_AVAILABLE

//
// Helper functions
//

__host__ __device__ __forceinline__
int cdiv(int a, int b) { return (a + b - 1) / b; }

// 128-byte aligned struct for shared memory
typedef struct __align__(128) {} Aligned128B;

//
// ldmatrix and MMA intrinsics
//

__device__ __forceinline__
void ldmatrix_x4(uint32_t reg[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3])
                 : "r"(addr));
}

__device__ __forceinline__
void mma_m16n8k16_f16(const uint32_t A[4], const uint32_t B[2], float C[4]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0, %1, %2, %3}, "
                 "{%4, %5, %6, %7}, "
                 "{%8, %9}, "
                 "{%0, %1, %2, %3};"
                 : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3])
                 : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                   "r"(B[0]), "r"(B[1]));
}

//
// Swizzle calculation for bank-conflict-free access
//

template <int SWIZZLE_WIDTH>
__device__ __forceinline__
uint32_t tma_swizzle(uint32_t addr) {
    const uint32_t row = (addr / 128) % (SWIZZLE_WIDTH / 16);
    return addr ^ (row << 4);
}

//
// TMA Matmul Kernel
//

#if TMA_AVAILABLE

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_WARP_M, int NUM_WARP_N>
__launch_bounds__(NUM_WARP_M * NUM_WARP_N * WARP_SIZE)
__global__ void tma_matmul_f16_kernel(
    const __grid_constant__ CUtensorMap A_tensor_map,
    const __grid_constant__ CUtensorMap B_tensor_map,
    float* __restrict__ C,
    int M, int N, int K
) {
    constexpr int MMA_M = 16;
    constexpr int MMA_N = 8;
    constexpr int MMA_K = 16;
    static_assert(BLOCK_M % NUM_WARP_M == 0);
    static_assert(BLOCK_N % NUM_WARP_N == 0);
    static_assert(BLOCK_K % MMA_K == 0);

    constexpr int WARP_M = BLOCK_M / NUM_WARP_M;
    constexpr int WARP_N = BLOCK_N / NUM_WARP_N;
    static_assert(WARP_M % MMA_M == 0);
    static_assert(WARP_N % MMA_N == 0);

    constexpr int TB_SIZE = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;
    constexpr int NUM_MMA_M = WARP_M / MMA_M;
    constexpr int NUM_MMA_N = WARP_N / MMA_N;
    constexpr int NUM_MMA_K = BLOCK_K / MMA_K;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    // Block indices
    const int num_blocks_n = cdiv(N, BLOCK_N);
    const int bid_m = blockIdx.x / num_blocks_n;
    const int bid_n = blockIdx.x % num_blocks_n;
    const int offset_m = bid_m * BLOCK_M;
    const int offset_n = bid_n * BLOCK_N;

    // Warp position
    const int warp_id_m = warp_id / NUM_WARP_N;
    const int warp_id_n = warp_id % NUM_WARP_N;

    // Shared memory
    extern __shared__ Aligned128B smem[];
    half* A_smem = reinterpret_cast<half*>(smem);
    half* B_smem = A_smem + BLOCK_M * BLOCK_K;

    // mbarrier in shared memory (at end)
    __shared__ __align__(8) uint64_t mbar;

    // Accumulators
    float acc[NUM_MMA_M][NUM_MMA_N][4] = {};
    uint32_t A_regs[NUM_MMA_M][NUM_MMA_K][4];
    uint32_t B_regs[NUM_MMA_N][NUM_MMA_K][2];

    // Pre-compute swizzled addresses for ldmatrix
    const int A_offm = (warp_id_m * WARP_M) + (lane_id % 16);
    const int A_offk = (lane_id / 16) * 8;
    uint32_t A_smem_thread = static_cast<uint32_t>(__cvta_generic_to_shared(A_smem));
    A_smem_thread += (A_offm * BLOCK_K + A_offk) * sizeof(half);
    A_smem_thread = tma_swizzle<BLOCK_K * sizeof(half)>(A_smem_thread);

    const int B_offn = (warp_id_n * WARP_N) + (lane_id % 8);
    const int B_offk = (lane_id / 8) * 8;
    uint32_t B_smem_thread = static_cast<uint32_t>(__cvta_generic_to_shared(B_smem));
    B_smem_thread += (B_offn * BLOCK_K + B_offk) * sizeof(half);
    B_smem_thread = tma_swizzle<BLOCK_K * sizeof(half)>(B_smem_thread);

    // Initialize mbarrier
    if (tid == 0) {
        mbarrier_init(&mbar, TB_SIZE);
        fence_proxy_async_shared();
    }
    __syncthreads();

    // Main K loop
    const int num_k_iters = cdiv(K, BLOCK_K);
    for (int k_iter = 0; k_iter < num_k_iters; k_iter++) {
        uint64_t token;

        if (tid == 0) {
            // Thread 0 initiates TMA copies
            tma_copy_2d_g2s(A_smem, &A_tensor_map, k_iter * BLOCK_K, offset_m, &mbar);
            tma_copy_2d_g2s(B_smem, &B_tensor_map, k_iter * BLOCK_K, offset_n, &mbar);

            // Arrive with expected bytes
            const uint32_t expected_bytes = (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(half);
            token = mbarrier_arrive_expect_tx(&mbar, expected_bytes);
        } else {
            // Other threads just arrive
            token = mbarrier_arrive(&mbar);
        }

        // Wait for TMA to complete
        mbarrier_wait(&mbar, token);

        // Load A tiles: shared -> registers
        #pragma unroll
        for (int mma_id_m = 0; mma_id_m < NUM_MMA_M; mma_id_m++) {
            #pragma unroll
            for (int mma_id_k = 0; mma_id_k < NUM_MMA_K; mma_id_k++) {
                uint32_t A_addr = A_smem_thread;
                A_addr += mma_id_m * MMA_M * BLOCK_K * sizeof(half);
                A_addr ^= mma_id_k * MMA_K * sizeof(half);
                ldmatrix_x4(A_regs[mma_id_m][mma_id_k], A_addr);
            }
        }

        // Load B tiles: shared -> registers
        #pragma unroll
        for (int mma_id_n = 0; mma_id_n < NUM_MMA_N; mma_id_n++) {
            #pragma unroll
            for (int mma_id_k = 0; mma_id_k < NUM_MMA_K; mma_id_k++) {
                uint32_t B_addr = B_smem_thread;
                B_addr += mma_id_n * MMA_N * BLOCK_K * sizeof(half);
                B_addr ^= mma_id_k * MMA_K * sizeof(half);
                // ldmatrix_x2 for B (8x16 tile)
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
                             : "=r"(B_regs[mma_id_n][mma_id_k][0]), "=r"(B_regs[mma_id_n][mma_id_k][1])
                             : "r"(B_addr));
            }
        }

        // MMA compute
        #pragma unroll
        for (int mma_id_m = 0; mma_id_m < NUM_MMA_M; mma_id_m++) {
            #pragma unroll
            for (int mma_id_n = 0; mma_id_n < NUM_MMA_N; mma_id_n++) {
                #pragma unroll
                for (int mma_id_k = 0; mma_id_k < NUM_MMA_K; mma_id_k++) {
                    mma_m16n8k16_f16(A_regs[mma_id_m][mma_id_k],
                                     B_regs[mma_id_n][mma_id_k],
                                     acc[mma_id_m][mma_id_n]);
                }
            }
        }

        __syncthreads();
    }

    // Write results to global memory (F32 output)
    float* C_base = C + (offset_m + warp_id_m * WARP_M) * N + (offset_n + warp_id_n * WARP_N);

    #pragma unroll
    for (int mma_id_m = 0; mma_id_m < NUM_MMA_M; mma_id_m++) {
        #pragma unroll
        for (int mma_id_n = 0; mma_id_n < NUM_MMA_N; mma_id_n++) {
            const int row = mma_id_m * MMA_M + (lane_id / 4);
            const int col = mma_id_n * MMA_N + (lane_id % 4) * 2;

            // Bounds check for non-power-of-2 dimensions
            if (offset_m + warp_id_m * WARP_M + row < M &&
                offset_n + warp_id_n * WARP_N + col < N) {

                float* C_local = C_base + row * N + col;
                C_local[0] = acc[mma_id_m][mma_id_n][0];
                C_local[1] = acc[mma_id_m][mma_id_n][1];
            }
            if (offset_m + warp_id_m * WARP_M + row + 8 < M &&
                offset_n + warp_id_n * WARP_N + col < N) {

                float* C_local = C_base + (row + 8) * N + col;
                C_local[0] = acc[mma_id_m][mma_id_n][2];
                C_local[1] = acc[mma_id_m][mma_id_n][3];
            }
        }
    }
}

#endif // TMA_AVAILABLE

//
// Host wrapper functions
//

bool ggml_cuda_should_use_tma(
    const ggml_tensor* src0,
    const ggml_tensor* src1,
    int cc
) {
    // Check architecture support
    if (!tma_supported_on_device(cc)) {
        return false;
    }

    // Only F16 supported currently
    if (src0->type != GGML_TYPE_F16) {
        return false;
    }

    // Need contiguous tensors
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
        return false;
    }

    // Minimum dimensions for TMA to be beneficial
    const int64_t M = src1->ne[1];
    const int64_t N = src0->ne[1];
    const int64_t K = src0->ne[0];

    // TMA has setup overhead, only use for larger matrices
    if (M < 64 || N < 64 || K < 64) {
        return false;
    }

    // K must be divisible by BLOCK_K (64)
    if (K % 64 != 0) {
        return false;
    }

    return true;
}

void ggml_cuda_mul_mat_tma(
    ggml_backend_cuda_context& ctx,
    const ggml_tensor* src0,
    const ggml_tensor* src1,
    ggml_tensor* dst
) {
#if TMA_AVAILABLE
    const int64_t ne00 = src0->ne[0];  // K
    const int64_t ne01 = src0->ne[1];  // N
    const int64_t ne11 = src1->ne[1];  // M

    const int M = ne11;
    const int N = ne01;
    const int K = ne00;

    const half* A = (const half*)src1->data;  // M x K
    const half* B = (const half*)src0->data;  // N x K (transposed)
    float* C = (float*)dst->data;             // M x N

    constexpr int BLOCK_M = 128;
    constexpr int BLOCK_N = 64;
    constexpr int BLOCK_K = 64;
    constexpr int NUM_WARP_M = 2;
    constexpr int NUM_WARP_N = 2;
    constexpr int TB_SIZE = NUM_WARP_M * NUM_WARP_N * WARP_SIZE;

    // Create tensor maps
    auto init_tensor_map = [&](CUtensorMap* tensor_map, const half* gmem_ptr,
                               uint64_t width, uint64_t height,
                               uint32_t box_width, uint32_t box_height) {
        constexpr uint32_t rank = 2;
        uint64_t size[rank] = {width, height};
        uint64_t stride[rank - 1] = {width * sizeof(half)};
        uint32_t box_size[rank] = {box_width, box_height};
        uint32_t elem_stride[rank] = {1, 1};

        CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_128B;

        CUresult res = cuTensorMapEncodeTiled(
            tensor_map,
            CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
            rank,
            (void*)gmem_ptr,
            size,
            stride,
            box_size,
            elem_stride,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            swizzle,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );

        if (res != CUDA_SUCCESS) {
            GGML_ABORT("cuTensorMapEncodeTiled failed");
        }
    };

    CUtensorMap A_tensor_map, B_tensor_map;
    init_tensor_map(&A_tensor_map, A, K, M, BLOCK_K, BLOCK_M);
    init_tensor_map(&B_tensor_map, B, K, N, BLOCK_K, BLOCK_N);

    const int grid_size = cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N);
    const int smem_size = (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(half) + 64;  // +64 for mbarrier

    cudaStream_t stream = ctx.stream();

    // Set max dynamic shared memory if needed
    if (smem_size > 48 * 1024) {
        CUDA_CHECK(cudaFuncSetAttribute(
            tma_matmul_f16_kernel<BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARP_M, NUM_WARP_N>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size));
    }

    tma_matmul_f16_kernel<BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARP_M, NUM_WARP_N>
        <<<grid_size, TB_SIZE, smem_size, stream>>>(
            A_tensor_map, B_tensor_map, C, M, N, K);

#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_ABORT("TMA not available on this architecture");
#endif
}
