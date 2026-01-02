#ifndef GGML_CUDA_TMA_CUH
#define GGML_CUDA_TMA_CUH

#include "common.cuh"

#if CUDART_VERSION >= 12000
// Create tensor map for 2D tile loads (K x N matrix, loading tiles of tile_k x tile_n)
//
// L2 Cache Promotion Options (l2_promotion parameter):
//   - CU_TENSOR_MAP_L2_PROMOTION_NONE:   No L2 promotion (data may bypass L2)
//   - CU_TENSOR_MAP_L2_PROMOTION_L2_64B: 64-byte L2 promotion granularity
//   - CU_TENSOR_MAP_L2_PROMOTION_L2_128B: 128-byte L2 promotion granularity
//   - CU_TENSOR_MAP_L2_PROMOTION_L2_256B: 256-byte L2 promotion granularity (default)
//
// For Flash Attention on Blackwell (96MB L2 cache on RTX 5090):
//   - L2_256B is optimal for large tile loads (nbatch_fa x nbatch_K2/V2)
//   - During decode, K/V cache benefits from L2 residency across iterations
//   - Larger promotion granularity reduces L2 tag pressure for streaming access
//
// Note: Stream-level L2 persistence (cudaStreamSetAttribute with accessPolicyWindow)
// could further benefit decode workloads but requires per-tensor configuration.
static inline __host__ CUresult ggml_cuda_create_tensor_map_2d(
    CUtensorMap* tensor_map,
    CUtensorMapDataType dtype,
    void* global_ptr,
    uint64_t dim_k,
    uint64_t dim_n,
    uint64_t stride_n_bytes,  // Typically dim_k * sizeof(element)
    uint32_t tile_k,
    uint32_t tile_n,
    CUtensorMapL2promotion l2_promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_256B)
{
#ifndef __CUDA_ARCH__
    const uint64_t dims[2] = {dim_k, dim_n};
    const uint64_t globalStrides[1] = {stride_n_bytes};
    const uint32_t tile_dims[2] = {tile_k, tile_n};
    const uint32_t elem_strides[2] = {1, 1};

    return cuTensorMapEncodeTiled(
        tensor_map,
        dtype,
        2,                              // 2D
        global_ptr,
        dims,
        globalStrides,                  // Only need stride for dim > 0
        tile_dims,
        elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,     // 128-byte swizzle for bank-conflict-free
        l2_promotion,                   // L2 cache promotion granularity
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
#else
    GGML_UNUSED_VARS(tensor_map, dtype, global_ptr, dim_k, dim_n, stride_n_bytes, tile_k, tile_n, l2_promotion);
    return (CUresult)0;
#endif // __CUDA_ARCH__
}
// Extended version with configurable swizzle parameter.
// For TMA on Blackwell with large tiles:
//   - CU_TENSOR_MAP_SWIZZLE_128B requires inner dim <= 128 bytes
//   - For larger tiles, use CU_TENSOR_MAP_SWIZZLE_NONE which has no size limit
//   - SWIZZLE_NONE is still faster than cp.async due to hardware bulk transfer
static inline __host__ CUresult ggml_cuda_create_tensor_map_2d_ex(
    CUtensorMap* tensor_map,
    CUtensorMapDataType dtype,
    void* global_ptr,
    uint64_t dim_k,
    uint64_t dim_n,
    uint64_t stride_n_bytes,
    uint32_t tile_k,
    uint32_t tile_n,
    CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_128B,
    CUtensorMapL2promotion l2_promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_256B)
{
#ifndef __CUDA_ARCH__
    const uint64_t dims[2] = {dim_k, dim_n};
    const uint64_t globalStrides[1] = {stride_n_bytes};
    const uint32_t tile_dims[2] = {tile_k, tile_n};
    const uint32_t elem_strides[2] = {1, 1};

    return cuTensorMapEncodeTiled(
        tensor_map,
        dtype,
        2,                              // 2D
        global_ptr,
        dims,
        globalStrides,
        tile_dims,
        elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle,
        l2_promotion,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
#else
    GGML_UNUSED_VARS(tensor_map, dtype, global_ptr, dim_k, dim_n, stride_n_bytes, tile_k, tile_n, swizzle, l2_promotion);
    return (CUresult)0;
#endif // __CUDA_ARCH__
}

// TMA tile size constants (legacy, kept for template API compatibility).
// These were used for chunking with SWIZZLE_128B but with SWIZZLE_NONE
// we can load full tiles in single TMA operations regardless of size.
static constexpr int TMA_SWIZZLE_128B_MAX_BYTES = 128;
static constexpr int TMA_HALF2_BYTES = 4;  // sizeof(half2)
static constexpr int TMA_MAX_HALF2_PER_CHUNK = TMA_SWIZZLE_128B_MAX_BYTES / TMA_HALF2_BYTES;  // 32

// Calculate number of TMA chunks needed for a given nbatch_K2/V2
static constexpr __host__ __device__ int tma_chunks_needed(const int nbatch_half2) {
    return (nbatch_half2 + TMA_MAX_HALF2_PER_CHUNK - 1) / TMA_MAX_HALF2_PER_CHUNK;
}

// Calculate chunk size (half2 elements per chunk)
static constexpr __host__ __device__ int tma_chunk_size(const int nbatch_half2, const int num_chunks) {
    // Distribute evenly across chunks
    return (nbatch_half2 + num_chunks - 1) / num_chunks;
}
#endif // CUDART_VERSION >= 12000

#ifdef BLACKWELL_TMA_AVAILABLE

// Async bulk load: global -> shared via tensor map
// Signals mbarrier when complete
// Note: shared::cluster is used even for non-clustered kernels (cluster size 1).
// This is the standard TMA addressing mode on Hopper/Blackwell.
__device__ __forceinline__ void tma_load_2d(
    void* __restrict__ smem_ptr,
    const CUtensorMap* __restrict__ tensor_map,
    uint64_t* __restrict__ mbar_ptr,
    int32_t coord_x,
    int32_t coord_y)
{
    uint32_t smem_addr = __cvta_generic_to_shared(smem_ptr);
    uint64_t tmap_addr = reinterpret_cast<uint64_t>(tensor_map);

    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3}], [%4] ;"
        :
        : "r"(smem_addr), "l"(tmap_addr), "r"(coord_x), "r"(coord_y),
          "l"(mbar_ptr)
        : "memory"
    );
}

// Fence to ensure TMA descriptor reads are visible
// On sm_120 (Blackwell), use __threadfence_system() to ensure tensor map descriptors
// copied via cudaMemcpyAsync from host are visible to the GPU before TMA operations.
// 
// Note: The fence.proxy.tensormap::generic.acquire instruction has PTX syntax issues
// with CUDA 13.1/sm_120. Using __threadfence_system() provides system-wide memory
// ordering which is appropriate since tensor maps are copied from host memory.
__device__ __forceinline__ void tma_fence_acquire(const void* tensor_map, uint32_t size_bytes) {
    (void)tensor_map;
    (void)size_bytes;
    __threadfence_system();
}

// Convenience overload for acquiring visibility of a single CUtensorMap
__device__ __forceinline__ void tma_fence_acquire(const CUtensorMap* tensor_map) {
    (void)tensor_map;
    __threadfence_system();
}

// Acquire visibility for multiple consecutive tensor maps (e.g., K and V maps)
__device__ __forceinline__ void tma_fence_acquire_maps(const void* tensor_maps, int num_maps) {
    (void)tensor_maps;
    (void)num_maps;
    __threadfence_system();
}

// Initialize barrier with expected transaction count
__device__ __forceinline__ void mbarrier_init(uint64_t* mbar, uint32_t count) {
    asm volatile(
        "mbarrier.init.shared::cta.b64 [%0], %1;"
        : 
        : "l"(mbar), "r"(count)
        : "memory"
    );
}

// Arrive and expect N bytes of async transactions
__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* mbar, uint32_t tx_bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        : 
        : "l"(mbar), "r"(tx_bytes)
        : "memory"
    );
}

// Arrive at barrier (no transaction update)
__device__ __forceinline__ void mbarrier_arrive(uint64_t* mbar) {
    asm volatile(
        "mbarrier.arrive.shared::cta.b64 _, [%0];"
        : 
        : "l"(mbar)
        : "memory"
    );
}

// Wait for phase (blocking)
__device__ __forceinline__ void mbarrier_wait(uint64_t* mbar, uint32_t phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "WAIT_LOOP:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
        "@!p bra WAIT_LOOP;\n\t"
        "}\n"
        : 
        : "l"(mbar), "r"(phase)
        : "memory"
    );
}

// Non-blocking try-wait
__device__ __forceinline__ bool mbarrier_try_wait(uint64_t* mbar, uint32_t phase) {
    uint32_t ready;
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
        "selp.u32 %2, 1, 0, p;\n\t"
        "}\n"
        : "=r"(ready)
        : "l"(mbar), "r"(phase)
        : "memory"
    );
    return ready != 0;
}

#else

__device__ __forceinline__ void tma_load_2d(
    void* __restrict__ smem_ptr,
    const CUtensorMap* __restrict__ tensor_map,
    uint64_t* __restrict__ mbar_ptr,
    int32_t coord_x,
    int32_t coord_y)
{
    GGML_UNUSED(smem_ptr);
    GGML_UNUSED(tensor_map);
    GGML_UNUSED(mbar_ptr);
    GGML_UNUSED(coord_x);
    GGML_UNUSED(coord_y);
    NO_DEVICE_CODE;
}

__device__ __forceinline__ void tma_fence_acquire(const void* tensor_map, uint32_t size_bytes) {
    GGML_UNUSED(tensor_map);
    GGML_UNUSED(size_bytes);
    NO_DEVICE_CODE;
}

__device__ __forceinline__ void tma_fence_acquire(const CUtensorMap* tensor_map) {
    GGML_UNUSED(tensor_map);
    NO_DEVICE_CODE;
}

__device__ __forceinline__ void tma_fence_acquire_maps(const void* tensor_maps, int num_maps) {
    GGML_UNUSED(tensor_maps);
    GGML_UNUSED(num_maps);
    NO_DEVICE_CODE;
}

__device__ __forceinline__ void mbarrier_init(uint64_t* mbar, uint32_t count) {
    GGML_UNUSED(mbar);
    GGML_UNUSED(count);
    NO_DEVICE_CODE;
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* mbar, uint32_t tx_bytes) {
    GGML_UNUSED(mbar);
    GGML_UNUSED(tx_bytes);
    NO_DEVICE_CODE;
}

__device__ __forceinline__ void mbarrier_arrive(uint64_t* mbar) {
    GGML_UNUSED(mbar);
    NO_DEVICE_CODE;
}

__device__ __forceinline__ void mbarrier_wait(uint64_t* mbar, uint32_t phase) {
    GGML_UNUSED(mbar);
    GGML_UNUSED(phase);
    NO_DEVICE_CODE;
}

__device__ __forceinline__ bool mbarrier_try_wait(uint64_t* mbar, uint32_t phase) {
    GGML_UNUSED(mbar);
    GGML_UNUSED(phase);
    NO_DEVICE_CODE;
    return false;
}

#endif // BLACKWELL_TMA_AVAILABLE
#endif // GGML_CUDA_TMA_CUH
