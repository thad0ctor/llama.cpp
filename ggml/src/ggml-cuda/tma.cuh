#ifndef GGML_CUDA_TMA_CUH
#define GGML_CUDA_TMA_CUH

#include "common.cuh"

#if CUDART_VERSION >= 12000
// Create tensor map for 2D tile loads (K x N matrix, loading tiles of tile_k x tile_n)
static inline __host__ CUresult ggml_cuda_create_tensor_map_2d(
    CUtensorMap* tensor_map,
    CUtensorMapDataType dtype,
    void* global_ptr,
    uint64_t dim_k,
    uint64_t dim_n,
    uint64_t stride_n_bytes,  // Typically dim_k * sizeof(element)
    uint32_t tile_k,
    uint32_t tile_n)
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
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
#else
    GGML_UNUSED_VARS(tensor_map, dtype, global_ptr, dim_k, dim_n, stride_n_bytes, tile_k, tile_n);
    return (CUresult)0;
#endif // __CUDA_ARCH__
}
#endif // CUDART_VERSION >= 12000

#ifdef BLACKWELL_TMA_AVAILABLE

// Async bulk load: global -> shared via tensor map
// Signals mbarrier when complete
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
__device__ __forceinline__ void tma_fence_acquire() {
    asm volatile("fence.proxy.tensormap::generic.acquire.gpu;");
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

__device__ __forceinline__ void tma_fence_acquire() {
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
