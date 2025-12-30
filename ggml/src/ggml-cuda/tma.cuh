#pragma once
#include "common.cuh"

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

#endif // BLACKWELL_TMA_AVAILABLE
