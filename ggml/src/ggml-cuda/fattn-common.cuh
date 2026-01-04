#pragma once

#include "common.cuh"
#include "convert.cuh"
#include "vecdotq.cuh"

#include <cstdint>

#define FATTN_KQ_STRIDE       256
#define HALF_MAX_HALF         __float2half(65504.0f/2) // Use neg. of this instead of -INFINITY to initialize KQ max vals to avoid NaN upon subtraction.
#define SOFTMAX_FTZ_THRESHOLD -20.0f                   // Softmax exp. of values smaller than this are flushed to zero to avoid NaNs.

// log(2) = 0.6931, by adding this to the KQ maximum used for the softmax the numerical range representable
//     by the VKQ accumulators is effectively being shifted up by a factor of 8.
// This reduces issues with numerical overflow but also causes larger values to be flushed to zero.
// However, as the output from FlashAttention will usually be used as an input for a matrix multiplication this should be negligible.
#define FATTN_KQ_MAX_OFFSET 0.6931f

typedef void (* fattn_kernel_t)(
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
                            const int32_t nb31, const int32_t nb32, const int64_t nb33);

typedef float (*vec_dot_KQ_t)(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8 , const void * __restrict__ Q_ds);

template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_f16(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8 , const void * __restrict__ Q_ds_v) {

    const half2 * K_h2 = (const half2 *) K_c;
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);

    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += nthreads*cpy_ne) {
        half2 tmp[cpy_ne];
        ggml_cuda_memcpy_1<sizeof(tmp)>(tmp, K_h2 + k_KQ_0 + (threadIdx.x % nthreads)*cpy_ne);
#pragma unroll
        for (int k_KQ_1 = 0; k_KQ_1 < cpy_ne; ++k_KQ_1) {
#ifdef V_DOT2_F32_F16_AVAILABLE
            ggml_cuda_mad(sum,                tmp[k_KQ_1] , ((const half2  *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#else
            ggml_cuda_mad(sum, __half22float2(tmp[k_KQ_1]), ((const float2 *) Q_v)[k_KQ_0/nthreads + k_KQ_1]);
#endif // V_DOT2_F32_F16_AVAILABLE
        }
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q4_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q4_0 * K_q4_0 = (const block_q4_0 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI4_0;
        const int shift = k_KQ & (QI8_1/2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int), 2>(&v, K_q4_0[ib].qs + sizeof(int)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;
        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];
        sum += __half2float(K_q4_0[ib].d) * (sumi*Q_ds.x - (8/QI8_1)*Q_ds.y);
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q4_1(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q4_1 * K_q4_1 = (const block_q4_1 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI4_1;
        const int shift = k_KQ & (QI8_1/2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int)>(&v, K_q4_1[ib].qs + sizeof(int)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;
        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 K_dm = __half22float2(K_q4_1[ib].dm);
        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];

        sum += K_dm.x*Q_ds.x*sumi + K_dm.y*Q_ds.y/QI8_1;
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q5_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q5_0 * K_q5_0 = (const block_q5_0 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI5_0;
        const int iqs8  = k_KQ %  QI8_1;
        const int shift = k_KQ & (QI8_1/2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int), 2>(&v, K_q5_0[ib].qs + sizeof(int)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;

        {
            int vh;
            ggml_cuda_memcpy_1<sizeof(int), 2>(&vh, K_q5_0[ib].qh);
            vh >>= iqs8 * QI5_0;

            v |= (vh <<  4) & 0x00000010; // 0 ->  4
            v |= (vh << 11) & 0x00001000; // 1 -> 12
            v |= (vh << 18) & 0x00100000; // 2 -> 20
            v |= (vh << 25) & 0x10000000; // 3 -> 28
        }

        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];

        sum += __half2float(K_q5_0[ib].d) * (sumi*Q_ds.x - (16/QI8_1)*Q_ds.y);
    }

    return sum;
}

template<int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q5_1(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q5_1 * K_q5_1 = (const block_q5_1 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI5_1;
        const int iqs8  = k_KQ %  QI8_1;
        const int shift = k_KQ & (QI8_1/2);

        int v;
        ggml_cuda_memcpy_1<sizeof(int)>(&v, K_q5_1[ib].qs + sizeof(int)*iqs4);
        v = (v >> shift) & 0x0F0F0F0F;

        {
            int vh;
            ggml_cuda_memcpy_1<sizeof(int)>(&vh, K_q5_1[ib].qh);
            vh >>= iqs8 * QI5_0;

            v |= (vh <<  4) & 0x00000010; // 0 ->  4
            v |= (vh << 11) & 0x00001000; // 1 -> 12
            v |= (vh << 18) & 0x00100000; // 2 -> 20
            v |= (vh << 25) & 0x10000000; // 3 -> 28
        }

        const int u = Q_q8[k_KQ_0/nthreads];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

        const float2 K_dm = __half22float2(K_q5_1[ib].dm);
        const float2 Q_ds = ((const float2 *) Q_ds_v)[k_KQ_0/nthreads];

        sum += K_dm.x*Q_ds.x*sumi + K_dm.y*Q_ds.y/QI8_1;
    }

    return sum;
}

template <int D, int nthreads>
static __device__ __forceinline__ float vec_dot_fattn_vec_KQ_q8_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q8_0 * K_q8_0 = (const block_q8_0 *) K_c;
    GGML_UNUSED(Q_v);

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += nthreads) {
        const int k_KQ = k_KQ_0 + (nthreads == WARP_SIZE ? threadIdx.x : threadIdx.x % nthreads);

        const int ib  = k_KQ / QI8_0;
        const int iqs = k_KQ % QI8_0;

        int v;
        ggml_cuda_memcpy_1<sizeof(v), 2>(&v, K_q8_0[ib].qs + 4*iqs);

        const float2 * Q_ds = (const float2 *) Q_ds_v;
        const float Q_d = Q_ds[k_KQ_0/nthreads].x;

        sum += vec_dot_q8_0_q8_1_impl<float, 1>(&v, &Q_q8[k_KQ_0/nthreads], K_q8_0[ib].d, Q_d);
    }

    return sum;
}

template <typename Tds, int ni>
static __device__ __forceinline__ void quantize_q8_1_to_shared(
    const float * __restrict__ x, const float scale, int * __restrict__ yq32, void * __restrict__ yds) {

    float vals[sizeof(int)] = {0.0f};
#pragma unroll
    for (int l = 0; l < int(sizeof(int)); ++l) {
        vals[l] = (ni == WARP_SIZE || threadIdx.x < ni) ? scale * x[4*threadIdx.x + l] : 0.0f;
    }

    float amax = fabsf(vals[0]);
    float sum  = vals[0];
#pragma unroll
    for (int l = 1; l < int(sizeof(int)); ++l) {
        amax = fmaxf(amax, fabsf(vals[l]));
        sum += vals[l];
    }
#pragma unroll
    for (int mask = QI8_1/2; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, mask, 32));
        sum +=             __shfl_xor_sync(0xFFFFFFFF, sum,  mask, 32);
    }

    const float d = amax / 127;
    int q32 = 0;
    int8_t * q8 = (int8_t *) &q32;

    if (d != 0.0f) {
#pragma unroll
        for (int l = 0; l < int(sizeof(int)); ++l) {
            q8[l] = roundf(vals[l] / d);
        }
    }

    yq32[threadIdx.x] = q32;
    if (threadIdx.x % QI8_1 == 0 && (ni == WARP_SIZE || threadIdx.x < ni)) {
        if (std::is_same<Tds, half2>::value) {
            ((half2  *) yds)[threadIdx.x/QI8_1] =  make_half2(d, sum);
        } else {
            ((float2 *) yds)[threadIdx.x/QI8_1] = make_float2(d, sum);
        }
    }
}

typedef void (*dequantize_V_t)(const void *, void *, const int64_t);

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_f16(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    if constexpr (std::is_same_v<T, half>) {
        ggml_cuda_memcpy_1<ne*sizeof(half)>(dst, (const half *) vx + i0);
    } else if constexpr (std::is_same_v<T, float>) {
        static_assert(ne % 2 == 0, "bad ne");
        half2 tmp[ne/2];
        ggml_cuda_memcpy_1<ne*sizeof(half)>(tmp, (const half *) vx + i0);
        float2 * dst_f2 = (float2 *) dst;
#pragma unroll
        for (int l = 0; l < ne/2; ++l) {
            dst_f2[l] = __half22float2(tmp[l]);
        }
    } else {
        static_assert(std::is_same_v<T, void>, "unsupported type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q4_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const int64_t ib    =  i0          /  QK4_0;
    const int     iqs   =  i0          % (QK4_0/2);
    const int     shift = (i0 % QK4_0) / (QK4_0/2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne, 2>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;
    q = __vsubss4(q, 0x08080808);

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 d = __half2half2(x[ib].d);

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]);
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float d = x[ib].d;

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = d * q8[l];
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q4_1(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const int64_t ib    =  i0          /  QK4_1;
    const int     iqs   =  i0          % (QK4_1/2);
    const int     shift = (i0 % QK4_1) / (QK4_1/2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 dm = x[ib].dm;
        const half2 d  = __half2half2( __low2half(dm));
        const half2 m  = __half2half2(__high2half(dm));

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]) + m;
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float2 dm = __half22float2(x[ib].dm);

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = dm.x * q8[l] + dm.y;
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q5_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const int64_t ib    =  i0          /  QK5_0;
    const int     idq   =  i0          %  QK5_0;
    const int     iqs   =  i0          % (QK5_0/2);
    const int     shift = (i0 % QK5_0) / (QK5_0/2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne, 2>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    {
        int qh;
        ggml_cuda_memcpy_1<ne, 2>(&qh, x[ib].qh);
#pragma unroll
        for (int l = 0; l < ne; ++l) {
            q |= ((qh >> (idq + l)) & 0x00000001) << (8*l + 4);
        }
    }

    q = __vsubss4(q, 0x10101010);

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 d = __half2half2(x[ib].d);

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]);
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float d = x[ib].d;

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = d * q8[l];
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q5_1(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const int64_t ib    =  i0          /  QK5_1;
    const int     idq   =  i0          %  QK5_1;
    const int     iqs   =  i0          % (QK5_1/2);
    const int     shift = (i0 % QK5_1) / (QK5_1/2);

    int q;
    static_assert(ne == 2 || ne == 4, "bad ne");
    ggml_cuda_memcpy_1<ne>(&q, x[ib].qs + iqs);
    q >>= 4*shift;
    q &= 0x0F0F0F0F;

    {
        int qh;
        ggml_cuda_memcpy_1<ne>(&qh, x[ib].qh);
#pragma unroll
        for (int l = 0; l < ne; ++l) {
            q |= ((qh >> (idq + l)) & 0x00000001) << (8*l + 4);
        }
    }

    const int8_t * q8 = (const int8_t *) &q;

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same_v<T, half>) {
        const half2 dm = x[ib].dm;
        const half2 d  = __half2half2( __low2half(dm));
        const half2 m  = __half2half2(__high2half(dm));

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(q8[l0 + 0], q8[l0 + 1]) + m;
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same_v<T, float>) {
        const float2 dm = __half22float2(x[ib].dm);

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = dm.x * q8[l] + dm.y;
        }
    } else {
        static_assert(std::is_same_v<T, void>, "bad type");
    }
}

template <typename T, int ne>
static __device__ __forceinline__ void dequantize_V_q8_0(const void * __restrict__ vx, void * __restrict__ dst, const int64_t i0) {
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const int64_t ib  = i0 / QK8_0;
    const int     iqs = i0 % QK8_0;

    static_assert(ne % 2 == 0, "bad ne");
    int8_t qs[ne];
    ggml_cuda_memcpy_1<ne, 2>(qs, x[ib].qs + iqs);

#ifdef FP16_AVAILABLE
    if constexpr (std::is_same<T, half>::value) {
        const half2 d = __half2half2(x[ib].d);

#pragma unroll
        for (int l0 = 0; l0 < ne; l0 += 2) {
            ((half2 *) dst)[l0/2] = d * make_half2(qs[l0 + 0], qs[l0 + 1]);
        }
    } else
#endif // FP16_AVAILABLE
    if constexpr (std::is_same<T, float>::value) {
        const float d = x[ib].d;

#pragma unroll
        for (int l = 0; l < ne; ++l) {
            ((float *) dst)[l] = d * qs[l];
        }
    } else {
        static_assert(std::is_same_v<T, void>, "unsupported type");
    }
}

template <ggml_type type_K, int D, int nthreads>
constexpr __device__ vec_dot_KQ_t get_vec_dot_KQ() {
    if constexpr (type_K == GGML_TYPE_F16) {
        return vec_dot_fattn_vec_KQ_f16<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q4_0) {
        return vec_dot_fattn_vec_KQ_q4_0<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q4_1) {
        return vec_dot_fattn_vec_KQ_q4_1<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q5_0) {
        return vec_dot_fattn_vec_KQ_q5_0<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q5_1) {
        return vec_dot_fattn_vec_KQ_q5_1<D, nthreads>;
    } else if constexpr (type_K == GGML_TYPE_Q8_0) {
        return vec_dot_fattn_vec_KQ_q8_0<D, nthreads>;
    } else {
        static_assert(type_K == -1, "bad type");
        return nullptr;
    }
}

template <ggml_type type_V, typename T, int ne>
constexpr __device__ dequantize_V_t get_dequantize_V() {
    if constexpr (type_V == GGML_TYPE_F16) {
        return dequantize_V_f16<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q4_0) {
        return dequantize_V_q4_0<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q4_1) {
        return dequantize_V_q4_1<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q5_0) {
        return dequantize_V_q5_0<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q5_1) {
        return dequantize_V_q5_1<T, ne>;
    } else if constexpr (type_V == GGML_TYPE_Q8_0) {
        return dequantize_V_q8_0<T, ne>;
    } else {
        static_assert(type_V == -1, "bad type");
        return nullptr;
    }
}

template <int ncols1>
__launch_bounds__(FATTN_KQ_STRIDE/2, 1)
static __global__ void flash_attn_mask_to_KV_max(
        const half2 * __restrict__ mask, int * __restrict__ KV_max, const int ne30, const int s31, const int s33) {
    const int ne31     = gridDim.x;
    const int tid      = threadIdx.x;
    const int sequence = blockIdx.y;
    const int jt       = blockIdx.x;

    mask += sequence*s33 + jt*ncols1*s31;

    __shared__ int buf_iw[WARP_SIZE];
    if (tid < WARP_SIZE) {
        buf_iw[tid] = 1;
    }
    __syncthreads();

    int KV_max_sj = (ne30 - 1) * FATTN_KQ_STRIDE;
    for (; KV_max_sj >= 0; KV_max_sj -= FATTN_KQ_STRIDE) {
        int all_inf = 1;

#pragma unroll
        for (int j = 0; j < ncols1; ++j) {
            const float2 tmp = __half22float2(mask[j*s31 + KV_max_sj/2 + tid]);
            all_inf = all_inf && int(isinf(tmp.x)) && int(isinf(tmp.y));
        }

        all_inf = warp_reduce_all(all_inf);
        if (tid % WARP_SIZE == 0) {
            buf_iw[tid / WARP_SIZE] = all_inf;
        }
        __syncthreads();
        all_inf = buf_iw[tid % WARP_SIZE];
        __syncthreads();
        all_inf = warp_reduce_all(all_inf);

        if (!all_inf) {
            break;
        }
    }

    // If the break in the loop was not triggered, KV_max_sj is now -FATTN_KQ_STRIDE.
    // If the break was triggered it's the lower edge of the tile with the first non-masked values.
    // In either case, walk back the decrementation by FATTN_KQ_STRIDE.
    KV_max_sj += FATTN_KQ_STRIDE;

    if (threadIdx.x != 0) {
        return;
    }

    KV_max[sequence*ne31 + jt] = KV_max_sj;
}

template<int D, int ncols1, int ncols2> // D == head size
__launch_bounds__(D, 1)
static __global__ void flash_attn_stream_k_fixup(
        float * __restrict__ dst, const float2 * __restrict__ dst_fixup, const int ne01, const int ne02, const int ne03, const int ne11,
        const int nbatch_fa) {
    constexpr int ncols = ncols1*ncols2;

    const int bidx0 = blockIdx.x;
    const int j     = blockIdx.y;
    const int c     = blockIdx.z;
    const int jc    = j*ncols2 + c;
    const int tid   = threadIdx.x;

    const float * dst_fixup_data = ((const float *) dst_fixup) + gridDim.x*(2*2*ncols);

    const int iter_k = (ne11 + (nbatch_fa - 1)) / nbatch_fa;
    const int iter_j = (ne01 + (ncols1    - 1)) / ncols1;

    const int kbc0      = int64_t(bidx0 + 0)*(iter_k*iter_j*(ne02/ncols2)*ne03) / gridDim.x;
    const int kbc0_stop = int64_t(bidx0 + 1)*(iter_k*iter_j*(ne02/ncols2)*ne03) / gridDim.x;

    const bool did_not_have_any_data   = kbc0 == kbc0_stop;
    const bool wrote_beginning_of_tile = kbc0 % iter_k == 0;
    const bool did_not_write_last      = kbc0/iter_k == kbc0_stop/iter_k && kbc0_stop % iter_k != 0;
    // DEBUG: Trace fixup kernel entry for first few blocks
    if (bidx0 < 2 && j == 0 && c == 0 && tid == 0) {
        printf("[FIXUP ENTRY] bidx0=%d j=%d c=%d: kbc0=%d kbc0_stop=%d iter_k=%d iter_j=%d\n",
               bidx0, j, c, kbc0, kbc0_stop, iter_k, iter_j);
        printf("[FIXUP ENTRY]   did_not_have_any_data=%d wrote_beginning_of_tile=%d did_not_write_last=%d\n",
               did_not_have_any_data, wrote_beginning_of_tile, did_not_write_last);
    }

    if (did_not_have_any_data || wrote_beginning_of_tile || did_not_write_last) {
        return;
    }

    // DEBUG: This block will modify the output!
    if (j == 0 && c == 0 && tid == 0) {
        printf("[FIXUP ACTIVE] bidx0=%d WILL MODIFY OUTPUT! kbc0=%d kbc0_stop=%d\n",
               bidx0, kbc0, kbc0_stop);
    }

    const int sequence = kbc0 / (iter_k*iter_j*(ne02/ncols2));
    const int head = (kbc0 - iter_k*iter_j*(ne02/ncols2)*sequence) / (iter_k*iter_j);
    const int jt = (kbc0 - iter_k*iter_j*(ne02/ncols2)*sequence - iter_k*iter_j*head) / iter_k; // j index of current tile.

    if (jt*ncols1 + j >= ne01) {
        return;
    }

    dst += sequence*ne02*ne01*D + jt*ne02*(ncols1*D) + head*(ncols2*D) + (j*ne02 + c)*D + tid;

    // Load the partial result that needs a fixup:
    float dst_val = 0.0f;
    float max_val = 0.0f;
    float rowsum  = 0.0f;
    {
        dst_val = *dst;

        const float2 tmp = dst_fixup[bidx0*ncols + jc];
        max_val = tmp.x;
        rowsum  = tmp.y;
    }

    // DEBUG: Show loaded values for first thread
    if (j == 0 && c == 0 && tid == 0) {
        printf("[FIXUP LOAD] bidx0=%d: dst_val=%f max_val=%f rowsum=%f\n",
               bidx0, dst_val, max_val, rowsum);
    }

    // Iterate over previous blocks and compute the combined results.
    // All CUDA blocks that get here must have a previous block that needs a fixup.
    int bidx = bidx0 - 1;
    int kbc_stop = kbc0;
    while(true) {
        const int kbc = int64_t(bidx)*(iter_k*iter_j*(ne02/ncols2)*ne03) / gridDim.x;
        if (kbc == kbc_stop) { // Did not have any data.
            bidx--;
            kbc_stop = kbc;
            continue;
        }

        const float dst_add = dst_fixup_data[bidx*ncols*D + jc*D + tid];

        const float2 tmp = dst_fixup[(gridDim.x + bidx)*ncols + jc];

        // Scale the current and new value accumulators depending on the max. values.
        const float max_val_new = fmaxf(max_val, tmp.x);

        const float diff_val = max_val - max_val_new;
        const float diff_add = tmp.x   - max_val_new;

        const float scale_val = diff_val >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_val) : 0.0f;
        const float scale_add = diff_add >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_add) : 0.0f;

        // DEBUG: Show scaling values
        if (j == 0 && c == 0 && tid == 0) {
            printf("[FIXUP COMBINE] bidx0=%d bidx=%d: max_val=%f tmp.x=%f diff_val=%f diff_add=%f\n",
                   bidx0, bidx, max_val, tmp.x, diff_val, diff_add);
            printf("[FIXUP COMBINE]   scale_val=%f scale_add=%f dst_add=%f tmp.y=%f\n",
                   scale_val, scale_add, dst_add, tmp.y);
        }

        dst_val = scale_val*dst_val + scale_add*dst_add;
        rowsum  = scale_val*rowsum  + scale_add*tmp.y;

        max_val = max_val_new;

        // If this block started in a previous tile we are done and don't need to combine additional partial results.
        if (kbc % iter_k == 0 || kbc/iter_k < kbc0/iter_k) {
            break;
        }
        bidx--;
        kbc_stop = kbc;
    }

    // DEBUG: Show final values before write
    if (j == 0 && c == 0 && tid == 0) {
        const float final_val = dst_val / rowsum;
        printf("[FIXUP WRITE] bidx0=%d: dst_val=%f rowsum=%f final=%f\n",
               bidx0, dst_val, rowsum, final_val);
        if (isnan(final_val) || isinf(final_val) || fabsf(final_val) < 1e-20f) {
            printf("[FIXUP WRITE] *** WARNING: Writing bad value! ***\n");
        }
    }

    // Write back final result:
    *dst = dst_val / rowsum;
}

template<int D> // D == head size
__launch_bounds__(D, 1)
static __global__ void flash_attn_combine_results(
        const float  * __restrict__ VKQ_parts,
        const float2 * __restrict__ VKQ_meta,
        float * __restrict__ dst,
        const int parallel_blocks) {
    // Dimension 0: threadIdx.x
    // Dimension 1: blockIdx.x
    // Dimension 2: blockIdx.y
    // Dimension 3: blockIdx.z
    // Memory layout is permuted with [0, 2, 1, 3]

    const int ne01 = gridDim.x;
    const int ne02 = gridDim.y;

    const int col      = blockIdx.x;
    const int head     = blockIdx.y;
    const int sequence = blockIdx.z;

    const int j_dst_unrolled = (sequence*ne01 + col)*ne02 + head;

    VKQ_parts += j_dst_unrolled * parallel_blocks*D;
    VKQ_meta  += j_dst_unrolled * parallel_blocks;
    dst       += j_dst_unrolled *                 D;

    const int tid = threadIdx.x;
    __builtin_assume(tid < D);

    extern __shared__ float2 meta[];
    for (int i = tid; i < 2*parallel_blocks; i += D) {
        ((float *) meta)[i] = ((const float *)VKQ_meta) [i];
    }

    __syncthreads();

    float kqmax = meta[0].x;
    for (int l = 1; l < parallel_blocks; ++l) {
        kqmax = max(kqmax, meta[l].x);
    }

    float VKQ_numerator   = 0.0f;
    float VKQ_denominator = 0.0f;
    for (int l = 0; l < parallel_blocks; ++l) {
        const float KQ_max_scale = expf(meta[l].x - kqmax);

        VKQ_numerator   += KQ_max_scale * VKQ_parts[l*D + tid];
        VKQ_denominator += KQ_max_scale * meta[l].y;
    }

    dst[tid] = VKQ_numerator / VKQ_denominator;
}

template <int DV, int ncols1, int ncols2, typename KernelFunc, typename... Args>
void launch_fattn(
    ggml_backend_cuda_context & ctx, ggml_tensor * dst, KernelFunc fattn_kernel, const int nwarps, const size_t nbytes_shared,
    const int nbatch_fa, const bool need_f16_K, const bool need_f16_V, const bool stream_k, const int warp_size,
    Args... args
) {
    constexpr int ncols = ncols1 * ncols2;

    const bool is_mla = DV == 512; // TODO better parameterization

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    GGML_ASSERT(V || is_mla);

    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    ggml_tensor * KQV = dst;

    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(KQV->type == GGML_TYPE_F32);

    GGML_ASSERT(      Q->nb[0] == ggml_element_size(Q));
    GGML_ASSERT(      K->nb[0] == ggml_element_size(K));
    GGML_ASSERT(!V || V->nb[0] == ggml_element_size(V));

    GGML_ASSERT(!mask || mask->type == GGML_TYPE_F16);

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t main_stream = ctx.stream();
    const int id  = ggml_cuda_get_device();
    const int cc  = ggml_cuda_info().devices[id].cc;
    const int nsm = ggml_cuda_info().devices[id].nsm;

    ggml_cuda_pool_alloc<half>   K_f16(pool);
    ggml_cuda_pool_alloc<half>   V_f16(pool);
    ggml_cuda_pool_alloc<int>    KV_max(pool);
    ggml_cuda_pool_alloc<float>  dst_tmp(pool);
    ggml_cuda_pool_alloc<float2> dst_tmp_meta(pool);

    const char * K_data = (const char *) K->data;
    size_t nb11 = K->nb[1];
    size_t nb12 = K->nb[2];
    size_t nb13 = K->nb[3];

    const char * V_data = V ? (const char *) V->data : nullptr;
    size_t nb21 = V ? V->nb[1] : nb11;
    size_t nb22 = V ? V->nb[2] : nb12;
    size_t nb23 = V ? V->nb[3] : nb13;

    if (need_f16_K && K->type != GGML_TYPE_F16) {
        const size_t bs = ggml_blck_size(K->type);
        const size_t ts = ggml_type_size(K->type);

        K_f16.alloc(ggml_nelements(K));
        if (ggml_is_contiguously_allocated(K)) {
            to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(K->type);
            to_fp16(K_data, K_f16.ptr, ggml_nelements(K), main_stream);

            nb11 = nb11*bs*sizeof(half)/ts;
            nb12 = nb12*bs*sizeof(half)/ts;
            nb13 = nb13*bs*sizeof(half)/ts;
        } else {
            GGML_ASSERT(K->nb[0] == ts);
            to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(K->type);
            const int64_t s01 = nb11 / ts;
            const int64_t s02 = nb12 / ts;
            const int64_t s03 = nb13 / ts;
            to_fp16(K_data, K_f16.ptr, K->ne[0], K->ne[1], K->ne[2], K->ne[3], s01, s02, s03, main_stream);

            nb11 = K->ne[0] * sizeof(half);
            nb12 = K->ne[1] * nb11;
            nb13 = K->ne[2] * nb12;
        }
        K_data = (char *) K_f16.ptr;
    }

    if (V && need_f16_V && V->type != GGML_TYPE_F16) {
        const size_t bs = ggml_blck_size(V->type);
        const size_t ts = ggml_type_size(V->type);

        V_f16.alloc(ggml_nelements(V));
        if (ggml_is_contiguously_allocated(V)) {
            to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(V->type);
            to_fp16(V_data, V_f16.ptr, ggml_nelements(V), main_stream);
            V_data = (char *) V_f16.ptr;

            nb21 = nb21*bs*sizeof(half)/ts;
            nb22 = nb22*bs*sizeof(half)/ts;
            nb23 = nb23*bs*sizeof(half)/ts;
        } else {
            GGML_ASSERT(V->nb[0] == ts);
            to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(V->type);
            const int64_t s01 = nb21 / ts;
            const int64_t s02 = nb22 / ts;
            const int64_t s03 = nb23 / ts;
            to_fp16(V_data, V_f16.ptr, V->ne[0], V->ne[1], V->ne[2], V->ne[3], s01, s02, s03, main_stream);

            nb21 = V->ne[0] * sizeof(half);
            nb22 = V->ne[1] * nb21;
            nb23 = V->ne[2] * nb22;
        }
        V_data = (char *) V_f16.ptr;
    }

    const int ntiles_x = ((Q->ne[1] + ncols1 - 1) / ncols1);
    const int ntiles_total = ntiles_x * (Q->ne[2] / ncols2) * Q->ne[3];

    // Optional optimization where the mask is scanned to determine whether part of the calculation can be skipped.
    // Only worth the overhead if there is at lease one FATTN_KQ_STRIDE x FATTN_KQ_STRIDE square to be skipped or
    //     multiple sequences of possibly different lengths.
    if (mask && K->ne[1] % FATTN_KQ_STRIDE == 0 && (Q->ne[1] >= 1024 || Q->ne[3] > 1)) {
        const int s31 = mask->nb[1] / sizeof(half2);
        const int s33 = mask->nb[3] / sizeof(half2);

        const dim3 blocks_num_KV_max(ntiles_x, Q->ne[3], 1);
        const dim3 block_dim_KV_max(FATTN_KQ_STRIDE/2, 1, 1);

        const int ne_KV_max = blocks_num_KV_max.x*blocks_num_KV_max.y;
        const int iter_k = K->ne[1] / FATTN_KQ_STRIDE;

        KV_max.alloc(ne_KV_max);
        flash_attn_mask_to_KV_max<ncols1><<<blocks_num_KV_max, block_dim_KV_max, 0, main_stream>>>
            ((const half2 *) mask->data, KV_max.ptr, iter_k, s31, s33);
        CUDA_CHECK(cudaGetLastError());
    }

    const dim3 block_dim(warp_size, nwarps, 1);
    int max_blocks_per_sm = 1; // Max. number of active blocks limited by occupancy.
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, fattn_kernel, block_dim.x * block_dim.y * block_dim.z, nbytes_shared));
    if (max_blocks_per_sm <= 0) {
        fprintf(stderr, "FATTN ERROR: max_blocks_per_sm = %d\n", max_blocks_per_sm);
        fprintf(stderr, "  nbytes_shared = %lu\n", nbytes_shared);
        fprintf(stderr, "  block_dim = %d, %d, %d (total %d)\n", block_dim.x, block_dim.y, block_dim.z, block_dim.x * block_dim.y * block_dim.z);
        int dev_id;
        cudaGetDevice(&dev_id);
        int max_shmem = 0;
        cudaDeviceGetAttribute(&max_shmem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev_id);
        fprintf(stderr, "  max_shmem_optin = %d\n", max_shmem);
        int max_regs = 0;
        cudaDeviceGetAttribute(&max_regs, cudaDevAttrMaxRegistersPerBlock, dev_id);
        fprintf(stderr, "  max_regs_per_block = %d\n", max_regs);
        
        cudaFuncAttributes attrs;
        if (cudaFuncGetAttributes(&attrs, fattn_kernel) == cudaSuccess) {
            fprintf(stderr, "  kernel regs = %d\n", attrs.numRegs);
            fprintf(stderr, "  kernel shmem (static) = %lu\n", attrs.sharedSizeBytes);
            fprintf(stderr, "  kernel local = %lu\n", attrs.localSizeBytes);
        }
    }
    GGML_ASSERT(max_blocks_per_sm > 0);
    int parallel_blocks = max_blocks_per_sm;

    dim3 blocks_num;
    if (stream_k) {
        // For short contexts it can be faster to have the SMs work on whole tiles because this lets us skip the fixup.
        const int max_blocks = max_blocks_per_sm*nsm;
        const int tiles_nwaves = (ntiles_total + max_blocks - 1) / max_blocks;
        const int tiles_efficiency_percent = 100 * ntiles_total / (max_blocks*tiles_nwaves);

        // Stream-K distributes work across K tiles too: iter_k * iter_j * (ne02/ncols2) * ne03
        // Cap blocks to actual work units to avoid blocks with no work (which causes uninitialized output)
        const int ntiles_KQ = (K->ne[1] + nbatch_fa - 1) / nbatch_fa; // iter_k
        const int total_work_stream_k = ntiles_KQ * ntiles_total;     // iter_k * iter_j * heads * batch
        const int nblocks_stream_k = std::min(max_blocks, total_work_stream_k);

        const bool use_stream_k = cc >= GGML_CUDA_CC_ADA_LOVELACE || tiles_efficiency_percent < 75;

        blocks_num.x = use_stream_k ? nblocks_stream_k : ntiles_total;
        blocks_num.y = 1;
        blocks_num.z = 1;

        // Debug: Show Stream-K work distribution
        static bool streamk_debug = false;
        if (!streamk_debug) {
            fprintf(stderr, "[STREAM-K DEBUG] max_blocks=%d, ntiles_KQ=%d, ntiles_total=%d\n",
                    max_blocks, ntiles_KQ, ntiles_total);
            fprintf(stderr, "[STREAM-K DEBUG] total_work_stream_k=%d, nblocks_stream_k=%d (capped: %s)\n",
                    total_work_stream_k, nblocks_stream_k,
                    (nblocks_stream_k < max_blocks) ? "YES" : "NO");
            fprintf(stderr, "[STREAM-K DEBUG] use_stream_k=%d, blocks_num.x=%d\n",
                    use_stream_k, blocks_num.x);
            streamk_debug = true;
        }

        dst_tmp_meta.alloc(blocks_num.x*ncols * (2*2 + DV) * sizeof(float));
    } else {
        const int ntiles_KQ = (K->ne[1] + nbatch_fa - 1) / nbatch_fa; // Max. number of parallel blocks limited by tensor size.

        // parallel_blocks must not be larger than what the tensor size allows:
        parallel_blocks = std::min(parallel_blocks, ntiles_KQ);

        // If ntiles_total % blocks_per_wave != 0 then some efficiency is lost due to tail effects.
        // Test whether parallel_blocks can be set to a higher value for better efficiency.
        const int blocks_per_wave = nsm * max_blocks_per_sm;
        int nwaves_best = 0;
        int efficiency_percent_best = 0;
        for (int parallel_blocks_test = parallel_blocks; parallel_blocks_test <= ntiles_KQ; ++parallel_blocks_test) {
            const int nblocks_total = ntiles_total * parallel_blocks_test;
            const int nwaves = (nblocks_total + blocks_per_wave - 1) / blocks_per_wave;
            const int efficiency_percent = 100 * nblocks_total / (nwaves*blocks_per_wave);

            // Stop trying configurations with more waves if we already have good efficiency to avoid excessive overhead.
            if (efficiency_percent_best >= 95 && nwaves > nwaves_best) {
                break;
            }

            if (efficiency_percent > efficiency_percent_best) {
                nwaves_best = nwaves;
                efficiency_percent_best = efficiency_percent;
                parallel_blocks = parallel_blocks_test;
            }
        }

        blocks_num.x = ntiles_x;
        blocks_num.y = parallel_blocks;
        blocks_num.z = (Q->ne[2]/ncols2)*Q->ne[3];

        if (parallel_blocks > 1) {
            dst_tmp.alloc(parallel_blocks*ggml_nelements(KQV));
            dst_tmp_meta.alloc(parallel_blocks*ggml_nrows(KQV));
        }
    }

    float scale         = 1.0f;
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;

    memcpy(&scale,         (const float *) KQV->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) KQV->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    const uint32_t n_head      = Q->ne[2];
    const uint32_t n_head_log2 = 1u << uint32_t(floorf(log2f(float(n_head))));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    // TODO other tensor dimensions after removal of WMMA kernel:
    const uint3 ne01 = init_fastdiv_values(Q->ne[1]);

    // L2 cache persistence for K/V cache on Blackwell (RTX 5090 has 96MB L2)
    // Persists K tensor in L2 during decode for faster repeated access
#if CUDART_VERSION >= 11000
    const size_t kv_cache_size = K->ne[0] * K->ne[1] * sizeof(half);
    ggml_cuda_l2_persist_guard l2_guard(main_stream, const_cast<char *>(K_data), kv_cache_size, cc);
#endif

    GGML_ASSERT(block_dim.x % warp_size == 0);
    
    // Debug: Print kernel launch parameters
    {
        static bool launch_debug_printed = false;
        if (!launch_debug_printed) {
            fprintf(stderr, "\n[LAUNCH DEBUG] ===== Flash Attention Kernel Launch =====\n");
            fprintf(stderr, "[LAUNCH DEBUG] Grid: (%u, %u, %u)\n", blocks_num.x, blocks_num.y, blocks_num.z);
            fprintf(stderr, "[LAUNCH DEBUG] Block: (%u, %u, %u) = %u threads\n", 
                    block_dim.x, block_dim.y, block_dim.z, 
                    block_dim.x * block_dim.y * block_dim.z);
            fprintf(stderr, "[LAUNCH DEBUG] Shared memory: %zu bytes (%.1f KB)\n", 
                    nbytes_shared, nbytes_shared / 1024.0);
            fprintf(stderr, "[LAUNCH DEBUG] Q->data=%p, K_data=%p, V_data=%p\n", 
                    Q->data, K_data, V_data);
            fprintf(stderr, "[LAUNCH DEBUG] Q dims: ne0=%ld, ne1=%ld, ne2=%ld, ne3=%ld\n",
                    (long)Q->ne[0], (long)Q->ne[1], (long)Q->ne[2], (long)Q->ne[3]);
            fprintf(stderr, "[LAUNCH DEBUG] K dims: ne0=%ld, ne1=%ld, ne2=%ld, ne3=%ld\n",
                    (long)K->ne[0], (long)K->ne[1], (long)K->ne[2], (long)K->ne[3]);
            
            // Check device shared memory limits
            int dev_id;
            cudaGetDevice(&dev_id);
            int max_shmem = 0;
            cudaDeviceGetAttribute(&max_shmem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev_id);
            fprintf(stderr, "[LAUNCH DEBUG] Device %d max shared mem (optin): %d bytes (%.1f KB)\n", 
                    dev_id, max_shmem, max_shmem / 1024.0);
            fprintf(stderr, "[LAUNCH DEBUG] Shared mem OK: %s\n", 
                    nbytes_shared <= (size_t)max_shmem ? "YES" : "NO - TOO LARGE!");
            
            // HOST SIDE: Verify K tensor values from GPU memory before kernel launch
            fprintf(stderr, "[K VERIFY HOST] Reading K tensor from GPU memory...\n");
            half k_row0[8], k_row1[8], k_row2[8], k_row3[8];
            cudaMemcpy(k_row0, K_data, sizeof(k_row0), cudaMemcpyDeviceToHost);
            cudaMemcpy(k_row1, (const char*)K_data + nb11, sizeof(k_row1), cudaMemcpyDeviceToHost);
            cudaMemcpy(k_row2, (const char*)K_data + 2*nb11, sizeof(k_row2), cudaMemcpyDeviceToHost);
            cudaMemcpy(k_row3, (const char*)K_data + 3*nb11, sizeof(k_row3), cudaMemcpyDeviceToHost);
            
            fprintf(stderr, "[K VERIFY HOST] K row 0: %f, %f, %f, %f | row 1: %f, %f, %f, %f\n",
                    __half2float(k_row0[0]), __half2float(k_row0[1]), 
                    __half2float(k_row0[2]), __half2float(k_row0[3]),
                    __half2float(k_row1[0]), __half2float(k_row1[1]),
                    __half2float(k_row1[2]), __half2float(k_row1[3]));
            fprintf(stderr, "[K VERIFY HOST] K row 2: %f, %f, %f, %f | row 3: %f, %f, %f, %f\n",
                    __half2float(k_row2[0]), __half2float(k_row2[1]), 
                    __half2float(k_row2[2]), __half2float(k_row2[3]),
                    __half2float(k_row3[0]), __half2float(k_row3[1]),
                    __half2float(k_row3[2]), __half2float(k_row3[3]));
            
            bool k_has_nan = false, k_row1_zero = true, k_row2_zero = true, k_row3_zero = true;
            for (int i = 0; i < 4; i++) {
                if (isnan(__half2float(k_row1[i])) || isnan(__half2float(k_row2[i])) || isnan(__half2float(k_row3[i]))) k_has_nan = true;
                if (__half2float(k_row1[i]) != 0.0f) k_row1_zero = false;
                if (__half2float(k_row2[i]) != 0.0f) k_row2_zero = false;
                if (__half2float(k_row3[i]) != 0.0f) k_row3_zero = false;
            }
            if (k_has_nan) fprintf(stderr, "[K VERIFY HOST] *** WARNING: K tensor has NaN! ***\n");
            if (k_row1_zero) fprintf(stderr, "[K VERIFY HOST] *** WARNING: K row 1 is all zeros! ***\n");
            if (k_row2_zero) fprintf(stderr, "[K VERIFY HOST] *** WARNING: K row 2 is all zeros! ***\n");
            if (k_row3_zero) fprintf(stderr, "[K VERIFY HOST] *** WARNING: K row 3 is all zeros! ***\n");
            
            fprintf(stderr, "[LAUNCH DEBUG] ==========================================\n\n");
            launch_debug_printed = true;
        }
    }
    
    // CRITICAL FIX: Explicit casts to match kernel signature types.
    // The kernel expects int32_t for nb01/nb02/nb03, nb11/nb12, nb21/nb22, nb31/nb32
    // but Q->nb[], K->nb[], V->nb[], mask->nb[] are size_t (64-bit).
    // Without casts, CUDA parameter buffer gets misaligned, corrupting all subsequent params!
    fattn_kernel<<<blocks_num, block_dim, nbytes_shared, main_stream>>>(
        (const char *) Q->data,
        K_data,
        V_data,
        mask ? ((const char *) mask->data) : nullptr,
        sinks ? ((const char *) sinks->data) : nullptr,
        KV_max.ptr,
        !stream_k && parallel_blocks > 1 ? dst_tmp.ptr : (float *) KQV->data, dst_tmp_meta.ptr,
        scale, max_bias, m0, m1, n_head_log2, logit_softcap,
        (int32_t)Q->ne[0], ne01, (int32_t)Q->ne[2], (int32_t)Q->ne[3],
        (int32_t)Q->nb[1], (int32_t)Q->nb[2], (int32_t)Q->nb[3],
        (int32_t)K->ne[0], (int32_t)K->ne[1], (int32_t)K->ne[2], (int32_t)K->ne[3],
        (int32_t)nb11, (int32_t)nb12, (int64_t)nb13,
        (int32_t)nb21, (int32_t)nb22, (int64_t)nb23,
        mask ? (int32_t)mask->ne[1] : 0, mask ? (int32_t)mask->ne[2] : 0, mask ? (int32_t)mask->ne[3] : 0,
        mask ? (int32_t)mask->nb[1] : 0, mask ? (int32_t)mask->nb[2] : 0, mask ? (int64_t)mask->nb[3] : 0,
        args...
    );
    
    // Sync and check error immediately for debugging
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        fprintf(stderr, "[LAUNCH DEBUG] !!! KERNEL LAUNCH FAILED !!!\n");
        fprintf(stderr, "[LAUNCH DEBUG] Error: %s\n", cudaGetErrorString(launch_err));
    }
    CUDA_CHECK(launch_err);

    // =========================================================================
    // DEBUG: Synchronize and check for runtime errors (illegal memory access)
    // This isolates whether flash attention or a later kernel (like MMQ) crashes.
    // REMOVE THIS BLOCK AFTER DEBUGGING - it kills performance!
    // =========================================================================
    {
        static bool sync_debug_enabled = true;  // Set to false to disable sync
        static bool params_printed = false;

        if (sync_debug_enabled) {
            // NOTE: Skip during CUDA graph capture (sync not allowed)
            cudaStreamCaptureStatus capture_status;
            cudaStreamIsCapturing(main_stream, &capture_status);
            if (capture_status != cudaStreamCaptureStatusNone) {
                // Skip sync during graph capture
            } else {
            // Print comprehensive parameters BEFORE sync (in case of crash)
            if (!params_printed) {
                fprintf(stderr, "\n[FATTN PARAMS] ============ Flash Attention Parameters ============\n");
                fprintf(stderr, "[FATTN PARAMS] Kernel Launch:\n");
                fprintf(stderr, "[FATTN PARAMS]   Grid: (%u, %u, %u) = %u blocks\n",
                        blocks_num.x, blocks_num.y, blocks_num.z,
                        blocks_num.x * blocks_num.y * blocks_num.z);
                fprintf(stderr, "[FATTN PARAMS]   Block: (%u, %u, %u) = %u threads\n",
                        block_dim.x, block_dim.y, block_dim.z,
                        block_dim.x * block_dim.y * block_dim.z);
                fprintf(stderr, "[FATTN PARAMS]   Shared mem: %zu bytes (%.2f KB)\n", nbytes_shared, nbytes_shared/1024.0);
                fprintf(stderr, "[FATTN PARAMS]   nwarps: %d, warp_size: %d\n", nwarps, warp_size);
                fprintf(stderr, "[FATTN PARAMS]   parallel_blocks: %d, stream_k: %d\n", parallel_blocks, stream_k);
                fprintf(stderr, "[FATTN PARAMS]   max_blocks_per_sm: %d, nsm: %d\n", max_blocks_per_sm, nsm);

                fprintf(stderr, "[FATTN PARAMS] Q tensor:\n");
                fprintf(stderr, "[FATTN PARAMS]   ne: [%ld, %ld, %ld, %ld]\n",
                        (long)Q->ne[0], (long)Q->ne[1], (long)Q->ne[2], (long)Q->ne[3]);
                fprintf(stderr, "[FATTN PARAMS]   nb: [%zu, %zu, %zu, %zu]\n",
                        Q->nb[0], Q->nb[1], Q->nb[2], Q->nb[3]);
                fprintf(stderr, "[FATTN PARAMS]   data: %p, type: %d\n", Q->data, Q->type);

                fprintf(stderr, "[FATTN PARAMS] K tensor:\n");
                fprintf(stderr, "[FATTN PARAMS]   ne: [%ld, %ld, %ld, %ld]\n",
                        (long)K->ne[0], (long)K->ne[1], (long)K->ne[2], (long)K->ne[3]);
                fprintf(stderr, "[FATTN PARAMS]   nb: [%zu, %zu, %zu, %zu]\n",
                        K->nb[0], K->nb[1], K->nb[2], K->nb[3]);
                fprintf(stderr, "[FATTN PARAMS]   K_data: %p (converted: %s), type: %d\n",
                        K_data, (K_data != (const char*)K->data) ? "yes" : "no", K->type);
                fprintf(stderr, "[FATTN PARAMS]   nb11: %zu, nb12: %zu, nb13: %zu\n", nb11, nb12, nb13);

                if (V) {
                    fprintf(stderr, "[FATTN PARAMS] V tensor:\n");
                    fprintf(stderr, "[FATTN PARAMS]   ne: [%ld, %ld, %ld, %ld]\n",
                            (long)V->ne[0], (long)V->ne[1], (long)V->ne[2], (long)V->ne[3]);
                    fprintf(stderr, "[FATTN PARAMS]   nb: [%zu, %zu, %zu, %zu]\n",
                            V->nb[0], V->nb[1], V->nb[2], V->nb[3]);
                    fprintf(stderr, "[FATTN PARAMS]   V_data: %p (converted: %s), type: %d\n",
                            V_data, (V_data != (const char*)V->data) ? "yes" : "no", V->type);
                    fprintf(stderr, "[FATTN PARAMS]   nb21: %zu, nb22: %zu, nb23: %zu\n", nb21, nb22, nb23);
                }

                fprintf(stderr, "[FATTN PARAMS] Output (KQV):\n");
                fprintf(stderr, "[FATTN PARAMS]   ne: [%ld, %ld, %ld, %ld]\n",
                        (long)KQV->ne[0], (long)KQV->ne[1], (long)KQV->ne[2], (long)KQV->ne[3]);
                fprintf(stderr, "[FATTN PARAMS]   data: %p, type: %d\n", KQV->data, KQV->type);
                fprintf(stderr, "[FATTN PARAMS]   nelements: %ld, nbytes: %zu\n",
                        (long)ggml_nelements(KQV), ggml_nbytes(KQV));

                if (mask) {
                    fprintf(stderr, "[FATTN PARAMS] Mask:\n");
                    fprintf(stderr, "[FATTN PARAMS]   ne: [%ld, %ld, %ld, %ld]\n",
                            (long)mask->ne[0], (long)mask->ne[1], (long)mask->ne[2], (long)mask->ne[3]);
                    fprintf(stderr, "[FATTN PARAMS]   data: %p\n", mask->data);
                }

                fprintf(stderr, "[FATTN PARAMS] Attention params:\n");
                fprintf(stderr, "[FATTN PARAMS]   scale: %f, max_bias: %f, logit_softcap: %f\n",
                        scale, max_bias, logit_softcap);
                fprintf(stderr, "[FATTN PARAMS]   n_head: %u, n_head_log2: %u\n", n_head, n_head_log2);
                fprintf(stderr, "[FATTN PARAMS]   m0: %f, m1: %f\n", m0, m1);

                fprintf(stderr, "[FATTN PARAMS] Template params:\n");
                fprintf(stderr, "[FATTN PARAMS]   DV: %d, ncols1: %d, ncols2: %d, ncols: %d\n",
                        DV, ncols1, ncols2, ncols);
                fprintf(stderr, "[FATTN PARAMS]   nbatch_fa: %d, ntiles_x: %d, ntiles_total: %d\n",
                        nbatch_fa, ntiles_x, ntiles_total);

                fprintf(stderr, "[FATTN PARAMS] Device info:\n");
                fprintf(stderr, "[FATTN PARAMS]   device: %d, cc: %d\n", id, cc);

                fprintf(stderr, "[FATTN PARAMS] =========================================================\n\n");
                params_printed = true;
            }

            fprintf(stderr, "[FATTN SYNC DEBUG] Waiting for flash attention kernel to complete...\n");
            cudaError_t sync_err = cudaDeviceSynchronize();
            if (sync_err != cudaSuccess) {
                fprintf(stderr, "[FATTN SYNC DEBUG] !!! FLASH ATTENTION KERNEL CRASHED !!!\n");
                fprintf(stderr, "[FATTN SYNC DEBUG] Error: %s\n", cudaGetErrorString(sync_err));
                fprintf(stderr, "[FATTN SYNC DEBUG] Grid: (%u, %u, %u), Block: (%u, %u, %u)\n",
                        blocks_num.x, blocks_num.y, blocks_num.z,
                        block_dim.x, block_dim.y, block_dim.z);
                fprintf(stderr, "[FATTN SYNC DEBUG] Shared mem: %zu bytes\n", nbytes_shared);
                fprintf(stderr, "[FATTN SYNC DEBUG] See [FATTN PARAMS] above for full parameter dump\n");
                GGML_ABORT("Flash attention kernel execution failed");
            }
            fprintf(stderr, "[FATTN SYNC DEBUG] Flash attention completed successfully!\n");

            // Double-check with cudaGetLastError after sync
            cudaError_t post_sync_err = cudaGetLastError();
            if (post_sync_err != cudaSuccess) {
                fprintf(stderr, "[FATTN SYNC DEBUG] Post-sync error: %s\n", cudaGetErrorString(post_sync_err));
                GGML_ABORT("Flash attention post-sync error");
            }
            } // else capture_status == cudaStreamCaptureStatusNone
        }
    }
    // =========================================================================

    if (stream_k) {
        if (ntiles_total % blocks_num.x != 0) { // Fixup is only needed if the SMs work on fractional tiles.
            const dim3 block_dim_combine(DV, 1, 1);
            const dim3 blocks_num_combine = {blocks_num.x, ncols1, ncols2};

            flash_attn_stream_k_fixup<DV, ncols1, ncols2>
                <<<blocks_num_combine, block_dim_combine, 0, main_stream>>>
                ((float *) KQV->data, dst_tmp_meta.ptr, Q->ne[1], Q->ne[2], Q->ne[3], K->ne[1], nbatch_fa);
        }
    } else if (parallel_blocks > 1) {
        const dim3 block_dim_combine(DV, 1, 1);
        const dim3 blocks_num_combine(Q->ne[1], Q->ne[2], Q->ne[3]);
        const size_t nbytes_shared_combine = parallel_blocks*sizeof(float2);

        flash_attn_combine_results<DV>
            <<<blocks_num_combine, block_dim_combine, nbytes_shared_combine, main_stream>>>
            (dst_tmp.ptr, dst_tmp_meta.ptr, (float *) KQV->data, parallel_blocks);
    }
    CUDA_CHECK(cudaGetLastError());

    // =========================================================================
    // DEBUG: UNCONDITIONALLY check FA output - will ALWAYS print
    // =========================================================================
    {
        cudaStreamCaptureStatus capture_status;
        cudaStreamIsCapturing(main_stream, &capture_status);
        if (capture_status == cudaStreamCaptureStatusNone) {
            CUDA_CHECK(cudaStreamSynchronize(main_stream));

            float debug_vals[8];
            CUDA_CHECK(cudaMemcpy(debug_vals, KQV->data, sizeof(debug_vals), cudaMemcpyDeviceToHost));

            bool has_nan = false;
            for (int i = 0; i < 8; i++) {
                if (isnan(debug_vals[i]) || isinf(debug_vals[i])) {
                    has_nan = true;
                    break;
                }
            }

            fprintf(stderr, "[FA OUTPUT CHECK] KQV->data=%p vals[0..7]: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %s\n",
                    KQV->data,
                    debug_vals[0], debug_vals[1], debug_vals[2], debug_vals[3],
                    debug_vals[4], debug_vals[5], debug_vals[6], debug_vals[7],
                    has_nan ? "*** HAS NaN ***" : "(OK)");
        }
    }
}
