#pragma once
// This file contains primitives that expose the tensor core PTX instructions for CUDA code.
// The primitives can be used in a similar way as the nvcuda::wmma interface but with a well-defined memory layout.
// The documentation for the PTX instructions can be found under:
//   https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-multiply-accumulate-operation-using-mma-instruction
//
// Like with nvcuda::wmma there are three types of matrix tiles: A, B, and C with A @ B = C.
// A is a row-major matrix with shape M x K.
// B is a column-major matrix with shape K x N.
// C is a column-major matrix with shape M x N.
// A, B, and C are represented using the same fundamental data type: a row-major matrix with I rows and J columns.
// Note that J is measured in physical 32 bit elements instead of logical elements.
// The methods get_i and get_j can be used to get the physical 32 bit index of the lth element of a thread within a tile.
// All matrix tiles have ne physical 32 bit elements per warp.
//
// As described in the PTX documentation, all pointers for load_ldmatrix must be to shared memory and aligned to 16 bytes.
// The API in this file also assumes that the pointers for load_generic are aligned to 16 bytes, unaligned pointers are considered undefined behavior.

#include "common.cuh"

// On Volta each warp is doing 4 8x8 mma operations in parallel.
// The basic memory layout for a 32x8 output tile is to stack 4 input tiles in I direction and to mirror the B tile.
// However, the i indices in this file are by default permuted to simplify the index calculations.
// #define GGML_CUDA_MMA_NO_VOLTA_PERM

#if CUDART_VERSION >= 11080

static __device__ __forceinline__ int ggml_cuda_movmatrix(const int x) {
    int ret = 0;

#ifdef TURING_MMA_AVAILABLE
    asm("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;"
        : "=r"(ret) : "r"(x));
#else
    GGML_UNUSED(x);
    NO_DEVICE_CODE;
#endif // defined(TURING_MMA_AVAILABLE)
    return ret;
}

#else

static __device__ __forceinline__ int ggml_cuda_movmatrix(const int x) {
    // Imagine transposing row-major matrix to column-major matrix.
    const int src_i_low  = 2 * (threadIdx.x % 4);
    const int src_i_high = src_i_low + 1;
    const int src_j      = threadIdx.x / 4;

    const int src_laneid_low  = src_i_low  * 4 + src_j / 2;
    const int src_laneid_high = src_i_high * 4 + src_j / 2;

    const int shift_low  = ((src_j + 0) % 2) * 16;
    const int shift_high = ((src_j + 1) % 2) * 16;

    const int ret_low  = (__shfl_sync(0xFFFFFFFF, x, src_laneid_low,  WARP_SIZE) >> shift_low)  & 0x0000FFFF;
    const int ret_high = (__shfl_sync(0xFFFFFFFF, x, src_laneid_high, WARP_SIZE) << shift_high) & 0xFFFF0000;

    return ret_low | ret_high;
}

#endif // CUDART_VERSION >= 11080

static __device__ __forceinline__ half2 ggml_cuda_movmatrix(const half2 x) {
    half2 ret;
    *((int *) &ret) = ggml_cuda_movmatrix(*((const int *) &x));
    return ret;
}

namespace ggml_cuda_mma {

    // Some architectures like Volta or CDNA3 perform multiple matrix multiplications per warp in parallel,
    //     effectively the warp is being split into subgroups of threads that each perform a single mma instruction.
    // In those cases the data can be split in different ways across the warp.
    enum data_layout {
        // By default the data uses the I direction as its major dimension and the J direction as its minor dimension.
        // For the A/C matrices this means I major == row major, J major == column major.
        // For the B matrix this means I major == column major, J major == row major.
        // MIRRORED == Each data value is held exactly once per thread subgroup.
        DATA_LAYOUT_I_MAJOR           =  0, // Always used for Turing, Ampere, Ada Lovelace, consumer Blackwell, matrix A&B for RDNA4 and CDNA.
        DATA_LAYOUT_J_MAJOR           = 10, // Matrix C for CDNA and RDNA4, int and float matrix C for RDNA3.
        DATA_LAYOUT_I_MAJOR_MIRRORED  = 20, // Volta, matrix A&B for RDNA3.
        DATA_LAYOUT_J_MAJOR_MIRRORED  = 30,
    };
    // Implemented mma combinations are:
    //   - (I_MAJOR, I_MAJOR)          -> I_MAJOR
    //   - (I_MAJOR, I_MAJOR_MIRRORED) -> I_MAJOR
    //   - (I_MAJOR, J_MAJOR_MIRRORED) -> I_MAJOR

    static constexpr bool is_i_major(const data_layout dl) {
        return dl == DATA_LAYOUT_I_MAJOR ||
               dl == DATA_LAYOUT_I_MAJOR_MIRRORED;
    }

    static constexpr __device__ data_layout get_input_data_layout() {
        // For CUDA (sm_86+), always use I_MAJOR layout
        return DATA_LAYOUT_I_MAJOR;
    }

    template <int I_, int J_, typename T, data_layout ds_=DATA_LAYOUT_I_MAJOR>
    struct tile {};

    template <int I_, int J_, typename T>
    struct tile<I_, J_, T, DATA_LAYOUT_I_MAJOR> {
        static constexpr int         I  = I_;
        static constexpr int         J  = J_;
        static constexpr data_layout dl = DATA_LAYOUT_I_MAJOR;

        // NVIDIA Turing+ layout (sm_75+)
        static constexpr int ne = I * J / 32;
        T x[ne] = {0};

        static constexpr __device__ bool supported() {
            if (I ==  8 && J ==  4) return true;
            if (I ==  8 && J ==  8) return true;
            if (I == 16 && J ==  8) return true;
            if (I == 16 && J == 16) return true;
            if (I == 32 && J ==  8) return true;
            return false;
        }

        static __device__ __forceinline__ int get_i(const int l) {
            if constexpr (I == 8 && J == 4) {
                return threadIdx.x / 4;
            } else if constexpr (I == 8 && J == 8) {
                return threadIdx.x / 4;
            } else if constexpr (I == 16 && J == 8) {
                return ((l / 2) * 8) + (threadIdx.x / 4);
            } else if constexpr (I == 16 && J == 16) {
                return (((l / 2) % 2) * 8) + (threadIdx.x / 4);
            } else if constexpr (I == 32 && J == 8) {
                return tile<16, 8, T>::get_i(l); // Memory layout simply repeated with same pattern in i direction.
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }

        static __device__ __forceinline__ int get_j(const int l) {
            if constexpr (I == 8 && J == 4) {
                return threadIdx.x % 4;
            } else if constexpr (I == 8 && J == 8) {
                return (l * 4) + (threadIdx.x % 4);
            } else if constexpr (I == 16 && J == 8) {
                return ((threadIdx.x % 4) * 2) + (l % 2);
            } else if constexpr (I == 16 && J == 16) {
                return ((l / 4) * 8) + ((threadIdx.x % 4) * 2) + (l % 2);
            } else if constexpr (I == 32 && J == 8) {
                return tile<16, 8, T>::get_j(l); // Memory layout simply repeated with same pattern in i direction.
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }
    };

    template <int I_, int J_>
    struct tile<I_, J_, half2, DATA_LAYOUT_I_MAJOR> {
        static constexpr int         I  = I_;
        static constexpr int         J  = J_;
        static constexpr data_layout dl = DATA_LAYOUT_I_MAJOR;

        // NVIDIA Turing+ layout (sm_75+)
        static constexpr int ne = I * J / WARP_SIZE;
        half2 x[ne] = {{0.0f, 0.0f}};

        static constexpr __device__ bool supported() {
            if (I ==  8 && J ==  4) return true;
            if (I ==  8 && J ==  8) return true;
            if (I == 16 && J ==  8) return true;
            if (I == 16 && J == 16) return true;
            if (I == 32 && J ==  8) return true;
            return false;
        }

        static __device__ __forceinline__ int get_i(const int l) {
            if constexpr (I == 8 && J == 8) {
                return threadIdx.x / 4;
            } else if constexpr (I == 16 && J == 4) {
                return (l * 8) + (threadIdx.x / 4);
            } else if constexpr (I == 16 && J == 8) {
                return ((l % 2) * 8) + (threadIdx.x / 4);
            } else if constexpr (I == 32 && J == 8) {
                return ((l / 4) * 16) + ((l % 2) * 8) + (threadIdx.x / 4);
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }

        static __device__ __forceinline__ int get_j(const int l) {
            if constexpr (I == 8 && J == 8) {
                return (l * 4) + (threadIdx.x % 4);
            } else if constexpr (I == 16 && J == 4) {
                return threadIdx.x % 4;
            } else if constexpr (I == 16 && J == 8) {
                return ((l / 2) * 4) + (threadIdx.x % 4);
            } else if constexpr (I == 32 && J == 8) {
                return ((l & 2) * 2) + (threadIdx.x % 4);
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }
    };

    template <int I_, int J_>
    struct tile<I_, J_, nv_bfloat162, DATA_LAYOUT_I_MAJOR> {
        static constexpr int         I  = I_;
        static constexpr int         J  = J_;
        static constexpr data_layout dl = DATA_LAYOUT_I_MAJOR;

        // NVIDIA Turing+ layout (sm_75+)
        static constexpr int ne = I * J / WARP_SIZE;
        nv_bfloat162 x[ne] = {{0.0f, 0.0f}};

        static constexpr __device__ bool supported() {
            if (I ==  8 && J ==  8) return true;
            if (I == 16 && J ==  4) return true;
            if (I == 16 && J ==  8) return true;
            return false;
        }

        static __device__ __forceinline__ int get_i(const int l) {
            if constexpr (I == 8 && J == 8) {
                return threadIdx.x / 4;
            } else if constexpr (I == 16 && J == 4) {
                return (l * 8) + (threadIdx.x / 4);
            } else if constexpr (I == 16 && J == 8) {
                return ((l % 2) * 8) + (threadIdx.x / 4);
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }

        static __device__ __forceinline__ int get_j(const int l) {
            if constexpr (I == 8 && J == 8) {
                return (l * 4) + (threadIdx.x % 4);
            } else if constexpr (I == 16 && J == 4) {
                return threadIdx.x % 4;
            } else if constexpr (I == 16 && J == 8) {
                return ((l / 2) * 4) + (threadIdx.x % 4);
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }
    };

    template <int I_, int J_, typename T>
    struct tile<I_, J_, T, DATA_LAYOUT_J_MAJOR> {
        static constexpr int         I  = I_;
        static constexpr int         J  = J_;
        static constexpr data_layout dl = DATA_LAYOUT_J_MAJOR;

        static constexpr int ne = tile<I_, J_, T, DATA_LAYOUT_I_MAJOR>::ne;
        T x[ne] = {0};

        static constexpr __device__ bool supported() {
            return tile<I_, J_, T, DATA_LAYOUT_I_MAJOR>::supported();
        }

        static __device__ __forceinline__ int get_i(const int l) {
            return tile<I_, J_, T, DATA_LAYOUT_I_MAJOR>::get_j(l);
        }

        static __device__ __forceinline__ int get_j(const int l) {
            return tile<I_, J_, T, DATA_LAYOUT_I_MAJOR>::get_i(l);
        }
    };

    template <int I_, int J_, typename T>
    struct tile<I_, J_, T, DATA_LAYOUT_I_MAJOR_MIRRORED> {
        static constexpr int         I  = I_;
        static constexpr int         J  = J_;
        static constexpr data_layout dl = DATA_LAYOUT_I_MAJOR_MIRRORED;

        // RDNA3
        static constexpr int         ne = I * J / 32 * 2;

        T x[ne] = {0};

        static constexpr __device__ bool supported() {
            if (I == 16 && J == 16) return true;
            if (I == 16 && J == 8)  return true;
            if (I == 16 && J == 4)  return true;
            return false;
        }

        static __device__ __forceinline__ int get_i(const int /*l*/) {
            if constexpr (supported()) {
                return threadIdx.x % 16;
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }

        static __device__ __forceinline__ int get_j(const int l) {
            if constexpr (supported()) {
                return l;
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }
    };

    template <int I_, int J_>
    struct tile<I_, J_, half2, DATA_LAYOUT_I_MAJOR_MIRRORED> {
        static constexpr int         I  = I_;
        static constexpr int         J  = J_;
        static constexpr data_layout dl = DATA_LAYOUT_I_MAJOR_MIRRORED;

        // Mirrored layout only used for RDNA3/Volta - not supported on Ampere+
        static constexpr int ne = I * J / (WARP_SIZE/4);
        half2 x[ne] = {{0.0f, 0.0f}};

        static constexpr __device__ bool supported() {
            return false;
        }

        static __device__ __forceinline__ int get_i(const int /*l*/) {
            NO_DEVICE_CODE;
            return -1;
        }

        static __device__ __forceinline__ int get_j(const int /*l*/) {
            NO_DEVICE_CODE;
            return -1;
        }
    };

    template <int I_, int J_>
    struct tile<I_, J_, nv_bfloat162, DATA_LAYOUT_I_MAJOR_MIRRORED> {
        static constexpr int         I  = I_;
        static constexpr int         J  = J_;
        static constexpr data_layout dl = DATA_LAYOUT_I_MAJOR_MIRRORED;
        static constexpr int         ne = tile<I_, J_, float, DATA_LAYOUT_I_MAJOR_MIRRORED>::ne;

        nv_bfloat162 x[ne] = {{0.0f, 0.0f}};

        static constexpr __device__ bool supported() {
            return tile<I_, J_, float, DATA_LAYOUT_I_MAJOR_MIRRORED>::supported();
        }

        static __device__ __forceinline__ int get_i(const int l) {
            return tile<I_, J_, float, DATA_LAYOUT_I_MAJOR_MIRRORED>::get_i(l);
        }

        static __device__ __forceinline__ int get_j(const int l) {
            return tile<I_, J_, float, DATA_LAYOUT_I_MAJOR_MIRRORED>::get_j(l);
        }
    };

    template <int I_, int J_>
    struct tile<I_, J_, half2, DATA_LAYOUT_J_MAJOR_MIRRORED> {
        static constexpr int         I  = I_;
        static constexpr int         J  = J_;
        static constexpr data_layout dl = DATA_LAYOUT_J_MAJOR_MIRRORED;
        static constexpr int         ne = I * J / (WARP_SIZE/4);

        half2 x[ne] = {{0.0f, 0.0f}};

        static constexpr __device__ bool supported() {
            if (I ==  8 && J ==  4) return true;
            return false;
        }

        static __device__ __forceinline__ int get_i(const int l) {
            if constexpr (I == 8 && J == 4) {
                return ((l / 2) * 4) + (threadIdx.x % 4);
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }

        static __device__ __forceinline__ int get_j(const int l) {
            if constexpr (I == 8 && J == 4) {
                return ((threadIdx.x / 16) * 2) + (l % 2);
            } else {
                NO_DEVICE_CODE;
                return -1;
            }
        }
    };

#if defined(TURING_MMA_AVAILABLE)
    template <int I, int J>
    static __device__ __forceinline__ tile<I, J/2, half2> get_half2(const tile<I, J, float> & tile_float) {
        tile<I, J/2, half2> ret;
#pragma unroll
        for (int l0 = 0; l0 < tile_float.ne; l0 += 2) {
            ret.x[l0/2] = make_half2(tile_float.x[l0 + 0], tile_float.x[l0 + 1]);
        }
        return ret;
    }

    static __device__ __forceinline__ tile<8, 8, half2> get_transposed(const tile<16, 4, half2> & t) {
        tile<8, 8, half2> ret;
        ret.x[0] = ggml_cuda_movmatrix(t.x[0]);
        ret.x[1] = ggml_cuda_movmatrix(t.x[1]);

        return ret;
    }
#else // Volta
    template <int I, int J>
    static __device__ __forceinline__ tile<I, J/2, half2> get_half2(const tile<I, J, float> & tile_float) {
        tile<I, J/2, half2> ret;
#pragma unroll
        for (int l0 = 0; l0 < tile_float.ne; l0 += 4) {
            ret.x[l0/2 + 0] = make_half2(tile_float.x[l0 + 0], tile_float.x[l0 + 1]);
            ret.x[l0/2 + 1] = make_half2(tile_float.x[l0 + 2], tile_float.x[l0 + 3]);

            // On Volta FP16 and FP32 tiles have a different memory layout,
            //     for the conversion threads with an offset of 2 need to exchange half their values:
            ret.x[l0/2 + (((threadIdx.x % 4) / 2) ^ 1)] = __shfl_xor_sync(
                0xFFFFFFFF, ret.x[l0/2 + (((threadIdx.x % 4) / 2) ^ 1)], 2, WARP_SIZE);
        }
        return ret;
    }
#endif // defined(TURING_MMA_AVAILABLE)

    template <int I, int J, typename T, data_layout dl>
    static __device__ __forceinline__ void load_generic(tile<I, J, T, dl> & t, const T * __restrict__ xs0, const int stride) {
        // NVIDIA path - simple element-by-element load
#pragma unroll
        for (int l = 0; l < t.ne; ++l) {
            t.x[l] = xs0[t.get_i(l)*stride + t.get_j(l)];
        }
    }

    template <typename T>
    static __device__ __forceinline__ void load_ldmatrix(
            tile<8, 8, T> & t, const T * __restrict__ xs0, const int stride) {
#ifdef TURING_MMA_AVAILABLE
        int * xi = (int *) t.x;
        const int * xs = (const int *) xs0 + (threadIdx.x % t.I) * stride + ((threadIdx.x / t.I) * (t.J / 2)) % t.J;
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
            : "=r"(xi[0]), "=r"(xi[1])
            : "l"(xs));
#else
        load_generic(t, xs0, stride);
#endif // TURING_MMA_AVAILABLE
    }

    template <typename T>
    static __device__ __forceinline__ void load_ldmatrix(
            tile<16, 4, T> & t, const T * __restrict__ xs0, const int stride) {
#ifdef TURING_MMA_AVAILABLE
        int * xi = (int *) t.x;
        const int * xs = (const int *) xs0 + (threadIdx.x % t.I) * stride;
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
            : "=r"(xi[0]), "=r"(xi[1])
            : "l"(xs));
#else
        load_generic(t, xs0, stride);
#endif // TURING_MMA_AVAILABLE
    }

    template <typename T, data_layout dl>
    static __device__ __forceinline__ void load_ldmatrix(
            tile<16, 8, T, dl> & t, const T * __restrict__ xs0, const int stride) {
#if defined(TURING_MMA_AVAILABLE)
        int * xi = (int * ) t.x;
        const int * xs = (const int *) xs0 + (threadIdx.x % t.I) * stride + (threadIdx.x / t.I) * (t.J / 2);
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
            : "=r"(xi[0]), "=r"(xi[1]), "=r"(xi[2]), "=r"(xi[3])
            : "l"(xs));
#else
        load_generic(t, xs0, stride);
#endif // TURING_MMA_AVAILABLE
    }

    static __device__ __forceinline__ void load_ldmatrix(
            tile<8, 4, half2, DATA_LAYOUT_I_MAJOR_MIRRORED> & t, const half2 * __restrict__ xs0, const int stride) {
        ggml_cuda_memcpy_1<4*sizeof(half2)>(t.x, xs0 + t.get_i(0)*stride);
    }

    static __device__ __forceinline__ void load_ldmatrix(
            tile<8, 4, half2, DATA_LAYOUT_J_MAJOR_MIRRORED> & t, const half2 * __restrict__ xs0, const int stride) {
#pragma unroll
        for (int l0 = 0; l0 < t.ne; l0 += 2) {
            ggml_cuda_memcpy_1<2*sizeof(half2)>(t.x + l0, xs0 + t.get_i(l0)*stride + t.get_j(l0));
        }
    }

    static __device__ __forceinline__ void load_ldmatrix(
            tile<32, 4, half2> & t, const half2 * __restrict__ xs0, const int stride) {
        // Volta-only function - not supported on Ampere+
        GGML_UNUSED_VARS(t, xs0, stride);
        NO_DEVICE_CODE;
    }

    template <typename T>
    static __device__ __forceinline__ void load_ldmatrix_trans(
            tile<16, 8, T> & t, const T * __restrict__ xs0, const int stride) {
#ifdef TURING_MMA_AVAILABLE
        int * xi = (int * ) t.x;
        const int * xs = (const int *) xs0 + (threadIdx.x % t.I) * stride + (threadIdx.x / t.I) * (t.J / 2);
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.b16 {%0, %1, %2, %3}, [%4];"
            : "=r"(xi[0]), "=r"(xi[2]), "=r"(xi[1]), "=r"(xi[3])
            : "l"(xs));
#else
        GGML_UNUSED_VARS(t, xs0, stride);
        NO_DEVICE_CODE;
#endif // TURING_MMA_AVAILABLE
    }

    // ============================================================================
    // SM_120 Optimized ldmatrix with Swizzle Support (Gau-Nernst V5 Style)
    // ============================================================================
    //
    // BUG FIX SUMMARY (2026-01-03):
    // -----------------------------
    // This code fixes a critical NaN bug in SM_120 (RTX 5090) flash attention.
    //
    // ROOT CAUSE:
    // The previous implementation had two bugs in the ldmatrix swizzle logic:
    //
    //   1. WRONG: Used ".shared.b16" modifier with "r" (32-bit) constraint
    //      FIXED: Use NO .shared modifier, let hardware infer state space
    //      The .shared.b16 variant appears buggy or has different semantics on SM_120.
    //
    //   2. WRONG: Used ADDITION for column offset navigation
    //             addr = swizzle(row * STRIDE + col)  // computed each time
    //      FIXED: Use XOR for column iteration (matches Gau-Nernst reference)
    //             swizzled = swizzle(row * STRIDE + thread_col)
    //             addr = smem_base + (swizzled ^ col_offset_bytes)
    //
    // WHY XOR WORKS:
    // The XOR-based swizzle pattern has a key mathematical property:
    //   swizzle(offset + col) == swizzle(offset) ^ col  (within same row)
    //
    // This allows efficient column iteration without recomputing the full swizzle.
    // Example trace (STRIDE_BYTES=256):
    //   - Write row 2, chunk 0: offset=512 → swizzle → 544, stored at smem+544
    //   - Write row 2, chunk 2: offset=544 → swizzle → 512, stored at smem+512
    //   - Read  row 2, col 0:   swizzle(512)=544, 544^0=544  ✓
    //   - Read  row 2, col 32:  swizzle(512)=544, 544^32=512 ✓
    //
    // WRITE PATH (fattn-mma-f16.cuh::flash_attn_ext_f16_load_tile_swizzle):
    //   offset = row * STRIDE_BYTES + chunk * 16
    //   offset ^= (xor_bits << 4)  // where xor_bits = (row % 8) / rows_per_swizzle
    //   cp_async(smem_base + offset, gmem_src)
    //
    // READ PATH (this file, load_ldmatrix_swizzle):
    //   linear_offset = thread_row * STRIDE_BYTES + thread_col_bytes
    //   swizzled_offset = swizzle(linear_offset)
    //   addr = smem_base + (swizzled_offset ^ col_offset_bytes)  // XOR not ADD!
    //   ldmatrix.sync.aligned.m8n8.x4.b16 [addr]  // NO .shared modifier!
    //
    // Reference: Gau-Nernst Flash Attention V5 implementation
    // https://gau-nern.st/blog/flash-attention-from-scratch.html
    // ============================================================================

    // Swizzle function matching the write path in flash_attn_ext_f16_load_tile_swizzle
    // This applies XOR to bits [4:6] based on row index for bank conflict avoidance
    // STRIDE_BYTES = row stride in bytes (e.g., DIM * sizeof(half))
    template <int STRIDE_BYTES>
    static __device__ __forceinline__ uint32_t swizzle_addr(uint32_t addr) {
        if constexpr (STRIDE_BYTES <= 16) {
            return addr;  // No swizzling for small strides
        } else {
            // XOR pattern: bits_to_xor = (row_idx / rows_per_swizzle_group) & 0x7
            // Then XOR with bits [4:6] of address
            constexpr int rows_per_swizzle = (64 / STRIDE_BYTES) > 0 ? (64 / STRIDE_BYTES) : 1;
            uint32_t row_idx = (addr / STRIDE_BYTES) % 8;
            uint32_t xor_bits = row_idx / rows_per_swizzle;
            return addr ^ (xor_bits << 4);
        }
    }

    // ldmatrix.x4 with swizzle for 16x8 tile (most common in Flash Attention)
    // Uses Gau-Nernst style: pre-computed swizzled base + XOR for column iteration
    //
    // CRITICAL: Uses 32-bit address with "r" constraint but NO .shared modifier!
    // This matches the reference code and allows hardware to infer state space.
    // The .shared.b16 variant has issues on SM_120.
    template <int STRIDE_BYTES, typename T, data_layout dl>
    static __device__ __forceinline__ void load_ldmatrix_swizzle(
            tile<16, 8, T, dl> & t, uint32_t smem_base, int row_offset, int col_offset_bytes) {
#ifdef TURING_MMA_AVAILABLE
        int * xi = (int *) t.x;
        
        // Thread position within 16x8 tile:
        // - Threads 0-15: rows 0-15, col group 0
        // - Threads 16-31: rows 0-15, col group 1 (offset by 8 half = 16 bytes)
        const int thread_row = (threadIdx.x % t.I) + row_offset;
        const int thread_col_bytes = (threadIdx.x / t.I) * (t.J / 2) * sizeof(int);  // 0 or 16 bytes
        
        // Compute linear offset, then apply swizzle (matching write path)
        uint32_t linear_offset = thread_row * STRIDE_BYTES + thread_col_bytes;
        uint32_t swizzled_offset = swizzle_addr<STRIDE_BYTES>(linear_offset);
        
        // For column iteration: XOR with col_offset_bytes (NOT addition!)
        // This correctly navigates the swizzled memory layout
        uint32_t addr = smem_base + (swizzled_offset ^ col_offset_bytes);
        
        // Use 32-bit address with "r" constraint, NO .shared modifier (matches Gau-Nernst reference)
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
            : "=r"(xi[0]), "=r"(xi[1]), "=r"(xi[2]), "=r"(xi[3])
            : "r"(addr));
#else
        GGML_UNUSED_VARS(t, smem_base, row_offset, col_offset_bytes);
        NO_DEVICE_CODE;
#endif // TURING_MMA_AVAILABLE
    }

    // ldmatrix.x4.trans with swizzle (for transposed B matrix loads)
    template <int STRIDE_BYTES, typename T>
    static __device__ __forceinline__ void load_ldmatrix_trans_swizzle(
            tile<16, 8, T> & t, uint32_t smem_base, int row_offset, int col_offset_bytes) {
#ifdef TURING_MMA_AVAILABLE
        int * xi = (int *) t.x;
        
        const int thread_row = (threadIdx.x % t.I) + row_offset;
        const int thread_col_bytes = (threadIdx.x / t.I) * (t.J / 2) * sizeof(int);
        
        uint32_t linear_offset = thread_row * STRIDE_BYTES + thread_col_bytes;
        uint32_t swizzled_offset = swizzle_addr<STRIDE_BYTES>(linear_offset);
        uint32_t addr = smem_base + (swizzled_offset ^ col_offset_bytes);
        
        // .trans variant reorders output registers for transposed load
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.b16 {%0, %1, %2, %3}, [%4];"
            : "=r"(xi[0]), "=r"(xi[2]), "=r"(xi[1]), "=r"(xi[3])
            : "r"(addr));
#else
        GGML_UNUSED_VARS(t, smem_base, row_offset, col_offset_bytes);
        NO_DEVICE_CODE;
#endif // TURING_MMA_AVAILABLE
    }

    // ldmatrix.x2 with swizzle for 8x8 tile
    template <int STRIDE_BYTES, typename T>
    static __device__ __forceinline__ void load_ldmatrix_x2_swizzle(
            tile<8, 8, T> & t, uint32_t smem_base, int row_offset, int col_offset_bytes) {
#ifdef TURING_MMA_AVAILABLE
        int * xi = (int *) t.x;
        
        const int thread_row = (threadIdx.x % t.I) + row_offset;
        const int thread_col_bytes = ((threadIdx.x / t.I) * (t.J / 2)) * sizeof(int);
        
        uint32_t linear_offset = thread_row * STRIDE_BYTES + thread_col_bytes;
        uint32_t swizzled_offset = swizzle_addr<STRIDE_BYTES>(linear_offset);
        uint32_t addr = smem_base + (swizzled_offset ^ col_offset_bytes);
        
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
            : "=r"(xi[0]), "=r"(xi[1])
            : "r"(addr));
#else
        GGML_UNUSED_VARS(t, smem_base, row_offset, col_offset_bytes);
        NO_DEVICE_CODE;
#endif // TURING_MMA_AVAILABLE
    }
    
    // Legacy function name for compatibility
    template <int STRIDE_BYTES>
    static __device__ __forceinline__ uint32_t ldmatrix_swizzle_addr(uint32_t base_addr, int row, int col_bytes) {
        uint32_t linear_offset = row * STRIDE_BYTES + col_bytes;
        return base_addr + swizzle_addr<STRIDE_BYTES>(linear_offset);
    }

    // ============================================================================
    // SM_120 Optimized ldmatrix.x4 with Padding Support (Gau-Nernst V4)
    // ============================================================================
    // These functions support the +8 half2 padding scheme for bank conflict avoidance
    // without XOR swizzling. The stride includes the padding.
    //
    // Key optimization: Pre-compute base address and offsets outside inner loops
    // to reduce ALU overhead. Use ldmatrix.x4 for maximum throughput.
    // ============================================================================

    // ldmatrix.x4 for 16x8 tile with padding (no swizzle)
    // STRIDE = actual stride in half2 elements (e.g., nbatch_K2 + 8 for padding)
    template <int STRIDE, typename T, data_layout dl = DATA_LAYOUT_I_MAJOR>
    static __device__ __forceinline__ void load_ldmatrix_x4_padded(
            tile<16, 8, T, dl> & t, const T * __restrict__ smem_ptr, int row_offset, int col_offset) {
#ifdef TURING_MMA_AVAILABLE
        int * xi = (int *) t.x;

        // Pre-compute thread's position in the 16x8 tile
        // Rows 0-15 are distributed across threads, columns 0-7 are loaded 4 at a time
        const int thread_row = (threadIdx.x % t.I) + row_offset;
        const int thread_col = (threadIdx.x / t.I) * (t.J / 2) + col_offset;

        // Calculate shared memory address with stride
        const int * xs = (const int *) smem_ptr + thread_row * STRIDE + thread_col;

        asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
            : "=r"(xi[0]), "=r"(xi[1]), "=r"(xi[2]), "=r"(xi[3])
            : "l"(xs));
#else
        GGML_UNUSED_VARS(t, smem_ptr, row_offset, col_offset);
        NO_DEVICE_CODE;
#endif // TURING_MMA_AVAILABLE
    }

    // ldmatrix.x4.trans for 16x8 tile with padding (transposed load for V matrix)
    template <int STRIDE, typename T>
    static __device__ __forceinline__ void load_ldmatrix_x4_trans_padded(
            tile<16, 8, T> & t, const T * __restrict__ smem_ptr, int row_offset, int col_offset) {
#ifdef TURING_MMA_AVAILABLE
        int * xi = (int *) t.x;

        const int thread_row = (threadIdx.x % t.I) + row_offset;
        const int thread_col = (threadIdx.x / t.I) * (t.J / 2) + col_offset;

        const int * xs = (const int *) smem_ptr + thread_row * STRIDE + thread_col;

        // .trans variant reorders output registers for transposed load
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.b16 {%0, %1, %2, %3}, [%4];"
            : "=r"(xi[0]), "=r"(xi[2]), "=r"(xi[1]), "=r"(xi[3])
            : "l"(xs));
#else
        GGML_UNUSED_VARS(t, smem_ptr, row_offset, col_offset);
        NO_DEVICE_CODE;
#endif // TURING_MMA_AVAILABLE
    }

    // ldmatrix.x4 with pre-computed base address (for loop optimization)
    // Call this version when base_addr is computed once outside the loop
    template <typename T, data_layout dl = DATA_LAYOUT_I_MAJOR>
    static __device__ __forceinline__ void load_ldmatrix_x4_preaddr(
            tile<16, 8, T, dl> & t, uint32_t base_addr, int col_offset_bytes) {
#ifdef TURING_MMA_AVAILABLE
        int * xi = (int *) t.x;

        // Column offset is in bytes, added directly to pre-computed base address
        uint32_t addr = base_addr + col_offset_bytes;

        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
            : "=r"(xi[0]), "=r"(xi[1]), "=r"(xi[2]), "=r"(xi[3])
            : "r"(addr));
#else
        GGML_UNUSED_VARS(t, base_addr, col_offset_bytes);
        NO_DEVICE_CODE;
#endif // TURING_MMA_AVAILABLE
    }

    // ldmatrix.x4.trans with pre-computed base address
    template <typename T>
    static __device__ __forceinline__ void load_ldmatrix_x4_trans_preaddr(
            tile<16, 8, T> & t, uint32_t base_addr, int col_offset_bytes) {
#ifdef TURING_MMA_AVAILABLE
        int * xi = (int *) t.x;

        uint32_t addr = base_addr + col_offset_bytes;

        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
            : "=r"(xi[0]), "=r"(xi[2]), "=r"(xi[1]), "=r"(xi[3])
            : "r"(addr));
#else
        GGML_UNUSED_VARS(t, base_addr, col_offset_bytes);
        NO_DEVICE_CODE;
#endif // TURING_MMA_AVAILABLE
    }

    // Helper: Pre-compute thread's base address for ldmatrix.x4 operations
    // Returns the shared memory address for this thread's row/column position
    // Use this outside the inner loop, then call load_ldmatrix_x4_preaddr with col offsets
    template <int STRIDE_BYTES>
    static __device__ __forceinline__ uint32_t precompute_ldmatrix_addr(
            uint32_t smem_base, int row_offset, int base_col_offset_bytes = 0) {
        // For 16x8 tile: 16 rows, threads 0-15 get rows 0-15, threads 16-31 add column offset
        constexpr int TILE_I = 16;
        constexpr int TILE_J_HALF = 4;  // J/2 = 8/2 = 4, * sizeof(int) = 16 bytes

        const int thread_row = (threadIdx.x % TILE_I) + row_offset;
        const int thread_col_bytes = (threadIdx.x / TILE_I) * TILE_J_HALF * sizeof(int);

        return smem_base + thread_row * STRIDE_BYTES + base_col_offset_bytes + thread_col_bytes;
    }

    static __device__ __forceinline__ void mma(
            tile<16, 8, int> & D, const tile<16, 4, int> & A, const tile<8, 4, int> & B) {
#ifdef TURING_MMA_AVAILABLE
#if __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
        asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
            : "+r"(D.x[0]), "+r"(D.x[1]), "+r"(D.x[2]), "+r"(D.x[3])
            : "r"(A.x[0]), "r"(A.x[1]), "r"(B.x[0]));
#else
        // On Turing m16n8k16 mma is not available, use 2x m8n8k16 mma instead:
        asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
            : "+r"(D.x[0]), "+r"(D.x[1])
            : "r"(A.x[0]), "r"(B.x[0]));
        asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
            : "+r"(D.x[2]), "+r"(D.x[3])
            : "r"(A.x[1]), "r"(B.x[0]));
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
#else
        GGML_UNUSED_VARS(D, A, B);
        NO_DEVICE_CODE;
#endif // TURING_MMA_AVAILABLE
    }

    static __device__ __forceinline__ void mma(
            tile<16, 8, int> & D, const tile<16, 8, int> & A, const tile<8, 8, int> & B) {
#ifdef TURING_MMA_AVAILABLE
#if __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
        asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
            : "+r"(D.x[0]), "+r"(D.x[1]), "+r"(D.x[2]), "+r"(D.x[3])
            : "r"(A.x[0]), "r"(A.x[1]), "r"(A.x[2]), "r"(A.x[3]), "r"(B.x[0]), "r"(B.x[1]));
#else
        // On Turing m16n8k32 mma is not available, use 4x m8n8k16 mma instead:
        asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
            : "+r"(D.x[0]), "+r"(D.x[1])
            : "r"(A.x[0]), "r"(B.x[0]));
        asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
            : "+r"(D.x[2]), "+r"(D.x[3])
            : "r"(A.x[1]), "r"(B.x[0]));
        asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
            : "+r"(D.x[0]), "+r"(D.x[1])
            : "r"(A.x[2]), "r"(B.x[1]));
        asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
            : "+r"(D.x[2]), "+r"(D.x[3])
            : "r"(A.x[3]), "r"(B.x[1]));
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
#else
        GGML_UNUSED_VARS(D, A, B);
        NO_DEVICE_CODE;
#endif // TURING_MMA_AVAILABLE
    }

    static __device__ __forceinline__ void mma(
            tile<16, 4, half2> & D, const tile<16, 8, half2> & A, const tile<8, 8, half2> & B) {
#ifdef TURING_MMA_AVAILABLE
        const int * Axi = (const int *) A.x;
        const int * Bxi = (const int *) B.x;
        int       * Dxi = (int       *) D.x;
#if __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
        asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};"
            : "+r"(Dxi[0]), "+r"(Dxi[1])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[0]), "r"(Bxi[1]));
#else
        // On Turing m16n8k16 mma is not available, use 2x m8n8k8 mma instead:
        asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3}, {%4}, {%0, %1};"
            : "+r"(Dxi[0]), "+r"(Dxi[1])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Bxi[0]));
        asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3}, {%4}, {%0, %1};"
            : "+r"(Dxi[0]), "+r"(Dxi[1])
            : "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[1]));
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
#else
        GGML_UNUSED_VARS(D, A, B);
        NO_DEVICE_CODE;
#endif // TURING_MMA_AVAILABLE
    }

    static __device__ __forceinline__ void mma(
            tile<16, 8, half2> & D, const tile<16, 8, half2> & A, const tile<16, 8, half2> & B) {
#ifdef TURING_MMA_AVAILABLE
        const int * Axi = (const int *) A.x;
        const int * Bxi = (const int *) B.x;
        int       * Dxi = (int       *) D.x;
#if __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
        asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};"
            : "+r"(Dxi[0]), "+r"(Dxi[1])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[0]), "r"(Bxi[2]));
        asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};"
            : "+r"(Dxi[2]), "+r"(Dxi[3])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[1]), "r"(Bxi[3]));
#else
        // On Turing m16n8k16 mma is not available, use 4x m8n8k8 mma instead:
        asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3}, {%4}, {%0, %1};"
            : "+r"(Dxi[0]), "+r"(Dxi[1])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Bxi[0]));
        asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3}, {%4}, {%0, %1};"
            : "+r"(Dxi[0]), "+r"(Dxi[1])
            : "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[2]));
        asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3}, {%4}, {%0, %1};"
            : "+r"(Dxi[2]), "+r"(Dxi[3])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Bxi[1]));
        asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3}, {%4}, {%0, %1};"
            : "+r"(Dxi[2]), "+r"(Dxi[3])
            : "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[3]));
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
#else
        GGML_UNUSED_VARS(D, A, B);
        NO_DEVICE_CODE;
#endif // TURING_MMA_AVAILABLE
    }

    template <data_layout dl_ab, data_layout dl_d>
    static __device__ __forceinline__ void mma(
            tile<16, 8, float, dl_d> & D, const tile<16, 8, float, dl_ab> & A, const tile<8, 8, float, dl_ab> & B) {
#ifdef AMPERE_MMA_AVAILABLE
        const int * Axi = (const int *) A.x;
        const int * Bxi = (const int *) B.x;
        int       * Dxi = (int       *) D.x;
        asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
            : "+r"(Dxi[0]), "+r"(Dxi[1]), "+r"(Dxi[2]), "+r"(Dxi[3])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[0]), "r"(Bxi[1]));
#else
        GGML_UNUSED_VARS(D, A, B);
        NO_DEVICE_CODE;
#endif // AMPERE_MMA_AVAILABLE
    }

    static __device__ __forceinline__ void mma_block_scaled(tile<16, 8, float> &     D,
                                                            const tile<16, 8, int> & A,
                                                            const tile<8, 8, int> &  B,
                                                            uint32_t                 a_scale,
                                                            uint32_t                 b_scale) {
#ifdef BLACKWELL_MMA_AVAILABLE
        const int * Axi = (const int *) A.x;
        const int * Bxi = (const int *) B.x;
        float *     Dxi = (float *) D.x;

        asm volatile(
            "mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
            "%10, {0, 0}, %11, {0, 0};"
            : "+f"(Dxi[0]), "+f"(Dxi[1]), "+f"(Dxi[2]), "+f"(Dxi[3])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[0]), "r"(Bxi[1]), "r"(a_scale), "r"(b_scale));
#else
        GGML_UNUSED_VARS(D, A, B, a_scale, b_scale);
#endif  // BLACKWELL_MMA_AVAILABLE
    }

    static __device__ __forceinline__ void mma(
            tile<16, 8, float> & D, const tile<16, 8, half2> & A, const tile<8, 8, half2> & B) {
#ifdef TURING_MMA_AVAILABLE
        const int * Axi = (const int *) A.x;
        const int * Bxi = (const int *) B.x;
        int       * Dxi = (int       *) D.x;
#if __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
        asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
            : "+r"(Dxi[0]), "+r"(Dxi[1]), "+r"(Dxi[2]), "+r"(Dxi[3])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[0]), "r"(Bxi[1]));
#else
        // On Turing m16n8k16 mma is not available, use 2x m8n8k8 mma instead:
        asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
            : "+r"(Dxi[0]), "+r"(Dxi[1]), "+r"(Dxi[2]), "+r"(Dxi[3])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Bxi[0]));
        asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
            : "+r"(Dxi[0]), "+r"(Dxi[1]), "+r"(Dxi[2]), "+r"(Dxi[3])
            : "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[1]));
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
#else
        GGML_UNUSED_VARS(D, A, B);
        NO_DEVICE_CODE;
#endif // TURING_MMA_AVAILABLE
    }

    static __device__ __forceinline__ void mma(
            tile<16, 8, float> & D, const tile<16, 8, nv_bfloat162> & A, const tile<8, 8, nv_bfloat162> & B) {
#ifdef AMPERE_MMA_AVAILABLE
        const int * Axi = (const int *) A.x;
        const int * Bxi = (const int *) B.x;
        int       * Dxi = (int       *) D.x;
        asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
            : "+r"(Dxi[0]), "+r"(Dxi[1]), "+r"(Dxi[2]), "+r"(Dxi[3])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[0]), "r"(Bxi[1]));
#else
        GGML_UNUSED_VARS(D, A, B);
        NO_DEVICE_CODE;
#endif // AMPERE_MMA_AVAILABLE
    }

    template <data_layout dl_ab, data_layout dl_d>
    static __device__ __forceinline__ void mma(
            tile<16, 16, float, dl_d> & D, const tile<16, 8, half2, dl_ab> & A, const tile<16, 8, half2, dl_ab> & B) {
#ifdef TURING_MMA_AVAILABLE
        const int * Axi = (const int *) A.x;
        const int * Bxi = (const int *) B.x;
        int       * Dxi = (int       *) D.x;
        // Ampere+ m16n8k16 MMA instructions
        asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
            : "+r"(Dxi[0]), "+r"(Dxi[1]), "+r"(Dxi[2]), "+r"(Dxi[3])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[0]), "r"(Bxi[2]));
        asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
            : "+r"(Dxi[4]), "+r"(Dxi[5]), "+r"(Dxi[6]), "+r"(Dxi[7])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[1]), "r"(Bxi[3]));
#else
        GGML_UNUSED_VARS(D, A, B);
        NO_DEVICE_CODE;
#endif // TURING_MMA_AVAILABLE
    }

    template <data_layout dl_ab, data_layout dl_d>
    static __device__ __forceinline__ void mma(
            tile<16, 16, float, dl_d> & D, const tile<16, 8, nv_bfloat162, dl_ab> & A, const tile<16, 8, nv_bfloat162, dl_ab> & B) {
        // No NVIDIA implementation for this signature - AMD-only (WMMA)
        GGML_UNUSED_VARS(D, A, B);
        NO_DEVICE_CODE;
    }

    template <data_layout dl_d, data_layout dl_ab>
    static __device__ __forceinline__ void mma(
            tile<16, 16, int, dl_d> & D, const tile<16, 8, int, dl_ab> & A, const tile<16, 8, int, dl_ab> & B) {
        // No NVIDIA implementation for this signature - AMD-only (MFMA/WMMA)
        GGML_UNUSED_VARS(D, A, B);
        NO_DEVICE_CODE;
    }

    static __device__ __forceinline__ void mma(
            tile<32, 32, int> & D, const tile<32, 4, int> & A, const tile<32, 4, int> & B) {
        // No NVIDIA implementation for this signature - AMD-only (MFMA)
        GGML_UNUSED_VARS(D, A, B);
        NO_DEVICE_CODE;
    }

    template <typename T1, typename T2, int J, int K>
    static __device__ __forceinline__ void mma(
            tile<32, J, T1> & D, const tile<32, K, T2> & A, const tile<J, K, T2> & B) {
        tile      <16, J, T1> * D16 = reinterpret_cast<      tile<16, J, T1> *>(&D);
        const tile<16, K, T2> * A16 = reinterpret_cast<const tile<16, K, T2> *>(&A);
        mma(D16[0], A16[0], B);
        mma(D16[1], A16[1], B);
    }

    static __device__ __forceinline__ void mma(
            tile<32, 8, float> & D, const tile<32, 4, half2> & A, const tile<8, 4, half2, DATA_LAYOUT_I_MAJOR_MIRRORED> & B) {
        // Volta-only function - not supported on Ampere+
        GGML_UNUSED_VARS(D, A, B);
        NO_DEVICE_CODE;
    }

    static __device__ __forceinline__ void mma(
            tile<32, 4, half2> & D, const tile<32, 4, half2> & A, const tile<8, 4, half2, DATA_LAYOUT_J_MAJOR_MIRRORED> & B) {
        // Volta-only function - not supported on Ampere+
        GGML_UNUSED_VARS(D, A, B);
        NO_DEVICE_CODE;
    }

    template <data_layout dl_d, data_layout dl_ab>
    static __device__ __forceinline__ void mma(
            tile<16, 16, int, dl_d> & D, const tile<16, 4, int, dl_ab> & A, const tile<16, 4, int, dl_ab> & B) {
        // No NVIDIA implementation for this signature - AMD-only (WMMA)
        GGML_UNUSED(D);
        GGML_UNUSED(A);
        GGML_UNUSED(B);
        NO_DEVICE_CODE;
    }
}

namespace ggml_cuda_wgmma {

#ifdef BLACKWELL_WGMMA_AVAILABLE

// Create shared memory descriptor for WGMMA operand B
// smem_ptr must be 16-byte aligned, swizzle_bits from 0-3
__device__ __forceinline__ uint64_t make_smem_desc(
    const void* smem_ptr, 
    uint32_t leading_dim_bytes,
    uint32_t stride_dim_bytes,
    uint32_t swizzle_bits = 3)  // 128-byte swizzle default
{
    uint32_t addr = __cvta_generic_to_shared(smem_ptr);
    // Descriptor format per PTX ISA 8.5
    uint64_t desc = addr;
    desc |= (uint64_t(leading_dim_bytes >> 4) << 16);
    desc |= (uint64_t(stride_dim_bytes >> 4) << 32);
    desc |= (uint64_t(swizzle_bits) << 62);
    return desc;
}

// WGMMA m64n64k16 FP16 -> FP32 accumulator
// Requires 4 consecutive warps (warpgroup)
__device__ __forceinline__ void wgmma_m64n64k16_f16_f32(
    float* accum,           // 64 floats in registers per thread
    const half* A_smem,     // 64x16 in shared memory
    uint64_t B_desc,        // Descriptor for 16x64 B matrix
    bool scale_D = true)
{
    uint32_t scale = scale_D ? 1 : 0;

    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31}, "
        "[%32], %33, %34;"
        : "+f"(accum[0]),  "+f"(accum[1]),  "+f"(accum[2]),  "+f"(accum[3]),
          "+f"(accum[4]),  "+f"(accum[5]),  "+f"(accum[6]),  "+f"(accum[7]),
          "+f"(accum[8]),  "+f"(accum[9]),  "+f"(accum[10]), "+f"(accum[11]),
          "+f"(accum[12]), "+f"(accum[13]), "+f"(accum[14]), "+f"(accum[15]),
          "+f"(accum[16]), "+f"(accum[17]), "+f"(accum[18]), "+f"(accum[19]),
          "+f"(accum[20]), "+f"(accum[21]), "+f"(accum[22]), "+f"(accum[23]),
          "+f"(accum[24]), "+f"(accum[25]), "+f"(accum[26]), "+f"(accum[27]),
          "+f"(accum[28]), "+f"(accum[29]), "+f"(accum[30]), "+f"(accum[31])
        : "l"(A_smem), "l"(B_desc), "r"(scale)
        : "memory"
    );
}

// Commit all pending WGMMA operations
__device__ __forceinline__ void wgmma_commit_group() {
    asm volatile("wgmma.commit_group.sync.aligned;");
}

// Wait for N groups to complete (0 = wait for all)
__device__ __forceinline__ void wgmma_wait_group(int n = 0) {
    if (n == 0) {
        asm volatile("wgmma.wait_group.sync.aligned 0;");
    } else if (n == 1) {
        asm volatile("wgmma.wait_group.sync.aligned 1;");
    }
    // Add more cases as needed
}

// Fence before reading accumulator results
__device__ __forceinline__ void wgmma_fence() {
    asm volatile("wgmma.fence.sync.aligned;");
}

#else

__device__ __forceinline__ uint64_t make_smem_desc(
    const void* smem_ptr, 
    uint32_t leading_dim_bytes,
    uint32_t stride_dim_bytes,
    uint32_t swizzle_bits = 3)
{
    GGML_UNUSED(smem_ptr);
    GGML_UNUSED(leading_dim_bytes);
    GGML_UNUSED(stride_dim_bytes);
    GGML_UNUSED(swizzle_bits);
    NO_DEVICE_CODE;
    return 0;
}

__device__ __forceinline__ void wgmma_m64n64k16_f16_f32(
    float* accum,
    const half* A_smem,
    uint64_t B_desc,
    bool scale_D = true)
{
    GGML_UNUSED(accum);
    GGML_UNUSED(A_smem);
    GGML_UNUSED(B_desc);
    GGML_UNUSED(scale_D);
    NO_DEVICE_CODE;
}

__device__ __forceinline__ void wgmma_commit_group() {
    NO_DEVICE_CODE;
}

__device__ __forceinline__ void wgmma_wait_group(int n = 0) {
    GGML_UNUSED(n);
    NO_DEVICE_CODE;
}

__device__ __forceinline__ void wgmma_fence() {
    NO_DEVICE_CODE;
}

#endif // BLACKWELL_WGMMA_AVAILABLE

} // namespace ggml_cuda_wgmma
