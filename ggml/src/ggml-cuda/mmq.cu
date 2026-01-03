#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"
#include "mmid.cuh"

// DEBUG: Counter to identify which MMQ call crashes
static int mmq_call_count = 0;

static void ggml_cuda_mul_mat_q_switch_type(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    switch (args.type_x) {
        case GGML_TYPE_Q4_0:
            mul_mat_q_case<GGML_TYPE_Q4_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_q_case<GGML_TYPE_Q4_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_q_case<GGML_TYPE_Q5_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_q_case<GGML_TYPE_Q5_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_q_case<GGML_TYPE_Q8_0>(ctx, args, stream);
            break;
        case GGML_TYPE_MXFP4:
            mul_mat_q_case<GGML_TYPE_MXFP4>(ctx, args, stream);
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_q_case<GGML_TYPE_Q2_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_q_case<GGML_TYPE_Q3_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_q_case<GGML_TYPE_Q4_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_q_case<GGML_TYPE_Q5_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_q_case<GGML_TYPE_IQ2_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_q_case<GGML_TYPE_IQ2_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_q_case<GGML_TYPE_IQ2_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_q_case<GGML_TYPE_IQ3_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_q_case<GGML_TYPE_IQ3_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ1_S:
            mul_mat_q_case<GGML_TYPE_IQ1_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_q_case<GGML_TYPE_IQ4_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_q_case<GGML_TYPE_IQ4_NL>(ctx, args, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }

    // =========================================================================
    // DEBUG: Synchronize and check for runtime errors in MMQ kernel
    // This isolates whether MMQ itself crashes or reads corrupted data from FA.
    // REMOVE THIS BLOCK AFTER DEBUGGING - it kills performance!
    // =========================================================================
    {
        static bool mmq_sync_debug_enabled = true;  // Set to false to disable sync

        if (mmq_sync_debug_enabled) {
            // Increment call counter BEFORE anything else
            mmq_call_count++;
            const int current_call = mmq_call_count;

            const int id = ggml_cuda_get_device();
            const int cc = ggml_cuda_info().devices[id].cc;

            // Print params for EVERY call with the call number
            fprintf(stderr, "\n[MMQ #%d] ============ MMQ Call #%d ============\n", current_call, current_call);
            fprintf(stderr, "[MMQ #%d] Device: %d, cc: %d, is_consumer_blackwell: %d\n",
                    current_call, id, cc, ggml_cuda_is_consumer_blackwell(cc));
            fprintf(stderr, "[MMQ #%d] type_x=%d, nrows_x=%d, ncols_x=%d, ncols_y=%d\n",
                    current_call, args.type_x, args.nrows_x, args.ncols_x, args.ncols_y);
            fprintf(stderr, "[MMQ #%d] nrows_dst=%d, ncols_dst=%d, ncols_max=%d\n",
                    current_call, args.nrows_dst, args.ncols_dst, args.ncols_max);
            fprintf(stderr, "[MMQ #%d] stride_row_x=%d, use_stream_k=%d\n",
                    current_call, args.stride_row_x, args.use_stream_k);
            fprintf(stderr, "[MMQ #%d] x=%p, y=%p, dst=%p\n",
                    current_call, args.x, (void*)args.y, (void*)args.dst);

            cudaError_t launch_err = cudaGetLastError();
            if (launch_err != cudaSuccess) {
                fprintf(stderr, "[MMQ #%d] !!! LAUNCH FAILED: %s\n", current_call, cudaGetErrorString(launch_err));
            }

            fprintf(stderr, "[MMQ #%d] Syncing...\n", current_call);
            cudaError_t sync_err = cudaDeviceSynchronize();
            if (sync_err != cudaSuccess) {
                fprintf(stderr, "[MMQ #%d] !!! CRASHED !!! Error: %s\n", current_call, cudaGetErrorString(sync_err));
                fprintf(stderr, "[MMQ CRASH] Call #%d failed! Check params above.\n", current_call);
                GGML_ABORT("MMQ kernel execution failed");
            }
            fprintf(stderr, "[MMQ #%d] OK\n", current_call);
        }
    }
    // =========================================================================
}

void ggml_cuda_mul_mat_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {

    // =========================================================================
    // DEBUG: Check if CUDA context is already corrupted BEFORE we do anything
    // This catches errors from previous kernels (like Flash Attention)
    // =========================================================================
    {
        static int mmq_entry_count = 0;
        mmq_entry_count++;
        cudaError_t pre_err = cudaDeviceSynchronize();
        if (pre_err != cudaSuccess) {
            fprintf(stderr, "[MMQ PRE-CHECK #%d] !!! Context already corrupted BEFORE MMQ !!!\n", mmq_entry_count);
            fprintf(stderr, "[MMQ PRE-CHECK #%d] Error: %s\n", mmq_entry_count, cudaGetErrorString(pre_err));
            fprintf(stderr, "[MMQ PRE-CHECK #%d] Was about to run MMQ with nrows_x=%d ncols_x=%d\n",
                    mmq_entry_count, (int)src0->ne[1], (int)src0->ne[0]);
            fprintf(stderr, "[MMQ PRE-CHECK #%d] src0 type=%d, src1 type=%d\n",
                    mmq_entry_count, src0->type, src1->type);
            GGML_ABORT("CUDA context corrupted before MMQ entry");
        }
    }
    // =========================================================================

    // DEBUG: Track if this is MoE or regular matmul
    {
        static int mmq_internal_call = 0;
        mmq_internal_call++;
        fprintf(stderr, "\n[MMQ INTERNAL #%d] ids=%p (MoE=%s)\n",
                mmq_internal_call, (void*)ids, ids ? "YES" : "NO");
        fprintf(stderr, "[MMQ INTERNAL #%d] src0: ne=[%lld,%lld,%lld,%lld] type=%d\n",
                mmq_internal_call,
                (long long)src0->ne[0], (long long)src0->ne[1],
                (long long)src0->ne[2], (long long)src0->ne[3], src0->type);
        fprintf(stderr, "[MMQ INTERNAL #%d] src1: ne=[%lld,%lld,%lld,%lld]\n",
                mmq_internal_call,
                (long long)src1->ne[0], (long long)src1->ne[1],
                (long long)src1->ne[2], (long long)src1->ne[3]);
    }

    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    const char  * src0_d = (const char  *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       *  dst_d = (float       *)  dst->data;

    // If src0 is a temporary compute buffer, clear any potential padding.
    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    // Stream-K decomposition provides better load balancing but has fixup overhead.
    // For single-token decode (ne1=1) and small batches, the fixup overhead dominates.
    // Only use stream-k when batch size is large enough to amortize the overhead.
    constexpr int64_t MMQ_STREAM_K_MIN_BATCH = 8;  // Threshold: skip stream-k for ne1 < 8
    const bool use_stream_k = ((GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA)
                            || GGML_CUDA_CC_IS_CDNA(cc))
                            && ne1 >= MMQ_STREAM_K_MIN_BATCH;

    // TODO: tighter pool buffer size vs q8 path
    const bool use_native_mxfp4 = blackwell_mma_available(cc) && src0->type == GGML_TYPE_MXFP4;

    if (!ids) {
        const size_t nbytes_src1_q8_1 = ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1 +
            get_mmq_x_max_host(cc)*sizeof(block_q8_1_mmq);
        ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);

        {
            const int64_t s11 = src1->nb[1] / ts_src1;
            const int64_t s12 = src1->nb[2] / ts_src1;
            const int64_t s13 = src1->nb[3] / ts_src1;
            if (use_native_mxfp4) {
                static_assert(sizeof(block_fp4_mmq) == 4 * sizeof(block_q8_1));
                quantize_mmq_mxfp4_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded,
                                        ne11, ne12, ne13, stream);

            } else {
                quantize_mmq_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded,
                                       ne11, ne12, ne13, stream);
            }
            CUDA_CHECK(cudaGetLastError());
        }

        // Stride depends on quantization format
        const int64_t s12 = use_native_mxfp4 ?
                                ne11 * ne10_padded * sizeof(block_fp4_mmq) /
                                    (8 * QK_MXFP4 * sizeof(int))  // block_fp4_mmq holds 256 values (8 blocks of 32)
                                :
                                ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
        const int64_t s13 = ne12*s12;

        const mmq_args args = {
            src0_d, src0->type, (const int *) src1_q8_1.ptr, nullptr, nullptr, dst_d,
            ne00, ne01, ne1, s01, ne11, s1,
            ne02, ne12, s02, s12, s2,
            ne03, ne13, s03, s13, s3,
            use_stream_k, ne1};
        ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
        return;
    }

    GGML_ASSERT(ne13 == 1);
    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2  % nb1  == 0);

    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows = ne12 * n_expert_used;
    GGML_ASSERT(ne1 == n_expert_used);

    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool(), ne02 + 1);

    {
        GGML_ASSERT(ids->nb[0] == ggml_element_size(ids));
        const int si1  = ids->nb[1] / ggml_element_size(ids);
        const int sis1 = nb12 / nb11;

        fprintf(stderr, "\n[MMQ MoE DEBUG] Launching mm_ids_helper...\n");
        fprintf(stderr, "[MMQ MoE DEBUG] ne02=%ld, ne12=%ld, n_expert_used=%ld, ne11=%ld\n",
                (long)ne02, (long)ne12, (long)n_expert_used, (long)ne11);
        fprintf(stderr, "[MMQ MoE DEBUG] si1=%d, sis1=%d\n", si1, sis1);

        // DEBUG: Validate ids tensor strides - THIS IS LIKELY THE BUG!
        fprintf(stderr, "[MMQ MoE DEBUG] ids tensor strides: nb=[%zu, %zu, %zu, %zu]\n",
                ids->nb[0], ids->nb[1], ids->nb[2], ids->nb[3]);
        fprintf(stderr, "[MMQ MoE DEBUG] ids tensor ne=[%lld, %lld, %lld, %lld]\n",
                (long long)ids->ne[0], (long long)ids->ne[1], (long long)ids->ne[2], (long long)ids->ne[3]);

        // Expected stride for contiguous [n_expert_used, n_tokens] tensor:
        // nb[1] should be n_expert_used * sizeof(int32) = 8 * 4 = 32
        // si1 should be 8, NOT 128!
        const size_t expected_nb1 = ids->ne[0] * sizeof(int32_t);
        const int expected_si1 = (int)ids->ne[0];  // = n_expert_used = 8
        if ((size_t)ids->nb[1] != expected_nb1) {
            fprintf(stderr, "[MMQ MoE CRITICAL] ids tensor has WRONG stride!\n");
            fprintf(stderr, "[MMQ MoE CRITICAL] Expected nb[1]=%zu (si1=%d), got nb[1]=%zu (si1=%d)\n",
                    expected_nb1, expected_si1, ids->nb[1], si1);
            fprintf(stderr, "[MMQ MoE CRITICAL] This will cause mm_ids_helper to read garbage!\n");

            // Print first few rows of ids tensor to show the problem
            CUDA_CHECK(cudaStreamSynchronize(stream));
            std::vector<int32_t> ids_data(std::min((int64_t)256, ids->ne[0] * ids->ne[1]));
            CUDA_CHECK(cudaMemcpy(ids_data.data(), ids->data, ids_data.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));

            fprintf(stderr, "[MMQ MoE DEBUG] ids tensor raw data (first 256 elements as linear array):\n");
            for (int row = 0; row < std::min(4, (int)ids->ne[1]); row++) {
                fprintf(stderr, "  Row %d (with current si1=%d, offset %d): ", row, si1, row * si1);
                for (int col = 0; col < std::min(8, (int)ids->ne[0]); col++) {
                    int idx = row * si1 + col;
                    if (idx < (int)ids_data.size()) {
                        fprintf(stderr, "%d ", ids_data[idx]);
                    } else {
                        fprintf(stderr, "OOB ");
                    }
                }
                fprintf(stderr, "\n");
            }
            fprintf(stderr, "  Row %d (with correct si1=%d, offset %d): ", 0, expected_si1, 0);
            for (int col = 0; col < std::min(8, (int)ids->ne[0]); col++) {
                fprintf(stderr, "%d ", ids_data[col]);
            }
            fprintf(stderr, "\n");
            fprintf(stderr, "  Row %d (with correct si1=%d, offset %d): ", 1, expected_si1, expected_si1);
            for (int col = 0; col < std::min(8, (int)ids->ne[0]); col++) {
                fprintf(stderr, "%d ", ids_data[expected_si1 + col]);
            }
            fprintf(stderr, "\n");
        }

        fprintf(stderr, "[MMQ MoE DEBUG] ids_src1=%p (size=%ld elements)\n",
                (void*)ids_src1.get(), (long)ne_get_rows);

        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), expert_bounds.get(),
            ne02, ne12, n_expert_used, ne11, si1, sis1, stream);
        CUDA_CHECK(cudaGetLastError());
        
        // DEBUG: Sync and validate mm_ids_helper output - CHECK ALL VALUES!
        {
            CUDA_CHECK(cudaStreamSynchronize(stream));
            fprintf(stderr, "[MMQ MoE DEBUG] mm_ids_helper completed\n");

            // Read back ALL values to verify they're valid indices
            std::vector<int32_t> ids_host(ne_get_rows);
            CUDA_CHECK(cudaMemcpy(ids_host.data(), ids_src1.get(), ids_host.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
            fprintf(stderr, "[MMQ MoE DEBUG] First 16 ids_src1 values: ");
            for (size_t i = 0; i < std::min((size_t)16, ids_host.size()); i++) {
                fprintf(stderr, "%d ", ids_host[i]);
            }
            fprintf(stderr, "\n");

            // Check ALL values (should be in range [0, ne12))
            // For MoE, ids_src1[i] = token_index, which must be < ne12 (number of tokens)
            int64_t max_valid = ne12;  // Token indices must be < ne12
            int bad_count = 0;
            int first_bad_idx = -1;
            int32_t first_bad_val = 0;
            int32_t max_seen = 0;
            int32_t min_seen = INT32_MAX;

            for (size_t i = 0; i < ids_host.size(); i++) {
                int32_t val = ids_host[i];
                if (val > max_seen) max_seen = val;
                if (val < min_seen) min_seen = val;

                if (val < 0 || val >= max_valid) {
                    bad_count++;
                    if (first_bad_idx < 0) {
                        first_bad_idx = (int)i;
                        first_bad_val = val;
                    }
                }
            }

            fprintf(stderr, "[MMQ MoE DEBUG] ids_src1 stats: count=%zu, min=%d, max=%d, expected_max=%ld\n",
                    ids_host.size(), min_seen, max_seen, (long)(max_valid - 1));

            if (bad_count > 0) {
                fprintf(stderr, "[MMQ MoE CRITICAL] Found %d INVALID ids_src1 values!\n", bad_count);
                fprintf(stderr, "[MMQ MoE CRITICAL] First bad: ids_src1[%d] = %d (must be in [0, %ld))\n",
                        first_bad_idx, first_bad_val, (long)max_valid);

                // Print surrounding context
                int ctx_start = std::max(0, first_bad_idx - 5);
                int ctx_end = std::min((int)ids_host.size(), first_bad_idx + 10);
                fprintf(stderr, "[MMQ MoE CRITICAL] Context around first bad value [%d..%d]:\n  ", ctx_start, ctx_end);
                for (int j = ctx_start; j < ctx_end; j++) {
                    if (j == first_bad_idx) fprintf(stderr, "[");
                    fprintf(stderr, "%d", ids_host[j]);
                    if (j == first_bad_idx) fprintf(stderr, "]");
                    fprintf(stderr, " ");
                }
                fprintf(stderr, "\n");

                // Check if the bad index would cause the actual OOB read
                // base_idx = ids[i1] * s01 + i00, where s01 = 2048 for this case
                int64_t bad_base_idx = (int64_t)first_bad_val * 2048;
                int64_t max_safe_base = ne12 * 2048;  // = 346 * 2048 = 708608
                fprintf(stderr, "[MMQ MoE CRITICAL] Bad value %d would access base_idx=%lld (max safe=%lld)\n",
                        first_bad_val, (long long)bad_base_idx, (long long)max_safe_base);
            } else {
                fprintf(stderr, "[MMQ MoE DEBUG] All %zu ids_src1 values are valid [0, %ld)\n",
                        ids_host.size(), (long)max_valid);
            }
        }
    }

    const size_t nbytes_src1_q8_1 = ne12*n_expert_used*ne10_padded * sizeof(block_q8_1)/QK8_1 +
        get_mmq_x_max_host(cc)*sizeof(block_q8_1_mmq);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);

    const int64_t ne11_flat = ne12*n_expert_used;
    const int64_t ne12_flat = 1;
    const int64_t ne13_flat = 1;

    {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;

        if (use_native_mxfp4) {
            quantize_mmq_mxfp4_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src0->type, ne10, s11, s12, s13,
                                    ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
        } else {
            quantize_mmq_q8_1_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src0->type, ne10, s11, s12, s13,
                                   ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
        }
        CUDA_CHECK(cudaGetLastError());
    }

    const int64_t s12 = use_native_mxfp4 ? ne11 * ne10_padded * sizeof(block_fp4_mmq) / (8 * QK_MXFP4 * sizeof(int)) :
                                           ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
    const int64_t s13 = ne12*s12;

    // Note that ne02 is used instead of ne12 because the number of y channels determines the z dimension of the CUDA grid.
    const mmq_args args = {
        src0_d, src0->type, (const int *) src1_q8_1.get(), ids_dst.get(), expert_bounds.get(), dst_d,
        ne00, ne01, ne_get_rows, s01, ne_get_rows, s1,
        ne02, ne02, s02, s12, s2,
        ne03, ne13, s03, s13, s3,
        use_stream_k, ne12};

    ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
}

void ggml_cuda_op_mul_mat_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    const int64_t row_diff = row_high - row_low;
    const int64_t stride01 = ne00 / ggml_blck_size(src0->type);

    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    // The stream-k decomposition is only faster for recent NVIDIA GPUs.
    // Also its fixup needs to allocate a temporary buffer in the memory pool.
    // There are multiple parallel CUDA streams for src1_ncols != ne11 which would introduce a race condition for this buffer.
    // Skip stream-k for small batches where fixup overhead dominates.
    constexpr int64_t MMQ_STREAM_K_MIN_BATCH_OP = 8;  // Same threshold as main path
    const bool use_stream_k = ((GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA)
                            || GGML_CUDA_CC_IS_CDNA(cc))
                            && src1_ncols == ne11
                            && src1_ncols >= MMQ_STREAM_K_MIN_BATCH_OP;
    const mmq_args args = {
        src0_dd_i, src0->type, (const int *) src1_ddq_i, nullptr, nullptr, dst_dd_i,
        ne00, row_diff, src1_ncols, stride01, ne11, nrows_dst,
        1, 1, 0, 0, 0,
        1, 1, 0, 0, 0,
        use_stream_k, src1_ncols};

    ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);

    GGML_UNUSED_VARS(src1, dst, src1_ddf_i, src1_padded_row_size);
}

bool ggml_cuda_should_use_mmq(enum ggml_type type, int cc, int64_t ne11, int64_t n_experts) {
#ifdef GGML_CUDA_FORCE_CUBLAS
    return false;
#endif // GGML_CUDA_FORCE_CUBLAS

    bool mmq_supported;

    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
            mmq_supported = true;
            break;
        default:
            mmq_supported = false;
            break;
    }

    if (!mmq_supported) {
        return false;
    }

    if (turing_mma_available(cc)) {
        return true;
    }

    if (ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A) {
        return false;
    }

#ifdef GGML_CUDA_FORCE_MMQ
    return true;
#endif //GGML_CUDA_FORCE_MMQ

    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
    }

    if (amd_mfma_available(cc)) {
        // As of ROCM 7.0 rocblas/tensile performs very poorly on CDNA3 and hipblaslt (via ROCBLAS_USE_HIPBLASLT)
        // performs better but is currently suffering from a crash on this architecture.
        // TODO: Revisit when hipblaslt is fixed on CDNA3
        if (GGML_CUDA_CC_IS_CDNA3(cc)) {
            return true;
        }
        if (n_experts > 64 || ne11 <= 128) {
            return true;
        }
        if (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q5_1) {
            return true;
        }
        if (ne11 <= 256 && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K)) {
            return true;
        }
        return false;
    }

    if (amd_wmma_available(cc)) {
        return true;
    }

    return (!GGML_CUDA_CC_IS_CDNA(cc)) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;

}
