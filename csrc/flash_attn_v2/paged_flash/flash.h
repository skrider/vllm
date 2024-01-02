/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cuda.h>
#include <vector>

#include "attention_atom.h"

constexpr int TOTAL_DIM = 0;
constexpr int H_DIM = 1;
constexpr int D_DIM = 2;

#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Qkv_params {
    using index_t = uint32_t;
    // The QKV matrices.
    void* __restrict__ q_ptr;
    void* __restrict__ k_ptr;
    void* __restrict__ v_ptr;

    // The stride between rows of the Q, K and V matrices.
    index_t q_row_stride;
    index_t k_row_stride;
    index_t v_row_stride;
    index_t q_head_stride;
    index_t k_head_stride;
    index_t v_head_stride;

    // The number of heads.
    int h, h_k;
    // In the case of multi-query and grouped-query attention (MQA/GQA), nheads_k could be
    // different from nheads (query).
    int h_h_k_ratio;  // precompute h / h_k,
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Flash_fwd_params : public Qkv_params {
    // The O matrix (output).
    void* __restrict__ o_ptr;
    using block_index_t = int64_t;

    // The attention metadata
    // AttentionAtom* __restrict__ atoms;

    // Total attention atoms
    // int num_atoms;

    // PagedAttention metadata
    int num_seqs;
    int max_num_query;
    int max_context_len;
    int block_size;
    int max_num_blocks_per_seq;

    block_index_t* __restrict__ block_tables;
    index_t* __restrict__ context_lens;
    index_t* __restrict__ draft_lens;

    // The stride between rows of O.
    index_t o_row_stride;
    index_t o_head_stride;

    // The dimensions
    int d, d_rounded;

    // The scaling factors for the kernel.
    float scale_softmax;
    float scale_softmax_log2;

    bool is_bf16;
    bool is_causal;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, int Headdim>
void run_mha_fwd_(Flash_fwd_params& params, cudaStream_t stream);
