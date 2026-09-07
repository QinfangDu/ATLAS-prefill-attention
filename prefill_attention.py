"""Prefill FlashAttention operator (ATLang, general MPMD region).

Implements only the attention core described in
`PREFILL_FLASHATTENTION_DESIGN.md` section 5 ("方案 A: Q 驻留 + KV 环形轮转",
Ring Attention). QKV/O/FFN projections and their inter-core communication are
out of scope for this file; this is the standalone flash-attention operator
that a full prefill pipeline (`frontend/ops/atlang/cloud_prefill.py` in the
design doc's roadmap) would slot in as its "attention" region.

Mapping (4x4 mesh, `CORE_ID = row * core_array_size + col`):
  - row selects a token chunk (`S_c` tokens out of `total_tokens`)
  - col selects one active KV head when `kv_head_num <= core_array_size`; the
    ring then shrinks dynamically to `P = kv_head_num` and only the active
    columns participate in the KV rotation.
  - Q is resident per core for the whole kernel; KV rotates in a ring across
    the active rows of the same column (`CORE_ID +/- core_array_size`), so
    every core eventually sees every token's KV.
  - Online-softmax state (m, l, O) is carried across ring steps through a
    per-core DRAM scratch tensor (`flash_state_*`), since the design assumes
    the Q-tile SRAM budget cannot also hold state for every Q tile block
    simultaneously (see design doc section 4).

Stage-1 scope (per design doc section 6): no causal mask, no zig-zag load
balancing 
"""

import math
import multiprocessing
from typing import Tuple

import frontend.atlang.language as A

from frontend.hardware_parser import CloudSystemConfig
from frontend.ops.atlang.cloud import sanity_check


def prefill_flash_attention_inference(
    cloud_config: CloudSystemConfig,
    attention_shape: Tuple[int, int, int, int, int],
    dtype=A.float16,
    min_tQ: int = 128,
    min_tS: int = 128,
    intermediate_result_dir: str = "",
    gemm_tiling_cache_dir: str = "",
    num_workers: int = int(0.8 * multiprocessing.cpu_count()),
):
    """Build the prefill FlashAttention ATLang kernel.

    attention_shape = (total_tokens, kv_head_num, kv_group_num, qk_head_dim, v_head_dim)
      - total_tokens: prefill sequence length (single request, no batching)
      - kv_head_num: number of KV heads
      - kv_group_num: number of Q heads sharing one KV head (GQA/MQA group size)
      - qk_head_dim / v_head_dim: per-head dimensions for Q/K and V respectively

    Note: When kv_head_num < core_array_size, multiple columns share the same KV head.
    Partial softmax states (m, l, O) are merged across columns at the end.
    """
    sanity_check(cloud_config)
    core_num = cloud_config.chip_config.core_num
    core_array_size = int(math.sqrt(core_num))

    total_tokens, kv_head_num, kv_group_num, qk_head_dim, v_head_dim = attention_shape
    if kv_head_num <= 0:
        raise ValueError("kv_head_num must be positive.")
    

    # Dynamic ring-step shrink: when there are fewer KV heads than columns, only
    # the active KV-head columns participate in the ring; otherwise keep the full
    # core-array ring length.
    P = kv_head_num if kv_head_num < core_array_size else core_array_size
    S_c = A.ceildiv(total_tokens, core_array_size)  # tokens owned by each row-chunk
    bQ = A.ceildiv(S_c,min_tQ)
    bK = A.ceildiv(S_c, min_tS)
    tQ = min_tQ
    tS = min_tS
    kv_vec_dim = qk_head_dim + v_head_dim

    # Only the active KV-head columns are materialized; inactive columns simply
    # skip the ring computation and never allocate a per-head tile for them.
    padded_Gkv = kv_head_num // core_array_size if kv_head_num >= core_array_size else 1

    # ATLAS's 3D-DRAM is physically partitioned per core (non-contiguous global address space),
    # so each core's DRAM only ever holds its own row chunk (S_c tokens) and column chunk (Gkv KV
    # head groups) -- there is no globally addressable (total_tokens, kv_head_num*...) tensor.
    # These are therefore the *local, per-core* row strides (kv_head_num already divided down to
    # Gkv by the column split), matching the SPMD tensors in cloud.py which are declared with the
    # already-partitioned core_M/core_K/core_N sizes rather than the global GEMM shape.
    # When kv_head_num < core_array_size, only columns [0, kv_head_num) have actual KV heads;
    # remaining columns have padded_Gkv=1 but will skip g_kv iteration via boundary check.
    q_row_stride = padded_Gkv * kv_group_num * qk_head_dim
    kv_row_stride = padded_Gkv * kv_vec_dim
    o_row_stride = padded_Gkv * kv_group_num * v_head_dim

    core_array_kwargs = {
        "system_config": cloud_config,
        "dram_row_size": 128 * 1024,
        "flit_size": cloud_config.chip_config.noc_config.flit_size,
        "inter_chip_communication_list": [],
        "min_tS": min_tS,
        "num_workers": num_workers,
        "intermediate_result_dir": intermediate_result_dir,
        "gemm_tiling_cache_dir": gemm_tiling_cache_dir,
    }

    @A.main
    def prefill_flash_attention(
        input_q_prefill: A.Tensor,
        input_kv_prefill: A.Tensor,
        output_o_prefill: A.Tensor,
        flash_state_m: A.Tensor,
        flash_state_l: A.Tensor,
        flash_state_o: A.Tensor,
    ):
        with A.CoreArray(shape=(core_array_size, core_array_size), **core_array_kwargs):
            with A.MPMD(name="prefill_flash_attention", type="general") as CORE_ID:
                with A.Kernel(core_list=list(range(core_num))):
                    # 3D-DRAM per-core memory is physically partitioned and non-contiguous across
                    # cores, so every tensor declared here is already the *local* per-core slice --
                    # row-partitioned by S_c (this core's token chunk) and column-partitioned by
                    # Gkv (this core's KV head groups) -- there is no globally addressable
                    # (total_tokens, kv_head_num*...) tensor, mirroring how cloud.py's SPMD/MPMD
                    # regions declare tensors sized to the already-partitioned core_M/core_K/core_N.
                    input_q_prefill = A.Tensor(shape=(S_c, q_row_stride), strides=(q_row_stride, 1), dtype=dtype)
                    input_kv_prefill = A.Tensor(shape=(S_c, kv_row_stride), strides=(kv_row_stride, 1), dtype=dtype)
                    output_o_prefill = A.Tensor(shape=(S_c, o_row_stride), strides=(o_row_stride, 1), dtype=dtype)
                    # Per-core online-softmax scratch (already local -- no cross-core indexing needed).
                    flash_state_m = A.Tensor(
                        shape=(padded_Gkv, bQ, kv_group_num, tQ),
                        strides=(bQ * kv_group_num * tQ, kv_group_num * tQ, tQ, 1),
                        dtype=dtype,
                    )
                    flash_state_l = A.Tensor(
                        shape=(padded_Gkv, bQ, kv_group_num, tQ),
                        strides=(bQ * kv_group_num * tQ, kv_group_num * tQ, tQ, 1),
                        dtype=dtype,
                    )
                    flash_state_o = A.Tensor(
                        shape=(padded_Gkv, bQ, kv_group_num, tQ, v_head_dim),
                        strides=(
                            bQ * kv_group_num * tQ * v_head_dim,
                            kv_group_num * tQ * v_head_dim,
                            tQ * v_head_dim,
                            v_head_dim,
                            1,
                        ),
                        dtype=dtype,
                    )

                    core_row = CORE_ID // core_array_size
                    core_col = CORE_ID % core_array_size
                    # Wrap-around ring neighbors within the same column (row +/- 1 mod core_array_size).
                    next_core = ((core_row + 1) % core_array_size) * core_array_size + core_col
                    prev_core = ((core_row - 1 + core_array_size) % core_array_size) * core_array_size + core_col

                    # Q tile (resident per (qb, n) iteration), KV ring chunk (whole S_c owned/received per step),
                    # and the online-softmax working tiles -- mirrors the decode-attention body in cloud.py:825-846.
                    q_tile = A.alloc(shape=(tQ, qk_head_dim), dtype=dtype)
                    kv_buf = A.alloc(shape=(S_c, kv_vec_dim), dtype=dtype)
                    kv_recv_buf = A.alloc(shape=(S_c, kv_vec_dim), dtype=dtype)
                    acc_tile = A.alloc(shape=(tQ, tS), dtype=dtype)
                    scores_max_tile = A.alloc(shape=(tQ,), dtype=dtype)
                    scores_max_prev_tile = A.alloc(shape=(tQ,), dtype=dtype)
                    scores_scale_tile = A.alloc(shape=(tQ,), dtype=dtype)
                    log_sum_tile = A.alloc(shape=(tQ,), dtype=dtype)
                    log_sum_prev_tile = A.alloc(shape=(tQ,), dtype=dtype)
                    o_tile_new = A.alloc(shape=(tQ, v_head_dim), dtype=dtype)
                    o_tile = A.alloc(shape=(tQ, v_head_dim), dtype=dtype)

                    if core_col < kv_head_num or core_array_size <= kv_head_num :
                        for g_kv in A.Serial(padded_Gkv):
                            for step in A.Serial(core_array_size):
                                if step == 0:
                                    A.copy(
                                        input_kv_prefill[:, g_kv * kv_vec_dim: (g_kv + 1) * kv_vec_dim],
                                        kv_buf,
                                    )

                                for qb in A.Serial(bQ):
                                    for n in A.serial(kv_group_num):
                                        if step == 0:
                                            A.fill(scores_max_tile, -A.infinity(dtype))
                                            A.clear(log_sum_tile)
                                            A.clear(o_tile)
                                        else:
                                            A.copy(flash_state_m[g_kv, qb, n, :], scores_max_tile)
                                            A.copy(flash_state_l[g_kv, qb, n, :], log_sum_tile)
                                            A.copy(flash_state_o[g_kv, qb, n, :, :], o_tile)

                                        A.copy(
                                            input_q_prefill[
                                                qb * tQ: (qb + 1) * tQ,
                                                (g_kv * kv_group_num + n) * qk_head_dim:
                                                (g_kv * kv_group_num + n + 1) * qk_head_dim,
                                            ],
                                            q_tile,
                                        )

                                        for kb in A.Serial(bK):
                                            A.copy(scores_max_tile, scores_max_prev_tile)
                                            # 1st GEMM: Q . K^T
                                            A.gemm(
                                                q_tile,
                                                kv_buf[kb * tS: (kb + 1) * tS, :qk_head_dim],
                                                acc_tile,
                                                transpose_B=True,
                                            )
                                            # Online softmax (same recurrence as cloud.py:825-846)
                                            A.reduce_max(acc_tile, scores_max_tile, dim=1, clear=True)
                                            A.max(scores_max_tile, scores_max_prev_tile, scores_max_tile)
                                            A.sub(acc_tile, scores_max_tile, acc_tile)
                                            A.exp(acc_tile, acc_tile)
                                            A.sub(scores_max_prev_tile, scores_max_tile, scores_scale_tile)
                                            A.exp(scores_scale_tile, scores_scale_tile)
                                            A.mul(log_sum_tile, scores_scale_tile, log_sum_prev_tile)
                                            A.reduce_sum(acc_tile, log_sum_tile, dim=1, clear=True)
                                            A.add(log_sum_tile, log_sum_prev_tile, log_sum_tile)
                                            # 2nd GEMM: softmax(QK^T) . V
                                            A.gemm(acc_tile, kv_buf[kb * tS: (kb + 1) * tS, qk_head_dim:], o_tile_new)
                                            # O_i = (O_{i-1} * e^{m_{i-1}-m_i} + O_new) / l_i
                                            A.mul(o_tile, log_sum_prev_tile, o_tile)
                                            A.add(o_tile, o_tile_new, o_tile)
                                            A.div(o_tile, log_sum_tile, o_tile)

                                        if step == core_array_size - 1:
                                            # Last ring step already normalized -- write straight to the output.
                                            A.copy(
                                                o_tile,
                                                output_o_prefill[
                                                    qb * tQ: (qb + 1) * tQ,
                                                    (g_kv * kv_group_num + n) * v_head_dim:
                                                    (g_kv * kv_group_num + n + 1) * v_head_dim,
                                                ],
                                            )
                                        else:
                                            A.copy(scores_max_tile, flash_state_m[g_kv, qb, n, :])
                                            A.copy(log_sum_tile, flash_state_l[g_kv, qb, n, :])
                                            A.copy(o_tile, flash_state_o[g_kv, qb, n, :, :])

                                if step < core_array_size - 1:
                                    A.send(CORE_ID, next_core, kv_buf)
                                    A.recv(prev_core, CORE_ID, kv_recv_buf)
                                    A.copy(kv_recv_buf, kv_buf)

    return prefill_flash_attention


__all__ = ["prefill_flash_attention_inference"]
