import math

import torch
import triton
import triton.language as tl

from vllm.utils import direct_register_custom_op


@triton.jit
def _lora_expand_slices_kernel(
    input_ptr,
    lora_ptr,
    out_ptr,
    N,
    K,
    lora_seq_indices,
    lora_seq_counts,
    lora_seq_start_loc,
    lora_ids,
    xm_stride,
    xk_stride,  # 1
    l0_stride,  # hidden_size * max_rank * num_loras
    l1_stride,  # hidden_size*max_rank
    lora_n_stride,
    lora_k_stride,
    cm_stride,
    cn_stride,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    NUM_SLICES: tl.constexpr,
    MAX_LORAS: tl.constexpr,
    NUM_M_CTAS: tl.constexpr,
    NUM_N_CTAS: tl.constexpr,
    ADD_INPUTS: tl.constexpr,
    CAST_TYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    cta_s = pid // (MAX_LORAS * NUM_M_CTAS * NUM_N_CTAS)
    cta_l = (pid // (NUM_M_CTAS * NUM_N_CTAS)) % MAX_LORAS
    cta_n = (pid // NUM_M_CTAS) % NUM_N_CTAS
    cta_m = pid % NUM_M_CTAS

    lora_id = tl.load(lora_ids + cta_l)
    if lora_id == -1:
        # early exit for the no-lora case.
        return

    # lora m indices offsets
    if cta_l == 0:
        lora_m_indices_start = tl.cast(0, tl.int32)
    else:
        lora_m_indices_start = tl.load(lora_seq_start_loc + cta_l - 1)
    lora_m_size = tl.load(lora_seq_counts + cta_l)

    cta_m_offset = cta_m * BLOCK_M
    if cta_m_offset >= lora_m_size:
        # early exit CTA
        return

    cta_lora_seq_indices = (lora_seq_indices + lora_m_indices_start +
                            cta_m_offset)
    cta_m_size = min(BLOCK_M, lora_m_size - cta_m_offset)

    offset_k = tl.arange(0, BLOCK_K)

    offset_rm = tl.arange(0, BLOCK_M) % cta_m_size
    rm = tl.load(cta_lora_seq_indices + offset_rm)
    a_ptr = input_ptr + rm[:, None] * xm_stride + offset_k[None, :] * xk_stride

    offset_n = tl.arange(0, BLOCK_N) + cta_n * BLOCK_N
    rbn = tl.max_contiguous(tl.multiple_of(offset_n % N, BLOCK_N), BLOCK_N)
    b_ptr = (lora_ptr + l0_stride * cta_s + l1_stride * lora_id +
             offset_k[:, None] * lora_k_stride + rbn[None, :] * lora_n_stride)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            tiled_a = tl.load(a_ptr)
            tiled_b = tl.load(b_ptr)
        else:
            tiled_a = tl.load(a_ptr,
                              mask=offset_k[None, :] < K - k * BLOCK_K,
                              other=0)
            tiled_b = tl.load(b_ptr,
                              mask=offset_k[:, None] < K - k * BLOCK_K,
                              other=0)
        if CAST_TYPE:
            tiled_a = tiled_a.to(lora_ptr.dtype.element_ty)
        accumulator += tl.dot(
            tiled_a,
            tiled_b,
        )
        a_ptr += BLOCK_K * xk_stride
        b_ptr += BLOCK_K * lora_k_stride

    slice_offset = cta_s * N
    tiled_c = accumulator.to(lora_ptr.dtype.element_ty)
    offset_cm = tl.arange(0, BLOCK_M)
    offset_cn = tl.arange(0, BLOCK_N) + cta_n * BLOCK_N + slice_offset
    c_ptr = out_ptr + rm[:, None] * cm_stride + offset_cn[None, :] * cn_stride

    c_mask = (offset_cm[:, None] < cta_m_size) & (offset_cn[None, :] <
                                                  (slice_offset + N))
    if ADD_INPUTS:
        tiled_out = tl.load(c_ptr, mask=c_mask)
        tiled_c += tiled_out
    tl.store(c_ptr, tiled_c, mask=c_mask)


@torch.inference_mode()
def _lora_expand_slices(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_seq_indices: torch.Tensor,
    lora_seq_counts: torch.Tensor,
    lora_seq_start_loc: torch.Tensor,
    lora_ids: torch.Tensor,
    add_inputs: bool = False,
) -> None:
    """_summary_

    Args:
        inputs (torch.Tensor): input tensor
        lora_b_weights (torch.Tensor): lora'a weight
        output_tensor (torch.Tensor): output tensor

        token_lora_mapping_tensor: Each token's lora id as it appears in the
            A matrix.

        lora_seq_indices: sorted lora-token mapping. Tokens of the same lora
            appear next to each other. This is used so a thread block knows
            what tokens to put next to each other when constructing a matrix
            block. Essentially, 
            _, lora_seq_indices = torch.sort(token_lora_mapping, stable=True)

        lora_seq_counts: number of tokens per lora id. essentially,
            lora_ids, lora_seq_counts = torch.unique(indices,
                                              sorted=False,
                                              return_counts=True)

        lora_seq_start_loc: start index of each lora id in lora_seq_indices.
            essentially,
            lora_seq_start_loc = torch.cumsum(lora_seq_counts, dim = 0)

        lora_ids : Set of lora ids in order according to lora_seq_counts,
            and lora_seq_indices.
            lora_ids, lora_seq_counts = torch.unique(indices,
                                              sorted=False,
                                              return_counts=True)

        add_inputs (bool, optional): Defaults to False, adds the final lora 
            results to the output.
    """

    assert inputs.dtype in [torch.float16, torch.bfloat16, torch.float32]
    assert lora_b_weights.dtype in [
        torch.float16,
        torch.bfloat16,
    ]
    assert inputs.size(1) == lora_b_weights.size(-1)
    num_slices = lora_b_weights.size(0)
    assert output_tensor.size(1) == lora_b_weights.size(-2) * num_slices
    assert inputs.is_contiguous()
    assert output_tensor.is_contiguous()

    assert lora_b_weights.ndim == 4  # (nslices, lora_num, hidden-size, rank)
    assert lora_b_weights.is_contiguous()

    # TODO tuning this config
    N = lora_b_weights.size(-2)
    K = lora_b_weights.size(-1)
    NUM_SLICES = lora_b_weights.size(0)
    M = inputs.size(0)
    MAX_LORAS = lora_ids.size(0)

    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 16
    EVEN_K = K % BLOCK_K == 0
    NUM_M_CTAS = math.ceil(M / BLOCK_M)
    NUM_N_CTAS = math.ceil(N / BLOCK_N)

    ADD_INPUTS = add_inputs
    CAST_TYPE = False
    if inputs.dtype == torch.float32 and lora_b_weights.dtype in [
            torch.float16,
            torch.bfloat16,
    ]:
        CAST_TYPE = True

    grid = (NUM_SLICES * MAX_LORAS * NUM_M_CTAS * NUM_N_CTAS, )

    xm_stride = inputs.stride(0)
    xk_stride = inputs.stride(1)
    l0_stride = lora_b_weights.stride(0)  # slice stride
    l1_stride = lora_b_weights.stride(1)  # lora stride
    lora_n_stride = lora_b_weights.stride(2)
    lora_k_stride = lora_b_weights.stride(3)
    cm_stride = output_tensor.stride(0)
    cn_stride = output_tensor.stride(1)

    _lora_expand_slices_kernel[grid](
        inputs,
        lora_b_weights,
        output_tensor,
        N,
        K,
        lora_seq_indices,
        lora_seq_counts,
        lora_seq_start_loc,
        lora_ids,
        xm_stride,
        xk_stride,
        l0_stride,
        l1_stride,
        lora_n_stride,
        lora_k_stride,
        cm_stride,
        cn_stride,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        EVEN_K,
        NUM_SLICES,
        MAX_LORAS,
        NUM_M_CTAS,
        NUM_N_CTAS,
        ADD_INPUTS,
        CAST_TYPE,
    )
    return


try:
    lora_expand_slices = torch.library.custom_op(
        "lora::v1::lora_expand_slices",
        _lora_expand_slices,
        mutates_args=["output_tensor"])
except AttributeError:
    lora_expand_slices = _lora_expand_slices


def lora_expand_slices_fake(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_seq_indices: torch.Tensor,
    lora_seq_counts: torch.Tensor,
    lora_seq_start_loc: torch.Tensor,
    lora_ids: torch.Tensor,
    add_inputs: bool = False,
) -> None:
    return


try:
    direct_register_custom_op(
        op_name="lora_expand_slices",
        op_func=_lora_expand_slices,
        mutates_args=["output_tensor"],
        fake_impl=lora_expand_slices_fake,
    )
    lora_expand_slices = torch.ops.vllm.lora_expand_slices

except AttributeError:
    lora_expand = _lora_expand_slices
