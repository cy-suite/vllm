import torch
import triton
import triton.language as tl

from vllm.lora.ops.bgmv_shrink import bgmv_shrink
from vllm.utils import direct_register_custom_op


@triton.jit
def _lora_shrink_kernel(
    input_ptr,
    lora_ptr,
    out_ptr,
    N,
    K,
    token_indices_sorted_by_lora_ids,
    num_tokens_per_lora,
    lora_token_start_loc,
    lora_ids,
    scaling,
    xm_stride,
    xk_stride,
    l0_stride,
    lora_k_stride,
    lora_n_stride,
    cm_stride,
    cn_stride,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    SMALL_BLOCK_M: tl.constexpr,
    NUM_M_CTAS: tl.constexpr,
    NUM_N_CTAS: tl.constexpr,
):
    lora_idx = tl.program_id(1)
    lora_id = tl.load(lora_ids + lora_idx)
    if lora_id == -1:
        # early exit for the no-lora case.
        return

    pid = tl.program_id(0)
    cta_sk = pid // (NUM_M_CTAS * NUM_N_CTAS)
    cta_n = (pid // NUM_M_CTAS) % NUM_N_CTAS
    cta_m = pid % NUM_M_CTAS

    # lora m indices offsets
    lora_m_indices_start = tl.load(lora_token_start_loc + lora_idx)
    lora_m_size = tl.load(num_tokens_per_lora + lora_idx)

    cta_m_offset = cta_m * BLOCK_M
    if cta_m_offset >= lora_m_size:
        # early exit CTA
        return

    cta_lora_seq_indices = (token_indices_sorted_by_lora_ids +
                            lora_m_indices_start + cta_m_offset)
    cta_m_size = min(BLOCK_M, lora_m_size - cta_m_offset)

    offset_k = tl.max_contiguous(BLOCK_K * cta_sk + tl.arange(0, BLOCK_K),
                                 BLOCK_K)

    offset_rm = tl.arange(0, BLOCK_M) % cta_m_size
    rm = tl.load(cta_lora_seq_indices + offset_rm)
    a_ptr = input_ptr + rm[:, None] * xm_stride + offset_k[None, :] * xk_stride

    offset_n = tl.max_contiguous((cta_n * BLOCK_N) + tl.arange(0, BLOCK_N),
                                 BLOCK_N)
    rn = tl.max_contiguous(tl.multiple_of(offset_n % N, BLOCK_N), BLOCK_N)
    b_ptr = lora_ptr + lora_id * l0_stride + rn[
        None, :] * lora_n_stride + offset_k[:, None] * lora_k_stride

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    max_k = tl.cdiv(K, BLOCK_K * SPLIT_K)
    for k in range(0, max_k):
        if EVEN_K:
            b_tile = tl.load(b_ptr)
            a_tile = tl.load(a_ptr)
        else:
            b_mask = offset_k[:, None] < K
            b_tile = tl.load(b_ptr, mask=b_mask, other=0.0)

            a_mask = offset_k[None, :] < K
            a_tile = tl.load(a_ptr, mask=a_mask, other=0.0)

        # TODO (varun) : When a_tile and b_tile are float16s the output is
        # also float16. this can lead to infs in the output.
        if SMALL_BLOCK_M:
            #acc += tl.sum(a_tile * b_tile.T)
            acc += tl.sum(a_tile * b_tile.T, 1)
        else:
            acc += tl.dot(a_tile, b_tile)

        a_ptr += BLOCK_K * SPLIT_K * xk_stride
        b_ptr += BLOCK_K * SPLIT_K * lora_k_stride
        offset_k += BLOCK_K * SPLIT_K

    acc *= scaling

    offset_cm = tl.arange(0, BLOCK_M)
    c_ptr = out_ptr + rm[:, None] * cm_stride + offset_n[None, :] * cn_stride
    c_mask = (offset_cm[:, None] < cta_m_size) & (offset_n[None, :] < N)
    if SPLIT_K == 1:
        tl.store(c_ptr, acc, mask=c_mask)
    else:
        tl.atomic_add(c_ptr, acc, mask=c_mask)


@torch.inference_mode()
def _lora_shrink(
    inputs: torch.Tensor,
    lora_a_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    token_indices_sorted_by_lora_ids: torch.Tensor,  # inputs.size(0)
    num_tokens_per_lora: torch.Tensor,  # max-loras
    lora_token_start_loc: torch.Tensor,  # max-loras
    lora_ids: torch.Tensor,  # max-loras
    scaling: float,
) -> None:
    """
    Args:
        inputs (torch.Tensor): input tensor
        lora_a_weights (torch.Tensor): lora'a weight
        output_tensor (torch.Tensor): output tensor
        token_indices_sorted_by_lora_ids: Row/Token indices from the A matrix
        grouped by LoRA IDs.
        num_tokens_per_lora: num_tokens_per_lora[i] is the number of tokens
            that are to be processed by LoRA ID lora_ids[i] 
        lora_token_start_loc: A cumulative sum of num_tokens_per_lora.
            lora_token_start_loc[0] is always 0 so that lora_token_start_loc[i],
            along with num_tokens_per_lora[i] identifies the the region in
            token_indices_sorted_by_lora_ids that LoRA lora_ids[i] should
            process.
        lora_ids: LoRA ids to process.
        add_inputs (bool, optional): Defaults to False, adds the final lora 
            results to the output.
    """

    M = inputs.size(0)  # num tokens
    if M <= 16:
        # GemmV is better for smaller batchsizes
        return bgmv_shrink(inputs, lora_a_weights, output_tensor,
                           token_lora_mapping, scaling)

    assert inputs.dtype == lora_a_weights.dtype
    assert inputs.dtype in [torch.float16, torch.bfloat16]
    assert lora_a_weights.dtype in [
        torch.float16,
        torch.bfloat16,
    ]
    assert inputs.size(1) == lora_a_weights.size(-1)
    assert inputs.is_contiguous()

    if lora_a_weights.ndim == 4:  # shape:(lora_num,1,rank, size)
        assert lora_a_weights.size(1) == 1
        lora_a_weights = lora_a_weights.squeeze(dim=1)
    else:
        assert lora_a_weights.ndim == 3  # shape:(lora_num,rank, size)
    assert lora_a_weights.is_contiguous()
    assert output_tensor.is_contiguous()
    assert token_indices_sorted_by_lora_ids.size(0) == inputs.size(0)
    assert num_tokens_per_lora.size(0) == lora_ids.size(0)
    assert lora_token_start_loc.size(0) == lora_ids.size(0) + 1

    xm_stride = inputs.stride(0)
    xk_stride = inputs.stride(1)
    l0_stride = lora_a_weights.stride(0)
    lora_k_stride = lora_a_weights.stride(2)
    lora_n_stride = lora_a_weights.stride(1)
    cm_stride = output_tensor.stride(0)
    cn_stride = output_tensor.stride(1)

    # TODO tuning this config
    N = lora_a_weights.size(-2)
    K = lora_a_weights.size(-1)
    MAX_LORAS = lora_ids.size(0)

    BLOCK_M = 32
    BLOCK_N = 16

    if M < 128:
        BLOCK_K = 256
        SPLIT_K = 64
    else:
        BLOCK_K = 32
        SPLIT_K = 8

    num_warps = 4
    EVEN_K = K % (BLOCK_K * SPLIT_K) == 0
    SMALL_BLOCK_M = BLOCK_M < 16
    NUM_M_CTAS = triton.cdiv(M, BLOCK_M)
    NUM_N_CTAS = triton.cdiv(N, BLOCK_N)

    grid = (
        SPLIT_K * NUM_M_CTAS * NUM_N_CTAS,
        MAX_LORAS,
    )
    _lora_shrink_kernel[grid](
        inputs,
        lora_a_weights,
        output_tensor,
        N,
        K,
        token_indices_sorted_by_lora_ids,
        num_tokens_per_lora,
        lora_token_start_loc,
        lora_ids,
        scaling,
        xm_stride,
        xk_stride,
        l0_stride,
        lora_k_stride,
        lora_n_stride,
        cm_stride,
        cn_stride,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        EVEN_K,
        SPLIT_K,
        SMALL_BLOCK_M,
        NUM_M_CTAS,
        NUM_N_CTAS,
        num_warps=num_warps,
    )
    return


def lora_shrink_fake(
    inputs: torch.Tensor,
    lora_a_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    token_indices_sorted_by_lora_ids: torch.Tensor,
    num_tokens_per_lora: torch.Tensor,
    lora_token_start_loc: torch.Tensor,
    lora_ids: torch.Tensor,
    scaling: float,
) -> None:
    return


try:
    direct_register_custom_op(
        op_name="lora_shrink",
        op_func=_lora_shrink,
        mutates_args=["output_tensor"],
        fake_impl=lora_shrink_fake,
    )
    lora_shrink = torch.ops.vllm.lora_shrink

except AttributeError:
    lora_shrink = _lora_shrink
