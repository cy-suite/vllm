import torch
import triton
import triton.language as tl
import math
from vllm.utils import direct_register_custom_op

@triton.jit
def _lora_expand_kernel(
    input_ptr,
    lora_ptr,
    out_ptr,
    M,
    N,
    K,
    token_indices_sorted_by_lora_ids,
    num_tokens_per_lora,
    lora_token_start_loc,
    lora_ids,
    xm_stride,
    xk_stride,  # 1
    l0_stride, # hidden_size * max rank
    lora_n_stride,
    lora_k_stride,
    cm_stride,
    cn_stride,
    EVEN_K: tl.constexpr,
    ADD_INPUTS: tl.constexpr,
    CAST_TYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):

    NUM_M_CTAS = tl.cdiv(M, BLOCK_M) 
    NUM_N_CTAS = tl.cdiv(N, BLOCK_N)

    pid = tl.program_id(0)
    l = pid // (NUM_M_CTAS * NUM_N_CTAS)
    cta_n = (pid // NUM_M_CTAS) % NUM_N_CTAS
    cta_m = pid % NUM_M_CTAS

    lora_id = tl.load(lora_ids + l)
    if lora_id == -1:
        # early exit for the no-lora case.
        return

    # lora m indices offsets
    lora_m_indices_start = tl.load(lora_token_start_loc + l)
    lora_m_size = tl.load(num_tokens_per_lora + l) 

    cta_m_offset = cta_m * BLOCK_M 
    if cta_m_offset >= lora_m_size:
        # early exit CTA
        return

    cta_lora_seq_indices = token_indices_sorted_by_lora_ids + lora_m_indices_start + cta_m_offset
    cta_m_size = min(BLOCK_M, lora_m_size - cta_m_offset)

    offset_k = tl.arange(0, BLOCK_K)

    offset_rm = tl.arange(0, BLOCK_M) % cta_m_size 
    rm = tl.load(cta_lora_seq_indices + offset_rm)
    a_ptr = input_ptr + rm[:, None] * xm_stride + offset_k[None, :] * xk_stride

    offset_n = tl.arange(0, BLOCK_N) + cta_n * BLOCK_N
    rbn = tl.max_contiguous(tl.multiple_of(offset_n % N, BLOCK_N), BLOCK_N)
    b_ptr = (lora_ptr + l0_stride * lora_id +
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


    tiled_c = accumulator.to(lora_ptr.dtype.element_ty)
    offset_cm = tl.arange(0, BLOCK_M)
    offset_cn = tl.arange(0, BLOCK_N) + cta_n * BLOCK_N
    c_ptr = out_ptr + rm[:, None] * cm_stride + offset_n[None, :] * cn_stride

    c_mask = (offset_cm[:, None] < cta_m_size) & (offset_cn[None, :] < N)
    if ADD_INPUTS:
        tiled_out = tl.load(c_ptr, mask=c_mask)
        tiled_c += tiled_out
    tl.store(c_ptr, tiled_c, mask=c_mask)

@torch.inference_mode()
def _lora_expand(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    token_indices_sorted_by_lora_ids: torch.Tensor, # inputs.size(0)
    num_tokens_per_lora: torch.Tensor, # max-loras + 1
    lora_token_start_loc: torch.Tensor, # max-loras + 2
    lora_ids: torch.Tensor, # max-loras + 1
    add_inputs: bool = False,
) -> None:
    """
    Args:
        inputs (torch.Tensor): input tensor
        lora_b_weights (torch.Tensor): lora'b weight
        output_tensor (torch.Tensor): output tensor
        token_indices_sorted_by_lora_ids: Row/Token indices from the A matrix grouped by LoRA IDs.
        num_tokens_per_lora: num_tokens_per_lora[i] is the number of tokens that are to be
            processed by LoRA ID lora_ids[i] 
        lora_token_start_loc: A cumulative sum of num_tokens_per_lora. lora_token_start_loc[0] 
            is always 0 so that lora_token_start_loc[i], along with num_tokens_per_lora[i]
            identifies the the region in token_indices_sorted_by_lora_ids that LoRA lora_ids[i]
            should process.
        lora_ids: LoRA ids to process.
        add_inputs (bool, optional): Defaults to False, adds the final lora 
            results to the output.
    """

    assert inputs.dtype in [torch.float16, torch.bfloat16, torch.float32]
    assert lora_b_weights.dtype in [
        torch.float16,
        torch.bfloat16,
    ]
    assert inputs.size(1) == lora_b_weights.size(-1)
    assert inputs.is_contiguous()
    assert output_tensor.is_contiguous()
    assert token_indices_sorted_by_lora_ids.size(0) == inputs.size(0)
    assert num_tokens_per_lora.size(0) == lora_ids.size(0)
    assert lora_token_start_loc.size(0) == lora_ids.size(0) + 1

    if lora_b_weights.ndim == 4:  # shape:(lora_num,1,size,rank)
        assert lora_b_weights.size(1) == 1
        lora_b_weights = lora_b_weights.squeeze(dim=1)
    else:
        assert lora_b_weights.ndim == 3  # shape:(lora_num,size,rank)

    assert lora_b_weights.is_contiguous()

    # TODO tuning this config

    M = inputs.size(0)
    N = lora_b_weights.size(-2)
    K = lora_b_weights.size(-1)
    BLOCK_M = 16
    BLOCK_N = 64
    BLOCK_K = 16

    NUM_M_CTAS = math.ceil(M / BLOCK_M)  # Each BLOCK_M is its own CTA
    NUM_N_CTAS = math.ceil(N / BLOCK_N)
    MAX_LORAS = lora_ids.size(0)  

    EVEN_K = K % BLOCK_K == 0
    ADD_INPUTS = add_inputs
    CAST_TYPE = False
    if inputs.dtype == torch.float32 and lora_b_weights.dtype in [
            torch.float16,
            torch.bfloat16,
    ]:
        CAST_TYPE = True

    xm_stride = inputs.stride(0)
    xk_stride = inputs.stride(1)
    l0_stride = lora_b_weights.stride(0)
    lora_n_stride = lora_b_weights.stride(1)
    lora_k_stride = lora_b_weights.stride(2)
    cm_stride =  output_tensor.stride(0)
    cn_stride = output_tensor.stride(1)

    grid = (
        MAX_LORAS * NUM_M_CTAS * NUM_N_CTAS,
    )

    _lora_expand_kernel[grid](
        inputs,
        lora_b_weights,
        output_tensor,
        M,
        N,
        K,
        token_indices_sorted_by_lora_ids,
        num_tokens_per_lora,
        lora_token_start_loc,
        lora_ids,
        xm_stride,
        xk_stride,
        l0_stride,
        lora_n_stride,
        lora_k_stride,
        cm_stride,
        cn_stride,
        EVEN_K,
        ADD_INPUTS,
        CAST_TYPE,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
    )
    return

def lora_expand_fake(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    token_indices_sorted_by_lora_ids: torch.Tensor,
    num_tokens_per_lora: torch.Tensor,
    lora_token_start_loc: torch.Tensor,
    lora_ids: torch.Tensor,
    add_inputs: bool = False,
) -> None:
    return

try:
    direct_register_custom_op(
        op_name="lora_expand",
        op_func=_lora_expand,
        mutates_args=["output_tensor"],
        fake_impl=lora_expand_fake,
    )
    lora_expand = torch.ops.vllm.lora_expand

except AttributeError:
    lora_expand = _lora_expand