from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple, Union, final

import torch

from vllm.lora.layers import LoRAMapping
from vllm.triton_utils import HAS_TRITON

if HAS_TRITON:
    from vllm.lora.ops.v1.lora_expand import lora_expand
    from vllm.lora.ops.v1.lora_shrink import lora_shrink
    from vllm.lora.ops.v1.lora_expand_slice import lora_expand_slice

from .punica_base import PunicaWrapperBase

if TYPE_CHECKING:
    # avoid circuit import
    from vllm.lora.models import LongContextLoRAContext


@dataclass
class V1KernelMeta:
    token_lora_mapping: torch.Tensor
    token_indices_sorted_by_lora_ids: torch.Tensor
    active_lora_ids: torch.Tensor
    num_tokens_per_lora: torch.Tensor
    lora_token_start_loc: torch.Tensor

    @staticmethod
    def make(max_loras: int, max_num_tokens: int,
             device: torch.device) -> "V1KernelMeta":

        token_lora_mapping = torch.empty(max_num_tokens,
                                         dtype=torch.int32,
                                         device=device)

        token_indices_sorted_by_lora_ids = torch.empty(max_num_tokens,
                                                       dtype=torch.int32,
                                                       device=device)

        # +1 because "no-lora" is also a possibility
        # example: let max_loras be 3, active_lora_ids of [-1, 0, 2, 1]
        # is a possibility.
        active_lora_ids = torch.empty(max_loras + 1,
                                      dtype=torch.int32,
                                      device=device)

        # using running example, [3, 10, 5, 2] is a possibility.
        num_tokens_per_lora = torch.zeros(max_loras + 1,
                                          dtype=torch.int32,
                                          device=device)

        # +2 for this because, the first index is always 0
        # for example: let max loras be 3, then lora_token_start_loc,
        # can be [0, 3, 13, 18, 20].
        lora_token_start_loc = torch.zeros(max_loras + 2,
                                           dtype=torch.int32,
                                           device=device)
        return V1KernelMeta(
            token_lora_mapping=token_lora_mapping,
            token_indices_sorted_by_lora_ids=token_indices_sorted_by_lora_ids,
            active_lora_ids=active_lora_ids,
            num_tokens_per_lora=num_tokens_per_lora,
            lora_token_start_loc=lora_token_start_loc)

    def reset(self):
        self.active_lora_ids.fill_(-1)
        self.num_tokens_per_lora.fill_(0)
        self.lora_token_start_loc.fill_(0)

    def prepare_tensors(self, token_lora_mapping: torch.Tensor) -> None:
        num_tokens = token_lora_mapping.size(0)

        # copy token lora mapping
        self.token_lora_mapping[:num_tokens].copy_(token_lora_mapping,
                                                   non_blocking=True)

        # token_indices_sorted_by_lora_ids
        _, token_indices_sorted_by_lora_ids = torch.sort(token_lora_mapping,
                                                         stable=True)
        # start gpu transfer
        self.token_indices_sorted_by_lora_ids[:num_tokens].copy_(
            token_indices_sorted_by_lora_ids, non_blocking=True)

        # active_lora_ids, num_tokens_per_lora
        lora_ids, num_tokens_per_lora = torch.unique(token_lora_mapping,
                                                     sorted=False,
                                                     return_counts=True)
        self.active_lora_ids[:lora_ids.size(0)].copy_(lora_ids,
                                                      non_blocking=True)
        self.num_tokens_per_lora[:num_tokens_per_lora.size(0)].copy_(
            num_tokens_per_lora, non_blocking=True)

        # lora_token_start_loc
        lora_token_start_loc = torch.cumsum(num_tokens_per_lora, dim=0)
        self.lora_token_start_loc[1:1 + lora_token_start_loc.size(0)].copy_(
            lora_token_start_loc, non_blocking=True)

    def meta_args(
        self, num_tokens: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor]:
        return (self.token_lora_mapping[:num_tokens],
                self.token_indices_sorted_by_lora_ids[:num_tokens],
                self.num_tokens_per_lora, self.lora_token_start_loc,
                self.active_lora_ids)


@final
class V1LoRAGPU(PunicaWrapperBase):
    """
    TODO (varun)
    _summary_

    Args:
        PunicaWrapperBase (_type_): _description_
    """

    def __init__(self, max_num_batched_tokens: int, max_batches: int,
                 device: Union[torch.device, str], **kwargs):
        PunicaWrapperBase.__init__(self, max_num_batched_tokens, max_batches,
                                   device)
        self.max_loras = kwargs['max_loras']
        self.token_mapping_v1_meta = V1KernelMeta.make(self.max_loras,
                                                       max_num_batched_tokens,
                                                       device=device)
        self.prompt_mapping_v1_meta = V1KernelMeta.make(self.max_loras,
                                                        max_batches,
                                                        device=device)

    def update_metadata(
            self,
            mapping: LoRAMapping,
            lora_index_to_id: List[Optional[int]],
            max_loras: int,
            vocab_size: int,
            extra_vocab_size: int,
            long_lora_context: Optional["LongContextLoRAContext"] = None,
            **kwargs):

        # TODO (varun) : Make no_lora case work with torch compile
        #self.no_lora = all([x == -1 for x in mapping.prompt_mapping])
        #if self.no_lora:
        #    # no update required
        #    return
        #num_tokens: int = len(mapping.index_mapping)

        self.token_mapping_v1_meta.reset()
        self.prompt_mapping_v1_meta.reset()

        self.update_base_metadata(mapping, lora_index_to_id, max_loras,
                                  vocab_size, extra_vocab_size,
                                  long_lora_context)

        self.token_mapping_v1_meta.prepare_tensors(self.token_lora_indices)
        self.prompt_mapping_v1_meta.prepare_tensors(self.sampler_indices)

    def _shrink(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        scale: float,
    ):
        lora_shrink(x, w_t_all, y,
                    *self.token_mapping_v1_meta.meta_args(x.size(0)), scale)

    def _expand(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        add_inputs: bool,
    ):
        lora_expand(x, w_t_all, y,
                    *self.token_mapping_v1_meta.meta_args(x.size(0)),
                    add_inputs)

    def _expand_slice(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        y_offset: Optional[int],
        y_slice_size: Optional[int],
        add_inputs: bool,
    ):
        #No LoRA request, so return directly
        if self.no_lora:
            return

        lora_expand_slice(x, w_t_all, y,
                          *self.token_mapping_v1_meta.meta_args(x.size(0)),
                          y_offset, y_slice_size, add_inputs)

    def _apply_expand(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        w_t_all: torch.Tensor,
        y_offset: Optional[int],
        y_slice_size: Optional[int],
        add_inputs: bool = True,
    ):
        """
        Perform the ` y[:,y_offset:y_offset+y_slice_size]+=x@w_t_all` 
        computation, which is suitable for the
        GEMM of lora'b.
        """
        self._expand_slice(y, x, w_t_all, y_offset, y_slice_size, add_inputs)

    def _apply_shrink(self, y: torch.Tensor, x: torch.Tensor,
                      w_t_all: torch.Tensor, scale: float):
        """
        Perform the ` y+=x@w_t_all` computation, which is suitable for the
        GEMM of lora'a.
        When `is_prefill is` true, it indicates that it is currently the
        prefill stage, and the `_shrink_prefill` function should be called.
        Otherwise, it is the decode stage, and the _shrink_decode function
        should be called.
        """
        y_org = y
        y = y.view(-1, y.size(-1))
        self._shrink(y, x, w_t_all, scale)
        y = y.view_as(y_org)

    def add_shrink(self, y: Union[Tuple[torch.Tensor, ...], torch.Tensor],
                   x: torch.Tensor, lora_a_stacked: Tuple[torch.Tensor, ...],
                   scale: float, **kwargs):
        """
        Performs GEMM  for multiple slices of lora_a.
        When `is_prefill is` true, it indicates that it is currently the
        prefill stage, and the `_shrink_prefill` function should be called.
        Otherwise, it is the decode stage, and the _shrink_decode function
        should be called.
            
        Semantics:
        for i in range(len(lora_a_stacked)):
            y[i] += (x @ lora_a_stacked[i]) * scale
        
        Args:
            y (Union[Tuple[torch.Tensor, ...], torch.Tensor]): Output tensors
            x (torch.Tensor): Input tensor
            lora_a_stacked (Tuple[torch.Tensor, ...]): lora_a's weights
            scale (float): Scaling factor for the operation
        """

        x = x.view(-1, x.size(-1))
        # TODO fuse these kernels
        for slice_idx in range(len(lora_a_stacked)):
            self._apply_shrink(y[slice_idx], x, lora_a_stacked[slice_idx],
                               scale)

    def add_expand(self,
                   y: torch.Tensor,
                   x: Union[Tuple[torch.Tensor, ...], torch.Tensor],
                   lora_b_stacked: Tuple[torch.Tensor, ...],
                   lora_bias_stacked: Optional[Tuple[torch.Tensor, ...]],
                   output_slices: Tuple[int, ...],
                   offset_start: int = 0,
                   add_inputs=True,
                   **kwargs) -> None:
        """
        Performs GEMM and bias addition for multiple slices of lora_b.
      
        Semantics:
            for i in range(len(lora_b_stacked)):
                slice = output_slices[i]
                y[:, offset:offset+slice] += x[i] @ lora_b_stacked[i] + 
                    lora_bias_stacked[i] 
                offset += slice
            
        Args:
            y (torch.Tensor): Output tensor.
            x (Union[Tuple[torch.Tensor, ...], torch.Tensor]): Input tensors
            lora_b_stacked (Tuple[torch.Tensor, ...]): lora_b's weight
            lora_bias_stacked (Optional[Tuple[torch.Tensor, ...]]): 
                bias's weight
            output_slices (Tuple[int, ...]): Every slice's size
            add_inputs (bool):  Defaults to True.
        """
        y_org = y
        y = y.view(-1, y.size(-1))
        offset_left = offset_start
        if lora_bias_stacked is not None:
            assert isinstance(x, torch.Tensor)
            self._apply_bias(self._token_lora_indices[:x.size(0)], y,
                             output_slices, lora_bias_stacked)
        for slice_idx in range(len(lora_b_stacked)):
            self._apply_expand(
                y,
                x[slice_idx],
                lora_b_stacked[slice_idx],
                offset_left,
                output_slices[slice_idx],
                add_inputs=add_inputs,
            )
            offset_left += output_slices[slice_idx]
        y = y.view_as(y_org)

    def add_lora_embedding(self,
                           y: torch.Tensor,
                           x: torch.Tensor,
                           lora_b_stacked: torch.Tensor,
                           add_inputs: bool = True,
                           **kwargs) -> None:
        """
        Applies lora  specifically for VocabParallelEmbeddingWithLoRA.

        Semantics:
            y += x @ lora_b_stacked

        Args:
            y (torch.Tensor): Output tensor.
            x (torch.Tensor): Input tensor.
            lora_b_stacked (torch.Tensor): lora_b's weights.
            add_inputs (bool): Default to True.
        """

        # Embedding layer only need expand op
        self._expand(y, x, lora_b_stacked, add_inputs)

    def add_lora_linear(self,
                        y: torch.Tensor,
                        x: torch.Tensor,
                        lora_a_stacked: Tuple[torch.Tensor, ...],
                        lora_b_stacked: Tuple[torch.Tensor, ...],
                        lora_bias_stacked: Optional[Tuple[torch.Tensor, ...]],
                        scale: float,
                        output_slices: Tuple[int, ...],
                        *,
                        buffer: Optional[Tuple[torch.Tensor, ...]] = None,
                        **kwargs) -> None:
        """
        Applicable to linear-related lora. 

        Semantics:
            for i in range(len(lora_a_stacked)):
                y[i] += (
                    x[i].unsqueeze(0)
                    @ lora_a_stacked[indices[i], layer_idx, :, :]
                    @ lora_b_stacked[indices[i], layer_idx, :, :]
                    * scale
                    ).squeeze(0)+lora_bias_stacked[i]

        Args:
            y (torch.Tensor): Output tensor. Will be changed in-place.
            x (torch.Tensor): Input tensor
            lora_a_stacked (Tuple[torch.Tensor, ...]): lora_a's weight.
            lora_b_stacked (Tuple[torch.Tensor, ...]): lora_b's weight.
            lora_bias_stacked (Optional[Tuple[torch.Tensor, ...]]): lora's bias.
            scale (float): Scaling factor.
            output_slices (Tuple[int, ...]): Every slice's size.
            buffer (Optional[Tuple[torch.Tensor, ...]]): Defaults to None.
        """

        assert len(lora_a_stacked) == len(lora_b_stacked) == len(output_slices)
        if lora_bias_stacked is not None:
            assert len(lora_bias_stacked) == len(output_slices)
            y = self._apply_bias(self._token_lora_indices[:x.size(0)], y,
                                 output_slices, lora_bias_stacked)

        if buffer is None:
            r = lora_b_stacked[0].size(-1)
            # We set the buffer to be float32 by default ,refer to:
            # https://github.com/triton-lang/triton/issues/1387
            buffer = tuple(
                torch.zeros(
                    (x.size(0), r), dtype=torch.float32, device=x.device)
                for _ in range(len(output_slices)))
        self.add_shrink(buffer, x, lora_a_stacked, scale, **kwargs)
        self.add_expand(y,
                        buffer,
                        lora_b_stacked,
                        None,
                        output_slices,
                        add_inputs=True,
                        **kwargs)

    def add_lora_logits(self,
                        y: torch.Tensor,
                        x: torch.Tensor,
                        lora_a_stacked: torch.Tensor,
                        lora_b_stacked: torch.Tensor,
                        scale,
                        *,
                        buffer: Optional[torch.Tensor] = None,
                        **kwargs) -> None:
        """
        Applies lora  specifically for LogitsProcessorWithLoRA.
        
        Semantics:
            buffer = (x @ lora_a_stacked) * scale
            y += buffer @ lora_b_stacked

        Args:
            y (torch.Tensor): Output tensor.
            x (torch.Tensor): Input tensor.
            lora_a_stacked (torch.Tensor): lora_a's weights.
            lora_b_stacked (torch.Tensor):lora_b's weights.
            scale (float): Scaling factor.
            buffer (Optional[torch.Tensor]):Default to None.
        """
        y_org = y
        y = y.view(-1, y.size(-1))
        x = x.view(-1, x.size(-1))
        r = lora_b_stacked.size(-1)
        if buffer is None:
            # We set the buffer to be float32 by default ,refer to:
            # https://github.com/triton-lang/triton/issues/1387
            buffer = torch.zeros((x.size(0), r),
                                 dtype=torch.float32,
                                 device=x.device)

        lora_shrink(x, lora_a_stacked, buffer,
                    *self.prompt_mapping_v1_meta.meta_args(x.size(0)), scale)
        lora_expand(buffer,
                    lora_b_stacked,
                    y,
                    *self.prompt_mapping_v1_meta.meta_args(x.size(0)),
                    add_inputs=True)

        y = y.view_as(y_org)
