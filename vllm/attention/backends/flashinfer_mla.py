from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Type
import math
from functools import cached_property

from vllm.multimodal import MultiModalPlaceholderMap

try:
    from flashinfer import BatchDecodeMlaWithPagedKVCacheWrapper
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    FLASHINFER_WORKSPACE_BUFFER_SIZE = 256 * 1024 * 1024
except ImportError:
    BatchDecodeMlaWithPagedKVCacheWrapper = None
    FLASHINFER_WORKSPACE_BUFFER_SIZE = 0

import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata,
                                              AttentionMetadataBuilder,
                                              AttentionState, AttentionType)
from vllm.attention.backends.utils import (PAD_SLOT_ID, compute_slot_mapping,
                                           compute_slot_mapping_start_idx,
                                           is_block_tables_empty)
from vllm.attention.ops.paged_attn import PagedAttention
from vllm.utils import (async_tensor_h2d, get_kv_cache_torch_dtype,
                        make_tensor_with_pad)

if TYPE_CHECKING:
    from vllm.worker.model_runner import (ModelInputForGPUBuilder,
                                          ModelInputForGPUWithSamplingMetadata)


class FlashInferMLABackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER_MLA"

    @staticmethod
    def get_impl_cls() -> Type["FlashInferMLAImpl"]:
        return FlashInferMLAImpl

    @staticmethod
    def get_metadata_cls() -> Type["AttentionMetadata"]:
        return FlashInferMLAMetadata

    @staticmethod
    def get_builder_cls() -> Type["FlashInferMLAMetadataBuilder"]:
        return FlashInferMLAMetadataBuilder

    @staticmethod
    def get_state_cls() -> Type["FlashInferMLAState"]:
        return FlashInferMLAState

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        # NOTE(simon): we repurpose the "key" cache for latent,
        # and "value" cache for rope. Until we have hybrid memory
        # allocate, we are living with some memory waste.
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        PagedAttention.swap_blocks(src_kv_cache, dst_kv_cache, src_to_dst)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        PagedAttention.copy_blocks(kv_caches, src_to_dists)

    @staticmethod
    def get_supported_head_sizes() -> List[int]:
        return [512]

    @staticmethod
    def get_fp8_dtype_for_flashinfer(kv_cache_dtype: str) -> torch.dtype:
        if kv_cache_dtype in ("fp8", "fp8_e4m3"):
            return torch.float8_e4m3fn
        elif kv_cache_dtype == "fp8_e5m2":
            return torch.float8_e5m2
        else:
            raise ValueError(f"Unrecognized FP8 dtype: {kv_cache_dtype}")


class FlashInferMLAState(AttentionState):

    def __init__(self, runner):
        self.runner = runner

    @cached_property
    def _workspace_buffer(self):
        return torch.empty(
                FLASHINFER_WORKSPACE_BUFFER_SIZE,
                dtype=torch.uint8,
                device=self.runner.device)

    @cached_property
    def _decode_wrapper(self):
        return BatchDecodeMlaWithPagedKVCacheWrapper(
            self._workspace_buffer)

    @contextmanager
    def graph_capture(self, max_batch_size: int):
        raise NotImplementedError("FlashInferMLAState does not support graph capture")

    def graph_clone(self, batch_size: int):
        raise NotImplementedError("FlashInferMLAState does not support graph capture")

    def graph_capture_get_metadata_for_batch(
            self, batch_size: int, is_encoder_decoder_model: bool = False):
        raise NotImplementedError("FlashInferMLAState does not support graph capture")

    def get_graph_input_buffers(self,
                                attn_metadata,
                                is_encoder_decoder_model: bool = False):
        raise NotImplementedError("FlashInferMLAState does not support graph capture")

    def prepare_graph_input_buffers(self,
                                    input_buffers,
                                    attn_metadata,
                                    is_encoder_decoder_model: bool = False):
        raise NotImplementedError("FlashInferMLAState does not support graph capture")

    def begin_forward(self, model_input):
        model_input.attn_metadata.decode_wrapper = self._decode_wrapper
        model_input.attn_metadata.begin_forward()


@dataclass
class FlashInferMLAMetadata(AttentionMetadata):
    # Maximum sequence length among prefill batch. 0 if there are decoding
    # requests only.
    max_prefill_seq_len: int

    # Number of query tokens for each request in the batch.
    # Currently, we require that all requests have the same number of query
    # tokens during the decoding phase. When speculavie decoding is enabled,
    # decode_query_len might be greater than 1. In all other cases, it is 1.
    decode_query_len: Optional[int] = 1

    use_cuda_graph: bool = True

    decode_wrapper: Optional[BatchDecodeMlaWithPagedKVCacheWrapper] = None

    # Metadata for the prefill stage
    seq_start_loc: Optional[torch.Tensor] = None
    query_start_loc: Optional[torch.Tensor] = None
    block_tables: Optional[torch.Tensor] = None

    # used for GPU in-place advance_step
    seq_lens_tensor: Optional[torch.Tensor] = None
    block_table_bound: Optional[torch.Tensor] = None

    # An example for paged_kv_indices, paged_kv_indptr:
    # request 1, page indices [0, 5, 8]
    # request 2, page indices [1, 6, 7]
    # request 3, page indices [3, 4]
    # paged_kv_indices is a concatenation of page indices of all requests:
    # [0, 5, 8, 1, 6, 7, 3, 4]
    # paged_kv_indptr is used to index into paged_kv_indices:
    # [0, 3, 6, 8]
    # The indptr of the paged kv cache, shape: [batch_size + 1]
    paged_kv_indptr: Optional[torch.Tensor] = None
    # The page indices of the paged kv cache
    paged_kv_indices: Optional[torch.Tensor] = None
    # The number of entries in the last page of each request in
    # the paged kv cache, shape: [batch_size]
    paged_kv_last_page_len: Optional[torch.Tensor] = None
    # The number of query/output heads
    num_qo_heads: Optional[int] = None
    # The number of key/value heads
    num_kv_heads: Optional[int] = None
    # The dimension of the attention heads
    head_dim: Optional[int] = None
    # Block size of vllm
    page_size: Optional[int] = None
    # The data type of the paged kv cache
    data_type: torch.dtype = None
    # The data type of the query
    q_data_type: torch.dtype = None
    device: torch.device = torch.device("cuda")
    is_profile_run: bool = False

    def __post_init__(self):
        supported_head_sizes = FlashInferMLABackend.get_supported_head_sizes()
        if self.head_dim is not None and self.head_dim \
                not in supported_head_sizes:
            raise ValueError(
                f"Only {supported_head_sizes} are supported for head_dim,",
                f"received {self.head_dim}.")

    def begin_forward(self):
        if self.num_prefill_tokens > 0:
            # assert NotImplementedError("FlashInferMLAState does not support prefill")
            # NOTE: only in profiling, treating it as noop
            return

            if self.paged_kv_indices is None:
                return

            assert self.prefill_wrapper is not None
            assert self.query_start_loc is not None
            assert self.paged_kv_indices is not None
            assert self.paged_kv_indptr is not None
            assert self.paged_kv_last_page_len is not None
            assert self.block_table_bound is not None
            assert self.seq_lens_tensor is not None
            self.query_start_loc = self.query_start_loc[:self.num_prefills + 1]
            batch_size = self.query_start_loc.shape[0] - 1
            assert batch_size >= 0
            # We will use flash attention for profiling to
            # determine the number of blocks. Therefore,
            # we don't need to prepare the input for flashinfer for profile run.
            if not self.is_profile_run:
                self.paged_kv_indptr = self.paged_kv_indptr.to(self.device)
                self.paged_kv_last_page_len = self.paged_kv_last_page_len.to(
                    self.device)
                self.block_table_bound = self.block_table_bound.to(self.device)
                self.seq_lens_tensor = self.seq_lens_tensor.to(self.device)
                self.paged_kv_indices = self.paged_kv_indices.to(self.device)
                self.prefill_wrapper.plan(
                    self.paged_kv_indptr[:self.num_prefills + 1],
                    self.paged_kv_indices,
                    self.paged_kv_last_page_len[:self.num_prefills],
                    self.num_qo_heads,
                    self.head_dim,
                    self.page_size,
                    sm_scale=1.0 / math.sqrt(self.head_dim + self.head_dim//8),  # TODO(simon): should we explicitly pass this in?
                    data_type=self.data_type,
                    q_data_type=self.q_data_type)

        if self.num_decode_tokens > 0:
            assert self.paged_kv_indices is not None
            assert self.paged_kv_indptr is not None
            assert self.paged_kv_last_page_len is not None
            self.paged_kv_indices = self.paged_kv_indices.to(self.device)
            self.paged_kv_indptr = self.paged_kv_indptr.to(self.device)
            self.paged_kv_last_page_len = self.paged_kv_last_page_len.to(
                self.device)
            # handle model warmup path
            if self.block_table_bound is not None:
                self.block_table_bound = self.block_table_bound.to(self.device)
            if self.seq_lens_tensor is not None:
                self.seq_lens_tensor = self.seq_lens_tensor.to(self.device)

            assert self.decode_wrapper is not None
            # self.decode_wrapper.end_forward()

            self.decode_wrapper.plan(
                self.paged_kv_indptr[self.num_prefills:],
                self.paged_kv_indices,
                self.paged_kv_last_page_len[self.num_prefills:],
                self.num_qo_heads,
                # self.num_kv_heads,
                self.head_dim,
                self.page_size,
                sm_scale=1.0 / math.sqrt(self.head_dim + self.head_dim//8),  # TODO(simon): should we explicitly pass this in?
                # Disable flashinfer's pos encoding and use vllm's rope.
                # pos_encoding_mode="NONE",
                # kv-cache data type.
                data_type=self.data_type,
                # query data type.
                q_data_type=self.q_data_type)

    def asdict_zerocopy(self,
                        skip_fields: Optional[Set[str]] = None
                        ) -> Dict[str, Any]:
        if skip_fields is None:
            skip_fields = set()
        # We need to skip the prefill/decode_wrapper field since it cannot be
        # broadcasted with nccl when TP is enabled.
        skip_fields.add('decode_wrapper')
        return super().asdict_zerocopy(skip_fields)

    @property
    def prefill_metadata(self) -> Optional["FlashInferMLAMetadata"]:
        if self.num_prefills == 0:
            return None
        return self

    @property
    def decode_metadata(self) -> Optional["FlashInferMLAMetadata"]:
        if self.num_decode_tokens == 0:
            return None
        return self

    def advance_step(self,
                     model_input: "ModelInputForGPUWithSamplingMetadata",
                     sampled_token_ids: Optional[torch.Tensor],
                     block_size: int,
                     num_seqs: int,
                     num_queries: int,
                     turn_prefills_into_decodes: bool = False):
        """
        Update metadata in-place to advance one decode step.
        """
        raise NotImplementedError("FlashInferMLAMetadata does not support multi-step")


class FlashInferMLAMetadataBuilder(AttentionMetadataBuilder[FlashInferMLAMetadata]):

    def __init__(self, input_builder: "ModelInputForGPUBuilder"):
        self.slot_mapping: List[int] = []
        self.prefill_seq_lens: List[int] = []
        self.context_lens: List[int] = []
        self.block_tables: List[List[int]] = []
        self.curr_seq_lens: List[int] = []
        self.multimodal_placeholder_maps: Dict[
            str,
            MultiModalPlaceholderMap] = defaultdict(MultiModalPlaceholderMap)
        self.num_prefills = 0
        self.num_prefill_tokens = 0
        self.num_decode_tokens = 0

        self.input_builder = input_builder
        self.runner = input_builder.runner

        self.sliding_window = input_builder.sliding_window
        self.block_size = input_builder.block_size

        # Please follow https://docs.flashinfer.ai/tutorials/kv_layout.html#page-layout
        # for the precise definition of the following fields.
        # An example:
        # request 1, page indices [0, 5, 8]
        # request 2, page indices [1, 6, 7]
        # request 3, page indices [3, 4]
        # paged_kv_indices is a concatenation of page indices of all requests:
        # [0, 5, 8, 1, 6, 7, 3, 4]
        # paged_kv_indptr is used to index into paged_kv_indices:
        # [0, 3, 6, 8]
        self.paged_kv_indices: List[int] = []
        # 0 at the beginning of paged_kv_indptr indicates the start of the
        # first request’s page indices in the paged_kv_indices list.
        self.paged_kv_indptr: List[int] = [0]
        # paged_kv_last_page_len is the length of the last page of each request
        self.paged_kv_last_page_len: List[int] = []
        self.total_blocks = 0
        self.is_profile_run: bool = False

    def _add_seq_group(
            self, inter_data: "ModelInputForGPUBuilder.InterDataForSeqGroup",
            chunked_prefill_enabled: bool):
        """Add a sequence group to the metadata. Specifically update/append
        1. context length.
        2. block table.
        3. slot mapping.
        """
        is_prompt = inter_data.is_prompt
        block_tables = inter_data.block_tables
        computed_block_nums = inter_data.computed_block_nums

        for (seq_id, token_len, seq_len, curr_seq_len, query_len, context_len,
             curr_sliding_window_block) in zip(
                 inter_data.seq_ids, [len(t) for t in inter_data.input_tokens],
                 inter_data.orig_seq_lens, inter_data.seq_lens,
                 inter_data.query_lens, inter_data.context_lens,
                 inter_data.curr_sliding_window_blocks):
            self.context_lens.append(context_len)
            if is_prompt:
                mm_maps = inter_data.multi_modal_placeholder_maps
                if mm_maps:
                    for modality, placeholders in mm_maps.items():
                        self.multimodal_placeholder_maps[modality].extend(
                            placeholders)
                self.num_prefills += 1
                self.num_prefill_tokens += token_len
                self.prefill_seq_lens.append(seq_len)
            else:
                assert query_len == 1, (
                    "seq_len: {}, context_len: {}, query_len: {}".format(
                        seq_len, context_len, query_len))
                self.num_decode_tokens += query_len
                self.curr_seq_lens.append(curr_seq_len)

            # Compute block table.
            # TODO(sang): Combine chunked prefill and prefix caching by
            # only allowing multiple of block_size chunk size.
            # NOTE: This only works for oooooooxxx style attention.
            block_table = []
            if inter_data.prefix_cache_hit:
                block_table = computed_block_nums
            elif ((chunked_prefill_enabled or not is_prompt)
                  and block_tables is not None):
                block_table = block_tables[seq_id][-curr_sliding_window_block:]
            self.block_tables.append(block_table)

            is_profile_run = is_block_tables_empty(block_tables)

            # Compute slot mapping.
            start_idx = compute_slot_mapping_start_idx(is_prompt, query_len,
                                                       context_len,
                                                       self.sliding_window)
            compute_slot_mapping(is_profile_run, self.slot_mapping, seq_id,
                                 seq_len, context_len, start_idx,
                                 self.block_size, inter_data.block_tables)

            # It is not necessary to add paged_kv_indices, paged_kv_indptr,
            # and paged_kv_last_page_len for profile run because we will
            # create dummy inputs.
            if is_profile_run:
                self.is_profile_run = is_profile_run
                return

            block_table = block_tables[seq_id]
            self._update_paged_kv_tensors(block_table, seq_len)

    def _update_paged_kv_tensors(self, block_table: List[int], seq_len: int):
        # Get the number of valid blocks based on sequence length.
        # If seq_len = 16, block_size = 16,
        # block_table_bound is 1 with 1 valid block.
        # If seq_len = 15, block_size = 16,
        # block_table_bound is 0 + 1 with 1 valid block.
        self.total_blocks += len(block_table)
        block_table_bound = seq_len // self.block_size + 1 \
                            if seq_len % self.block_size != 0 \
                            else seq_len // self.block_size
        self.paged_kv_indices.extend(block_table[:block_table_bound])
        self.paged_kv_indptr.append(self.paged_kv_indptr[-1] +
                                    block_table_bound)

        last_page_len = seq_len % self.block_size
        if last_page_len == 0:
            last_page_len = self.block_size
        self.paged_kv_last_page_len.append(last_page_len)

    def build(self, seq_lens: List[int], query_lens: List[int],
              cuda_graph_pad_size: int, batch_size: int):
        """Build attention metadata with on-device tensors.

        Args:
            seq_lens: The maybe padded sequence lengths of the input sequences.
            query_lens: The query lengths of the input sequences.
            cuda_graph_pad_size: The padding size for cuda graph.
                                 -1 if cuda graph is not used.
            batch_size: The maybe padded batch size.
        """
        for inter_data in self.input_builder.inter_data_list:
            self._add_seq_group(inter_data,
                                self.input_builder.chunked_prefill_enabled)

        device = self.runner.device
        use_captured_graph = cuda_graph_pad_size != -1

        max_prefill_seq_len = max(self.prefill_seq_lens, default=0)
        num_decode_tokens = self.num_decode_tokens
        decode_query_len = max(query_lens[self.num_prefills:], default=1)

        if use_captured_graph:
            self.slot_mapping.extend([PAD_SLOT_ID] * cuda_graph_pad_size)
            self.block_tables.extend([] * cuda_graph_pad_size)
            num_decode_tokens = batch_size - self.num_prefill_tokens

            # The shape of graph_block_tables is
            # [max batch size, max context len // block size].
            input_block_tables = self.runner.graph_block_tables[:batch_size]
            max_blocks = input_block_tables.shape[1]
            for i, block_table in enumerate(self.block_tables):
                if block_table:
                    num_blocks = len(block_table)
                    if num_blocks <= max_blocks:
                        input_block_tables[i, :num_blocks] = block_table
                    else:
                        # It may be possible to have more blocks allocated due
                        # to lookahead slots of multi-step, however, they are
                        # not used anyway, so can be safely ignored.
                        input_block_tables[
                            i, :max_blocks] = block_table[:max_blocks]

            block_tables = torch.from_numpy(input_block_tables).to(
                device, non_blocking=True)

            last_paged_kv_indptr = self.paged_kv_indptr[-1]
            self.paged_kv_indptr.extend([last_paged_kv_indptr] *
                                        cuda_graph_pad_size)
            self.paged_kv_last_page_len.extend([0] * cuda_graph_pad_size)
        else:
            block_tables = make_tensor_with_pad(
                self.block_tables,
                pad=0,
                dtype=torch.int,
                device=device,
            )

        assert device is not None
        seq_lens_tensor = async_tensor_h2d(seq_lens, torch.int, device,
                                           self.runner.pin_memory)
        query_lens_tensor = async_tensor_h2d(query_lens, torch.long, device,
                                             self.runner.pin_memory)
        slot_mapping_tensor = async_tensor_h2d(self.slot_mapping, torch.long,
                                               device, self.runner.pin_memory)
        query_start_loc = torch.zeros(query_lens_tensor.shape[0] + 1,
                                      dtype=torch.int32,
                                      device=device)
        seq_start_loc = torch.zeros(seq_lens_tensor.shape[0] + 1,
                                    dtype=torch.int32,
                                    device=device)
        placeholder_index_maps = {
            modality: placeholder_map.index_map()
            for modality, placeholder_map in
            self.multimodal_placeholder_maps.items()
        }
        torch.cumsum(seq_lens_tensor,
                     dim=0,
                     dtype=seq_start_loc.dtype,
                     out=seq_start_loc[1:])
        torch.cumsum(query_lens_tensor,
                     dim=0,
                     dtype=query_start_loc.dtype,
                     out=query_start_loc[1:])

        if len(self.paged_kv_indptr) > 0:
            # extend to the maximum number of blocks as returned by the
            # scheduler
            self.paged_kv_indices.extend(
                [0] * (self.total_blocks - len(self.paged_kv_indices)))
            paged_kv_indices_tensor = torch.tensor(self.paged_kv_indices,
                                                   device="cpu",
                                                   dtype=torch.int)
            paged_kv_indptr_tensor = torch.tensor(self.paged_kv_indptr,
                                                  device="cpu",
                                                  dtype=torch.int)
            paged_kv_last_page_len_tensor = torch.tensor(
                self.paged_kv_last_page_len, device="cpu", dtype=torch.int)
            block_table_bound_tensor = torch.zeros(len(self.paged_kv_indptr) -
                                                   1,
                                                   device="cpu",
                                                   dtype=torch.int)
        else:
            paged_kv_indices_tensor = None
            paged_kv_indptr_tensor = None
            paged_kv_last_page_len_tensor = None
            block_table_bound_tensor = None

        if self.runner.kv_cache_dtype.startswith("fp8"):
            kv_cache_dtype = FlashInferMLABackend.get_fp8_dtype_for_flashinfer(
                self.runner.kv_cache_dtype)
        else:
            kv_cache_dtype = get_kv_cache_torch_dtype(
                self.runner.kv_cache_dtype, self.runner.model_config.dtype)

        return FlashInferMLAMetadata(
            decode_query_len=decode_query_len,
            num_prefills=self.num_prefills,
            slot_mapping=slot_mapping_tensor,
            multi_modal_placeholder_index_maps=placeholder_index_maps,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            max_prefill_seq_len=max_prefill_seq_len,
            block_tables=block_tables,
            paged_kv_indptr=paged_kv_indptr_tensor,
            paged_kv_indices=paged_kv_indices_tensor,
            paged_kv_last_page_len=paged_kv_last_page_len_tensor,
            block_table_bound=block_table_bound_tensor,
            seq_lens_tensor=seq_lens_tensor,
            num_qo_heads=self.runner.model_config.get_num_attention_heads(
                self.runner.parallel_config),
            num_kv_heads=self.runner.model_config.get_num_kv_heads(
                self.runner.parallel_config),
            head_dim=self.runner.model_config.get_head_size(),
            page_size=self.block_size,
            seq_start_loc=seq_start_loc,
            query_start_loc=query_start_loc,
            device=device,
            data_type=kv_cache_dtype,
            q_data_type=self.runner.model_config.dtype,
            use_cuda_graph=use_captured_graph,
            is_profile_run=self.is_profile_run)


class FlashInferMLAImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype

        unsupported_features = [alibi_slopes, sliding_window, blocksparse_params, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(f"FlashInferMLAImpl does not support {unsupported_features}")

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashInferMLAMetadata,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        attn_type: str = AttentionType.DECODER,
    ) -> torch.Tensor:
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "FlashInferMLAImpl")
        key_rope = value
        del value

        num_tokens, N, LR = query.shape
        assert N == self.num_heads
        assert LR == self.head_size + self.head_size//8
        qk_rope_head_dim = LR - self.head_size
        assert qk_rope_head_dim == 64
        # hidden_size = N * self.head_size

        num_heads: int = self.num_heads
        head_size: int = self.head_size
        num_kv_heads: int = self.num_kv_heads
        assert self.num_kv_heads == 1
        kv_cache_dtype: str = self.kv_cache_dtype
        # softmax_scale: float = self.scale
        # window_size = self.sliding_window
        # alibi_slopes = self.alibi_slopes
        # logits_soft_cap = self.logits_soft_cap

        # num_tokens, hidden_size = query.shape
        query = query.view(-1, num_heads, LR)
        key = key.view(-1, num_kv_heads, head_size)
        key_rope = key_rope.view(-1, num_kv_heads,
                                 head_size)  # this is padded!

        if kv_cache.numel() > 0:
            # Use the same reshape and cache kernel as flash attention.
            ops.reshape_and_cache_flash(
                key,
                key_rope,
                kv_cache[:, 0],
                kv_cache[:, 1],
                attn_metadata.slot_mapping.flatten(),
                kv_cache_dtype,
                k_scale,
                v_scale,
            )
            # The FlashInfer api requires data to be in fp8_e4m3 or fp8_e5m2
            # to process the cache when the kv_cache_dtype is fp8
            if kv_cache_dtype.startswith("fp8"):
                torch_dtype = FlashInferMLABackend.get_fp8_dtype_for_flashinfer(
                    kv_cache_dtype)
                kv_cache = kv_cache.view(torch_dtype)

        num_prefill_tokens = attn_metadata.num_prefill_tokens
        num_decode_tokens = attn_metadata.num_decode_tokens
        assert num_prefill_tokens == 0 and num_decode_tokens > 0, "only mla decode"

        assert key.shape[0] == num_prefill_tokens + num_decode_tokens, \
                    f"key : {key.shape} : #prefill tokens {num_prefill_tokens} : #decode tokens {num_decode_tokens}" # noqa
        assert key_rope.shape[0] == num_prefill_tokens + num_decode_tokens, \
                    f"value : {key_rope.shape} : #prefill toks {num_prefill_tokens} : #decode toks {num_decode_tokens}" # noqa

        query = query.contiguous()  # Flashinfer requires query to be contiguous
        # Query for decode. KV is not needed because it is already cached.
        # QKV for prefill.
        # query = query[:num_prefill_tokens]
        decode_query = query#[num_prefill_tokens:]
        # assert query.shape[0] == num_prefill_tokens
        assert decode_query.shape[0] == num_decode_tokens, f"{decode_query.shape=}, {num_decode_tokens=}"

        # query_nope = query[:, :, :head_size].contiguous()
        # query_pe = query[:, :, head_size:].contiguous()
        decode_query_nope = decode_query[:, :, :head_size].contiguous()
        decode_query_pe = decode_query[:, :, head_size:].contiguous()

        # window_left = window_size[0] if window_size is not None else -1

        prefill_output: Optional[torch.Tensor] = None
        decode_output: Optional[torch.Tensor] = None
        if prefill_meta := attn_metadata.prefill_metadata:
            # only in prefill, use run FA
            prefill_output = flash_attn_varlen_func(
                    q=query,
                    k=key,
                    v=value,
                    cu_seqlens_q=prefill_meta.seq_start_loc,
                    cu_seqlens_k=prefill_meta.seq_start_loc,
                    max_seqlen_q=prefill_meta.max_prefill_seq_len,
                    max_seqlen_k=prefill_meta.max_prefill_seq_len,
                    causal=True,
            )
        if decode_meta := attn_metadata.decode_metadata:
            assert decode_meta is not None
            assert decode_meta.decode_wrapper is not None
            # paged_kpe_cache, _ = kv_cache[:, 1].split(
            #     [qk_rope_head_dim, head_size - qk_rope_head_dim], dim=-1)
            # paged_kpe_cache = paged_kpe_cache.contiguous() # this is making of entire KV cache noooo
            # # note: this shouldn't matter b/c FI assumes head_dim_kpe == head_dim_ckv//8

            decode_output = decode_meta.decode_wrapper.run(
                q_nope=decode_query_nope,
                q_pe=decode_query_pe,
                paged_ckv_cache=kv_cache[:, 0],
                paged_kpe_cache=kv_cache[:, 1],
                # paged_kpe_cache=paged_kpe_cache,
                # sm_scale=softmax_scale,
                # logits_soft_cap=logits_soft_cap,
                # k_scale=k_scale,
                # v_scale=v_scale,
                # window_left=window_left
            )

        if prefill_output is None and decode_output is not None:
            # Decode only batch.
            output, num_tokens = decode_output, num_decode_tokens
        elif decode_output is None and prefill_output is not None:
            # Prefill only batch.
            output, num_tokens = prefill_output, num_prefill_tokens
        else:
            # Chunked prefill batch does not work with speculative decoding in
            # FlashInfer backend, so the query length for decode should be 1.
            assert prefill_output is not None
            assert decode_output is not None
            assert decode_meta is not None
            assert decode_meta.decode_query_len == 1
            decode_output = decode_output.squeeze(1)
            output = torch.cat([prefill_output, decode_output], dim=0)
        assert output.shape == (
            num_tokens, N,
            head_size), f"{output.shape=}!={num_tokens=}, {N=}, {head_size=}"
        return output
