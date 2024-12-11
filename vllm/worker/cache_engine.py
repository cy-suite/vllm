"""CacheEngine class for managing the KV cache."""
from collections import deque
from typing import Any, Dict, List

import torch

from vllm.attention import get_attn_backend
from vllm.config import CacheConfig, DeviceConfig, ModelConfig, ParallelConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, get_dtype_size,
                        is_pin_memory_available)

logger = init_logger(__name__)


class CacheEngine:
    """Manages the KV cache.

    This class is responsible for initializing and managing the GPU and CPU KV
    caches. It also provides methods for performing KV cache operations, such
    as swapping and copying.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        device_config: DeviceConfig,
    ) -> None:
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.device_config = device_config

        self.head_size = model_config.get_head_size()
        # Models like Jamba, have mixed typed layers, E.g Mamba
        self.num_attention_layers = model_config.get_num_attention_layers(
            parallel_config)
        self.num_kv_heads = model_config.get_num_kv_heads(parallel_config)

        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        if self.num_gpu_blocks:
            self.num_gpu_blocks //= parallel_config.pipeline_parallel_size
        self.num_cpu_blocks = cache_config.num_cpu_blocks
        if self.num_cpu_blocks:
            self.num_cpu_blocks //= parallel_config.pipeline_parallel_size

        if cache_config.cache_dtype == "auto":
            self.dtype = model_config.dtype
        else:
            self.dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        # Get attention backend.
        self.attn_backend = get_attn_backend(self.head_size,
                                             model_config.dtype,
                                             cache_config.cache_dtype,
                                             self.block_size,
                                             model_config.is_attention_free)

        # Initialize the cache.
        self.gpu_cache = self._allocate_kv_cache(
            self.num_gpu_blocks, self.device_config.device_type)
        self.cpu_cache = self._allocate_kv_cache(self.num_cpu_blocks, "cpu")

    def _allocate_kv_cache(
        self,
        num_blocks: int,
        device: str,
    ) -> List[torch.Tensor]:
        """Allocates KV cache on the specified device."""
        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
            num_blocks, self.block_size, self.num_kv_heads, self.head_size)
        pin_memory = is_pin_memory_available() if device == "cpu" else False
        kv_cache: List[torch.Tensor] = []
        for _ in range(self.num_attention_layers):
            # null block in CpuGpuBlockAllocator requires at least that
            # block to be zeroed-out.
            # We zero-out everything for simplicity.
            kv_cache.append(
                torch.zeros(kv_cache_shape,
                            dtype=self.dtype,
                            pin_memory=pin_memory,
                            device=device))
        return kv_cache

    def swap_in(self,
                src_to_dst: torch.Tensor,
                offsets: torch.Tensor = None,
                sequence_ids: torch.Tensor = None) -> None:
        for i in range(self.num_attention_layers):
            self.attn_backend.swap_blocks(self.cpu_cache[i], self.gpu_cache[i],
                                          src_to_dst)

    def swap_out(self, src_to_dst: torch.Tensor) -> None:
        for i in range(self.num_attention_layers):
            self.attn_backend.swap_blocks(self.gpu_cache[i], self.cpu_cache[i],
                                          src_to_dst)

    def swap_in_sync(self, sequence_ids: torch.Tensor) -> None:
        pass

    def swap_out_sync(self) -> None:
        pass

    def copy(self, src_to_dsts: torch.Tensor) -> None:
        self.attn_backend.copy_blocks(self.gpu_cache, src_to_dsts)

    @staticmethod
    def get_cache_block_size(
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        head_size = model_config.get_head_size()
        num_heads = model_config.get_num_kv_heads(parallel_config)
        num_attention_layers = model_config.get_num_attention_layers(
            parallel_config)

        key_cache_block = cache_config.block_size * num_heads * head_size
        value_cache_block = key_cache_block
        total = num_attention_layers * (key_cache_block + value_cache_block)
        if cache_config.cache_dtype == "auto":
            dtype = model_config.dtype
        else:
            dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
        dtype_size = get_dtype_size(dtype)
        return dtype_size * total


class EventPool:

    def __init__(self, reserve_num_events: int, device: torch.device):
        self.reserve_num_events = reserve_num_events
        self.event_queue: deque[torch.cuda.Event] = deque()
        self.device = device
        with torch.cuda.device(device):
            for i in range(reserve_num_events):
                event = torch.cuda.Event()
                # create the detail new event
                event.record()
                event.synchronize()
                self.event_queue.append(event)

    def get_event(self) -> torch.cuda.Event:
        if (len(self.event_queue) == 0):
            with torch.cuda.device(self.device):
                event = torch.cuda.Event()
                # create the detail new event
                event.record()
                event.synchronize()
                self.event_queue.append(event)
        return self.event_queue.popleft()

    def put_event(self, event: torch.cuda.Event):
        self.event_queue.append(event)

    def get_events(self, num_events: int) -> list[torch.cuda.Event]:
        ret = []
        for i in range(num_events):
            ret.append(self.get_event())
        return ret

    def put_events(self, events: list[torch.cuda.Event]):
        for event in events:
            self.event_queue.append(event)


class GPUCacheEngine(CacheEngine):

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        device_config: DeviceConfig,
    ) -> None:
        super().__init__(cache_config, model_config, parallel_config,
                         device_config)
        self.use_fast_path = False
        # only these *PUs support fast path
        if (current_platform.is_cuda()) or \
                (current_platform.is_rocm()):
            self.use_fast_path = True
        self.swap_in_stream = None
        self.swap_in_event_pool = None
        self.swap_in_event_map: Dict[int, Any] = {}
        self.swap_out_stream = None
        self.swap_out_event = None
        self.device = None
        self._cur_swap_in_sync_layer = 0
        self._cur_swap_out_layer = 0
        if (not self.use_fast_path):
            return
        # create device streams and events
        self.device = torch.device(torch.cuda.current_device())
        with torch.cuda.device(self.device):
            self.swap_in_stream = torch.cuda.Stream()
            self.swap_in_event_pool = EventPool(64 * self.num_attention_layers,
                                                self.device)
            self.swap_out_stream = torch.cuda.Stream()
            self.swap_out_event = torch.cuda.Event()

    def swap_in(self,
                src_to_dst: torch.Tensor,
                offsets: torch.Tensor = None,
                sequence_ids: torch.Tensor = None) -> None:
        if (not self.use_fast_path) or \
                (sequence_ids is None) or (sequence_ids.numel() == 0):
            super().swap_in(src_to_dst)
            return
        sequence_ids_numpy = sequence_ids.numpy()
        for seq_id in sequence_ids_numpy:
            # the first one
            if (seq_id == -1):
                continue
            assert (self.swap_in_event_map.get(seq_id) is None)
            assert (self.swap_in_event_pool is not None)
            tmp_event_list = self.swap_in_event_pool.get_events(
                self.num_attention_layers)
            self.swap_in_event_map[seq_id] = tmp_event_list
        offsets_numpy = offsets.numpy()
        forward_stream = torch.cuda.current_stream()
        for idx, seq_id in enumerate(sequence_ids_numpy):
            start_idx = offsets_numpy[idx]
            last_idx = offsets_numpy[idx + 1]
            num_blocks = last_idx - start_idx
            swap_in_blocks = src_to_dst.narrow(0, start_idx, num_blocks)
            for layer_idx in range(self.num_attention_layers):
                if (seq_id == -1):
                    with torch.cuda.stream(forward_stream):
                        self.attn_backend.swap_blocks(
                            self.cpu_cache[layer_idx],
                            self.gpu_cache[layer_idx], swap_in_blocks)
                else:
                    with torch.cuda.stream(self.swap_in_stream):
                        self.attn_backend.swap_blocks(
                            self.cpu_cache[layer_idx],
                            self.gpu_cache[layer_idx], swap_in_blocks)
                        self.swap_in_event_map[seq_id][layer_idx].record(
                            self.swap_in_stream)

    def swap_out(
        self,
        src_to_dst: torch.Tensor,
    ) -> None:
        if (src_to_dst.numel() == 0):
            return
        if (not self.use_fast_path):
            cur_layer = self._cur_swap_out_layer
            self.attn_backend.swap_blocks(self.gpu_cache[cur_layer],
                                          self.cpu_cache[cur_layer],
                                          src_to_dst)
        else:
            forward_stream = torch.cuda.current_stream()
            assert (self.swap_out_event is not None)
            self.swap_out_event.record(forward_stream)
            self.swap_out_event.wait(self.swap_out_stream)
            with torch.cuda.stream(self.swap_out_stream):
                cur_layer = self._cur_swap_out_layer
                self.attn_backend.swap_blocks(self.gpu_cache[cur_layer],
                                              self.cpu_cache[cur_layer],
                                              src_to_dst)
        self._cur_swap_out_layer = \
            (self._cur_swap_out_layer + 1) % self.num_attention_layers

    def _swap_in_layer_sync_with_seq_ids(self, layer_id: int,
                                         seq_ids: torch.Tensor) -> None:
        for seq_id in seq_ids.numpy():
            if (self.swap_in_event_map.get(seq_id) is None):
                continue
            self.swap_in_event_map[seq_id][layer_id].synchronize()
        if (layer_id == self.num_attention_layers - 1):
            # recycle the events
            for seq_id in seq_ids.numpy():
                if (self.swap_in_event_map.get(seq_id) is None):
                    continue
                event_list = self.swap_in_event_map[seq_id]
                assert (self.swap_in_event_pool is not None)
                self.swap_in_event_pool.put_events(event_list)
                del self.swap_in_event_map[seq_id]

    def _swap_in_layer_all_sync(self, layer_id: int) -> None:
        for event_list in self.swap_in_event_map.values():
            event_list[layer_id].synchronize()
        # recycle the events
        if (layer_id == self.num_attention_layers - 1):
            for event_list in self.swap_in_event_map.values():
                assert (self.swap_in_event_pool is not None)
                self.swap_in_event_pool.put_events(event_list)
            self.swap_in_event_map.clear()

    def swap_in_sync(self, sequence_ids: torch.Tensor) -> None:
        if (not self.use_fast_path):
            return
        if (sequence_ids.numel() == 0):
            self._swap_in_layer_all_sync(self._cur_swap_in_sync_layer)
        else:
            self._swap_in_layer_sync_with_seq_ids(self._cur_swap_in_sync_layer,
                                                  sequence_ids)
        self._cur_swap_in_sync_layer = \
            (self._cur_swap_in_sync_layer + 1) % self.num_attention_layers

    def swap_out_sync(self) -> None:
        if (not self.use_fast_path):
            return
        assert (self.swap_out_stream is not None)
        self.swap_out_stream.synchronize()
