"""
Attention computation layer with attention sink logic,
as described in https://github.com/mit-han-lab/streaming-llm.
Currently works for Llama (should eventually work for all RoPE models).
"""
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from vllm._C import cache_ops
from vllm.attention import AttentionMetadata
from vllm.attention.ops.paged_attn import PagedAttention
from vllm.attention.selector import _Backend, _which_attn_to_use
from vllm.config import CacheConfig
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.utils import make_tensor_with_pad


class StreamingAttentionSink(nn.Module):
    def __init__(
        self,
        model_context_len: int,
        block_size: int,
        kv_cache_dtype: str,
        attn_backend: _Backend,
        num_kv_heads: int,
        head_dim: int,
        kv_scale: float,
        rotary_emb_layer: Optional[RotaryEmbedding],
        attn_layer,
    ) -> None:
        super().__init__()
        self.model_context_len = model_context_len
        self.block_size = block_size
        self.kv_cache_dtype = kv_cache_dtype
        self.attn_backend = attn_backend
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.kv_scale = kv_scale
        self.rotary_emb = rotary_emb_layer
        self.use_alibi = rotary_emb_layer is None
        self.attn = attn_layer

        if attn_backend not in (_Backend.XFORMERS, _Backend.FLASH_ATTN):
            raise NotImplementedError(
                'Attention sinks is only supported for '
                'XFormers and FlashAttention currently.')

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        positions: Optional[torch.Tensor],
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        if self.use_alibi:
            return self._forward_alibi(q, k, v, kv_cache, attn_metadata)
        else:
            return self._forward_rope(q, k, v, positions, kv_cache, attn_metadata)
    
    def _forward_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        # q k v all have shape [num_tokens, num_heads * head_size] i.e. [1, 4096] for decode
        if kv_cache is not None:
            if self.attn_backend == _Backend.FLASH_ATTN:
            # key cache shape: [num_blocks, block_size, num_heads, head_size]
                key_cache, value_cache = kv_cache
            elif self.attn_backend == _Backend.XFORMERS:
            # key cache shape: [num_blocks, num_heads, head_size/x, block_size, x]
                key_cache, value_cache = PagedAttention.split_kv_cache(
                    kv_cache, self.num_kv_heads, self.head_dim)
        
        # what if metadata has both prefill and decode?
        if attn_metadata.prefill_metadata is not None:
            k_original = k.clone()
            q, k = self.rotary_emb(positions, q, k)
            attn_output = self.attn(q, k, v, kv_cache, attn_metadata, self.kv_scale)
            
            if kv_cache is not None:
                k_original = k_original.view(-1, self.num_kv_heads, self.head_dim)
                v = v.view(-1, self.num_kv_heads, self.head_dim)

                if self.attn_backend == _Backend.FLASH_ATTN:
                    cache_ops.reshape_and_cache_flash(
                        k_original,
                        v,
                        key_cache,
                        value_cache,
                        attn_metadata.slot_mapping.flatten(),
                        self.kv_cache_dtype,
                    )
                elif self.attn_backend == _Backend.XFORMERS:
                    PagedAttention.write_to_paged_cache(
                        k_original,
                        v,
                        key_cache,
                        value_cache,
                        attn_metadata.slot_mapping,
                        self.kv_cache_dtype,
                        self.kv_scale
                    )

            return attn_output

        elif attn_metadata.decode_metadata is not None:
            k_original = k.clone()
            device = q.device
            block_size = self.block_size
            model_context_len = self.model_context_len

            # cache seq_lens
            if hasattr(attn_metadata, 'seq_lens_clone'):
                seq_lens = attn_metadata.seq_lens_clone
            else:
                seq_lens = attn_metadata.seq_lens_tensor.tolist()
                attn_metadata.seq_lens_clone = seq_lens

            # cache phys_bnums
            if hasattr(attn_metadata, 'phys_bnums_list'):
                phys_bnums_list = attn_metadata.phys_bnums_list
            else:
                phys_bnums_list = []
            
            block_tables_tensor = attn_metadata.decode_metadata.block_tables

            # batch size = num sequences
            batch_size = block_tables_tensor.shape[0]
            original_keys: List[Tuple[torch.Tensor]] = []
            for i in range(batch_size):
                num_past_tokens = seq_lens[i] - 1  # assumes decode only yields 1 token
                within_context_len = num_past_tokens < model_context_len
                block_table = block_tables_tensor[i]
                
                if hasattr(attn_metadata, 'phys_bnums_list'):
                    phys_bnums = phys_bnums_list[i]
                else:
                    phys_bnums = block_table[:-1]
                    phys_bnums_list.append(phys_bnums)
                
                rem = num_past_tokens % block_size
                rem_phys_bnum = block_table[-1]
                
                # read unrotated keys from cache
                # FA shape: [len(phys_bnums), block_size, num_heads, head_size]
                # XF shape: [len(phys_bnums), num_heads, head_size/x, block_size, x]
                full_past_keys = torch.index_select(key_cache, 0, phys_bnums)
                if self.attn_backend == _Backend.FLASH_ATTN:
                    rem_past_keys = key_cache[rem_phys_bnum, :rem, :, :]
                elif self.attn_backend == _Backend.XFORMERS:
                    rem_past_keys = key_cache[rem_phys_bnum, :, :, :rem, :]
                original_keys.append((full_past_keys.clone(), rem_past_keys.clone()))
                
                # use positions within cache (capped by context length)
                pos_start = 0 if within_context_len else 2 * block_size - 1 - rem
                pos_end = min(num_past_tokens, model_context_len - 1)
                pos = torch.arange(pos_start, pos_end, device=device)
                if not within_context_len:
                    # pos: [0, 16] + [31 - rem, 4095)
                    pos_sink = torch.arange(0, block_size, device=device)
                    pos = torch.cat((pos_sink, pos))
                
                # reshape for rotary embedding kernel
                if self.attn_backend == _Backend.FLASH_ATTN:
                    full_past_keys = full_past_keys.flatten(0, 1)
                elif self.attn_backend == _Backend.XFORMERS:
                    full_past_keys = full_past_keys.permute((0, 3, 1, 2, 4)).flatten(0, 1)
                    rem_past_keys = rem_past_keys.permute((2, 0, 1, 3))
                    
                # combine full and remainder keys
                full_past_keys = torch.cat((full_past_keys, rem_past_keys), dim=0)
                full_past_keys = full_past_keys.flatten(1, -1)
                # shape: [pos_end - pos_start, num_heads * head_size]
                
                # rotate keys with new positions
                dummy_q = torch.zeros_like(full_past_keys)
                _, full_past_keys = self.rotary_emb(pos, dummy_q, full_past_keys)
                
                # reshape for writing back to cache
                if self.attn_backend == _Backend.FLASH_ATTN:
                    full_past_keys = full_past_keys.unflatten(1, (key_cache.shape[2], key_cache.shape[3]))
                elif self.attn_backend == _Backend.XFORMERS:
                    full_past_keys = full_past_keys.unflatten(1, (key_cache.shape[1], key_cache.shape[2], key_cache.shape[4]))
                
                # split into full and remainder keys
                full_past_keys, rem_past_keys = torch.split(full_past_keys, [len(phys_bnums) * block_size, rem])
                full_past_keys = full_past_keys.unflatten(0, (len(phys_bnums), block_size))
                
                # write rotated keys to cache for attention computation
                if self.attn_backend == _Backend.FLASH_ATTN:
                    key_cache.index_put_((phys_bnums,), full_past_keys)
                    key_cache[rem_phys_bnum, :rem, :, :] = rem_past_keys
                elif self.attn_backend == _Backend.XFORMERS:
                    full_past_keys = full_past_keys.permute((0, 2, 3, 1, 4))
                    rem_past_keys = rem_past_keys.permute((1, 2, 0, 3))
                    key_cache.index_put_((phys_bnums,), full_past_keys)
                    key_cache[rem_phys_bnum, :, :, :rem, :] = rem_past_keys
                
                if not within_context_len:
                    # cap number of tokens to consider with model context len
                    attn_metadata.seq_lens_tensor[i] = model_context_len - block_size + rem + 1
                    positions[i] = model_context_len - 1

            if not hasattr(attn_metadata, 'phys_bnums_list'):
                attn_metadata.phys_bnums_list = phys_bnums_list

            # compute attention in kernel
            q, k = self.rotary_emb(positions, q, k)
            attn_output = self.attn(q, k, v, kv_cache, attn_metadata, self.kv_scale)
                        
            # put original pre-rotated keys back in cache
            for i in range(batch_size):
                num_past_tokens = seq_lens[i] - 1
                within_context_len = num_past_tokens < model_context_len
                block_table = block_tables_tensor[i]
                phys_bnums = phys_bnums_list[i]

                rem = num_past_tokens % block_size
                rem_phys_bnum = block_table[-1]

                full_past_keys, rem_past_keys = original_keys[i]
                key_cache.index_put_((phys_bnums,), full_past_keys)
                if self.attn_backend == _Backend.FLASH_ATTN:
                    key_cache[rem_phys_bnum, :rem, :, :] = rem_past_keys
                elif self.attn_backend == _Backend.XFORMERS:
                    key_cache[rem_phys_bnum, :, :, :rem, :] = rem_past_keys
            
            k_original = k_original.view(-1, self.num_kv_heads, self.head_dim)
            v = v.view(-1, self.num_kv_heads, self.head_dim)

            if self.attn_backend == _Backend.FLASH_ATTN:
                cache_ops.reshape_and_cache_flash(
                    k_original,
                    v,
                    key_cache,
                    value_cache,
                    attn_metadata.slot_mapping.flatten(),
                    self.kv_cache_dtype,
                )
            elif self.attn_backend == _Backend.XFORMERS:
                PagedAttention.write_to_paged_cache(
                    k_original,
                    v,
                    key_cache,
                    value_cache,
                    attn_metadata.slot_mapping,
                    self.kv_cache_dtype,
                    self.kv_scale
                )
            
            # revert seq_lens inside metadata
            # so that next attn layer starts with same seq lens
            attn_metadata.seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int, device=device)
            
            return attn_output

    def _forward_alibi(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        if attn_metadata.prefill_metadata is not None:
            return self.attn(q, k, v, kv_cache, attn_metadata, self.kv_scale)
        elif attn_metadata.decode_metadata is not None:
            device = q.device
            block_size = self.block_size
            model_context_len = self.model_context_len

            # cache seq_lens
            if hasattr(attn_metadata, 'seq_lens_clone'):
                seq_lens = attn_metadata.seq_lens_clone
            else:
                seq_lens = attn_metadata.seq_lens_tensor.tolist()
                attn_metadata.seq_lens_clone = seq_lens

            block_tables_tensor = attn_metadata.decode_metadata.block_tables

            # batch size = num sequences
            batch_size = block_tables_tensor.shape[0]
            for i in range(batch_size):
                num_past_tokens = seq_lens[i] - 1  # assumes decode only yields 1 token
                if num_past_tokens < model_context_len: continue

                # cap number of tokens to consider with model context len
                rem = num_past_tokens % block_size
                attn_metadata.seq_lens_tensor[i] = model_context_len - block_size + rem + 1

            # compute attention in kernel
            attn_output = self.attn(q, k, v, kv_cache, attn_metadata, self.kv_scale)
            
            # revert seq_lens inside metadata
            # so that next attn layer starts with same seq lens
            attn_metadata.seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int, device=device)
            
            return attn_output


def get_attention_sink(
    model_attn: nn.Module,
    cache_config: Optional[CacheConfig],
    sliding_window: Optional[int],
    model_context_len: int
) -> StreamingAttentionSink:
    if cache_config is not None:
        kv_cache_dtype = cache_config.cache_dtype
        block_size = cache_config.block_size
    else:
        kv_cache_dtype = "auto"
        block_size = 16
    
    num_kv_heads = getattr(model_attn, "num_kv_heads", model_attn.num_heads)
    attn_backend = _which_attn_to_use(
        model_attn.num_heads,
        model_attn.head_dim,
        num_kv_heads,
        sliding_window,
        torch.get_default_dtype(),
        kv_cache_dtype,
        block_size
    )

    return StreamingAttentionSink(
        model_context_len,
        block_size,
        kv_cache_dtype,
        attn_backend,
        num_kv_heads,
        model_attn.head_dim,
        getattr(model_attn, "kv_scale", 1.0),
        getattr(model_attn, "rotary_emb", None),
        model_attn.attn
    )