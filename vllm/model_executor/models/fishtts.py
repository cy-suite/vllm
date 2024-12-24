from array import array
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from transformers import LlamaConfig

from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig, VllmConfig
from vllm.inputs import INPUT_REGISTRY, InputContext, DummyData
from vllm.inputs.registry import InputContext
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.multi_head_sampler import MultiheadSampler
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.layers.vocab_parallel_embedding import DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.models.interfaces import SupportsLoRA
from vllm.model_executor.models.llama import LlamaModel
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.speech import SpeechPlugin
from vllm.sequence import VLLM_TOKEN_ID_ARRAY_TYPE, IntermediateTensors
from .interfaces import SupportsMultiModal

from einops import rearrange
from transformers.generation import TopKLogitsWarper, TopPLogitsWarper

import lzma
import numpy as np

def dummy_data_for_ttsllm(ctx: InputContext, seq_len: int, mm_counts: Mapping[str, int]):

    from vllm.sequence import SequenceData


    dummy_seq_data = SequenceData([0] * seq_len)
    dummy_multi_modal_data = {"audio": SpeechPlugin.sample_random_speaker()}

    return DummyData(dummy_seq_data, dummy_multi_modal_data, None)

def get_max_speech_tokens(ctx: InputContext):
    return 16

@MULTIMODAL_REGISTRY.register_speech_input_mapper()
@INPUT_REGISTRY.register_dummy_data(dummy_data_for_ttsllm)
@MULTIMODAL_REGISTRY.register_max_speech_tokens(get_max_speech_tokens)
class FishTtsLlm(nn.Module, SupportsLoRA, SupportsMultiModal):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj", "o_proj", "gate_up_proj", "down_proj", "embed_tokens",
        "lm_head"
    ]
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }
    embedding_padding_modules = ["lm_head"]
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        
        config = vllm_config.model_config.hf_config
        lora_config = vllm_config.lora_config
        quant_config = vllm_config.quant_config

        self.config = config
        
        # static parameters, put them in config later
        self.num_audio_tokens = config.num_audio_tokens
        self.num_text_tokens = config.num_text_tokens
        self.num_output_head = config.num_output_head
        self.audio_start_token_id = config.audio_start_token_id
        self.audio_ref_token_id = config.audio_ref_start_token_id

        self.gpt = LlamaModel(vllm_config=vllm_config, prefix=prefix)
        self.model_dim = self.gpt.config.hidden_size
        self.emb_text = VocabParallelEmbedding(self.num_text_tokens, self.model_dim) 
        self.emb_code = nn.ModuleList([
            VocabParallelEmbedding(self.num_audio_tokens, self.model_dim) for _ in range(self.num_output_head)
        ])
        
        unpadded_vocab_size = self.num_audio_tokens
        if lora_config:
            unpadded_vocab_size += lora_config.lora_extra_vocab_size
        self.lm_head = nn.ModuleList([
            ParallelLMHead(
                unpadded_vocab_size,
                self.model_dim,
                org_num_embeddings=self.num_audio_tokens,
                padding_size=DEFAULT_VOCAB_PADDING_SIZE
                # We need bigger padding if using lora for kernel
                # compatibility
                if not lora_config else lora_config.lora_vocab_padding_size,
                quant_config=quant_config,
            ) for _ in range(self.num_output_head)
        ])
        self.logits_processor = nn.ModuleList([LogitsProcessor(unpadded_vocab_size, self.num_audio_tokens) for _ in range(self.num_output_head)])
        self.sampler = MultiheadSampler()

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            # (".gate_up_proj", ".gate_proj", 0),
            # (".gate_up_proj", ".up_proj", 1),
        ]
        
        if getattr(self.config, "use_fused_mlp", True):
            stacked_params_mapping.extend(
                [
                    (".gate_up_proj", ".gate_proj", 0),
                    (".gate_up_proj", ".up_proj", 1)
                ]
            )
        
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                try:
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)
                except KeyError:
                    pass
                break
            else:
                try:
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
                except KeyError:
                    pass

    def get_input_embeddings(self, input_ids: torch.Tensor, audio_ref: torch.Tensor, is_prompt: bool) -> torch.Tensor:
        if is_prompt:
            emb: torch.Tensor = self.emb_text(input_ids)
            audio_start = torch.tensor([1024, 1024], device=input_ids.device)
            code_emb = [
                self.emb_code[i](audio_start[i])
                for i in range(self.num_output_head)
            ]
            start_token = torch.stack(code_emb, 1).sum(1).to(emb.dtype)

            # find the index of the audio BOS token
            emb[-1] = start_token

            # batch size = 2
            # inpudId 7004 7004  XXXX 7004 7001  1 2 34 | 7003 7004 7004  XXXX 7004 7001  1 2 34 7003
            # speaker ref [16*2, 1536]
            emb[0:16] = audio_ref[0].to(emb.dtype)

        else:
            code_emb = [
                self.emb_code[i](input_ids[:,i]) for i in range(self.num_output_head)
            ]
            emb = torch.stack(code_emb, 2).sum(2)
        return emb

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = [
            self.logits_processor[i](self.lm_head[i], hidden_states, sampling_metadata)
            for i in range(self.num_output_head)
        ]
        logits = torch.stack(logits, 0).permute(1, 0, 2)
        return logits
    
    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
        is_prompt: bool = False,
        **kwargs: object
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            audio_ref = kwargs.get("audio", None)
            hidden_states = self.get_input_embeddings(input_ids, audio_ref, is_prompt)
        model_output = self.gpt(
            input_ids=input_ids,
            inputs_embeds=hidden_states,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            intermediate_tensors=intermediate_tensors
        )
        return model_output

    def apply_spk_emb(
        self,
        emb: torch.Tensor,
        spk_emb: torch.Tensor,
        attn_metadata: AttentionMetadata,
        input_ids: torch.Tensor,
    ):
        audio_start = torch.tensor([1024, 1024], device=input_ids.device)
        code_emb = [
            self.emb_code[i](audio_start[i])
            for i in range(self.num_output_head)
        ]
        start_token = torch.stack(code_emb, 1).sum(1).to(emb.dtype)

        # find the index of the speaker token
        indices = (input_ids == self.audio_start_token_id).nonzero(as_tuple=True)
        if indices[0].size(0) == 0:
            return
        emb.index_put_(indices, start_token)

    def merge_sample_results(
        self,
        source: SamplerOutput,
        target: SamplerOutput,
    ):
        for o_a, o_b in zip(source.outputs, target.outputs):
            for s_a, s_b in zip(o_a.samples, o_b.samples):
                s_a.output_tokens.append(s_b.output_token)
    