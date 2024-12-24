"""A layer that samples the next tokens from the model's outputs."""
from array import array
import itertools
from math import inf
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from vllm.model_executor.layers.sampler import MaybeDeferredSampleResultType, SampleResultArgsType, SampleReturnType, SamplerOutput, _apply_top_k_top_p, _get_bin_counts_and_mask
from vllm.triton_utils import HAS_TRITON

from vllm.model_executor.sampling_metadata import (SamplingMetadata,
                                                   SamplingTensors,
                                                   SequenceGroupToSample)
from vllm.sampling_params import SamplingType
from vllm.sequence import (CompletionSequenceGroupOutput, Logprob,
                           PromptLogprobs, SampleLogprobs, SequenceOutput)
from vllm.utils import (PyObjectCache, async_tensor_h2d,
                        is_pin_memory_available, make_tensor_with_pad)

from einops import rearrange

_SAMPLING_EPS = 1e-5

class MultiheadSampler(nn.Module):
    def __init__(self):
        super().__init__()
        # Whether or not the SamplerOutput should have on-device tensors
        # containing the sampled token ids and probabilities. This is used by
        # speculative decoding.
        self.include_gpu_probs_tensor = False
    
    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        batch_size, num_heads, vocab_size = logits.size()
        logits = logits.reshape(batch_size * num_heads, vocab_size)

        self._init_sampling_tensors(num_heads, vocab_size, logits, sampling_metadata)
        sampling_tensors = self._sampling_tensors
        do_penalties = self._do_penalties
        do_top_p_top_k = self._do_top_p_top_k
        is_prompt = self._is_prompt

        if not is_prompt and do_penalties:
            logits = self._apply_penalties(logits, sampling_tensors.output_tokens, sampling_tensors.repetition_penalties)

        # Use float32 to apply temperature scaling.
        # Use in-place division to avoid creating a new tensor.
        logits = logits.to(torch.float)
        logits.div_(sampling_tensors.temperatures.unsqueeze(dim=1))

        if do_top_p_top_k:
            logits = _apply_top_k_top_p(logits, sampling_tensors.top_ps,
                                        sampling_tensors.top_ks)

        # We use float32 for probabilities and log probabilities.
        # Compute the probabilities.
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        # Compute the log probabilities.
        logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)
        
        # Sample the next tokens.
        maybe_deferred_sample_results, maybe_sampled_tokens_tensor = self._sample(
            probs,
            logprobs,
            sampling_metadata,
            sampling_tensors,
            include_gpu_probs_tensor=self.include_gpu_probs_tensor,
            modify_greedy_probs=False
        )
        
        sampled_token_ids_tensor = maybe_sampled_tokens_tensor.reshape(-1, num_heads)
        id_next = sampled_token_ids_tensor.cpu().numpy()

        if self.include_gpu_probs_tensor:
            # Since we will defer sampler result Pythonization,
            # preserve GPU-side tensors in support of later
            # deferred pythonization of logprobs
            sampled_token_ids_tensor = sampled_token_ids_tensor.to(dtype=torch.long, device=probs.device)
            on_device_tensors = (probs, logprobs, sampled_token_ids_tensor)
        else:
            # Since Pythonization has already happened, don't preserve
            # GPU-side tensors.
            on_device_tensors = None

        return self.build_sampler_output(id_next, sampling_metadata, 
                                         on_device_tensors=on_device_tensors,
                                         maybe_deferred_sample_results=maybe_deferred_sample_results,
                                         skip_sampler_cpu_output=sampling_metadata.skip_sampler_cpu_output)

    def _sample(
        self,
        probs: torch.Tensor,
        logprobs: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        sampling_tensors: SamplingTensors,
        include_gpu_probs_tensor: bool,
        modify_greedy_probs: bool,
    ) -> SampleReturnType:
        id_next_tensor = torch.multinomial(probs, 1).to(dtype=torch.long, device=probs.device)
 
        maybe_deferred_args = SampleResultArgsType(
            sampling_metadata=sampling_metadata,
            sample_metadata=None,
            multinomial_samples=None,
            greedy_samples=id_next_tensor,
            beam_search_logprobs=None,
            sample_results_dict={})
        
        if not sampling_metadata.skip_sampler_cpu_output:
            return id_next_tensor, id_next_tensor
        else:
            return maybe_deferred_args, id_next_tensor

    def _init_sampling_tensors(self,
                               num_heads: int,
                               vocab_size: int,
                               logits: torch.Tensor,
                               sampling_metadata: SamplingMetadata):
        self._sampling_tensors = None
        sampling_tensors, do_penalties, do_top_p_top_k, is_prompt = self.from_sampling_metadata(
            num_heads, vocab_size, logits, sampling_metadata
        )
        
        self._sampling_tensors = sampling_tensors
        self._do_penalties = do_penalties
        self._do_top_p_top_k = do_top_p_top_k
        self._is_prompt = is_prompt
        
    def from_sampling_metadata(self,
                               num_heads: int,
                               vocab_size: int,
                               logits: torch.Tensor,
                               sampling_metadata: SamplingMetadata) -> Tuple["SamplingTensors", bool, bool, bool]:
        dtype = logits.dtype
        device = logits.device

        output_tokens: List[array] = []
        top_ks: List[int] = []
        temperatures: List[float] = []
        top_ps: List[float] = []
        repetition_penalties: List[float] = []
        do_penalties = False
        do_top_p_top_k = False
        is_prompt = False

        for seq_group in sampling_metadata.seq_groups:
            seq_ids = seq_group.seq_ids
            sampling_params = seq_group.sampling_params
            temperature = sampling_params.temperature
            r = sampling_params.repetition_penalty
            top_p = sampling_params.top_p

            # k should not be greater than the vocab size.
            top_k = min(sampling_params.top_k, vocab_size)
            top_k = vocab_size if top_k == -1 else top_k
            if temperature < _SAMPLING_EPS:
                # NOTE: Zero temperature means deterministic sampling
                # (i.e., greedy sampling or beam search).
                # Set the temperature to 1 to avoid division by zero.
                temperature = 1.0
            if not do_top_p_top_k and (top_p < 1.0 - _SAMPLING_EPS
                                       or top_k != vocab_size):
                do_top_p_top_k = True
            if not do_penalties and (abs(r - 1.0) >= _SAMPLING_EPS):
                do_penalties = True

            is_prompt = seq_group.is_prompt
            if seq_group.do_sample:
                sample_lens = len(seq_group.sample_indices)
                assert sample_lens == len(seq_ids)
                temperatures += [temperature] * len(seq_ids) * num_heads
                top_ps += [top_p] * len(seq_ids) * num_heads
                top_ks += [top_k] * len(seq_ids) * num_heads
                repetition_penalties += [r] * len(seq_ids) * num_heads
                
        if do_penalties:
            for seq_group in sampling_metadata.seq_groups:
                seq_ids = seq_group.seq_ids
                repetition_window = seq_group.sampling_params.repetition_window
                if seq_group.do_sample:
                    for seq_id in seq_ids:
                        seq_data = seq_group.seq_data[seq_id]
                        token_ids_in_window = seq_data.output_token_ids_array[-repetition_window:]
                        if token_ids_in_window:
                            for head_id in range(num_heads):
                                output_tokens.append([row[head_id] for row in token_ids_in_window])
        
        pin_memory = is_pin_memory_available()
        if do_penalties:
            output_t = make_tensor_with_pad(
                output_tokens,
                vocab_size,
                device="cpu",
                dtype=torch.int64,
                pin_memory=pin_memory,
            )
        else:
            empty_tensor = torch.empty(0, device=device, dtype=torch.long)
            output_t = empty_tensor
            
        temperatures_t = torch.tensor(
            temperatures,
            device="cpu",
            dtype=dtype,
            pin_memory=pin_memory,
        )
        top_ps_t = torch.tensor(
            top_ps,
            device="cpu",
            dtype=dtype,
            pin_memory=pin_memory,
        )
        repetition_penalties_t = torch.tensor(
            repetition_penalties,
            device="cpu",
            dtype=dtype,
            pin_memory=pin_memory,
        )
        top_ks_t = torch.tensor(
            top_ks,
            device="cpu",
            dtype=torch.int,
            pin_memory=pin_memory,
        )
        
        sampling_tensors = SamplingTensors(
            output_tokens=output_t.to(device=device, non_blocking=True),
            temperatures=temperatures_t.to(device=device, non_blocking=True),
            top_ps=top_ps_t.to(device=device, non_blocking=True),
            repetition_penalties=repetition_penalties_t.to(device=device, non_blocking=True),
            top_ks=top_ks_t.to(device=device, non_blocking=True),

            min_ps=None,
            presence_penalties=None,
            frequency_penalties=None,
            prompt_tokens=None
        )
        
        return (sampling_tensors, do_penalties, do_top_p_top_k, is_prompt)
    
    def _apply_penalties(self, logits: torch.Tensor,
                     output_tokens_tensor: torch.Tensor,
                     repetition_penalties: torch.Tensor) -> torch.Tensor:
        num_seqs, vocab_size = logits.shape
        output_bin_counts, output_mask = _get_bin_counts_and_mask(
            output_tokens_tensor, vocab_size, num_seqs)

        repetition_penalties = repetition_penalties[:, None].repeat(1, vocab_size)
        repetition_penalties[~(output_mask)] = 1.0
        logits = torch.where(logits > 0, logits / repetition_penalties,
                            logits * repetition_penalties)

        return logits
    
    def build_sampler_output(self,
                             sample_results: List[List[int]],
                             sampling_metadata: SamplingMetadata,
                             maybe_deferred_sample_results: MaybeDeferredSampleResultType = None,
                             on_device_tensors: Optional[Tuple[torch.Tensor, torch.Tensor,torch.Tensor]] = None,
                             skip_sampler_cpu_output: bool = False) -> SamplerOutput:
        sampler_output: List[CompletionSequenceGroupOutput] = []
        if skip_sampler_cpu_output:
            pass
        else:
            for seq_group, sample_result in zip(sampling_metadata.seq_groups, sample_results):
                seq_ids = seq_group.seq_ids
                parent_id = 0 # no beam search for now
                seq_outputs: List[SequenceOutput] = []
                log_prob = { sample_result[0]: Logprob(logprob=inf, rank=None, decoded_token=None) }
                seq_output = SequenceOutput(seq_ids[parent_id], sample_result[0], log_prob)
                seq_output.output_tokens = sample_result.tolist()
                seq_outputs.append(seq_output)
                sampler_output.append(CompletionSequenceGroupOutput(seq_outputs, prompt_logprobs=None))
        
        # If not specified, store None values in SamplerOutput.
        if on_device_tensors is not None:
            (sampled_token_probs, logprobs_tensor,
            sampled_token_ids) = on_device_tensors
        else:
            sampled_token_probs, logprobs_tensor, sampled_token_ids = (None, None,
                                                                    None)
        return SamplerOutput(
            outputs=sampler_output,
            sampled_token_probs=sampled_token_probs,
            sampled_token_ids=sampled_token_ids,
            logprobs=logprobs_tensor,
            deferred_sample_results_args=maybe_deferred_sample_results,
        )