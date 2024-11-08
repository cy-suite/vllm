import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from vllm.config import LoRAConfig
# yapf conflicts with isort for this block
# yapf: disable
from vllm.lora.layers import (LoRAMapping,
                             BaseLayerWithLoRA,
                              VocabParallelEmbeddingWithLoRA)
# yapf: enable
from vllm.lora.punica_wrapper import get_punica_wrapper
from vllm.model_executor.utils import set_random_seed

from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)

from vllm.lora.models import (LongContextLoRAContext, LoRALayerWeights,
                              PackedLoRALayerWeights)

from utils import DummyLoRAManager
from vllm.distributed.parallel_state import ensure_model_parallel_initialized, init_distributed_environment
from conftest import _dist_init

def get_random_id_to_index(num_loras: int,
                           num_slots: int,
                           log: bool = True) -> List[Optional[int]]:
    """Creates a random lora_id_to_index mapping.

    Args:
        num_loras: The number of active loras in the mapping.
        num_slots: The number of slots in the mapping. Must be larger
            than num_loras.
        log: Whether to log the output.
    """

    if num_loras > num_slots:
        raise ValueError(
            f"num_loras is higher than num_slots: {num_loras} > {num_slots}. "
            "num_loras must be less than or equal to num_slots.")

    slots: List[Optional[int]] = [None] * num_slots
    random_slot_selections = (torch.randperm(num_slots)[:num_loras]).tolist()
    for lora_id, slot_idx in enumerate(random_slot_selections, start=1):
        slots[slot_idx] = lora_id

    if log:
        print(f"Created lora_id_to_index mapping: {slots}.")

    return slots

def populate_loras(
    id_to_index: List[Optional[int]],
    layer: BaseLayerWithLoRA,
    layer_weights: torch.Tensor,
    generate_embeddings_tensor: int = 0,
    repeats: int = 1,
) -> Tuple[Dict[int, LoRALayerWeights], Dict[int, List[LoRALayerWeights]]]:
    """This method populates the lora layers with lora weights.

    Args:
        id_to_index: a list of lora ids. The index of the lora id
            represents which memory slot the lora matrices are
            stored in. A None value indicates a free slot.
        layer: the LoRAlayer to populate.
        layer_weights: the PyTorch tensor containing the layer's
            weights.
        generate_embeddings_tensor: whether to generate an
            embeddings tensor for each LoRA.
        repeats: must only be set for column parallel packed
            layers. Indicates the number of loras to compose
            together to create a single lora layer.
    """

    # Dictionary that maps the lora ID to the
    # corresponding lora weights.
    lora_dict: Dict[int, LoRALayerWeights] = dict()

    # Dictionary that maps the lora ID to the
    # corresponding subloras.
    sublora_dict: Dict[int, List[LoRALayerWeights]] = dict()

    for slot_idx, lora_id in enumerate(id_to_index):
        if lora_id is not None:
            subloras: List[LoRALayerWeights] = []
            sublora_len = layer_weights.shape[0] // repeats
            for i in range(repeats):
                sublora = DummyLoRAManager(
                    layer_weights.device).init_random_lora(
                        module_name=f"fake_{i}",
                        weight=layer_weights,
                        generate_embeddings_tensor=generate_embeddings_tensor,
                    )
                sublora.lora_b = sublora.lora_b[:, (sublora_len *
                                                    i):(sublora_len * (i + 1))]
                sublora.optimize()
                subloras.append(sublora)

            lora = PackedLoRALayerWeights.pack(
                subloras) if repeats > 1 else subloras[0]

            layer.set_lora(
                slot_idx,
                lora_a=lora.lora_a,
                lora_b=lora.lora_b,
                embeddings_tensor=lora.embeddings_tensor,
            )

            lora_dict[lora_id] = lora
            sublora_dict[lora_id] = subloras

    return lora_dict, sublora_dict

def create_random_inputs(
    active_lora_ids: List[int],
    num_inputs: int,
    input_size: Tuple[int, ...],
    input_range: Tuple[float, float],
    input_type: torch.dtype = torch.int,
    device: torch.device = "cuda"
) -> Tuple[List[torch.Tensor], List[int], List[int]]:
    """Creates random inputs.

    Args:
        active_lora_ids: lora IDs of active lora weights.
        num_inputs: the number of inputs to create.
        input_size: the size of each individual input.
        input_range: the range of values to include in the input.
            input_range[0] <= possible input values < input_range[1]
        input_type: the type of values in the input.
    """

    low, high = input_range

    inputs: List[torch.Tensor] = []
    index_mapping: List[int] = []
    prompt_mapping: List[int] = []

    for _ in range(num_inputs):
        if input_type == torch.int:
            inputs.append(
                torch.randint(low=int(low),
                              high=int(high),
                              size=input_size,
                              device=device))
        else:
            inputs.append(
                torch.rand(size=input_size, dtype=input_type, device=device) *
                high + low)

        lora_id = random.choice(active_lora_ids)
        index_mapping += [lora_id] * input_size[0]
        prompt_mapping += [lora_id]

    return inputs, index_mapping, prompt_mapping


num_loras = 4
vocab_size = 512
is_prefill = True
max_loras = 8
device="cuda:0"

def custom_pass(graph: torch.fx.Graph) -> torch.fx.Graph:
    print("Pre-pass:")
    print(graph)

    return graph


def custom_backend(graph: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("Graph entering custom_backend:")
    print(graph.print_readable())
    from torch._inductor import config
    current_config = config.shallow_copy_dict()
    from torch._inductor.compile_fx import compile_fx
    current_config['post_grad_custom_post_pass'] = custom_pass
    return compile_fx(graph, example_inputs, config_patches=current_config)

@torch.inference_mode()
def test_embeddings() -> None:

    torch.cuda.set_device(device)
    torch.set_default_device(device)

    init_distributed_environment(1, 0)
    ensure_model_parallel_initialized(1,1)

    max_loras = 8
    punica_wrapper = get_punica_wrapper(8192, 256, device)
    lora_config = LoRAConfig(max_loras=max_loras,
                             max_lora_rank=8,
                             lora_dtype=torch.float16)

    def create_random_embedding_layer():
        embedding = VocabParallelEmbedding(vocab_size, 256)
        embedding.weight.data = torch.rand_like(embedding.weight.data)
        embedding.weight.data[vocab_size:, :] = 0
        lora_embedding = VocabParallelEmbeddingWithLoRA(embedding)
        lora_embedding.create_lora_weights(max_loras, lora_config)

        return embedding, lora_embedding

    id_to_index = get_random_id_to_index(num_loras, max_loras)
    embedding, lora_embedding = create_random_embedding_layer()

    lora_embedding.set_mapping(punica_wrapper)
    lora_dict, _ = populate_loras(
        id_to_index,
        layer=lora_embedding,
        layer_weights=embedding.weight.T,
    )

    inputs, index_mapping, prompt_mapping = create_random_inputs(
        active_lora_ids=list(lora_dict.keys()),
        num_inputs=num_loras * 3,
        input_size=(200, ),
        input_range=(1, vocab_size),
        device=device)
    lora_mapping = LoRAMapping(index_mapping,
                               prompt_mapping,
                               is_prefill=True)
    punica_wrapper.update_metadata(lora_mapping, id_to_index, max_loras,
                                   vocab_size,
                                   lora_config.lora_extra_vocab_size)

    lora_embedding_compiled = torch.compile(lora_embedding, backend=custom_backend)

    embedding_compiled = torch.compile(embedding, backend=custom_backend)

    input = torch.cat(inputs)
    torch._dynamo.mark_dynamic(input, 0)

    lr = embedding_compiled(input)
    lora_result = lora_embedding_compiled(input)

if __name__ == '__main__':
    with _dist_init():
        test_embeddings()