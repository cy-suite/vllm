import dataclasses
from abc import ABC, abstractmethod
from typing import (TYPE_CHECKING, Any, Dict, Generic, List, Optional, Type,
                    TypeVar, Union)

import torch

from vllm.sequence import SamplerOutput, SequenceGroupMetadata

if TYPE_CHECKING:
    from vllm.attention import AttentionMetadata
    from vllm.attention.backends.abstract import AttentionBackend
    from vllm.model_executor import SamplingMetadata

T = TypeVar('T', bound="ModelInputBase")


def _add_attn_metadata_broadcastable_dict(
        tensor_dict: Dict[str, Union[int, torch.Tensor]],
        attn_metadata: Optional["AttentionMetadata"]) -> None:
    """
    Helper method to update tensor_dict with broadcastable
    AttentionMetadata fields.
    """
    if attn_metadata is not None:
        tensor_dict.update(attn_metadata.asdict_zerocopy())


def _init_attn_metadata_from_kwargs(attn_backend: "AttentionBackend",
                                    **kwargs) -> Dict[str, Any]:
    """
    Helper method to initialize AttentionMetadata based on an
    AttentionBackend and broadcastable AttentionMetadata fields.
    """
    # Extract the fields used to create AttentionMetadata.
    valid_attn_kwargs = {}
    for field in dataclasses.fields(attn_backend.get_metadata_cls()):
        val = kwargs.pop(field.name, None)
        if val is not None:
            valid_attn_kwargs[field.name] = val

    attn_metadata = attn_backend.make_metadata(**valid_attn_kwargs)
    kwargs["attn_metadata"] = attn_metadata
    return kwargs


def _init_sampling_metadata_from_kwargs(  # type: ignore
        selected_token_indices: torch.Tensor = None,
        **kwargs) -> Dict[str, Any]:
    """
    Helper method to initialize SamplingMetadata based on broadcastable
    SamplingMetadata fields.
    """
    from vllm.model_executor import SamplingMetadata

    # An empty SamplingMetadata to signal that the worker should skip
    # sampling.
    sampling_metadata = SamplingMetadata(
        seq_groups=None,
        selected_token_indices=selected_token_indices,
        categorized_sample_indices=None,
        num_prompts=0,
    )
    kwargs["sampling_metadata"] = sampling_metadata
    return kwargs


def _add_sampling_metadata_broadcastable_dict(
        tensor_dict: Dict[str, Union[int, torch.Tensor]],
        sampling_metadata: Optional["SamplingMetadata"]) -> None:
    """
    Helper method to update tensor_dict with broadcastable
    SamplingMetadata fields.
    """
    if sampling_metadata is not None:
        tensor_dict["selected_token_indices"] = (
            sampling_metadata.selected_token_indices)


@dataclasses.dataclass(frozen=True)
class ModelInputBase(ABC):
    """Local inputs to each worker's model runner. May contain
    device-specific data. Different worker backends may have different methods
    of converting from the global ExecuteModelRequest produced by the LLM
    engine to the worker-local ModelInputBase objects.

    Model runners that support multi-GPU execution should define a
    ModelInputBase subclass, add their required fields, and specify how to
    serialize/deserialize a ModelInput for broadcast between workers.
    """

    @classmethod
    @abstractmethod
    def new(cls: Type[T], **kwargs) -> T:
        """
        Create a new instance of this class. Populate the new instance with
        the given kwargs.
        """
        raise NotImplementedError

    def replace(self: T, **kwargs) -> T:
        """
        Replace current fields with fields in kwargs.
        """
        return dataclasses.replace(self, **kwargs)

    def as_broadcastable_tensor_dict(
            self) -> Dict[str, Union[int, torch.Tensor]]:
        """
        Extract broadcastable fields. Override for fields that require some
        custom deserialization.
        """
        raise NotImplementedError

    def _get_attrs(self, attrs: List[str]) -> Dict[str, Any]:
        """
        Helper method to get a dictionary from attribute name to value.
        Attributes whose values are None will not be added to the returned
        dictionary.
        """
        tensor_dict: Dict[str, Union[int, torch.Tensor]] = {}
        for attr in attrs:
            val = getattr(self, attr, None)
            if val is not None:
                tensor_dict[attr] = val

        return tensor_dict


class ModelRunnerBase(ABC, Generic[T]):
    """
    Model runner interface that abstracts a particular hardware and/or type of
    model. Model execution may communicate data with model runners in other
    processes, but it should not include control plane metadata communication.

    Each ModelRunnerBase subclass should define a corresponding ModelInputBase
    subclass.
    """

    @abstractmethod
    def make_model_input(self,
                         make_attn_metadata: bool = False,
                         **model_input_fields) -> T:
        """
        Make an instance of a ModelInputBase from the given fields. If
        make_attn_metadata=True, then AttentionMetadata will be created from
        fields extracted from model_input_fields.
        """
        raise NotImplementedError

    @abstractmethod
    def prepare_model_input(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> T:
        """
        Prepare the inputs to ModelRunnerBase.execute_model from an execution
        request. This method may move data to the worker's local device. It is
        not allowed to communicate with other workers or devices.
        """
        raise NotImplementedError

    @torch.inference_mode()
    def execute_model(
        self,
        model_input: T,
        kv_caches: Optional[List[torch.Tensor]],
    ) -> Optional[SamplerOutput]:
        """
        Execute the model on the given input.
        """
        raise NotImplementedError
