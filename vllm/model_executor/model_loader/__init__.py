import os
from typing import Optional

from torch import nn

from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, LoRAConfig,
                         ModelConfig, MultiModalConfig, ParallelConfig,
                         SchedulerConfig)
from vllm.model_executor.model_loader.loader import (BaseModelLoader,
                                                     get_model_loader)
from vllm.model_executor.model_loader.utils import (
    get_architecture_class_name, get_model_architecture)
from vllm.model_executor.model_optimizer.model_optimizer import optimizer


def get_model(*, model_config: ModelConfig, load_config: LoadConfig,
              device_config: DeviceConfig, parallel_config: ParallelConfig,
              scheduler_config: SchedulerConfig,
              lora_config: Optional[LoRAConfig],
              multimodal_config: Optional[MultiModalConfig],
              cache_config: CacheConfig) -> nn.Module:
    loader = get_model_loader(load_config)
    m = loader.load_model(model_config=model_config,
                          device_config=device_config,
                          lora_config=lora_config,
                          multimodal_config=multimodal_config,
                          parallel_config=parallel_config,
                          scheduler_config=scheduler_config,
                          cache_config=cache_config)
    if "VLLM_DISABLE_MODEL_OPTIMIZER" in os.environ:
        return m
    else:
        return optimizer(m)  #, fullgraph=True)


__all__ = [
    "get_model", "get_model_loader", "BaseModelLoader",
    "get_architecture_class_name", "get_model_architecture"
]
