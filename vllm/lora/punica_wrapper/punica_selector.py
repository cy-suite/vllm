from vllm.platforms import current_platform
from vllm.utils import print_info_once
import vllm.envs as envs

from .punica_base import PunicaWrapperBase


def get_punica_wrapper(*args, **kwargs) -> PunicaWrapperBase:
    if current_platform.is_cuda_alike():
        # Lazy import to avoid ImportError
        if envs.VLLM_USE_V1:
            from vllm.lora.punica_wrapper.v1_gpu import V1LoRAGPU
            print_info_once("Using V1LoRAGPU.")
            return V1LoRAGPU(*args, **kwargs)
        else:
            from vllm.lora.punica_wrapper.punica_gpu import PunicaWrapperGPU
            print_info_once("Using PunicaWrapperGPU.")
            return PunicaWrapperGPU(*args, **kwargs)
    elif current_platform.is_hpu():
        # Lazy import to avoid ImportError
        from vllm.lora.punica_wrapper.punica_hpu import PunicaWrapperHPU
        print_info_once("Using PunicaWrapperHPU.")
        return PunicaWrapperHPU(*args, **kwargs)
    else:
        raise NotImplementedError
