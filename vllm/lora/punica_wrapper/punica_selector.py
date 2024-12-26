from vllm.platforms import current_platform

from .punica_base import PunicaWrapperBase


def get_punica_wrapper(*args, **kwargs) -> PunicaWrapperBase:
    return current_platform.get_punica_wrapper(*args, **kwargs)
