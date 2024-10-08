from dataclasses import dataclass
from typing import List

import torch

from vllm.model_executor.prefill_only.engine_io import RequestOutput


@dataclass
class EncodeOnlyRequestOutput(RequestOutput):
    prompt_token_ids: List[int]
    outputs: torch.Tensor