import asyncio
from vllm import AsyncLLMEngine
import grpc

from vllm.grpc.server import UNIX_SOCKET
from .pb import generate_pb2_grpc, generate_pb2
from typing import AsyncIterator, List, Optional, Mapping

from vllm.inputs import PromptInputs
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.outputs import CompletionOutput
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams

import time

MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

class TextGenerationClient(AsyncLLMEngine):
    def __init__(self):
        # channel = grpc.aio.insecure_channel("localhost:5543")
        channel = grpc.aio.insecure_channel(UNIX_SOCKET)
        self.stub = generate_pb2_grpc.TextGenerationServiceStub(channel)
        self.engine_use_ray = False
        self.worker_use_ray = False
        self.log_requests = False
        self.engine = None

        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)

    @property
    def is_running(self) -> bool:
        return True
    
    @property
    def is_stopped(self) -> bool:
        return False

    @property
    def errored(self) -> bool:
        return False
    
    def start_background_loop(self):
        # TODO something lol
        pass
    
    async def get_tokenizer(
        self,
        lora_request: Optional[LoRARequest] = None,
    ) -> "PreTrainedTokenizer":
        # TODO: what to return :/
        return self.tokenizer

    async def generate(
        self,
        inputs: PromptInputs,
        sampling_params: SamplingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None
    ) -> AsyncIterator[RequestOutput]:
        
        
        prompt: str = inputs.get('prompt', "")
        prompt_token_ids: List[int] = inputs.get('prompt_token_ids', [])

        generate_stream = self.stub.Generate(
            generate_pb2.GenerateRequest(
                prompt_inputs=generate_pb2.PromptInputs(
                    prompt=prompt,
                    prompt_token_ids=prompt_token_ids,
                ),
                request_id=request_id,
            )
        )

        async for generate_response in generate_stream:
            completion_outputs = [
                CompletionOutput(
                    index=output.index,
                    text=output.text,
                    token_ids=output.token_ids,
                    cumulative_logprob=0.0,
                    logprobs=None,
                    finish_reason=output.finish_reason,
                ) for output in generate_response.outputs
            ]

            yield RequestOutput(
                request_id=request_id,
                prompt_token_ids=[],
                outputs=completion_outputs,
                finished=(completion_outputs[0].finish_reason != ""),
                prompt_logprobs=None,
                prompt=prompt,
            )

        # print(f"TTFT: {ttft}")
        # print(f"TPOT: {sum(tpots)/len(tpots)}")