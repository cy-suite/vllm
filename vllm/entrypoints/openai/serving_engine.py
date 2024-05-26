import json
from dataclasses import dataclass
from http import HTTPStatus
from typing import Dict, Iterable, Iterator, List, Optional, TypedDict, Union

from pydantic import Field
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from typing_extensions import Annotated

from vllm.config import ModelConfig
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              CompletionRequest,
                                              EmbeddingRequest, ErrorResponse,
                                              LogProbs, ModelCard, ModelList,
                                              ModelPermission)
from vllm.inputs import parse_and_batch_prompt
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.sequence import Logprob
from vllm.transformers_utils.tokenizer import get_tokenizer

logger = init_logger(__name__)


class TextTokensPrompt(TypedDict):
    prompt: str
    prompt_token_ids: List[int]


@dataclass
class LoRAModulePath:
    name: str
    local_path: str


class OpenAIServing:

    def __init__(
        self,
        engine: AsyncLLMEngine,
        model_config: ModelConfig,
        served_model_names: List[str],
        lora_modules: Optional[List[LoRAModulePath]],
        *,
        log_requests: bool,
        max_log_len: Optional[int],
    ):
        self.engine = engine
        self.max_model_len = model_config.max_model_len

        # A separate tokenizer to map token IDs to strings.
        self.tokenizer = get_tokenizer(
            model_config.tokenizer,
            tokenizer_mode=model_config.tokenizer_mode,
            tokenizer_revision=model_config.tokenizer_revision,
            trust_remote_code=model_config.trust_remote_code,
            truncation_side="left")

        self.served_model_names = served_model_names

        if lora_modules is None:
            self.lora_requests = []
        else:
            self.lora_requests = [
                LoRARequest(
                    lora_name=lora.name,
                    lora_int_id=i,
                    lora_local_path=lora.local_path,
                ) for i, lora in enumerate(lora_modules, start=1)
            ]

        self.log_requests = log_requests
        self.max_log_len = max_log_len

    async def show_available_models(self) -> ModelList:
        """Show available models. Right now we only have one model."""
        model_cards = [
            ModelCard(id=served_model_name,
                      root=self.served_model_names[0],
                      permission=[ModelPermission()])
            for served_model_name in self.served_model_names
        ]
        lora_cards = [
            ModelCard(id=lora.lora_name,
                      root=self.served_model_names[0],
                      permission=[ModelPermission()])
            for lora in self.lora_requests
        ]
        model_cards.extend(lora_cards)
        return ModelList(data=model_cards)

    def _create_logprobs(
        self,
        token_ids: List[int],
        top_logprobs: List[Optional[Dict[int, Logprob]]],
        num_output_top_logprobs: Optional[int] = None,
        initial_text_offset: int = 0,
    ) -> LogProbs:
        """Create OpenAI-style logprobs."""
        logprobs = LogProbs()
        last_token_len = 0
        if num_output_top_logprobs:
            logprobs.top_logprobs = []

        for i, token_id in enumerate(token_ids):
            step_top_logprobs = top_logprobs[i]
            if step_top_logprobs is None:
                token = self.tokenizer.decode(token_id)
                logprobs.tokens.append(token)
                logprobs.token_logprobs.append(None)
                assert logprobs.top_logprobs is not None
                logprobs.top_logprobs.append(None)
            else:
                token_logprob = step_top_logprobs[token_id].logprob
                token = step_top_logprobs[token_id].decoded_token
                logprobs.tokens.append(token)
                token_logprob = max(token_logprob, -9999.0)
                logprobs.token_logprobs.append(token_logprob)

                if num_output_top_logprobs:
                    assert logprobs.top_logprobs is not None
                    logprobs.top_logprobs.append({
                        # Convert float("-inf") to the
                        # JSON-serializable float that OpenAI uses
                        p.decoded_token: max(p.logprob, -9999.0)
                        for i, p in step_top_logprobs.items()
                    } if step_top_logprobs else None)

            if len(logprobs.text_offset) == 0:
                logprobs.text_offset.append(initial_text_offset)
            else:
                logprobs.text_offset.append(logprobs.text_offset[-1] +
                                            last_token_len)
            last_token_len = len(token)
        return logprobs

    def create_error_response(
            self,
            message: str,
            err_type: str = "BadRequestError",
            status_code: HTTPStatus = HTTPStatus.BAD_REQUEST) -> ErrorResponse:
        return ErrorResponse(message=message,
                             type=err_type,
                             code=status_code.value)

    def create_streaming_error_response(
            self,
            message: str,
            err_type: str = "BadRequestError",
            status_code: HTTPStatus = HTTPStatus.BAD_REQUEST) -> str:
        json_str = json.dumps({
            "error":
            self.create_error_response(message=message,
                                       err_type=err_type,
                                       status_code=status_code).model_dump()
        })
        return json_str

    async def _check_model(
        self, request: Union[CompletionRequest, ChatCompletionRequest,
                             EmbeddingRequest]
    ) -> Optional[ErrorResponse]:
        if request.model in self.served_model_names:
            return None
        if request.model in [lora.lora_name for lora in self.lora_requests]:
            return None
        return self.create_error_response(
            message=f"The model `{request.model}` does not exist.",
            err_type="NotFoundError",
            status_code=HTTPStatus.NOT_FOUND)

    def _maybe_get_lora(
        self, request: Union[CompletionRequest, ChatCompletionRequest,
                             EmbeddingRequest]
    ) -> Optional[LoRARequest]:
        if request.model in self.served_model_names:
            return None
        for lora in self.lora_requests:
            if request.model == lora.lora_name:
                return lora
        # if _check_model has been called earlier, this will be unreachable
        raise ValueError(f"The model `{request.model}` does not exist.")

    def _normalize_prompt_text_to_input(
        self,
        request: Union[ChatCompletionRequest, CompletionRequest,
                       EmbeddingRequest],
        prompt: str,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]],
        add_special_tokens: bool,
    ) -> TextTokensPrompt:
        if truncate_prompt_tokens is None:
            encoded = tokenizer(prompt, add_special_tokens=add_special_tokens)
        else:
            encoded = tokenizer(prompt,
                                add_special_tokens=add_special_tokens,
                                truncation=True,
                                max_length=truncate_prompt_tokens)

        input_ids = encoded.input_ids

        input_text = prompt

        return self._validate_input(request, input_ids, input_text)

    def _normalize_prompt_tokens_to_input(
        self,
        request: Union[ChatCompletionRequest, CompletionRequest,
                       EmbeddingRequest],
        prompt_ids: List[int],
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]],
    ) -> TextTokensPrompt:
        if truncate_prompt_tokens is None:
            input_ids = prompt_ids
        else:
            input_ids = prompt_ids[-truncate_prompt_tokens:]

        input_text = tokenizer.decode(prompt_ids)

        return self._validate_input(request, input_ids, input_text)

    def _validate_input(
        self,
        request: Union[ChatCompletionRequest, CompletionRequest,
                       EmbeddingRequest],
        input_ids: List[int],
        input_text: str,
    ) -> TextTokensPrompt:
        token_num = len(input_ids)

        # Note: EmbeddingRequest doesn't have max_tokens
        if isinstance(request, EmbeddingRequest):
            if token_num > self.max_model_len:
                raise ValueError(
                    f"This model's maximum context length is "
                    f"{self.max_model_len} tokens. However, you requested "
                    f"{token_num} tokens in the input for embedding "
                    f"generation. Please reduce the length of the input.")
            return TextTokensPrompt(prompt=input_text,
                                    prompt_token_ids=input_ids)

        if request.max_tokens is None:
            if token_num >= self.max_model_len:
                raise ValueError(
                    f"This model's maximum context length is "
                    f"{self.max_model_len} tokens. However, you requested "
                    f"{token_num} tokens in the messages, "
                    f"Please reduce the length of the messages.")
            request.max_tokens = self.max_model_len - token_num

        if token_num + request.max_tokens > self.max_model_len:
            raise ValueError(
                f"This model's maximum context length is "
                f"{self.max_model_len} tokens. However, you requested "
                f"{request.max_tokens + token_num} tokens "
                f"({token_num} in the messages, "
                f"{request.max_tokens} in the completion). "
                f"Please reduce the length of the messages or completion.")

        return TextTokensPrompt(prompt=input_text, prompt_token_ids=input_ids)

    def _tokenize_prompt_input(
        self,
        request: Union[ChatCompletionRequest, CompletionRequest,
                       EmbeddingRequest],
        prompt_input: Union[str, List[int]],
        truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None,
        add_special_tokens: bool = True,
    ) -> TextTokensPrompt:
        """A simpler implementation of
        :meth:`~vllm.entrypoints.openai.serving_engine.OpenAIServing._tokenize_prompt_input_or_inputs`
        that assumes single input."""
        return next(
            self._tokenize_prompt_inputs(
                request,
                [prompt_input],
                truncate_prompt_tokens=truncate_prompt_tokens,
                add_special_tokens=add_special_tokens,
            ))

    def _tokenize_prompt_inputs(
        self,
        request: Union[ChatCompletionRequest, CompletionRequest,
                       EmbeddingRequest],
        prompt_inputs: Iterable[Union[str, List[int]]],
        truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None,
        add_special_tokens: bool = True,
    ) -> Iterator[TextTokensPrompt]:
        """A simpler implementation of
        :meth:`~vllm.entrypoints.openai.serving_engine.OpenAIServing._tokenize_prompt_input_or_inputs`
        that assumes multiple inputs."""
        tokenizer = self.tokenizer

        for text in prompt_inputs:
            if isinstance(text, str):
                yield self._normalize_prompt_text_to_input(
                    request,
                    prompt=text,
                    tokenizer=tokenizer,
                    truncate_prompt_tokens=truncate_prompt_tokens,
                    add_special_tokens=add_special_tokens,
                )
            else:
                yield self._normalize_prompt_tokens_to_input(
                    request,
                    prompt_ids=text,
                    tokenizer=tokenizer,
                    truncate_prompt_tokens=truncate_prompt_tokens,
                )

    def _tokenize_prompt_input_or_inputs(
        self,
        request: Union[ChatCompletionRequest, CompletionRequest,
                       EmbeddingRequest],
        input_or_inputs: Union[str, List[str], List[int], List[List[int]]],
        truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None,
        add_special_tokens: bool = True,
    ) -> Iterator[TextTokensPrompt]:
        """Tokenize/detokenize depending on the input format.

        According to `OpenAI API <https://platform.openai.com/docs/api-reference/embeddings/create>`_
        , each input can be a string or array of tokens. Note that each request
        can pass one or more inputs.
        """
        tokenizer = self.tokenizer

        for prompt_input in parse_and_batch_prompt(input_or_inputs):
            # Although our type checking is based on mypy,
            # VSCode Pyright extension should still work properly
            # "is True" is required for Pyright to perform type narrowing
            # See: https://github.com/microsoft/pyright/issues/7672
            if prompt_input["is_tokens"] is False:
                yield self._normalize_prompt_text_to_input(
                    request,
                    prompt=prompt_input["content"],
                    tokenizer=tokenizer,
                    truncate_prompt_tokens=truncate_prompt_tokens,
                    add_special_tokens=add_special_tokens,
                )
            else:
                yield self._normalize_prompt_tokens_to_input(
                    request,
                    prompt_ids=prompt_input["content"],
                    tokenizer=tokenizer,
                    truncate_prompt_tokens=truncate_prompt_tokens,
                )

    def _log_inputs(
        self,
        request_id: str,
        inputs: TextTokensPrompt,
        params: Union[SamplingParams, PoolingParams],
        lora_request: Optional[LoRARequest],
    ) -> None:
        if self.log_requests:
            if isinstance(inputs, str):
                shortened_prompt = inputs
                shortened_token_ids = None
            else:
                shortened_prompt = inputs.get("prompt")
                shortened_token_ids = inputs.get("prompt_token_ids")

            max_log_len = self.max_log_len
            if max_log_len is not None:
                if shortened_prompt is not None:
                    shortened_prompt = shortened_prompt[:max_log_len]
                if shortened_token_ids is not None:
                    shortened_token_ids = shortened_token_ids[:max_log_len]

            logger.info(
                "Received request %s: prompt: %r, "
                "params: %s, prompt_token_ids: %s, "
                "lora_request: %s.", request_id, shortened_prompt, params,
                shortened_token_ids, lora_request)
