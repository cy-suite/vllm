import asyncio
from typing import AsyncGenerator, Dict, List, Mapping, Optional, Type, Union

from vllm.config import ModelConfig, VllmConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.metrics_types import StatLoggerBase, Stats
from vllm.engine.protocol import EngineClient
from vllm.envs import VLLM_STATS_ENGINE_POLLING_INTERVAL_S
from vllm.inputs import INPUT_REGISTRY, InputRegistry, PromptType
from vllm.inputs.preprocess import InputPreprocessor
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.outputs import EmbeddingRequestOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.transformers_utils.tokenizer_group import init_tokenizer_from_configs
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.engine.async_stream import AsyncStream
from vllm.v1.engine.core_client import EngineCoreClient
from vllm.v1.engine.detokenizer import Detokenizer
from vllm.v1.engine.processor import Processor
from vllm.v1.executor.gpu_executor import GPUExecutor
from vllm.v1.stats.stats_manager import (
    NoopEngineStatsManager,
    ThreadSafeEngineStatsManager,
)

logger = init_logger(__name__)


class AsyncLLM(EngineClient):

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: Type[GPUExecutor],
        log_stats: bool,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[Dict[str, StatLoggerBase]] = None,
        input_registry: InputRegistry = INPUT_REGISTRY,
        use_cached_outputs: bool = False,
        log_requests: bool = True,
        start_engine_loop: bool = True,
    ) -> None:
        assert start_engine_loop

        self.log_requests = log_requests
        self.log_stats = vllm_config.observability_config.log_stats
        self.stat_loggers = stat_loggers
        self.model_config = vllm_config.model_config

        # Tokenizer (+ ensure liveness if running in another process).
        self.tokenizer = init_tokenizer_from_configs(
            model_config=vllm_config.model_config,
            scheduler_config=vllm_config.scheduler_config,
            parallel_config=vllm_config.parallel_config,
            enable_lora=bool(vllm_config.lora_config))
        self.tokenizer.ping()

        # Request streams (map of request_id -> AsyncStream).
        self.request_streams: Dict[str, AsyncStream] = {}
        # List of cancelled request ids to be aborted.
        self.client_aborted_requests: List[str] = []

        # Processor (converts Inputs --> EngineCoreRequests).
        self.processor = Processor(vllm_config.model_config,
                                   vllm_config.lora_config, self.tokenizer,
                                   input_registry)

        # Detokenizer (converts EngineCoreOutputs --> RequestOutput).
        self.detokenizer = Detokenizer(vllm_config.model_config.tokenizer)

        # EngineCore (starts the engine in background process).
        self.engine_core = EngineCoreClient.make_client(
            vllm_config=vllm_config,
            executor_class=executor_class,
            usage_context=usage_context,
            multiprocess_mode=True,
            asyncio_mode=True,
            log_stats=self.log_stats,
        )

        # Async tasks that run in the background.
        # We don't initialize the handlers until we have a request so
        # they are None first.
        self.output_handler: Optional[asyncio.Task] = None
        self.stats_handler: Optional[asyncio.Task] = None

        if vllm_config.observability_config.log_stats:
            # The stats manager will be invoked from different threads:
            # 1. stats_handler: a background busy event loop that pulls stats
            # from the EngineCore periodically. The stats logging also happens
            # on the background thread.
            # 2. the main event loop when a request is added (and input
            # processed).
            self.stats_manager = ThreadSafeEngineStatsManager(
                vllm_config, stat_loggers
            )
        else:
            self.stats_manager = NoopEngineStatsManager()

    def __del__(self):
        self.shutdown()

    @classmethod
    def from_engine_args(
        cls,
        engine_args: AsyncEngineArgs,
        engine_config: Optional[VllmConfig] = None,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[Dict[str, StatLoggerBase]] = None,
    ) -> "AsyncLLMEngine":
        """Create an AsyncLLM from the EngineArgs."""

        # Create the engine configs.
        if engine_config is None:
            vllm_config = engine_args.create_engine_config(usage_context)
        else:
            vllm_config = engine_config

        executor_class = cls._get_executor_cls(vllm_config)

        # Create the AsyncLLM.
        return cls(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_requests=not engine_args.disable_log_requests,
            log_stats=not engine_args.disable_log_stats,
            start_engine_loop=start_engine_loop,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
        )

    def shutdown(self):
        """Shutdown, cleaning up the background proc and IPC."""

        self.engine_core.shutdown()

        if handler := getattr(self, "output_handler", None):
            handler.cancel()

        if handler := getattr(self, "stats_handler", None):
            handler.cancel()

    @classmethod
    def _get_executor_cls(cls, vllm_config: VllmConfig):
        return GPUExecutor

    async def add_request(
        self,
        request_id: str,
        prompt: PromptType,
        params: Union[SamplingParams, PoolingParams],
        arrival_time: Optional[float] = None,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
    ) -> AsyncGenerator[Union[RequestOutput, EmbeddingRequestOutput], None]:
        """Add new request to the AsyncLLM."""

        if self.detokenizer.is_request_active(request_id):
            raise KeyError(f"Request {request_id} already exists.")

        # 1) Create a new AsyncStream for the request.
        stream = self._add_request_to_streams(request_id)

        # 2) Convert input --> DetokenizerRequest / EngineCoreRequest.
        detokenizer_req, engine_core_req = self.processor.process_inputs(
            request_id, prompt, params, arrival_time, lora_request,
            trace_headers, prompt_adapter_request, priority)
        self.stats_manager.record_engine_input(engine_core_req)

        # 3) Add the request to Detokenizer (this process).
        self.detokenizer.add_request(detokenizer_req)

        # 4) Add the EngineCoreRequest to EngineCore (separate process).
        await self.engine_core.add_request_async(engine_core_req)

        # 5) Return the generator.
        return stream.generator()

    # TODO: we should support multiple prompts in one call, as you
    # can do with LLM.generate. So that for multi-prompt completion
    # requests we don't need to send multiple messages to core proc,
    # and so we don't need multiple streams which then get
    # re-multiplexed in the API server anyhow.
    async def generate(
        self,
        prompt: PromptType,
        sampling_params: SamplingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
    ) -> AsyncGenerator[RequestOutput, None]:
        """
        Main function called by the API server to kick off a request
            * 1) Making an AsyncStream corresponding to the Request.
            # 2) Processing the Input.
            * 3) Adding the Request to the Detokenizer.
            * 4) Adding the Request to the EngineCore (separate process).

        A separate output_handler loop runs in a background AsyncIO task, 
        pulling outputs from EngineCore and putting them into the 
        per-request AsyncStream.

        The caller of generate() iterates the returned AsyncGenerator,
        returning the RequestOutput back to the caller.
        """

        # We start the output_handler on the first call to generate() so that
        # we can call __init__ before the event loop starts, which enables us
        # to handle startup failure gracefully in the OpenAI server.
        if self.output_handler is None:
            self.output_handler = asyncio.create_task(
                self._run_output_handler())

        if self.stats_handler is None and self.log_stats:
            self.stats_handler = asyncio.create_task(self._run_stats_handler())

        async for output in await self.add_request(
                request_id,
                prompt,
                sampling_params,
                lora_request=lora_request,
                trace_headers=trace_headers,
                prompt_adapter_request=prompt_adapter_request,
                priority=priority,
        ):
            yield output

    def _finish_stream(self, request_id: str):
        stream = self.request_streams.pop(request_id, None)
        if stream is not None:
            stream.finish()

    def _add_request_to_streams(
        self,
        request_id: str,
    ) -> AsyncStream:

        if request_id in self.request_streams:
            raise ValueError(f"Request id {request_id} already running.")

        # Add the request to the stats manager to start tracking.
        self.stats_manager.add_request(request_id)

        # Avoid streams having circular ref to parent AsyncLLM object.
        aborted_reqs = self.client_aborted_requests
        stream = AsyncStream(request_id, aborted_reqs.append)
        self.request_streams[request_id] = stream

        if self.log_requests:
            logger.info("Added request %s.", request_id)

        return stream

    async def _process_cancellations(self) -> None:
        """
        Process requests cancelled from user disconnecting.

        When a client disconnects, AsyncStream._cancel() is called.
        We passed a callback to AsyncStream(), which appends to 
        self.client_aborted_requests.

        As a result, if any requests are canceled from the user side
        the request_id will show up in self.client_aborted_requests.
        """

        # Avoid streams having circular ref to parent AsyncLLM object.
        if not self.client_aborted_requests:
            return
        reqs_to_abort = self.client_aborted_requests.copy()
        self.client_aborted_requests.clear()

        # Remove from Detokenizer.
        self.detokenizer.abort_requests(reqs_to_abort)

        # Remove from RequestStreams.
        for request_id in reqs_to_abort:
            if self.log_requests:
                logger.info("User-cancelled request %s.", request_id)
            self._finish_stream(request_id)

        # Remove from EngineCore.
        await self.engine_core.abort_requests_async(reqs_to_abort)

    def _process_request_outputs(self, request_outputs: List[RequestOutput]):
        """Process outputs by putting them into per-request AsyncStreams."""

        for request_output in request_outputs:
            request_id = request_output.request_id
            assert request_id in self.request_streams
            self.stats_manager.record_request_output(request_output)

            # Each request in the API server pulls from the per-request stream.
            stream = self.request_streams.get(request_id)
            if stream is not None:
                stream.put(request_output)

                # If finished, remove from the tracker.
                if request_output.finished:
                    if self.log_requests:
                        logger.info("Finished request %s.", request_id)
                    self._finish_stream(request_id)

    async def _run_output_handler(self):
        """Background loop: pulls from EngineCore and pushes to AsyncStreams."""

        try:
            while True:
                # 1) Pull EngineCoreOutput from the EngineCore.
                outputs: EngineCoreOutputs = (
                    await self.engine_core.get_output_async())
                for output in outputs.outputs:
                    self.stats_manager.record_engine_output(output)

                # 2) Detokenize based on the output.
                request_outputs, reqs_to_abort = self.detokenizer.step(
                    outputs.outputs)

                # 3) Put the RequestOutputs into the per-request AsyncStreams.
                self._process_request_outputs(request_outputs)

                # 4) Abort any requests that finished due to stop strings.
                await self.engine_core.abort_requests_async(reqs_to_abort)

                # 5) Abort any requests due to client cancellations.
                await self._process_cancellations()

        except asyncio.CancelledError:
            logger.info("Engine shutting down.")
            self.shutdown()
        except BaseException as e:
            logger.error(e)
            raise e

    async def _run_stats_handler(self):
        while True:
            try:
                # Pull the stats from the EngineCore every X seconds.
                await asyncio.sleep(VLLM_STATS_ENGINE_POLLING_INTERVAL_S)

                # Pull the stats from the EngineCore before finalizing the
                # snapshot for an updated view of the current stats.
                stats_snapshot_from_engine = (
                    await self.engine_core.poll_stats_async()
                )

                # Make the stats.
                stats = self.stats_manager.make_stats(
                    stats_snapshot_from_engine
                )
                self.stats_manager.log_stats(stats)
            except Exception:
                logger.exception(
                    "Error in stats handler. Suppressing exceptions and "
                    "exiting stats handler. Please file an issue at "
                    "https://github.com/vllm-project/vllm/issues/new/choose"
                )
                continue

    def _log_stats(self, stats: Stats):
        for stat_logger in self.stat_loggers.values():
            # TODO(rickyx): we here assume loggers are lightweight and
            # non-blocking. To make this more robust, we should really
            # have an async logger interface, which implements actual
            # logging that could be cpu-heavy in a separate process
            # to minimize the latency impact on the frontend engine's
            # event loop.
            stat_logger.log(stats)

    async def abort(self, request_id: str) -> None:
        # Note: Who Calls this? I dont think this is actually used.
        raise ValueError("Not Supported on V1 yet.")

    def encode(
        self,
        prompt: PromptType,
        pooling_params: PoolingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        priority: int = 0,
    ):
        raise ValueError("Not Supported on V1 yet.")

    async def get_model_config(self) -> ModelConfig:
        return self.model_config

    async def get_decoding_config(self):
        raise ValueError("Not Supported on V1 yet.")

    async def get_input_preprocessor(self) -> InputPreprocessor:
        return self.processor.input_preprocessor

    async def get_tokenizer(
        self,
        lora_request: Optional[LoRARequest] = None,
    ) -> AnyTokenizer:
        assert lora_request is None
        return self.detokenizer.tokenizer

    async def is_tracing_enabled(self) -> bool:
        return False

    async def do_log_stats(
        self,
        scheduler_outputs=None,
        model_output=None,
    ) -> None:
        raise NotImplementedError(
            "V1 stats logging should not be called by user. "
            "The engine client handles logging internally.")

    async def check_health(self) -> None:
        logger.debug("Called check_health.")

    async def start_profile(self) -> None:
        await self.engine_core.profile(True)

    async def stop_profile(self) -> None:
        await self.engine_core.profile(False)

    @property
    def is_running(self) -> bool:
        return True

    @property
    def is_stopped(self) -> bool:
        return False

    @property
    def errored(self) -> bool:
        return False

    @property
    def dead_error(self) -> BaseException:
        return Exception


# Retain V0 name for backwards compatibility.
AsyncLLMEngine = AsyncLLM
