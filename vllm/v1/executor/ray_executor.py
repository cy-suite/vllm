import os
from collections import defaultdict
from itertools import islice, repeat
from typing import (TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple,
                    Type)

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils import get_distributed_init_method, get_ip, get_open_port
from vllm.v1.executor.ray_utils import RayWorkerWrapper, ray
from vllm.v1.outputs import ModelRunnerOutput
from vllm.worker.worker_base import WorkerBase

if ray is not None:
    from ray.exceptions import RayChannelTimeoutError
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup

    from vllm.v1.core.scheduler import SchedulerOutput

logger = init_logger(__name__)


class RayExecutor:

    def __init__(self, vllm_config: VllmConfig) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.speculative_config = vllm_config.speculative_config
        self.prompt_adapter_config = vllm_config.prompt_adapter_config
        self.observability_config = vllm_config.observability_config
        self.compiled_dag_refs = []
        self._init_executor()

    def _init_executor(self) -> None:
        self.forward_dag: Optional[ray.dag.CompiledDAG] = None
        placement_group = self.parallel_config.placement_group

        # Disable Ray usage stats collection.
        ray_usage = os.environ.get("RAY_USAGE_STATS_ENABLED", "0")
        if ray_usage != "1":
            os.environ["RAY_USAGE_STATS_ENABLED"] = "0"

        # Create the parallel GPU workers.
        self._init_workers_ray(placement_group)

    def _init_workers_ray(self, placement_group: "PlacementGroup",
                          **ray_remote_kwargs):
        if (self.parallel_config.tensor_parallel_size == 1
                and self.parallel_config.pipeline_parallel_size == 1):
            # For single GPU case, we use a ray worker with constrained memory.
            num_gpus = self.cache_config.gpu_memory_utilization
        else:
            # Otherwise, the ray workers are allocated with a full GPU.
            num_gpus = 1

        # A list of workers to run a model.
        self.workers: List[RayWorkerWrapper] = []
        if self.parallel_config.ray_workers_use_nsight:
            ray_remote_kwargs = self._configure_ray_workers_use_nsight(
                ray_remote_kwargs)

        # Create the workers.
        driver_ip = get_ip()
        worker_wrapper_kwargs = self._get_worker_wrapper_args()
        for bundle_id, bundle in enumerate(placement_group.bundle_specs):
            if not bundle.get("GPU", 0):
                continue
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=bundle_id,
            )

            worker = ray.remote(
                num_cpus=0,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
                **ray_remote_kwargs,
            )(RayWorkerWrapper).remote(**worker_wrapper_kwargs)
            self.workers.append(worker)

        logger.debug("workers: %s", self.workers)
        worker_ips = [
            ray.get(worker.get_node_ip.remote())  # type: ignore[attr-defined]
            for worker in self.workers
        ]
        ip_counts: Dict[str, int] = {}
        for ip in worker_ips:
            ip_counts[ip] = ip_counts.get(ip, 0) + 1

        def sort_by_driver_then_worker_ip(worker):
            """
            Sort the workers based on 3 properties:
            1. If the worker is on the same node as the driver (vllm engine),
                it should be placed first.
            2. Then, if the worker is on a node with fewer workers, it should
                be placed first.
            3. Finally, if the work is on a node with smaller IP address, it
                should be placed first.
            """
            ip = ray.get(worker.get_node_ip.remote())
            return (ip != driver_ip, ip_counts[ip], ip)

        # After sorting, the workers on the same node will be
        # close to each other, and the workers on the driver
        # node will be placed first.
        self.workers = sorted(self.workers, key=sort_by_driver_then_worker_ip)

        # Get the set of GPU IDs used on each node.
        worker_node_and_gpu_ids = self._run_workers("get_node_and_gpu_ids")

        node_workers = defaultdict(list)  # node id -> list of worker ranks
        node_gpus = defaultdict(list)  # node id -> list of gpu ids

        for i, (node_id, gpu_ids) in enumerate(worker_node_and_gpu_ids):
            node_workers[node_id].append(i)
            # `gpu_ids` can be a list of strings or integers.
            # convert them to integers for consistency.
            # NOTE: gpu_ids can be larger than 9 (e.g. 16 GPUs),
            # string sorting is not sufficient.
            # see https://github.com/vllm-project/vllm/issues/5590
            gpu_ids = [int(x) for x in gpu_ids]
            node_gpus[node_id].extend(gpu_ids)

        for node_id, gpu_ids in node_gpus.items():
            node_gpus[node_id] = sorted(gpu_ids)

        all_ips = set(worker_ips)
        n_ips = len(all_ips)
        n_nodes = len(node_workers)

        if n_nodes != n_ips:
            raise RuntimeError(
                f"Every node should have a unique IP address. Got {n_nodes}"
                f" nodes with node ids {list(node_workers.keys())} and "
                f"{n_ips} unique IP addresses {all_ips}. Please check your"
                " network configuration. If you set `VLLM_HOST_IP` or "
                "`HOST_IP` environment variable, make sure it is unique for"
                " each node.")

        # VLLM_INSTANCE_ID = get_vllm_instance_id()

        # Set environment variables for the driver and workers.
        all_args_to_update_environment_variables = [
            (
                {
                    "CUDA_VISIBLE_DEVICES":
                    ",".join(map(str, node_gpus[node_id])),
                    # "VLLM_INSTANCE_ID":
                    # VLLM_INSTANCE_ID,
                    "VLLM_TRACE_FUNCTION":
                    str(envs.VLLM_TRACE_FUNCTION),
                    "VLLM_USE_V1":
                    str(int(envs.VLLM_USE_V1)),
                    **({
                        "VLLM_ATTENTION_BACKEND": envs.VLLM_ATTENTION_BACKEND
                    } if envs.VLLM_ATTENTION_BACKEND is not None else {})
                }, ) for (node_id, _) in worker_node_and_gpu_ids
        ]

        self._env_vars_for_all_workers = (
            all_args_to_update_environment_variables)

        self._run_workers("update_environment_variables",
                          all_args=self._get_env_vars_to_be_updated())

        if len(node_gpus) == 1:
            # in single node case, we don't need to get the IP address.
            # the loopback address is sufficient
            # NOTE: a node may have several IP addresses, one for each
            # network interface. `get_ip()` might return any of them,
            # while they might not work for communication inside the node
            # if the network setup is complicated. Using the loopback address
            # solves this issue, as it always works for communication inside
            # the node.
            driver_ip = "127.0.0.1"
        distributed_init_method = get_distributed_init_method(
            driver_ip, get_open_port())

        # Initialize the actual workers inside worker wrapper.
        init_worker_all_kwargs = [
            self._get_worker_kwargs(
                local_rank=node_workers[node_id].index(rank),
                rank=rank,
                distributed_init_method=distributed_init_method,
            ) for rank, (node_id, _) in enumerate(worker_node_and_gpu_ids)
        ]
        self._run_workers("init_worker", all_kwargs=init_worker_all_kwargs)
        self._run_workers("initialize")
        self._run_workers("load_model")

    def _get_env_vars_to_be_updated(self):
        return self._env_vars_for_all_workers

    def _get_worker_module_and_class(
            self) -> Tuple[str, str, Optional[Callable[[], Type[WorkerBase]]]]:
        worker_module_name = "vllm.v1.worker.gpu_worker"
        worker_class_name = "Worker"
        return worker_module_name, worker_class_name

    def _get_worker_kwargs(
            self,
            local_rank: int = 0,
            rank: int = 0,
            distributed_init_method: Optional[str] = None) -> Dict[str, Any]:
        """Return worker init args for a given rank."""
        if distributed_init_method is None:
            distributed_init_method = get_distributed_init_method(
                get_ip(), get_open_port())
        return dict(
            vllm_config=self.vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
        )

    def _get_worker_wrapper_args(self) -> Dict[str, Any]:
        worker_module_name, worker_class_name = (
            self._get_worker_module_and_class())

        return dict(
            worker_module_name=worker_module_name,
            worker_class_name=worker_class_name,
            trust_remote_code=self.model_config.trust_remote_code,
        )

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine the number of available KV blocks.
        This invokes `determine_num_available_blocks` on each worker and takes
        the min of the results, guaranteeing that the selected cache sizes are
        compatible with all workers.
        Returns:
            - tuple[num_gpu_blocks, num_cpu_blocks]
        """
        # Get the maximum number of blocks that can be allocated on GPU and CPU.
        num_blocks = self._run_workers("determine_num_available_blocks")

        # Since we use a shared centralized controller, we take the minimum
        # number of blocks across all workers to make sure all the memory
        # operators can be applied to all workers.
        num_gpu_blocks = min(b[0] for b in num_blocks)
        return num_gpu_blocks, 0

    def initialize(self, num_gpu_blocks: int) -> None:
        """Initialize the KV cache in all workers.
        """
        # NOTE: This is logged in the executor because there can be >1 worker
        # with other executors. We could log in the engine level, but work
        # remains to abstract away the device for non-GPU configurations.
        logger.info("# GPU blocks: %d", num_gpu_blocks)
        self._run_workers("initialize_cache", num_gpu_blocks)
        self._run_workers("compile_or_warm_up_model")

    def save_sharded_state(
        self,
        path: str,
        pattern: Optional[str] = None,
        max_size: Optional[int] = None,
    ) -> None:
        self._run_workers("save_sharded_state",
                          path=path,
                          pattern=pattern,
                          max_size=max_size)

    def _run_workers(
        self,
        method: str,
        *args,
        all_args: Optional[List[Tuple[Any, ...]]] = None,
        all_kwargs: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers. Can be used in the following
        ways:
        Args:
        - async_run_tensor_parallel_workers_only: If True the method will be
          run only in the remote TP workers, not the driver worker.
          It will also be run asynchronously and return a list of futures
          rather than blocking on the results.
        - args/kwargs: All workers share the same args/kwargs
        - all_args/all_kwargs: args/kwargs for each worker are specified
          individually
        """
        count = len(self.workers)
        all_worker_args = repeat(args, count) if all_args is None \
            else islice(all_args, 0, None)
        all_worker_kwargs = repeat(kwargs, count) if all_kwargs is None \
            else islice(all_kwargs, 0, None)

        # Start the ray workers first.
        ray_workers = self.workers
        ray_worker_outputs = [
            worker.execute_method.remote(method, *worker_args, **worker_kwargs)
            for (worker, worker_args, worker_kwargs
                 ) in zip(ray_workers, all_worker_args, all_worker_kwargs)
        ]

        # Get the results of the ray workers.
        if self.workers:
            ray_worker_outputs = ray.get(ray_worker_outputs)

        return ray_worker_outputs

    def execute_model(
        self,
        scheduler_output,
    ) -> ModelRunnerOutput:
        if self.forward_dag is None:
            self.forward_dag = self._compiled_ray_dag()
        # All workers are supposed to produce the same output. Only
        # get the first output.
        output = ray.get(self.forward_dag.execute(scheduler_output))[0]
        return output

    def execute_model_pipelined(
            self, scheduler_output
    ) -> List[Tuple[ModelRunnerOutput, SchedulerOutput]]:
        if self.forward_dag is None:
            self.forward_dag = self._compiled_ray_dag()

        outputs = []
        for dag_ref, schd_output in self.compiled_dag_refs:
            try:
                outputs.append((ray.get(dag_ref, timeout=0)[0], schd_output))
            except RayChannelTimeoutError:
                break
        self.compiled_dag_refs = self.compiled_dag_refs[len(outputs):]
        if len(self.compiled_dag_refs
               ) == self.parallel_config.pipeline_parallel_size:
            outputs.append((ray.get(self.compiled_dag_refs[0],
                                    timeout=-1)[0], scheduler_output))
        ref = self.forward_dag.execute(scheduler_output)
        self.compiled_dag_refs.append((ref, scheduler_output))
        return outputs

    def profile(self, is_start=True):
        raise NotImplementedError

    def shutdown(self):
        if hasattr(self, "forward_dag") and self.forward_dag is not None:
            self.forward_dag.teardown()
            import ray
            for worker in self.workers:
                ray.kill(worker)
            self.forward_dag = None

    def check_health(self) -> None:
        raise NotImplementedError

    def _check_ray_compiled_graph_installation(self):
        # TODO: We should check versions that support compiled graph.
        import importlib.util
        adag_spec = importlib.util.find_spec(
            "ray.experimental.compiled_dag_ref")
        if adag_spec is None:
            raise ValueError("Ray accelerated DAG is not installed. "
                             "Run `pip install ray[adag]` to install it.")

        cupy_spec = importlib.util.find_spec("cupy")
        if cupy_spec is None and envs.VLLM_USE_RAY_COMPILED_DAG_NCCL_CHANNEL:
            raise ValueError(
                "cupy is not installed but required since "
                "VLLM_USE_RAY_COMPILED_DAG_NCCL_CHANNEL is set."
                "Run `pip install ray[adag]` and check cupy installation.")

    def _compiled_ray_dag(self):
        assert self.parallel_config.use_ray
        self._check_ray_compiled_graph_installation()
        from ray.dag import InputNode, MultiOutputNode

        with InputNode() as input_batches:
            outputs = [
                worker.execute_model.bind(input_batches)
                for worker in self.workers
            ]
            forward_dag = MultiOutputNode(outputs)

        return forward_dag.experimental_compile()

    def __del__(self):
        self.shutdown()
