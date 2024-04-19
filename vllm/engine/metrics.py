import time
from dataclasses import dataclass
from typing import Counter as CollectionsCounter
from typing import Dict, List, Protocol

import numpy as np
from prometheus_client import (REGISTRY, Counter, Gauge, Histogram, Info,
                               disable_created_metrics)

from vllm.logger import init_logger

logger = init_logger(__name__)

disable_created_metrics()

# The begin-* and end* here are used by the documentation generator
# to extract the metrics definitions.


# begin-metrics-definitions
class Metrics:
    labelname_finish_reason = "finished_reason"

    def __init__(self, labelnames: List[str], max_model_len: int):
        # Unregister any existing vLLM collectors
        for collector in list(REGISTRY._collector_to_names):
            if hasattr(collector, "_name") and "vllm" in collector._name:
                REGISTRY.unregister(collector)

        # Config Information
        self.info_cache_config = Info(
            name='vllm:cache_config',
            documentation='information of cache_config')

        # System stats
        self.gauge_scheduler_running = Gauge(
            name="vllm:num_requests_running",
            documentation="Number of requests currently running on GPU.",
            labelnames=labelnames)
        self.gauge_scheduler_swapped = Gauge(
            name="vllm:num_requests_swapped",
            documentation="Number of requests swapped to CPU.",
            labelnames=labelnames)
        self.gauge_scheduler_waiting = Gauge(
            name="vllm:num_requests_waiting",
            documentation="Number of requests waiting to be processed.",
            labelnames=labelnames)
        self.gauge_gpu_cache_usage = Gauge(
            name="vllm:gpu_cache_usage_perc",
            documentation="GPU KV-cache usage. 1 means 100 percent usage.",
            labelnames=labelnames)
        self.gauge_cpu_cache_usage = Gauge(
            name="vllm:cpu_cache_usage_perc",
            documentation="CPU KV-cache usage. 1 means 100 percent usage.",
            labelnames=labelnames)

        # Raw stats from last model iteration
        self.counter_request_success = Counter(
            name="vllm:request_success",
            documentation="Count of successfully processed requests.",
            labelnames=labelnames + [Metrics.labelname_finish_reason])
        self.histogram_request_prompt_tokens = Histogram(
            name="vllm:request_prompt_tokens",
            documentation="Number of prefill tokens processed.",
            labelnames=labelnames,
            buckets=build_1_2_5_buckets(max_model_len),
        )
        self.histogram_request_generation_tokens = Histogram(
            name="vllm:request_generation_tokens",
            documentation="Number of generation tokens processed.",
            labelnames=labelnames,
            buckets=build_1_2_5_buckets(max_model_len),
        )
        self.histogram_time_to_first_token = Histogram(
            name="vllm:time_to_first_token_seconds",
            documentation="Histogram of time to first token in seconds.",
            labelnames=labelnames,
            buckets=[
                0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5,
                0.75, 1.0, 2.5, 5.0, 7.5, 10.0
            ])
        self.histogram_time_per_output_token = Histogram(
            name="vllm:time_per_output_token_seconds",
            documentation="Histogram of time per output token in seconds.",
            labelnames=labelnames,
            buckets=[
                0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75,
                1.0, 2.5
            ])
        self.histogram_e2e_request_latency = Histogram(
            name="vllm:e2e_request_latency_seconds",
            documentation="Histogram of end to end request latency in seconds.",
            labelnames=labelnames,
            buckets=[1.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        self.histogram_request_n = Histogram(
            name="vllm:request_params_n",
            documentation="Histogram of the n request parameter.",
            labelnames=labelnames,
            buckets=[1, 2, 5, 10, 20],
        )
        self.histogram_request_best_of = Histogram(
            name="vllm:request_params_best_of",
            documentation="Histogram of the best_of request parameter.",
            labelnames=labelnames,
            buckets=[1, 2, 5, 10, 20],
        )

        # Deprecated in favor of vllm:request_prompt_tokens_sum
        self.counter_prompt_tokens = Counter(
            name="vllm:prompt_tokens_total",
            documentation="Number of prefill tokens processed.",
            labelnames=labelnames)
        # Deprecated in favor of vllm:request_generation_tokens_sum
        self.counter_generation_tokens = Counter(
            name="vllm:generation_tokens_total",
            documentation="Number of generation tokens processed.",
            labelnames=labelnames)
        # Deprecated in favor of vllm:prompt_tokens_total
        self.gauge_avg_prompt_throughput = Gauge(
            name="vllm:avg_prompt_throughput_toks_per_s",
            documentation="Average prefill throughput in tokens/s.",
            labelnames=labelnames,
        )
        # Deprecated in favor of vllm:generation_tokens_total
        self.gauge_avg_generation_throughput = Gauge(
            name="vllm:avg_generation_throughput_toks_per_s",
            documentation="Average generation throughput in tokens/s.",
            labelnames=labelnames,
        )


# end-metrics-definitions


def build_1_2_5_buckets(max_value: int):
    """
    Builds a list of buckets with increasing powers of 10 multiplied by 
    mantissa values (1, 2, 5) until the value exceeds the specified maximum.

    Example:
    >>> build_1_2_5_buckets(100)
    [1, 2, 5, 10, 20, 50, 100]
    """
    mantissa_lst = [1, 2, 5]
    exponent = 0
    buckets = []
    while True:
        for m in mantissa_lst:
            value = m * 10**exponent
            if value <= max_value:
                buckets.append(value)
            else:
                return buckets
        exponent += 1


@dataclass
class Stats:
    """Created by LLMEngine for use by StatLogger."""
    now: float

    # System stats.
    num_running: int
    num_waiting: int
    num_swapped: int
    gpu_cache_usage: float
    cpu_cache_usage: float

    # Raw stats from last model iteration.
    finished_reason_lst: List[str]
    num_prompt_tokens_lst: List[int]
    num_generation_tokens_lst: List[int]
    request_n: List[int]
    request_best_of: List[int]
    time_to_first_tokens: List[float]
    time_per_output_tokens: List[float]
    time_e2e_requests: List[float]


class SupportsMetricsInfo(Protocol):

    def metrics_info(self) -> Dict[str, str]:
        ...


class StatLogger:
    """StatLogger is used LLMEngine to log to Promethus and Stdout."""

    def __init__(self, local_interval: float, labels: Dict[str, str],
                 max_model_len: int) -> None:
        # Metadata for logging locally.
        self.last_local_log = time.time()
        self.local_interval = local_interval

        # Tracked stats over current local logging interval.
        self.num_prompt_tokens: List[int] = []
        self.num_generation_tokens: List[int] = []

        # Prometheus metrics
        self.labels = labels
        self.metrics = Metrics(labelnames=list(labels.keys()),
                               max_model_len=max_model_len)

    def info(self, type: str, obj: SupportsMetricsInfo) -> None:
        if type == "cache_config":
            self.metrics.info_cache_config.info(obj.metrics_info())

    def _get_throughput(self, tracked_stats: List[int], now: float) -> float:
        return float(np.sum(tracked_stats) / (now - self.last_local_log))

    def _local_interval_elapsed(self, now: float) -> bool:
        elapsed_time = now - self.last_local_log
        return elapsed_time > self.local_interval

    def _log_prometheus(self, stats: Stats) -> None:
        # Set system stat gauges.
        self.metrics.gauge_scheduler_running.labels(**self.labels).set(
            stats.num_running)
        self.metrics.gauge_scheduler_swapped.labels(**self.labels).set(
            stats.num_swapped)
        self.metrics.gauge_scheduler_waiting.labels(**self.labels).set(
            stats.num_waiting)
        self.metrics.gauge_gpu_cache_usage.labels(**self.labels).set(
            stats.gpu_cache_usage)
        self.metrics.gauge_cpu_cache_usage.labels(**self.labels).set(
            stats.cpu_cache_usage)

        # Add to token counters.
        self.metrics.counter_prompt_tokens.labels(**self.labels).inc(
            sum(stats.num_prompt_tokens_lst))
        self.metrics.counter_generation_tokens.labels(**self.labels).inc(
            sum(stats.num_generation_tokens_lst))

        # Add to request counters.
        finished_reason_counter = CollectionsCounter(stats.finished_reason_lst)
        for finished_reason, count in finished_reason_counter.items():
            self.metrics.counter_request_success.labels(**{
                **self.labels,
                Metrics.labelname_finish_reason:
                finished_reason,
            }).inc(count)

        # Observe number of tokens in histograms.
        for val in stats.num_prompt_tokens_lst:
            self.metrics.histogram_request_prompt_tokens.labels(
                **self.labels).observe(val)
        for val in stats.num_generation_tokens_lst:
            self.metrics.histogram_request_generation_tokens.labels(
                **self.labels).observe(val)

        # Observe sampling params in histograms.
        for n in stats.request_n:
            self.metrics.histogram_request_n.labels(**self.labels).observe(n)
        for best_of in stats.request_best_of:
            self.metrics.histogram_request_best_of.labels(
                **self.labels).observe(best_of)

        # Observe request level latencies in histograms.
        for ttft in stats.time_to_first_tokens:
            self.metrics.histogram_time_to_first_token.labels(
                **self.labels).observe(ttft)
        for tpot in stats.time_per_output_tokens:
            self.metrics.histogram_time_per_output_token.labels(
                **self.labels).observe(tpot)
        for e2e in stats.time_e2e_requests:
            self.metrics.histogram_e2e_request_latency.labels(
                **self.labels).observe(e2e)

    def _log_prometheus_interval(self, prompt_throughput: float,
                                 generation_throughput: float) -> None:
        # Logs metrics to prometheus that are computed every logging_interval.
        # Support legacy gauge metrics that make throughput calculations on
        # the vLLM side. Moving forward, we should use counters like
        # counter_prompt_tokens, counter_generation_tokens
        # Which log raw data and calculate summaries using rate() on the
        # grafana/prometheus side. See
        # https://github.com/vllm-project/vllm/pull/2316#discussion_r1464204666
        self.metrics.gauge_avg_prompt_throughput.labels(
            **self.labels).set(prompt_throughput)
        self.metrics.gauge_avg_generation_throughput.labels(
            **self.labels).set(generation_throughput)

    def log(self, stats: Stats) -> None:
        """Called by LLMEngine.
           Logs to prometheus and tracked stats every iteration.
           Logs to Stdout every self.local_interval seconds."""

        # Log to prometheus.
        self._log_prometheus(stats)

        # Save tracked stats for token counters.
        self.num_prompt_tokens.append(sum(stats.num_prompt_tokens_lst))
        self.num_generation_tokens.append(sum(stats.num_generation_tokens_lst))

        # Log locally every local_interval seconds.
        if self._local_interval_elapsed(stats.now):
            # Compute summary metrics for tracked stats (and log them
            # to promethus if applicable).
            prompt_throughput = self._get_throughput(self.num_prompt_tokens,
                                                     now=stats.now)
            generation_throughput = self._get_throughput(
                self.num_generation_tokens, now=stats.now)
            self._log_prometheus_interval(
                prompt_throughput=prompt_throughput,
                generation_throughput=generation_throughput)

            # Log to stdout.
            logger.info(
                f"Avg prompt throughput: {prompt_throughput:.1f} tokens/s, "
                f"Avg generation throughput: "
                f"{generation_throughput:.1f} tokens/s, "
                f"Running: {stats.num_running} reqs, "
                f"Swapped: {stats.num_swapped} reqs, "
                f"Pending: {stats.num_waiting} reqs, "
                f"GPU KV cache usage: {stats.gpu_cache_usage * 100:.1f}%, "
                f"CPU KV cache usage: {stats.cpu_cache_usage * 100:.1f}%")

            # Reset tracked stats for next interval.
            self.num_prompt_tokens = []
            self.num_generation_tokens = []
            self.last_local_log = stats.now
