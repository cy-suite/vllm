from vllm.entrypoints.openai.protocol import EngineMetrics
from vllm.logger import init_logger
import vllm.entrypoints.openai.orca_load_report_pb2 as orca_load_report_pb2
import base64
from typing import Optional, Tuple, List, Mapping
import json

logger = init_logger(__name__)


def create_orca_header(
    format: str, named_metrics: List[Tuple[str, float]], metadata_fields=None
) -> Optional[Mapping[str, str]]:
    """
    Creates ORCA headers named 'endpoint-load-metrics' in the specified format and adds custom metrics to named_metrics.
    ORCA headers format description: https://docs.google.com/document/d/1C1ybMmDKJIVlrbOLbywhu9iRYo4rilR-cT50OTtOFTs/edit?tab=t.0
    ORCA proto https://github.com/cncf/xds/blob/main/xds/data/orca/v3/orca_load_report.proto

    Parameters:
    - format (str): The format of the header ('BIN', 'TEXT', 'JSON').
    - named_metrics (List[Tuple[str, float]]): List of tuples with metric names and their corresponding double values.
    - metadata_fields (list): List of additional metadata fields (currently unsupported).

    Returns:
    - Optional[Mapping[str,str]]: A dictionary with header key as 'endpoint-load-metrics' and values as the ORCA header strings with format prefix and data in  with named_metrics in.
    """

    if metadata_fields:
        logger.warning(
            "Warning: `metadata_fields` are not supported in the ORCA response header yet."
        )

    if format not in ["BIN", "TEXT", "JSON"]:
        logger.warning(
            "Warning: `%s` format is not supported in the ORCA response header",
            format,
        )
        return

    header = {}
    orca_report = {
        "named_metrics": {
            metric_name: value
            for metric_name, value in named_metrics
            if isinstance(metric_name, str) and isinstance(value, float)
        }
    }
    # output example:
    # endpoint-load-metrics: BIN
    # CZqZmZmZmbk/MQAAAAAAAABAQg4KA2ZvbxGamZmZmZm5P0IOCgNiYXIRmpmZmZmZyT8=
    if format == "BIN":
        load_proto = orca_load_report_pb2.OrcaLoadReport()
        for metric_name, value in named_metrics:
            if isinstance(metric_name,str) and isinstance(value, float):
                load_proto.named_metrics[metric_name] = value
        binary_data = load_proto.SerializeToString()
        base64_encoded_data = base64.b64encode(binary_data).decode("utf-8")
        header["endpoint-load-metrics"] = f"BIN {base64_encoded_data}"

    # output example:
    # endpoint-load-metrics: TEXT named_metrics.kv_cache_utilization=0.4
    elif format == "TEXT":
        native_http_header = ", ".join(
            [
                f"named_metrics.{metric_name}={value}"
                for metric_name, value in named_metrics
                if isinstance(metric_name, str) and isinstance(value, float)
            ]
        )
        header["endpoint-load-metrics"] = f"TEXT {native_http_header}"

    # output example:
    # endpoint-load-metrics: JSON “named_metrics”: {“custom-metric-util”: 0.4}
    elif format == "JSON":
        header["endpoint-load-metrics"] = f"JSON {json.dumps(orca_report)}"

    return header


def metrics_header(m: EngineMetrics) -> Optional[Mapping[str, str]]:
    named_metrics = []
    for metric, val in vars(m).items():
        if isinstance(val, float) and metric != "format":
            named_metrics.append([metric, val])
    return create_orca_header(m.format, named_metrics)
