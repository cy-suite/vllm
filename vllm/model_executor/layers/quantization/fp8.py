from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.quantization.gptq_marlin import (
    GPTQ_MARLIN_MAX_PARALLEL, GPTQ_MARLIN_MIN_THREAD_N, GPTQMarlinState,
    marlin_permute_scales)
from vllm.model_executor.utils import set_weight_attrs
from vllm.utils import print_warning_once

ACTIVATION_SCHEMES = ["static", "dynamic"]

logger = init_logger(__name__)


def cutlass_fp8_supported() -> bool:
    capability = torch.cuda.get_device_capability()
    capability = capability[0] * 10 + capability[1]

    return ops.cutlass_scaled_mm_supports_fp8(capability)


class Fp8Config(QuantizationConfig):
    """Config class for FP8."""

    def __init__(
        self,
        is_checkpoint_fp8_serialized: bool = False,
        activation_scheme: str = "dynamic",
    ) -> None:
        self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized
        if is_checkpoint_fp8_serialized:
            logger.warning("Detected fp8 checkpoint. Please note that the "
                           "format is experimental and subject to change.")
        if activation_scheme not in ACTIVATION_SCHEMES:
            raise ValueError(
                f"Unsupported activation scheme {activation_scheme}")
        self.activation_scheme = activation_scheme

    @classmethod
    def get_name(cls) -> str:
        return "fp8"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Fp8Config":
        quant_method = cls.get_from_keys(config, ["quant_method"])
        is_checkpoint_fp8_serialized = ("fp8" in quant_method)
        activation_scheme = cls.get_from_keys(config, ["activation_scheme"])
        return cls(is_checkpoint_fp8_serialized=is_checkpoint_fp8_serialized,
                   activation_scheme=activation_scheme)

    def get_quant_method(
            self, layer: torch.nn.Module) -> Optional["QuantizeMethodBase"]:
        from vllm.attention.layer import Attention  # Avoid circular import

        if isinstance(layer, LinearBase):
            return Fp8LinearMethod(self)
        if isinstance(layer, Attention):
            return Fp8KVCacheMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class Fp8LinearMethod(LinearMethodBase):
    """Linear method for FP8.
    Supports loading FP8 checkpoints with static weight scale and
    dynamic/static activation scale.

    Also supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.

    Limitations:
    1. Only support per-tensor quantization due to torch._scaled_mm support.
    2. Only support float8_e4m3fn data type due to the limitation of
       torch._scaled_mm (https://github.com/pytorch/pytorch/blob/2e48b39603411a41c5025efbe52f89560b827825/aten/src/ATen/native/cuda/Blas.cpp#L854-L856)

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: Fp8Config):
        self.quant_config = quant_config
        self.cutlass_fp8_supported = cutlass_fp8_supported()

        capability = torch.cuda.get_device_capability()
        capability = capability[0] * 10 + capability[1]
        # self.use_marlin = capability < 89
        self.use_marlin = True

    def _create_scale_param(
        self,
        scale_name: str,
        layer: torch.nn.Module,
        output_partition_sizes: List[int],
        **extra_weight_attrs,
    ) -> None:
        scale = Parameter(torch.empty(len(output_partition_sizes),
                                      dtype=torch.float32),
                          requires_grad=False)
        layer.register_parameter(scale_name, scale)
        set_weight_attrs(
            scale, {
                **extra_weight_attrs,
                "fp8_scales_shard_indexer":
                self.scales_shard_indexer,
            })

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del input_size, output_size
        output_size_per_partition = sum(output_partition_sizes)

        layer.process_after_load = True
        layer.logical_widths = output_partition_sizes

        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        # WEIGHT
        weight_dtype = (torch.float8_e4m3fn
                        if self.quant_config.is_checkpoint_fp8_serialized else
                        params_dtype)
        weight = Parameter(torch.empty(output_size_per_partition,
                                       input_size_per_partition,
                                       dtype=weight_dtype),
                           requires_grad=False)
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, {
            **extra_weight_attrs,
            "input_dim": 1,
            "output_dim": 0,
        })

        # If checkpoint is serialized fp8, load them.
        # Otherwise, wait until process_weights_after_loading.
        if self.quant_config.is_checkpoint_fp8_serialized:
            # WEIGHT SCALE
            self._create_scale_param(
                scale_name="weight_scale",
                layer=layer,
                output_partition_sizes=output_partition_sizes,
                **extra_weight_attrs)

            # INPUT ACTIVATION SCALE
            if self.quant_config.activation_scheme == "static":
                self._create_scale_param(
                    scale_name="input_scale",
                    layer=layer,
                    output_partition_sizes=output_partition_sizes,
                    **extra_weight_attrs)

        # For GPUs without FP8 hardware support, we use Marlin for fast
        # fused dequantization
        if self.use_marlin:
            layer.marlin_state = GPTQMarlinState.REPACK

    def scales_shard_indexer(
            self, param: torch.Tensor, loaded_weight: torch.Tensor,
            shard_id: Union[str, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        qkv_idxs = {"q": 0, "k": 1, "v": 2}

        if isinstance(shard_id, int):
            pass
        elif isinstance(shard_id, str):
            if shard_id not in qkv_idxs:
                raise ValueError(f"Unknown shard_id: {shard_id}")
            shard_id = qkv_idxs[shard_id]
        else:
            ValueError(f"Shard id must be int or str but got {type(shard_id)}")

        return param[shard_id], loaded_weight

    def process_weights_after_loading(self, layer: Module) -> None:
        if (not hasattr(layer, "process_after_load")
                or not layer.process_after_load):
            return

        # If checkpoint is fp/bf16 (not serialized fp8), quantize the weights.
        if not self.quant_config.is_checkpoint_fp8_serialized:
            qweight, weight_scale = ops.scaled_fp8_quant(layer.weight,
                                                         scale=None)
            layer.weight = Parameter(qweight.t(), requires_grad=False)
            layer.weight_scale = Parameter(weight_scale, requires_grad=False)
            layer.logical_widths = None
            layer.input_scale = None
            return

        # If checkpoint is fp8, requantize the separately quantized logical
        # weights into a single fp8 weight with a single weight scale.
        else:
            # WEIGHT_SCALE / WEIGHT
            #   Loop over logical weights, requantizing with single scale.
            max_w_scale = layer.weight_scale.max()
            start = 0
            for idx, logical_width in enumerate(layer.logical_widths):
                end = start + logical_width
                weight_dq = per_tensor_dequantize(layer.weight[start:end, :],
                                                  layer.weight_scale[idx])

                layer.weight[start:end, :] = per_tensor_quantize(
                    weight_dq, layer.weight_scale.max())
                start = end
            layer.weight_scale = Parameter(max_w_scale, requires_grad=False)

            # WEIGHT
            #   Transpose weight for passing to torch._scaled_mm
            weight = layer.weight
            layer.weight = Parameter(weight.t(), requires_grad=False)

            # INPUT ACTIVATION SCALE
            #   Dynamic: set to None (required input to ops.scaled_fp8_quant).
            #   Static:  set to max of the input_scales (since they are equal).
            if self.quant_config.activation_scheme == "dynamic":
                layer.input_scale = None
            elif self.quant_config.activation_scheme == "static":
                if not all_close_1d(layer.input_scale):
                    raise ValueError(
                        "All the input_scales for the logical weights of a "
                        f"layer must be equal. But got {layer.input_scale}")
                layer.input_scale = Parameter(layer.input_scale.max(),
                                              requires_grad=False)
            else:
                raise ValueError(
                    f"Unknown scheme {self.quant_config.activation_scheme}")

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:

        if self.use_marlin:
            reshaped_x = x.reshape(-1, x.shape[-1])

            size_m = reshaped_x.shape[0]
            part_size_n = layer.output_size_per_partition
            part_size_k = layer.input_size_per_partition

            out_shape = x.shape[:-1] + (part_size_n, )

            if layer.marlin_state == GPTQMarlinState.REPACK:
                print("REPACK")
                layer.marlin_state = GPTQMarlinState.READY

                device = layer.weight.device

                # NO ACT ORDER
                # Reset g_idx related tensors to zero
                layer.g_idx = Parameter(
                    torch.empty(0, dtype=torch.int, device=device),
                    requires_grad=False,
                )
                layer.g_idx_sort_indices = Parameter(
                    torch.empty(0, dtype=torch.int, device=device),
                    requires_grad=False,
                )

                # WEIGHTS
                # FP8 is always 8 bits
                weight_bits = 8
                # # Repack weights to gptq format (packed int32 elements)
                fp8_weights = layer.weight
                # # View the tensor as uint8 to access the raw bits
                # tensor_int8 = orig_fp8_weights.view(torch.int8)
                # # Reshape to group every four elements together, with padding
                # num_elements = tensor_int8.numel()
                # padded_size = (orig_weight_shape[0] + 3) // 4 * 4
                # # Pad the tensor to ensure it is a multiple of 4
                # tensor_padded = torch.nn.functional.pad(
                #     tensor_int8, (padded_size - orig_weight_shape[0], 0))
                # print(tensor_int8.shape)
                # print(tensor_padded.shape)
                # # Reshape the padded tensor to (4, N)
                # tensor_reshaped = tensor_padded.reshape(4, -1)
                # # Pack the 4 uint8 values into 1 int32
                # tensor_packed = (
                #     tensor_reshaped[0, :].to(torch.int32) & 0xFF
                # ) | ((tensor_reshaped[1, :].to(torch.int32) & 0xFF) << 8) | (
                #     (tensor_reshaped[2, :].to(torch.int32) & 0xFF) << 16) | (
                #         (tensor_reshaped[3, :].to(torch.int32) & 0xFF) << 24)
                # # Reshape the packed tensor back to the desired shape
                # tensor_packed = tensor_packed.view(-1, orig_weight_shape[1])

                print("ORIG FP8 WEIGHT", fp8_weights.shape)
                fp8_uint8 = fp8_weights.view(dtype=torch.uint8).cpu().numpy()

                i = 0
                row = 0
                gptq_qweight = np.zeros(
                    (fp8_weights.shape[0] // 32 * weight_bits,
                     fp8_weights.shape[1]),
                    dtype=np.uint32)
                while row < gptq_qweight.shape[0]:
                    for j in range(i, i + (32 // weight_bits)):
                        gptq_qweight[row] |= fp8_uint8[j] << (weight_bits *
                                                              (j - i))
                    i += 32 // weight_bits
                    row += 1

                gptq_qweight = torch.from_numpy(gptq_qweight.astype(
                    np.int32)).to(device)
                print("GPTQ PACKED WEIGHT", gptq_qweight.shape)

                # Repack weights to marlin format
                marlin_qweight = ops.gptq_marlin_repack(
                    gptq_qweight,
                    layer.g_idx_sort_indices,
                    part_size_k,
                    part_size_n,
                    weight_bits,
                )
                layer.weight = Parameter(marlin_qweight, requires_grad=False)

                # WEIGHT SCALES
                # Currently Marlin doesn't support per-tensor scales, so we
                # expand it to channelwise
                scales_size_k = part_size_k
                scales_size_n = part_size_n
                scales = layer.weight_scale.repeat(1, scales_size_n).to(
                    torch.float16)
                # Permute scales
                group_size = -1
                marlin_scales = marlin_permute_scales(
                    scales,
                    scales_size_k,
                    scales_size_n,
                    group_size,
                    weight_bits,
                )
                layer.weight_scale = Parameter(marlin_scales,
                                               requires_grad=False)

                # Allocate marlin workspace
                max_workspace_size = (part_size_n // GPTQ_MARLIN_MIN_THREAD_N
                                      ) * GPTQ_MARLIN_MAX_PARALLEL
                workspace = torch.zeros(max_workspace_size,
                                        dtype=torch.int,
                                        requires_grad=False)

                layer.workspace = workspace
                # K is always full due to full alignment with
                # group-size and shard of scales/zp
                layer.is_k_full = True

                torch.cuda.synchronize()

            print("MARLIN FP8")
            print("RESHAPE X", reshaped_x.shape)
            print("WEIGHT", layer.weight.shape)
            print("SCALES", layer.weight_scale.shape)
            print("G IDX", layer.g_idx.shape)
            print("G IDX SORT INDICES", layer.g_idx_sort_indices.shape)
            print("WORKSPACE", layer.workspace.shape)
            print("M", size_m)
            print("N", part_size_n)
            print("K", part_size_k)
            print("IS K FULL", layer.is_k_full)
            torch.cuda.synchronize()
            output = ops.fp8_marlin_gemm(
                reshaped_x,
                layer.weight,
                layer.weight_scale,
                layer.g_idx,
                layer.g_idx_sort_indices,
                layer.workspace,
                weight_bits,
                size_m,
                part_size_n,
                part_size_k,
                layer.is_k_full,
            )
            torch.cuda.synchronize()

            if bias is not None:
                output.add_(bias)  # In-place add

            return output.reshape(out_shape)

        else:

            # ops.scaled_fp8_quant supports both dynamic and static quant.
            # If dynamic, layer.input_scale is None and x_scale computed from x
            # If static, layer.input_scale is scalar and x_scale is input_scale

            if bias is None and self.cutlass_fp8_supported:
                qinput, x_scale = ops.scaled_fp8_quant(x, layer.input_scale)

                # Fused GEMM_DQ
                output = ops.cutlass_scaled_mm(
                    qinput,
                    layer.weight,
                    out_dtype=x.dtype,
                    scale_a=x_scale,
                    scale_b=layer.weight_scale,
                )

            else:
                qinput, x_scale = ops.scaled_fp8_quant(x,
                                                       layer.input_scale,
                                                       batch_dim_padding=17)

                # Fused GEMM_DQ -- note we padded the input above because
                # torch._scaled_mm is more performant for matrices with
                # batch dimension > 16. Note that this could change
                # in the future.
                output, _ = torch._scaled_mm(
                    qinput,
                    layer.weight,
                    out_dtype=x.dtype,
                    scale_a=x_scale,
                    scale_b=layer.weight_scale,
                    bias=bias,
                )

        return torch.narrow(output, 0, 0, x.shape[0])


class Fp8KVCacheMethod(QuantizeMethodBase):
    """Supports loading kv-cache scaling factors from FP8 checkpoints.
    """

    def __init__(self, quant_config: Fp8Config):
        self.quant_config = quant_config

    def create_weights(self, layer: torch.nn.Module):
        """Create "weight" (aka kv_scale) for an attention layer.

        Args:
            layer: The layer that is using the QuantizeMethodBase factory.
        """
        # Initialize the KV cache scale to 1.0 as the default value.
        # If the kv_scale appears in the checkpoint, it will be
        # overwritten when loading weights.
        layer.kv_scale = Parameter(torch.tensor(1.0), requires_grad=False)

    def apply(self, layer: torch.nn.Module) -> torch.Tensor:
        raise RuntimeError("Fp8KVCacheMethod.apply should not be called.")

    def process_weights_after_loading(self, layer: Module) -> None:
        # If the kv-cache dtype is auto, we enforce the kv-scale to be 1.0
        # regardless whether the kv-scale is available in the checkpoint.
        if layer.kv_cache_dtype != "auto":
            kv_scale = layer.kv_scale.to("cpu").tolist()
            if not isinstance(kv_scale, float):
                raise ValueError("Only support per-tensor scaling factor "
                                 "for fp8 KV cache")
            layer._kv_scale = kv_scale
            if layer._kv_scale == 1.0 and "e5m2" not in layer.kv_cache_dtype:
                print_warning_once(
                    "Using KV cache scaling factor 1.0 for fp8_e4m3. This may "
                    "cause accuracy issues. Please make sure kv-cache scaling "
                    "factor is available in the fp8 checkpoint.")
        del layer.kv_scale


def all_close_1d(x: torch.Tensor) -> bool:
    assert len(x.shape) == 1
    return all(torch.allclose(x[0], x[i]) for i in range(x.shape[0]))


def per_tensor_quantize(tensor: torch.Tensor,
                        inv_scale: Union[float, torch.Tensor]) -> torch.Tensor:
    finfo = torch.finfo(torch.float8_e4m3fn)
    qweight = (tensor / inv_scale).clamp(min=finfo.min, max=finfo.max)
    return qweight.to(torch.float8_e4m3fn)


def per_tensor_dequantize(
        tensor: torch.Tensor, inv_scale: Union[float,
                                               torch.Tensor]) -> torch.Tensor:
    fake_qweight = tensor.to(torch.float16)
    dq_weight = fake_qweight * inv_scale
    return dq_weight
