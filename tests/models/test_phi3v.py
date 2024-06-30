import re
from typing import List, Optional, Tuple, Type

import pytest
from transformers import AutoConfig, AutoTokenizer

from vllm.config import VisionLanguageConfig
from vllm.model_executor.models.phi3v import get_phi3v_image_feature_size
from vllm.multimodal.image import ImagePixelData
from vllm.multimodal.utils import rescale_image_size
from vllm.utils import is_cpu

from ..conftest import IMAGE_ASSETS, HfRunner, VllmRunner, _ImageAssets
from .utils import check_outputs_equal_xfail

pytestmark = pytest.mark.vlm

HF_IMAGE_PROMPTS = IMAGE_ASSETS.prompts({
    "stop_sign":
    "<|user|>\n<|image_1|>\nWhat's the content of the image?<|end|>\n<|assistant|>\n",  # noqa: E501
    "cherry_blossom":
    "<|user|>\n<|image_1|>\nWhat is the season?<|end|>\n<|assistant|>\n",
})


def iter_phi3v_configs(model_name: str):
    image_hw_to_feature_size = {
        (1008, 1344): 1921,
        (2016, 2688): 1933,
    }

    for (h, w), f in image_hw_to_feature_size.items():
        for input_type, input_shape in [
            (VisionLanguageConfig.ImageInputType.PIXEL_VALUES, (1, 3, h, w)),
        ]:
            yield (model_name,
                   VisionLanguageConfig(image_input_type=input_type,
                                        image_feature_size=f,
                                        image_token_id=32044,
                                        image_input_shape=input_shape,
                                        image_processor=model_name,
                                        image_processor_revision=None))


model_and_vl_config = [
    *iter_phi3v_configs("microsoft/Phi-3-vision-128k-instruct"),
]


def vllm_to_hf_output(vllm_output: Tuple[List[int], str],
                      vlm_config: VisionLanguageConfig, model_id: str,
                      image_feature_size: int):
    """Sanitize vllm output to be comparable with hf output.
    The function reduces `input_ids` from 1, 32000, 32000, ..., 32000,
    x1, x2, x3 ... to 1, 32000, x1, x2, x3 ...
    It also reduces `output_str` from "<image><image>bla" to "bla".
    """
    output_ids, output_str = vllm_output
    output_str_without_image = re.sub(r"(<\|image_\d+\|>)+", " ", output_str)

    hf_output_str = output_str_without_image.replace("<|user|>", "") \
        .replace("<|end|>\n<|assistant|>", " ")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    hf_output_ids = tokenizer.encode(output_str_without_image)
    hf_output_ids = hf_output_ids[:4] + [0] * image_feature_size \
        + [1] + hf_output_ids[4:]

    return hf_output_ids, hf_output_str


target_dtype = "half"
if is_cpu():
    target_dtype = "bfloat16"


def run_test(
    hf_runner: Type[HfRunner],
    vllm_runner: Type[VllmRunner],
    image_assets: _ImageAssets,
    model_and_config: Tuple[str, VisionLanguageConfig],
    *,
    size_factors: List[float],
    dtype: str,
    max_tokens: int,
    tensor_parallel_size: int,
    distributed_executor_backend: Optional[str] = None,
):
    """Inference result should be the same between hf and vllm.

    All the image fixtures for the test is under tests/images.
    For huggingface runner, we provide the PIL images as input.
    For vllm runner, we provide MultiModalData objects and corresponding
    vision language config as input.
    Note, the text input is also adjusted to abide by vllm contract.
    The text output is sanitized to be able to compare with hf.
    """
    model_id, vlm_config = model_and_config
    hf_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    hf_images = [asset.for_hf() for asset in image_assets]
    vllm_images = [asset.for_vllm(vlm_config) for asset in image_assets]

    image_inputs_per_size_factors = [[(
        prompt,
        rescale_image_size(hf_image, factor),
        ImagePixelData(image=rescale_image_size(vllm_image.image, factor)),
    ) for hf_image, vllm_image, prompt in zip(
        hf_images, vllm_images, HF_IMAGE_PROMPTS)] for factor in size_factors]
    hf_inputs_per_size_factors = [(
        [prompt for prompt, hf_image, vllm_image in image_inputs],
        [hf_image for prompt, hf_image, vllm_image in image_inputs],
    ) for image_inputs in image_inputs_per_size_factors]
    vllm_inputs_per_size_factors = [(
        [prompt for prompt, hf_image, vllm_image in image_inputs],
        [vllm_image for prompt, hf_image, vllm_image in image_inputs],
    ) for image_inputs in image_inputs_per_size_factors]

    # max_model_len should be greater than image_feature_size
    with vllm_runner(model_id,
                     max_model_len=4096,
                     dtype=dtype,
                     enforce_eager=True,
                     **vlm_config.as_cli_args_dict()) as vllm_model:
        vllm_outputs_per_size_factors = [
            vllm_model.generate_greedy(prompts, max_tokens, images=vllm_images)
            for prompts, vllm_images in vllm_inputs_per_size_factors
        ]

    # use eager mode for hf runner, since phi3_v didn't work with flash_attn
    hf_model_kwargs = {"_attn_implementation": "eager"}
    with hf_runner(model_id, dtype=dtype,
                   model_kwargs=hf_model_kwargs) as hf_model:
        eos_token_id = hf_model.processor.tokenizer.eos_token_id
        hf_outputs_per_size_factors = [
            hf_model.generate_greedy(prompts,
                                     max_tokens,
                                     images=hf_images,
                                     eos_token_id=eos_token_id)
            for prompts, hf_images in hf_inputs_per_size_factors
        ]
        hf_dummy_outputs_per_size_factors = [
            hf_model.generate_greedy(prompts,
                                     max_tokens=1,
                                     images=hf_images,
                                     eos_token_id=eos_token_id)
            for prompts, hf_images in hf_inputs_per_size_factors
        ]

    # Since we use _attn_implementation="eager", there is numeric
    # difference for longer context (max_tokens=128) and test can't pass
    for image_inputs, vllm_outputs, hf_outputs, hf_dummy_outputs in zip(
            image_inputs_per_size_factors,
            vllm_outputs_per_size_factors,
            hf_outputs_per_size_factors,
            hf_dummy_outputs_per_size_factors,
    ):
        image_feature_sizes = [
            get_phi3v_image_feature_size(
                hf_config,
                input_height=hf_image.height,
                input_width=hf_image.width,
            ) for _, hf_image, _ in image_inputs
        ]

        check_outputs_equal_xfail(
            outputs_0_lst=hf_outputs,
            outputs_1_lst=[
                vllm_to_hf_output(vllm_output,
                                  vlm_config,
                                  model_id,
                                  image_feature_size=image_feature_size)
                for vllm_output, image_feature_size in zip(
                    vllm_outputs, image_feature_sizes)
            ],
            outputs_num_prefix_tokens=[
                len(hf_dummy_output[0]) - 1
                for hf_dummy_output in hf_dummy_outputs
            ],
            name_0="hf",
            name_1="vllm",
            min_tokens_to_xfail=1,
            min_tokens_to_pass=max_tokens,
        )


@pytest.mark.parametrize("model_and_config", model_and_vl_config)
@pytest.mark.parametrize(
    "size_factors",
    [
        # No image
        [],
        # Single-scale
        [1.0],
        # Single-scale, batched
        [1.0, 1.0, 1.0],
        # Multi-scale
        [0.25, 0.5, 1.0],
    ],
)
@pytest.mark.parametrize("dtype", [target_dtype])
@pytest.mark.parametrize("max_tokens", [128])
def test_models(hf_runner, vllm_runner, image_assets, model_and_config,
                size_factors, dtype: str, max_tokens: int) -> None:
    run_test(
        hf_runner,
        vllm_runner,
        image_assets,
        model_and_config,
        size_factors=size_factors,
        dtype=dtype,
        max_tokens=max_tokens,
        tensor_parallel_size=1,
    )
