from typing import List, Optional, Tuple, Type

import pytest
from transformers import AutoTokenizer

from vllm.config import VisionLanguageConfig
from vllm.multimodal.image import ImagePixelData
from vllm.multimodal.utils import rescale_image_size

from ..conftest import IMAGE_ASSETS, HfRunner, VllmRunner, _ImageAssets
from .utils import check_outputs_equal_xfail

pytestmark = pytest.mark.vlm

HF_IMAGE_PROMPTS = IMAGE_ASSETS.prompts({
    "stop_sign":
    "USER: <image>\nWhat's the content of the image?\nASSISTANT:",
    "cherry_blossom":
    "USER: <image>\nWhat is the season?\nASSISTANT:",
})


def iter_llava_configs(model_name: str):
    image_hw_to_feature_size = {
        (336, 336): 576,
    }

    for (h, w), f in image_hw_to_feature_size.items():
        for input_type, input_shape in [
            (VisionLanguageConfig.ImageInputType.PIXEL_VALUES, (1, 3, h, w)),
        ]:
            yield (model_name,
                   VisionLanguageConfig(image_input_type=input_type,
                                        image_feature_size=f,
                                        image_token_id=32000,
                                        image_input_shape=input_shape,
                                        image_processor=model_name,
                                        image_processor_revision=None))


model_and_vl_config = [
    *iter_llava_configs("llava-hf/llava-1.5-7b-hf"),
]


def vllm_to_hf_output(vllm_output: Tuple[List[int], str],
                      vlm_config: VisionLanguageConfig, model_id: str):
    """Sanitize vllm output to be comparable with hf output.
    The function reduces `input_ids` from 1, 32000, 32000, ..., 32000,
    x1, x2, x3 ... to 1, 32000, x1, x2, x3 ...
    It also reduces `output_str` from "<image><image>bla" to "bla".
    """
    output_ids, output_str = vllm_output
    image_token_id = vlm_config.image_token_id

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    image_token_str = tokenizer.decode(image_token_id)

    hf_output_ids = [
        token_id for idx, token_id in enumerate(output_ids)
        if token_id != image_token_id or output_ids[idx - 1] != image_token_id
    ]
    hf_output_str = output_str \
        .replace(image_token_str * vlm_config.image_feature_size, " ")

    return hf_output_ids, hf_output_str


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
                     dtype=dtype,
                     tensor_parallel_size=tensor_parallel_size,
                     distributed_executor_backend=distributed_executor_backend,
                     enforce_eager=True,
                     **vlm_config.as_cli_args_dict()) as vllm_model:
        vllm_outputs_per_size_factors = [
            vllm_model.generate_greedy(prompts, max_tokens, images=vllm_images)
            for prompts, vllm_images in vllm_inputs_per_size_factors
        ]

    with hf_runner(model_id, dtype=dtype, is_vision_model=True) as hf_model:
        hf_outputs_per_size_factors = [
            hf_model.generate_greedy(prompts, max_tokens, images=hf_images)
            for prompts, hf_images in hf_inputs_per_size_factors
        ]
        hf_dummy_outputs_per_size_factors = [
            hf_model.generate_greedy(prompts, max_tokens=1, images=hf_images)
            for prompts, hf_images in hf_inputs_per_size_factors
        ]

    # There may be numeric differences for multiscale images due to
    # our implementation of CLIPVisionModel
    for image_inputs, vllm_outputs, hf_outputs, hf_dummy_outputs in zip(
            image_inputs_per_size_factors,
            vllm_outputs_per_size_factors,
            hf_outputs_per_size_factors,
            hf_dummy_outputs_per_size_factors,
    ):
        check_outputs_equal_xfail(
            outputs_0_lst=hf_outputs,
            outputs_1_lst=[
                vllm_to_hf_output(vllm_output, vlm_config, model_id)
                for vllm_output in vllm_outputs
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
@pytest.mark.parametrize("dtype", ["half"])
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
