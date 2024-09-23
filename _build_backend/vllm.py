import os
from textwrap import dedent

from setuptools import build_meta as build_meta_orig
from setuptools.build_meta import *

VLLM_TARGET_DEVICE = os.getenv(
    "VLLM_TARGET_DEVICE",
    "cuda",  # match the default value in vllm/envs.py
)

BASE_REQUIREMENTS = (
    "setuptools>=61",
    "setuptools-scm>=8",
)
BUILD_REQUIREMENTS_EXTENSIONS = (
    "cmake>=3.26",
    "ninja",
    "packaging",
    "wheel",
)


def check_for_index_url_env_var(index_url: str):
    if os.getenv("PIP_INDEX_EXTRA_URL"):
        return

    msg = dedent(
        """
        ***
        PIP_INDEX_EXTRA_URL is not defined, but might be required for this build.
        If building fails because of dependency issues, try setting

            PIP_EXTRA_INDEX_URL={index_url}

        in your environment before starting the build.
        ***""",  # noqa: E501
    )

    import warnings

    warnings.warn(msg.format(index_url=index_url), stacklevel=2)


def get_requires_for_build_wheel(  # type: ignore[no-redef]
    config_settings=None,
):
    requirements_extras = []
    if VLLM_TARGET_DEVICE == "cpu" or VLLM_TARGET_DEVICE == "openvino":
        check_for_index_url_env_var(
            index_url="https://download.pytorch.org/whl/cpu"
        )

        requirements_extras.append("torch==2.4.0+cpu")
    elif VLLM_TARGET_DEVICE == "cuda":
        requirements_extras.append("torch==2.4.0")
    elif VLLM_TARGET_DEVICE == "rocm":
        # TODO: add support for multiple ROCM versions (5.2 and?)
        rocm_supported_versions = ("6.1",)
        requested_version = os.getenv("ROCM_VERSION")
        if not requested_version:
            raise RuntimeError(
                "Set ROCM_VERSION env var. "
                f"Supported versions={rocm_supported_versions}"
            )
        if requested_version not in rocm_supported_versions:
            raise ValueError(
                "Invalid ROCM_VERSION. "
                f"Supported versions={rocm_supported_versions}"
            )

        check_for_index_url_env_var(
            index_url=f"https://download.pytorch.org/whl/nightly/rocm{requested_version}"
        )
        requirements_extras.extend(
            [
                "torch==2.5.0.dev20240726",
                "torchvision==0.20.0.dev20240726",
            ]
        )
    elif VLLM_TARGET_DEVICE == "neuron":
        requirements_extras.append("torch==2.1.2")
    elif VLLM_TARGET_DEVICE == "tpu":
        if not os.getenv("PIP_FIND_LINKS"):
            import warnings

            msg = dedent(
                """
                ***
                TPU builds require modules from google, if the build fails with dependency issues, try setting

                    PIP_FIND_LINKS="https://storage.googleapis.com/libtpu-releases/index.html https://storage.googleapis.com/jax-releases/jax_nightly_releases.html https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html"

                in your environment before starting the build.
                ***""",  # noqa: E501
            )
            warnings.warn(msg, stacklevel=1)
        requirements_extras.extend(
            [
                "torch==2.5.0",
                "torch_xla[tpu,pallas]",
            ]
        )
    elif VLLM_TARGET_DEVICE == "xpu":
        requirements_extras.append(
            "torch @ https://intel-extension-for-pytorch.s3.amazonaws.com/ipex_dev/xpu/torch-2.1.0.post1%2Bcxx11.abi-cp310-cp310-linux_x86_64.whl",
        )
    else:
        raise RuntimeError(f"Unknown runtime environment {VLLM_TARGET_DEVICE=}")

    requirements = build_meta_orig.get_requires_for_build_wheel(config_settings)

    complete_requirements = [
        *BASE_REQUIREMENTS,
        *BUILD_REQUIREMENTS_EXTENSIONS,
        *requirements,
        *requirements_extras,
    ]
    print(
        f"vllm build-backend: resolved build dependencies to: {complete_requirements}"  # noqa: E501
    )
    return complete_requirements


def get_requires_for_build_sdist(  # type: ignore[no-redef]
    config_settings=None,
):
    requirements = build_meta_orig.get_requires_for_build_sdist(config_settings)

    return [
        *BASE_REQUIREMENTS,
        *requirements,
    ]
