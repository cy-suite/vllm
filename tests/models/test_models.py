"""Compare the outputs of HF and vLLM when using greedy sampling.

This test only tests small models. Big models such as 7B should be tested from
test_big_models.py because it could use a larger instance to run tests.

Run `pytest tests/models/test_models.py`.
"""
import pytest

MODELS = [
    # baichuan
    "bigscience/bloom-560m",
    # chatglm
    # command-r
    # dbrx
    # decilm
    "deepseek-ai/deepseek-coder-1.3b-instruct",
    # falcon
    # "google/gemma-1.1-2b-it", # < broken
    "gpt2",
    "bigcode/tiny_starcoder_py",
    # gpt-j
    "EleutherAI/pythia-70m",
    # internlm2
    # jais
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    # llava
    "openbmb/MiniCPM-2B-128k",
    # mixtral
    # mixtral-quant
    # mpt
    "allenai/OLMo-1B",
    "facebook/opt-125m",
    # orion
    "microsoft/phi-2",
    "Qwen/Qwen-1_8B",
    "Qwen/Qwen1.5-1.8B",
    # qwen2 moe
    "stabilityai/stablelm-2-1_6b-chat",
    "bigcode/starcoder2-3b",
    # xverse
]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["float"])
@pytest.mark.parametrize("max_tokens", [96])
def test_models(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
) -> None:
    # To pass the small model tests, we need full precision.
    assert dtype == "float"

    hf_model = hf_runner(model, dtype=dtype)
    hf_outputs = hf_model.generate_greedy(example_prompts, max_tokens)
    del hf_model

    vllm_model = vllm_runner(model, dtype=dtype)
    vllm_outputs = vllm_model.generate_greedy(example_prompts, max_tokens)
    del vllm_model

    for i in range(len(example_prompts)):
        hf_output_ids, hf_output_str = hf_outputs[i]
        vllm_output_ids, vllm_output_str = vllm_outputs[i]
        assert hf_output_str == vllm_output_str, (
            f"Test{i}:\nHF: {hf_output_str!r}\nvLLM: {vllm_output_str!r}")
        assert hf_output_ids == vllm_output_ids, (
            f"Test{i}:\nHF: {hf_output_ids}\nvLLM: {vllm_output_ids}")
