# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.model_executor.models.whisper_causal import select_block_pooling_backend
from vllm.v1.attention.backends.cpu_attn import CPUAttentionBackend
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
from vllm.v1.attention.backends.flex_attention import FlexAttentionBackend
from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend

# core_model so the per-PR multimodal job runs it; it needs no GPU.
pytestmark = [pytest.mark.core_model, pytest.mark.cpu_test]

# FlexAttention stands in for FlashInfer (the backend FP8 KV selects on
# sm_89): both are real backends the block-pooling wrapper cannot wrap, and
# FlexAttention imports without flashinfer installed.
UNSUPPORTED = FlexAttentionBackend


@pytest.mark.parametrize("kv_cache_dtype", ["fp8", "fp8_e4m3", "auto"])
def test_unsupported_backend_falls_back_to_triton_on_cuda(kv_cache_dtype):
    selected = select_block_pooling_backend(UNSUPPORTED, kv_cache_dtype, True)
    assert selected is TritonAttentionBackend


@pytest.mark.parametrize(
    "backend", [CPUAttentionBackend, FlashAttentionBackend, TritonAttentionBackend]
)
@pytest.mark.parametrize("is_cuda", [True, False])
def test_supported_backend_is_kept(backend, is_cuda):
    assert select_block_pooling_backend(backend, "fp8", is_cuda) is backend


@pytest.mark.parametrize(
    ("kv_cache_dtype", "is_cuda"),
    [
        ("fp8", False),  # ROCm, CPU, XPU: no fallback
        ("nvfp4", True),  # Triton cannot read an NVFP4 KV cache
    ],
)
def test_unsupported_backend_without_fallback_raises(kv_cache_dtype, is_cuda):
    with pytest.raises(NotImplementedError, match="not yet supported"):
        select_block_pooling_backend(UNSUPPORTED, kv_cache_dtype, is_cuda)
