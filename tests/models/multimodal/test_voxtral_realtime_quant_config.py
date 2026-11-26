# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for loading mistral-format compressed-tensors checkpoints of
Voxtral Realtime (`mistralai/Voxtral-Mini-4B-Realtime-2602`)."""

import pytest
from transformers import LlamaConfig

from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsConfig,
    CompressedTensorsLinearMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    should_ignore_layer,
)
from vllm.model_executor.models.mistral import MistralDecoderLayer
from vllm.model_executor.models.voxtral_realtime import VoxtralRealtimeGeneration

pytestmark = [pytest.mark.core_model, pytest.mark.cpu_test]

ENC = "mm_streams_embeddings.embedding_module"
ENC_LAYER = f"{ENC}.whisper_encoder.transformer.layers.0"
VLLM_ENC_LAYER = "whisper_encoder.whisper_encoder.layers.0.layers"


def _w4a16_config(ignore: list[str]) -> CompressedTensorsConfig:
    return CompressedTensorsConfig.from_config(
        {
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 4,
                        "type": "int",
                        "symmetric": True,
                        "strategy": "group",
                        "group_size": 128,
                        "dynamic": False,
                    },
                }
            },
            "ignore": ignore,
        }
    )


def test_realtime_ignore_names_are_remapped():
    """Mistral-format ignore entries of the realtime model must resolve to vLLM
    module names; otherwise the bf16 encoder silently loads into quantized
    params."""
    quant_config = _w4a16_config(
        [
            "output",
            "layers.0.ada_rms_norm_t_cond.2",
            f"{ENC_LAYER}.attention.wq",
            f"{ENC_LAYER}.attention.wk",
            f"{ENC_LAYER}.attention.wv",
            f"{ENC_LAYER}.attention.wo",
            f"{ENC_LAYER}.feed_forward.w1",
            f"{ENC_LAYER}.feed_forward.w2",
            f"{ENC_LAYER}.feed_forward.w3",
            f"{ENC}.audio_language_projection.0",
            f"{ENC}.audio_language_projection.2",
        ]
    )

    # The method does not use `self`.
    quant_config = VoxtralRealtimeGeneration.maybe_update_quant_config(
        None, quant_config
    )

    assert quant_config.ignore == [
        "language_model.lm_head",
        "language_model.model.layers.0.ada_rms_norm_t_cond.2",
        f"{VLLM_ENC_LAYER}.self_attn.q_proj",
        f"{VLLM_ENC_LAYER}.self_attn.k_proj",
        f"{VLLM_ENC_LAYER}.self_attn.v_proj",
        f"{VLLM_ENC_LAYER}.self_attn.out_proj",
        f"{VLLM_ENC_LAYER}.mlp.gate_proj",
        f"{VLLM_ENC_LAYER}.mlp.down_proj",
        f"{VLLM_ENC_LAYER}.mlp.up_proj",
        "audio_language_adapter.w_in",
        "audio_language_adapter.w_out",
    ]
    for fused_layer in ("self_attn.qkv_proj", "mlp.gate_up_proj"):
        assert should_ignore_layer(
            f"{VLLM_ENC_LAYER}.{fused_layer}",
            ignore=quant_config.ignore,
            fused_mapping=VoxtralRealtimeGeneration.packed_modules_mapping,
        )


def test_ada_rms_norm_t_cond_uses_quant_config(dist_init, tmp_path):
    """The AdaRMSNorm conditioning linears must get the model's quant config
    and their own prefix, so quantized checkpoint tensors have params to load
    into and ignore entries can target them."""
    LlamaConfig(
        architectures=["MistralForCausalLM"],
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        vocab_size=128,
        max_position_embeddings=64,
        ada_rms_norm_t_cond=True,
        ada_rms_norm_t_cond_dim=32,
    ).save_pretrained(tmp_path)
    prefix = "language_model.model.layers.0"
    vllm_config = VllmConfig(
        model_config=ModelConfig(
            model=str(tmp_path), skip_tokenizer_init=True, max_model_len=64
        ),
        quant_config=_w4a16_config([f"{prefix}.ada_rms_norm_t_cond.2"]),
    )

    with set_current_vllm_config(vllm_config):
        layer = MistralDecoderLayer(vllm_config, prefix=prefix)

    assert layer.ada_rms_norm_t_cond is not None
    linear1, _, linear2 = layer.ada_rms_norm_t_cond
    assert linear1.prefix == f"{prefix}.ada_rms_norm_t_cond.0"
    assert isinstance(linear1.quant_method, CompressedTensorsLinearMethod)
    assert linear2.prefix == f"{prefix}.ada_rms_norm_t_cond.2"
    assert isinstance(linear2.quant_method, UnquantizedLinearMethod)
