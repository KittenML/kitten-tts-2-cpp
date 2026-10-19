"""Synthetic Qwen3-shaped checkpoints for CI-safe converter tests.

Weights are generated directly as BF16 bit patterns (``uint16``) so the tests
control every edge case exactly: all-zero groups, single-sign groups, ``-0.0``,
subnormal and maximal scales. The tensor set mirrors the real checkpoint:
embedding, per-layer norms (input / post-attention / q_norm / k_norm), the
seven projections per layer, the final norm and the speaker projection
(``spk_proj.0`` Linear + ``spk_proj.1`` LayerNorm).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from kt2n_assets.safetensors_raw import write_safetensors

# Interesting BF16 magnitudes for scales.
SCALE_SUBNORMAL_MIN = 0x0001
SCALE_SUBNORMAL_MAX = 0x007F
SCALE_MIN_NORMAL = 0x0080
SCALE_MAX_FINITE = 0x7F7F


@dataclass
class SynthSpec:
    layers: int = 2
    hidden: int = 256
    intermediate: int = 768
    vocab: int = 1024
    heads: int = 4
    kv_heads: int = 2
    head_dim: int = 64
    spk_dim: int = 512
    seed: int = 0
    neg_zero_per_matrix: int = 3  # -0.0 values sprinkled into zero positions
    edge_groups: bool = True  # all-zero, single-sign, subnormal/max-scale groups


def f32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even float32 -> bf16 bits (finite inputs)."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    rounded = u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))
    return (rounded >> np.uint32(16)).astype(np.uint16)


def ternary_bits(rng: np.random.Generator, n: int, k: int, *, neg_zero: int = 0,
                 edge_groups: bool = True, p_zero: float = 0.4) -> np.ndarray:
    """A random exactly-ternary BF16 matrix ``[n, k]`` as raw bits."""
    g = k // 128
    # Per-group scale: positive normal bf16 around 0.01..0.1.
    scale_f = np.exp(rng.uniform(np.log(0.005), np.log(0.2), size=(n, g))).astype(np.float32)
    scales = f32_to_bf16_bits(scale_f) & np.uint16(0x7FFF)
    r = rng.random((n, k))
    codes = np.where(r < p_zero, 0, np.where(r < p_zero + (1 - p_zero) / 2, 1, 2)).astype(np.uint16)
    if edge_groups and n >= 2 and g >= 2:
        groups = codes.reshape(n, g, 128)
        groups[0, 0, :] = 0  # all-zero group (scale must come out 0)
        groups[0, 1, :] = 1  # all positive
        groups[1, 0, :] = 2  # all negative
        groups[1, 1, :] = 0
        groups[1, 1, 5] = 2  # single non-zero, negative
        if n >= 4:
            scales[2, 0] = SCALE_SUBNORMAL_MIN
            scales[2, 1] = SCALE_SUBNORMAL_MAX
            scales[3, 0] = SCALE_MIN_NORMAL
            scales[3, 1] = SCALE_MAX_FINITE
            groups[2:4, 0:2, :][..., 0] = 1  # make sure each special-scale group is non-empty
    expanded = np.repeat(scales, 128, axis=1)
    bits = np.where(codes == 0, np.uint16(0), expanded | ((codes == 2).astype(np.uint16) << np.uint16(15)))
    bits = bits.astype(np.uint16)
    if neg_zero:
        zeros = np.flatnonzero(bits.reshape(-1) == 0)
        pick = rng.choice(zeros, size=min(neg_zero, zeros.size), replace=False)
        bits.reshape(-1)[pick] = 0x8000
    return bits


def dense_bits(rng: np.random.Generator, shape: tuple[int, ...], scale: float = 0.02, offset: float = 0.0) -> np.ndarray:
    return f32_to_bf16_bits(rng.normal(offset, scale, size=shape).astype(np.float32))


def qwen3_config(spec: SynthSpec) -> dict[str, Any]:
    return {
        "architectures": ["Qwen3ForCausalLM"],
        "attention_bias": False,
        "head_dim": spec.head_dim,
        "hidden_act": "silu",
        "hidden_size": spec.hidden,
        "intermediate_size": spec.intermediate,
        "max_position_embeddings": 4096,
        "model_type": "qwen3",
        "num_attention_heads": spec.heads,
        "num_hidden_layers": spec.layers,
        "num_key_value_heads": spec.kv_heads,
        "rms_norm_eps": 1e-06,
        "rope_theta": 1000000,
        "tie_word_embeddings": True,
        "torch_dtype": "bfloat16",
        "vocab_size": spec.vocab,
    }


def runtime_config(spec: SynthSpec) -> dict[str, Any]:
    base = spec.vocab - 64
    return {
        "format": "synthetic-test",
        "vocab_size": spec.vocab,
        "audio_id_base": base,
        "num_audio_tokens": 32,
        "speech_start_id": base - 6,
        "speech_end_id": base - 5,
    }


def make_tensors(spec: SynthSpec) -> dict[str, tuple[str, tuple[int, ...], bytes]]:
    rng = np.random.default_rng(spec.seed)
    h, i, d = spec.hidden, spec.intermediate, spec.head_dim
    out: dict[str, tuple[str, tuple[int, ...], bytes]] = {}

    def put(name: str, bits: np.ndarray) -> None:
        out[name] = ("BF16", tuple(bits.shape), bits.astype("<u2").tobytes())

    put("model.embed_tokens.weight", dense_bits(rng, (spec.vocab, h)))
    proj_shapes = {
        "self_attn.q_proj": (spec.heads * d, h),
        "self_attn.k_proj": (spec.kv_heads * d, h),
        "self_attn.v_proj": (spec.kv_heads * d, h),
        "self_attn.o_proj": (h, spec.heads * d),
        "mlp.gate_proj": (i, h),
        "mlp.up_proj": (i, h),
        "mlp.down_proj": (h, i),
    }
    for layer in range(spec.layers):
        p = f"model.layers.{layer}."
        # Alphabetical order, like the real checkpoint's header.
        put(p + "input_layernorm.weight", dense_bits(rng, (h,), 0.05, 1.0))
        for sub in ("mlp.down_proj", "mlp.gate_proj", "mlp.up_proj"):
            put(p + sub + ".weight", ternary_bits(rng, *proj_shapes[sub], neg_zero=spec.neg_zero_per_matrix,
                                                  edge_groups=spec.edge_groups))
        put(p + "post_attention_layernorm.weight", dense_bits(rng, (h,), 0.05, 1.0))
        put(p + "self_attn.k_norm.weight", dense_bits(rng, (d,), 0.05, 1.0))
        for sub in ("self_attn.k_proj", "self_attn.o_proj"):
            put(p + sub + ".weight", ternary_bits(rng, *proj_shapes[sub], neg_zero=spec.neg_zero_per_matrix,
                                                  edge_groups=spec.edge_groups))
        put(p + "self_attn.q_norm.weight", dense_bits(rng, (d,), 0.05, 1.0))
        for sub in ("self_attn.q_proj", "self_attn.v_proj"):
            put(p + sub + ".weight", ternary_bits(rng, *proj_shapes[sub], neg_zero=spec.neg_zero_per_matrix,
                                                  edge_groups=spec.edge_groups))
    put("model.norm.weight", dense_bits(rng, (h,), 0.05, 1.0))
    put("spk_proj.0.bias", dense_bits(rng, (h,)))
    put("spk_proj.0.weight", dense_bits(rng, (h, spec.spk_dim)))
    put("spk_proj.1.bias", dense_bits(rng, (h,)))
    put("spk_proj.1.weight", dense_bits(rng, (h,), 0.05, 1.0))
    return out


def make_checkpoint(root: str | Path, spec: SynthSpec | None = None, *,
                    tensors: dict[str, tuple[str, tuple[int, ...], bytes]] | None = None,
                    runtime_config_in_parent: bool = True) -> Path:
    """Write ``root/ckpt/effective-bf16/{model.safetensors,config.json,...}``.

    ``runtime_config.json`` goes in the parent directory (like the real layout)
    unless ``runtime_config_in_parent`` is False. Returns the source directory.
    """
    spec = spec or SynthSpec()
    src = Path(root) / "ckpt" / "effective-bf16"
    src.mkdir(parents=True, exist_ok=True)
    write_safetensors(src / "model.safetensors", tensors if tensors is not None else make_tensors(spec),
                      metadata={"format": "pt"})
    (src / "config.json").write_text(json.dumps(qwen3_config(spec), indent=2) + "\n")
    (src / "tokenizer.json").write_text('{"version": "1.0", "model": {"type": "BPE"}}\n')
    (src / "vocab.json").write_text('{"a": 0, "b": 1}\n')
    (src / "merges.txt").write_text("#version: 0.2\na b\n")
    rc_dir = src.parent if runtime_config_in_parent else src
    (rc_dir / "runtime_config.json").write_text(json.dumps(runtime_config(spec), indent=2) + "\n")
    return src
