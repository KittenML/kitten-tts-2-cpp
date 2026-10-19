"""VERBATIM copies of production's exact ternary packer (do not edit by hand).

Source: haaziq-sl/kittenml-api @ 4253ae55876262b92ae22ebc1b4bd91638d7b4c3
        kitten_tts_09/fast_lm/packed_ternary.py (git blob 552c43b6fcf4cb2039ff93e5c0a75ef1e342126a)
Symbols: GROUP, PROJECTION_RE, pack_exact, unpack

The retained symbols are verified against their hashes in PROVENANCE.json.
"""

from __future__ import annotations

import re
from typing import Any

GROUP = 128

PROJECTION_RE = re.compile(
    r"^layers\.(\d+)\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)$"
)


def pack_exact(weight: Any) -> tuple[Any, Any]:
    """2-bit codes (16 per int32, LSB first, code = +1 -> 1, -1 -> 2, 0 -> 0) + BF16 scales.

    Raises ``ValueError`` unless every 128-group is exactly ``{-s, 0, +s}``.
    """
    import torch

    if weight.dtype != torch.bfloat16 or weight.ndim != 2 or weight.shape[1] % 256:
        raise ValueError("Expected a BF16 matrix with K divisible by 256")
    # Detached: the model's Parameters still require grad, and MEASURED 24 Sep
    # on the lab an attached ``amax`` kept each matrix's full-size ``abs()``
    # alive through ``scales.grad_fn`` -- 2.41 GB of GPU memory for 0.28 GB of
    # packed weights.
    weight = weight.detach()
    n, k = weight.shape
    group = weight.reshape(n, k // GROUP, GROUP)
    scales = group.abs().amax(-1)
    reconstructed = torch.sign(group) * scales[..., None]
    if not torch.equal(reconstructed, group):
        raise ValueError("Weight is not exactly ternary per 128-group")
    codes = torch.where(weight > 0, 1, torch.where(weight < 0, 2, 0)).to(torch.int32)
    packed = torch.zeros((n, k // 16), device=weight.device, dtype=torch.int32)
    for offset in range(16):
        packed |= codes[:, offset::16] << (offset * 2)
    return packed.contiguous(), scales.contiguous()


def unpack(packed: Any, scales: Any, k: int) -> Any:
    """Inverse of ``pack_exact`` (for tests and the round-trip check)."""
    import torch

    n = packed.shape[0]
    words = packed.to(torch.int64) & 0xFFFFFFFF
    codes = torch.empty((n, k), dtype=torch.int64, device=packed.device)
    for offset in range(16):
        codes[:, offset::16] = (words >> (offset * 2)) & 3
    signs = torch.where(codes == 1, 1.0, torch.where(codes == 2, -1.0, 0.0))
    expanded = scales.float().repeat_interleave(GROUP, dim=1)
    return (signs * expanded).to(scales.dtype)
