"""Ternary packing: numpy packer == production pack_exact, bit for bit; edge cases; refusals."""

from __future__ import annotations

import numpy as np
import pytest

from kt2n_assets.convert import ConvertError, NotTernaryError, pack_ternary_bits
from kt2n_assets.reconstruct import ReconstructError, unpack_ternary_bits
from synth import (
    SCALE_MAX_FINITE,
    SCALE_MIN_NORMAL,
    SCALE_SUBNORMAL_MAX,
    SCALE_SUBNORMAL_MIN,
    ternary_bits,
)


def to_torch(bits: np.ndarray):
    import torch

    return torch.from_numpy(np.ascontiguousarray(bits).view(np.int16).copy()).view(torch.bfloat16)


def torch_bits(t) -> np.ndarray:
    import torch

    return t.contiguous().view(torch.int16).numpy().view(np.uint16)


def assert_matches_vendored(vendored, bits: np.ndarray) -> None:
    packed, scales, _, _ = pack_ternary_bits(bits)
    ref_packed, ref_scales = vendored.pack_exact(to_torch(bits))
    assert ref_packed.numpy().dtype == np.int32
    assert packed.dtype == np.dtype("<i4")
    assert packed.shape == tuple(ref_packed.shape)
    assert packed.tobytes() == ref_packed.numpy().tobytes()
    assert np.array_equal(scales, torch_bits(ref_scales))


def group_matrix(n: int, k: int, fill) -> np.ndarray:
    return np.full((n, k), fill, dtype=np.uint16)


# --- equality with production ---------------------------------------------------
@pytest.mark.parametrize("shape", [(1, 256), (8, 256), (16, 768), (5, 512), (4, 2048), (3, 6144)])
@pytest.mark.parametrize("seed", [0, 1])
def test_random_matches_vendored(vendored, shape, seed):
    rng = np.random.default_rng(seed)
    bits = ternary_bits(rng, *shape, neg_zero=4, edge_groups=True)
    assert_matches_vendored(vendored, bits)


@pytest.mark.parametrize("p_zero", [0.0, 0.5, 0.95, 1.0])
def test_sparsity_extremes_match_vendored(vendored, p_zero):
    rng = np.random.default_rng(7)
    bits = ternary_bits(rng, 6, 512, p_zero=p_zero, edge_groups=False)
    assert_matches_vendored(vendored, bits)


def test_word_layout_is_lsb_first(vendored):
    # Row 0: column c = 16*j + o carries code at bits 2o..2o+1 of word j.
    bits = np.zeros((1, 256), np.uint16)
    one = 0x3F80  # bf16 1.0
    bits[0, 0] = one  # word 0, bits 0..1 = 1
    bits[0, 15] = one | 0x8000  # word 0, bits 30..31 = 2
    bits[0, 17] = one  # word 1, bits 2..3 = 1
    bits[0, 255] = one | 0x8000  # word 15, bits 30..31 = 2
    packed, scales, _, _ = pack_ternary_bits(bits)
    words = packed.view(np.uint32)[0]
    assert words[0] == (1 | (2 << 30))
    assert words[1] == (1 << 2)
    assert words[15] == (2 << 30)
    assert packed[0, 0] < 0  # bit 31 set -> negative int32, same as torch
    assert list(scales[0]) == [one, one]
    assert_matches_vendored(vendored, bits)


def test_all_zero_group(vendored):
    bits = np.zeros((2, 256), np.uint16)
    bits[1, 128:] = 0x3C00  # second group of row 1 all positive
    packed, scales, zero_signs, neg = pack_ternary_bits(bits)
    assert scales[0, 0] == 0 and scales[0, 1] == 0 and scales[1, 0] == 0
    assert scales[1, 1] == 0x3C00
    assert not packed[0].any()
    assert neg == 0 and zero_signs.size == (256 + 128 + 7) // 8 and not zero_signs.any()
    assert_matches_vendored(vendored, bits)


@pytest.mark.parametrize("sign", [0, 1])
def test_single_sign_groups(vendored, sign):
    bits = group_matrix(3, 256, 0x3E00 | (sign << 15))
    packed, scales, _, _ = pack_ternary_bits(bits)
    expected_word = 0x55555555 if sign == 0 else 0xAAAAAAAA
    assert (packed.view(np.uint32) == expected_word).all()
    assert (scales == 0x3E00).all()
    assert_matches_vendored(vendored, bits)


def test_negative_zero_recorded_and_restored(vendored):
    rng = np.random.default_rng(3)
    bits = ternary_bits(rng, 4, 256, neg_zero=0, edge_groups=False, p_zero=0.5)
    zeros = np.flatnonzero(bits.reshape(-1) == 0)[:5]
    bits.reshape(-1)[zeros] = 0x8000
    bits[3, :128] = 0x8000  # a group of only -0.0: scale must be +0 (abs), like torch
    packed, scales, zero_signs, neg = pack_ternary_bits(bits)
    neg_pos = np.sort(np.concatenate([zeros, np.arange(3 * 256, 3 * 256 + 128)]))
    assert neg == neg_pos.size
    assert scales[3, 0] == 0
    assert (packed.view(np.uint32)[3, :8] == 0).all()  # -0.0 gets code 0 (torch: -0.0 < 0 is False)
    # zero_signs: one bit per zero weight in row-major order, LSB-first.
    zero_flat = np.flatnonzero((bits.reshape(-1) & 0x7FFF) == 0)
    expect = np.isin(zero_flat, neg_pos).astype(np.uint8)
    assert np.array_equal(zero_signs, np.packbits(expect, bitorder="little"))
    assert_matches_vendored(vendored, bits)
    # Without the bitmap: +0.0 everywhere (value-equal); with it: the exact bits.
    plain = unpack_ternary_bits(packed, scales, 256)
    assert np.array_equal(np.flatnonzero(plain.reshape(-1) != bits.reshape(-1)), neg_pos)
    assert np.array_equal(unpack_ternary_bits(packed, scales, 256, zero_signs), bits)


def test_zero_signs_are_validated():
    bits = np.zeros((1, 256), np.uint16)
    bits[0, :5] = 0x8000
    packed, scales, zero_signs, _ = pack_ternary_bits(bits)
    with pytest.raises(ReconstructError, match="bytes for"):
        unpack_ternary_bits(packed, scales, 256, zero_signs[:-1])
    bits[0, 0] = 0x3F80  # 255 zeros -> last byte has one padding bit
    packed, scales, zero_signs, _ = pack_ternary_bits(bits)
    bad = zero_signs.copy()
    bad[-1] |= 0x80
    with pytest.raises(ReconstructError, match="padding"):
        unpack_ternary_bits(packed, scales, 256, bad)


@pytest.mark.parametrize("scale", [SCALE_SUBNORMAL_MIN, SCALE_SUBNORMAL_MAX, SCALE_MIN_NORMAL, SCALE_MAX_FINITE])
def test_extreme_scales(vendored, scale):
    rng = np.random.default_rng(scale)
    codes = rng.integers(0, 3, size=(4, 256))
    codes[:, 0] = 1  # never an all-zero group
    codes[:, 128] = 2
    bits = np.where(codes == 0, 0, scale | np.where(codes == 2, 0x8000, 0)).astype(np.uint16)
    packed, scales, _, _ = pack_ternary_bits(bits)
    assert (scales == scale).all()
    assert_matches_vendored(vendored, bits)
    assert np.array_equal(unpack_ternary_bits(packed, scales, 256), bits)


# --- unpack ---------------------------------------------------------------------
@pytest.mark.parametrize("shape", [(8, 256), (16, 768), (2, 2048)])
def test_unpack_lossless(vendored, shape):
    import torch

    rng = np.random.default_rng(11)
    bits = ternary_bits(rng, *shape, neg_zero=3, edge_groups=True)
    packed, scales, zero_signs, neg = pack_ternary_bits(bits)
    assert neg == 3 * 1
    assert np.array_equal(unpack_ternary_bits(packed, scales, shape[1], zero_signs), bits)
    un = vendored.unpack(torch.from_numpy(packed.copy()), to_torch(scales), shape[1])
    assert torch.equal(un, to_torch(bits))  # value-equal (-0.0 == +0.0)
    diff = np.flatnonzero(torch_bits(un).reshape(-1) != bits.reshape(-1))
    assert np.array_equal(diff, np.flatnonzero(bits.reshape(-1) == 0x8000))  # only -0.0 differs in bits


def test_unpack_rejects_code_3():
    packed = np.full((1, 16), -1, np.int32)  # all codes 3
    with pytest.raises(ReconstructError):
        unpack_ternary_bits(packed, np.ones((1, 2), np.uint16), 256)


# --- refusals (same as production) ------------------------------------------------
def test_refuses_two_magnitudes(vendored):
    bits = group_matrix(2, 256, 0x3F80)
    bits[1, 200] = 0x3F00  # 0.5 next to 1.0 in one group
    with pytest.raises(NotTernaryError, match="row 1, group 1"):
        pack_ternary_bits(bits, "w")
    with pytest.raises(ValueError, match="not exactly ternary"):
        vendored.pack_exact(to_torch(bits))


@pytest.mark.parametrize("k", [128, 384, 640])
def test_refuses_k_not_multiple_of_256(vendored, k):
    bits = group_matrix(2, k, 0x3F80)
    with pytest.raises(ConvertError, match="divisible by 256"):
        pack_ternary_bits(bits, "w")
    with pytest.raises(ValueError, match="divisible by 256"):
        vendored.pack_exact(to_torch(bits))


@pytest.mark.parametrize("bad", [0x7FC0, 0xFFC1, 0x7F80, 0xFF80])  # NaN, -NaN, +Inf, -Inf
def test_refuses_non_finite(bad):
    bits = group_matrix(2, 256, 0x3F80)
    bits[0, 3] = bad
    with pytest.raises(NotTernaryError, match="non-finite"):
        pack_ternary_bits(bits, "w")


def test_refuses_all_inf_group_stricter_than_production(vendored):
    # pack_exact accepts a group of only +-Inf (sign*Inf == Inf); we refuse any Inf.
    bits = group_matrix(1, 256, 0x7F80)
    vendored.pack_exact(to_torch(bits))
    with pytest.raises(NotTernaryError):
        pack_ternary_bits(bits)


def test_refuses_nan_like_production(vendored):
    bits = group_matrix(1, 256, 0x3F80)
    bits[0, 0] = 0x7FC0
    with pytest.raises(ValueError):
        vendored.pack_exact(to_torch(bits))


@pytest.mark.parametrize("arr", [np.zeros((2, 256), np.float32), np.zeros(256, np.uint16), np.zeros((1, 2, 256), np.uint16)])
def test_refuses_wrong_dtype_or_rank(arr):
    with pytest.raises(ConvertError):
        pack_ternary_bits(arr)
