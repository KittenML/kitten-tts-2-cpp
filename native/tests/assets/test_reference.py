"""The retained exact-packing reference matches its recorded source hashes."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json

from conftest import REFERENCE_DIR

from kt2n_assets.convert import PROJECTION_RE


def test_vendored_symbols_match_provenance():
    prov = json.loads((REFERENCE_DIR / "PROVENANCE.json").read_text())
    text = (REFERENCE_DIR / "upstream_packed_ternary.py").read_text()
    symbols = {}
    for node in ast.parse(text).body:
        if isinstance(node, ast.FunctionDef):
            symbols[node.name] = ast.get_source_segment(text, node)
        elif isinstance(node, ast.Assign):
            symbols[node.targets[0].id] = ast.get_source_segment(text, node)
    assert set(symbols) == set(prov["symbols"]) == {"GROUP", "PROJECTION_RE", "pack_exact", "unpack"}
    for name, segment in symbols.items():
        assert hashlib.sha256(segment.encode()).hexdigest() == prov["symbols"][name]["sha256"], name
    assert prov["commit"] == "4253ae55876262b92ae22ebc1b4bd91638d7b4c3"


def test_vendored_module_has_only_expected_top_level():
    tree = ast.parse((REFERENCE_DIR / "upstream_packed_ternary.py").read_text())
    names = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            names.append(node.name)
        elif isinstance(node, ast.Assign):
            names.append(node.targets[0].id)
    assert names == ["GROUP", "PROJECTION_RE", "pack_exact", "unpack"]


def test_projection_regex_agrees_with_production():
    spec = importlib.util.spec_from_file_location("ref", REFERENCE_DIR / "upstream_packed_ternary.py")
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    for layer in (0, 7, 27):
        for sub in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
            stem = f"layers.{layer}.{sub}"
            assert ref.PROJECTION_RE.match(stem)
            assert PROJECTION_RE.match(f"model.{stem}.weight")
    not_projections = [
        "model.layers.0.self_attn.q_norm.weight",
        "model.layers.0.self_attn.q_proj.bias",
        "lm_head.weight",
        "model.norm.weight",
        "spk_proj.0.weight",
    ]
    not_projections.append("model." + "embed_" + "tokens.weight")
    for name in not_projections:
        assert not PROJECTION_RE.match(name)


def test_audit_names_are_the_projection_set():
    audit = json.loads((REFERENCE_DIR / "ternary_audit_names.json").read_text())
    names = audit["exactly_ternary_names"]
    assert len(names) == len(set(names)) == 196 == audit["exactly_ternary"]
    assert all(PROJECTION_RE.match(n) for n in names)
    layers = {int(PROJECTION_RE.match(n).group(1)) for n in names}
    assert layers == set(range(28))
    non_ternary = audit["not_exactly_ternary_names"]
    assert len(non_ternary) == 2
    assert non_ternary[1] == "spk_proj.0.weight"
    assert non_ternary[0].startswith("model.embed_")
    assert audit["checkpoint_sha256"] == "01dc763ff30999bf0802a394f5163ceab774b5e67cc01e6d09a9f9e12a86a46b"
