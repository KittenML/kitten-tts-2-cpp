# From kitten-inference (KittenML, Apache License 2.0): replaces the torch + transformers call in english.py.
"""The English fallback G2P (a character-level BART, g2p/g2p_bart_en_us in each model repository) in numpy.

Same computation as transformers' BartForConditionalGeneration.generate(input_ids, max_length=config
.max_position_embeddings, num_beams=1, do_sample=False): learned positions with offset 2, layernorm_embedding,
post-norm encoder/decoder layers, exact (erf) GELU, tied output embedding plus final_logits_bias, greedy decoding
from decoder_start_token_id, EOS forced at max_length - 1. Keeps torch out of the text front end.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from safetensors.numpy import load_file as load_safetensors

_erf = np.vectorize(math.erf, otypes=[np.float64])


def _gelu(x):
    return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))


def _ln(x, w, b, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * w + b


class BartG2P:
    def __init__(self, model_dir):
        with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
            self.config = cfg = json.load(f)
        if cfg.get("encoder_layers") is None or cfg.get("model_type") != "bart":
            raise ValueError(f"{model_dir}: not a BART config")
        if cfg.get("activation_function", "gelu") != "gelu":
            raise ValueError("only the erf GELU BART is supported")
        self.w = {k: v.astype(np.float64) for k, v in load_safetensors(os.path.join(model_dir, "model.safetensors")).items()}
        self.d = int(cfg["d_model"])
        self.heads_e = int(cfg["encoder_attention_heads"])
        self.heads_d = int(cfg["decoder_attention_heads"])
        self.n_enc, self.n_dec = int(cfg["encoder_layers"]), int(cfg["decoder_layers"])
        self.scale_emb = math.sqrt(self.d) if cfg.get("scale_embedding") else 1.0
        self.max_len = int(cfg["max_position_embeddings"])
        self.start = int(cfg.get("decoder_start_token_id", 2))
        self.eos = int(cfg["eos_token_id"])
        self.forced_eos = cfg.get("forced_eos_token_id")
        self.emb = self.w["model.shared.weight"]
        self.bias = self.w.get("final_logits_bias", np.zeros((1, self.emb.shape[0])))[0]

    def _lin(self, x, p):
        return x @ self.w[p + ".weight"].T + self.w[p + ".bias"]

    def _attn(self, x, kv, p, heads, causal):
        T, S, d = x.shape[0], kv.shape[0], self.d
        hd = d // heads
        q = self._lin(x, p + ".q_proj") * (hd ** -0.5)
        k = self._lin(kv, p + ".k_proj")
        v = self._lin(kv, p + ".v_proj")
        q = q.reshape(T, heads, hd).transpose(1, 0, 2)
        k = k.reshape(S, heads, hd).transpose(1, 0, 2)
        v = v.reshape(S, heads, hd).transpose(1, 0, 2)
        s = q @ k.transpose(0, 2, 1)
        if causal:
            s = s + np.triu(np.full((T, S), -np.inf), 1)
        s = s - s.max(-1, keepdims=True)
        a = np.exp(s)
        a /= a.sum(-1, keepdims=True)
        o = (a @ v).transpose(1, 0, 2).reshape(T, d)
        return self._lin(o, p + ".out_proj")

    def _embed(self, ids, side):
        pos = self.w[f"model.{side}.embed_positions.weight"][np.arange(len(ids)) + 2]
        x = self.emb[ids] * self.scale_emb + pos
        return _ln(x, self.w[f"model.{side}.layernorm_embedding.weight"], self.w[f"model.{side}.layernorm_embedding.bias"])

    def encode(self, ids):
        x = self._embed(np.asarray(ids), "encoder")
        for i in range(self.n_enc):
            p = f"model.encoder.layers.{i}"
            x = _ln(x + self._attn(x, x, p + ".self_attn", self.heads_e, False),
                    self.w[p + ".self_attn_layer_norm.weight"], self.w[p + ".self_attn_layer_norm.bias"])
            h = self._lin(_gelu(self._lin(x, p + ".fc1")), p + ".fc2")
            x = _ln(x + h, self.w[p + ".final_layer_norm.weight"], self.w[p + ".final_layer_norm.bias"])
        return x

    def _decode_last(self, dec_ids, enc):
        x = self._embed(np.asarray(dec_ids), "decoder")
        for i in range(self.n_dec):
            p = f"model.decoder.layers.{i}"
            x = _ln(x + self._attn(x, x, p + ".self_attn", self.heads_d, True),
                    self.w[p + ".self_attn_layer_norm.weight"], self.w[p + ".self_attn_layer_norm.bias"])
            x = _ln(x + self._attn(x, enc, p + ".encoder_attn", self.heads_d, False),
                    self.w[p + ".encoder_attn_layer_norm.weight"], self.w[p + ".encoder_attn_layer_norm.bias"])
            h = self._lin(_gelu(self._lin(x, p + ".fc1")), p + ".fc2")
            x = _ln(x + h, self.w[p + ".final_layer_norm.weight"], self.w[p + ".final_layer_norm.bias"])
        return x[-1] @ self.emb.T + self.bias

    def generate(self, input_ids, max_length=None):
        """Greedy decoding; returns the decoder ids including the start token (as transformers' generate)."""
        max_length = max_length or self.max_len
        enc = self.encode(input_ids)
        out = [self.start]
        while len(out) < max_length:
            logits = self._decode_last(out, enc)
            if self.forced_eos is not None and len(out) == max_length - 1:
                nxt = int(self.forced_eos)
            else:
                nxt = int(np.argmax(logits))
            out.append(nxt)
            if nxt == self.eos:
                break
        return out
