#!/usr/bin/env python3
"""Export KittenTTS2 assets. Python is used only for conversion and validation."""
import argparse
import contextlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'gguf-py'), str(ROOT / 'vendor/kitten-text-processing')]


def pack_ternary_q4(data):
    """Store exact {-gamma, 0, gamma} values in standard Q4_0 blocks."""
    import numpy as np
    shape = data.shape
    blocks = data.reshape(-1, 32).astype(np.float32)
    scales = np.abs(blocks).max(axis=1, keepdims=True)
    codes = np.sign(blocks).astype(np.int8)
    restored = codes * scales.astype(np.float16).astype(np.float32)
    if not np.array_equal(restored, blocks):
        raise ValueError('Weights are not losslessly representable as ternary Q4_0')
    biased = (codes + 8).astype(np.uint8)
    packed = biased[:, :16] | (biased[:, 16:] << 4)
    result = np.concatenate([scales.astype(np.float16).view(np.uint8), packed], axis=1)
    return result.reshape(*shape[:-1], shape[-1] // 32 * 18)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', type=pathlib.Path, required=True)
    p.add_argument('--python-package', type=pathlib.Path, required=True)
    p.add_argument('--output', type=pathlib.Path, required=True)
    p.add_argument('--outtype', choices=['f16', 'f32', 'ternary-q4_0', 'tq2_1'], default='tq2_1')
    p.add_argument('--stage', choices=['lm', 'decoder', 'all'], default='all')
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--decoder', choices=['default','student_w4','student_w8'], default='default')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu',
                   help='device to trace the decoder on; a CUDA decoder runs only with --decoder-device cuda')
    p.add_argument('--rng', choices=['device', 'cpu'], default='device',
                   help='cpu: draw decoder noise from the CPU generator and copy it to the device, '
                        'so one seed gives the same noise as a CPU decoder (for parity checks)')
    p.add_argument('--reuse-voices', type=pathlib.Path,
                   help='prepared voices.json to keep (for example the published cpp/<decoder>/voices.json) '
                        'instead of recomputing speaker projections and S3 conditioning')
    a = p.parse_args()
    if a.rng == 'cpu' and a.decoder != 'default':
        # student_w4 fails the trace self-check with CPU-drawn noise (max error 4.6e-4 > 2e-4); student_w8 is not checked
        p.error('--rng cpu is supported only with --decoder default; use --rng device for student decoders')
    sys.path.insert(0, str(a.python_package))
    import torch
    torch.set_num_threads(a.threads)
    config = json.loads((a.repo / 'config.json').read_text())
    config.pop('cpp', None)
    if config['type'] != 'KITTEN2':
        raise ValueError('Only KITTEN2 checkpoints are supported')
    a.output.mkdir(parents=True, exist_ok=True)
    existing = a.output / 'config.json'
    previous = json.loads(existing.read_text()) if existing.exists() else {}
    config['cpp_decoder'] = a.decoder if a.stage != 'lm' else previous.get('cpp_decoder', 'default')
    (a.output / 'config.json').write_text(json.dumps(config, indent=2))
    if a.stage in ('lm', 'all'):
        from conversion.qwen import Qwen3Model
        import gguf
        class KittenModel(Qwen3Model):
            model_arch = Qwen3Model.model_arch
            def modify_tensors(self, data_torch, name, bid):
                if name.startswith('spk_proj.'):
                    return
                yield from super().modify_tensors(data_torch, name, bid)

            def set_gguf_parameters(self):
                super().set_gguf_parameters()
                if a.outtype == 'tq2_1':
                    self.gguf_writer.add_file_type(gguf.LlamaFileType.MOSTLY_TQ2_1)
                if a.outtype == 'ternary-q4_0':
                    self.gguf_writer.add_file_type(gguf.LlamaFileType.MOSTLY_Q4_0)

            def tensor_force_quant(self, name, new_name, bid, n_dims):
                if name.startswith('model.layers.') and n_dims == 2:
                    if a.outtype == 'tq2_1':
                        return gguf.GGMLQuantizationType.TQ2_1
                    if a.outtype == 'ternary-q4_0':
                        return gguf.GGMLQuantizationType.Q4_0
                return super().tensor_force_quant(name, new_name, bid, n_dims)
        types = {'f16': gguf.LlamaFileType.MOSTLY_F16, 'f32': gguf.LlamaFileType.ALL_F32,
                 'ternary-q4_0': gguf.LlamaFileType.MOSTLY_F16, 'tq2_1': gguf.LlamaFileType.MOSTLY_F16}
        if a.outtype == 'ternary-q4_0':
            original_quantize = gguf.quants.quantize
            def quantize(data, qtype):
                if qtype == gguf.GGMLQuantizationType.Q4_0:
                    import numpy as np
                    from gguf.lazy import LazyNumpyTensor
                    if isinstance(data, LazyNumpyTensor):
                        pack = LazyNumpyTensor._wrap_fn(pack_ternary_q4, meta_noop=(np.uint8, lambda shape: (*shape[:-1], shape[-1] // 32 * 18)))
                        return pack(data)
                    return pack_ternary_q4(data)
                return original_quantize(data, qtype)
            gguf.quants.quantize = quantize
        model = KittenModel(a.repo / config.get('lm_dir', 'lm'), types[a.outtype],
                            a.output / ('model-' + a.outtype + '.gguf'))
        model.write()
    if a.stage in ('decoder', 'all'):
        export_decoder(a, config)


def export_decoder(a, config):
    import torch
    from kittenml.kittentts2.vocoder import S3Codec
    from kittenml.kittentts2 import s3gen
    # The reference .item() freezes sequence length when tracing. This equivalent
    # mask keeps arange's bound in the graph for variable-length C++ inference.
    def make_pad_mask(lengths, max_len=0):
        bound = max_len if max_len > 0 else lengths.max()
        return torch.arange(bound, device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(1)
    s3gen.make_pad_mask = make_pad_mask
    s3gen.tqdm = lambda iterable, **kwargs: iterable
    student_weights = None if a.decoder == 'default' else a.repo / config['decoders'][a.decoder]
    if a.device == 'cuda':
        # Match the CPU decoder: FP32 GEMMs and convolutions without TF32.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    codec = S3Codec(device=a.device, student_weights=student_weights)
    from kittenml.kittentts2 import student_flow
    student_flow.make_pad_mask = make_pad_mask
    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.flow = codec.s3gen.flow if codec.student is None else codec.student
            self.hift = codec.s3gen.mel2wav
            self.register_buffer('fade', codec.s3gen.trim_fade)
        def forward(self, tokens, prompt_token, prompt_feat, embedding):
            lengths = torch.ones(1, dtype=torch.long, device=tokens.device) * tokens.shape[1]
            prompt_lengths = torch.ones(1, dtype=torch.long, device=tokens.device) * prompt_token.shape[1]
            if codec.student is None:
                rng_device = 'cpu' if a.rng == 'cpu' else tokens.device
                noise = torch.randn(1, 80, tokens.shape[1] * 2, device=rng_device).to(tokens.device)
                mel, _ = self.flow.inference(tokens, lengths, prompt_token, prompt_lengths,
                    prompt_feat, None, embedding, True, n_timesteps=2, noised_mels=noise, meanflow=True)
            else:
                mel, _, _ = student_flow.meanflow_forward(self.flow, tokens, lengths,
                    prompt_token, prompt_lengths, prompt_feat, embedding, n_steps=1)
            wav, _ = self.hift.inference(mel, torch.zeros(1, 1, 0, device=tokens.device))
            wav[:, :self.fade.shape[0]] *= self.fade
            return wav
    if a.reuse_voices:
        export_voices = json.loads(a.reuse_voices.read_text())
        voice = next(iter(export_voices.values()))
        args = (torch.full((1, 53), 4299, dtype=torch.long), torch.tensor(voice['prompt_token'], dtype=torch.long),
                torch.tensor(voice['prompt_feat'], dtype=torch.float32), torch.tensor(voice['embedding'], dtype=torch.float32))
        print('Reusing', len(export_voices), 'voices from', a.reuse_voices, flush=True)
    else:
        export_voices, args = export_conditioning(a, config, codec)
    (a.output / 'voices.json').write_text(json.dumps(export_voices))
    args = tuple(x.to(a.device) for x in args)
    decoder = Decoder().eval()
    metadata = dict(device=a.device, rng=a.rng, decoder=a.decoder, torch=torch.__version__, sinegen='device')
    with torch.inference_mode(), rng_on_cpu(a.rng == 'cpu'), sinegen_on_device(s3gen):
        traced = torch.jit.trace(decoder, args, check_trace=False)
        traced = torch.jit.freeze(traced)
        traced.save(str(a.output / 'decoder.pending.pt'), _extra_files={'kitten_decoder.json': json.dumps(metadata)})
        results = []
        for n in [4, 17, 53, 101]:
            inputs = (torch.full((1, n), 4299, dtype=torch.long, device=a.device), *args[1:])
            torch.manual_seed(123)
            expected = decoder(*inputs)
            torch.manual_seed(123)
            actual = traced(*inputs)
            torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
            results.append(dict(tokens=n, samples=actual.numel(), max_error=float((actual-expected).abs().max())))
        (a.output / 'decoder.pending.pt').replace(a.output / 'decoder.pt')
        (a.output / 'decoder-parity.json').write_text(json.dumps(dict(metadata, results=results), indent=2))
        print(results)


@contextlib.contextmanager
def rng_on_cpu(enabled):
    """Trace randn_like as a CPU draw copied to the tensor's device, so a CUDA decoder uses the same noise as the CPU decoder for one seed.

    The CPU and CUDA generators give different numbers for one seed. The torch.rand calls in s3gen already use CPU tensors."""
    import torch
    saved = torch.randn_like
    if enabled:
        torch.randn_like = lambda x, **kw: saved(x, **dict(kw, device='cpu')).to(x.device)
    try:
        yield
    finally:
        torch.randn_like = saved


@contextlib.contextmanager
def sinegen_on_device(s3gen):
    """Create SineGen's harmonic buffer (9 floats per output sample) on f0's device.

    The reference creates it with torch.zeros on the CPU and copies it to f0's device on every call. The values are the same."""
    import torch
    forward = s3gen.SineGen.forward
    def patched(self, f0):
        zeros = torch.zeros
        torch.zeros = lambda *args, **kw: zeros(*args, **{'device': f0.device, **kw})
        try:
            return forward(self, f0)
        finally:
            torch.zeros = zeros
    s3gen.SineGen.forward = patched
    try:
        yield
    finally:
        s3gen.SineGen.forward = forward


def export_conditioning(a, config, codec):
    import numpy as np
    import torch
    from safetensors import safe_open
    voices_path = a.repo / config['voices']
    voices = json.loads(voices_path.read_text())
    head = {}
    for shard in sorted((a.repo / config['lm_dir']).glob('model*.safetensors')):
        with safe_open(shard, framework='pt') as f:
            for key in f.keys():
                if key.startswith('spk_proj.'):
                    head[key[9:]] = f.get_tensor(key)
    hidden = json.loads((a.repo / config['lm_dir'] / 'config.json').read_text())['hidden_size']
    dtype = torch.bfloat16 if config.get('dtype', 'bf16') == 'bf16' else torch.float32
    proj = torch.nn.Sequential(torch.nn.Linear(config.get('speaker_embedding_dim',512), hidden), torch.nn.LayerNorm(hidden)).to(dtype).eval()
    proj.load_state_dict(head, strict=True)
    export_voices = {}
    args = None
    for name, voice in voices.items():
        art = np.load(voices_path.parent / voice['artifacts'])
        with torch.inference_mode():
            spk = proj(torch.from_numpy(art['embedding']).reshape(1, -1).to(dtype)).float().flatten()
            cond = codec.reference_conditioning(voices_path.parent / voice['reference'])
        export_voices[name] = dict(transcript=voice['transcript'].strip(),
            reference_tokens=art['reference_tokens'].tolist(), speaker=spk.tolist(),
            prompt_token=cond['prompt_token'].long().tolist(), prompt_feat=cond['prompt_feat'].float().tolist(),
            embedding=cond['embedding'].float().tolist())
        args = (torch.full((1, 53), 4299, dtype=torch.long), cond['prompt_token'].long(),
                cond['prompt_feat'].float(), cond['embedding'].float())
        print('Exported voice', name, flush=True)
    return export_voices, args


if __name__ == '__main__':
    main()
