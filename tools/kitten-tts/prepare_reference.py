#!/usr/bin/env python3
"""Prepare a cloned voice once; the C++ runner then needs no Python."""
import argparse
import json
import pathlib
import sys

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--repo',type=pathlib.Path,required=True)
p.add_argument('--assets',type=pathlib.Path,required=True)
p.add_argument('--python-package',type=pathlib.Path,required=True)
p.add_argument('--reference',type=pathlib.Path,required=True)
p.add_argument('--transcript',required=True)
p.add_argument('--name',required=True)
p.add_argument('--threads',type=int,default=8)
a=p.parse_args()
root=pathlib.Path(__file__).resolve().parents[2]
sys.path[:0]=[str(a.python_package),str(root/'vendor/kitten-text-processing')]
import torch
from safetensors import safe_open
from kittenml.kittentts2.speaker import SpeakerEncoder
from kittenml.kittentts2.vocoder import S3Codec

torch.set_num_threads(a.threads)
config=json.loads((a.assets/'config.json').read_text())
if config['type']!='KITTEN2':raise ValueError('Only KITTEN2 is supported')
voices=json.loads((a.assets/'voices.json').read_text())
if a.name in voices:raise ValueError('Voice name already exists; use a new name')
if not a.transcript.strip():raise ValueError('Reference transcript must not be empty')
name=config.get('cpp_decoder','default')
student=None if name=='default' else a.repo/config['decoders'][name]
codec=S3Codec(device='cpu',student_weights=student)
encoder=SpeakerEncoder(a.repo/config['speaker_embedding'],device='cpu')
state={}
for shard in (a.repo/config['lm_dir']).glob('model*.safetensors'):
    with safe_open(shard,framework='pt') as f:
        for k in f.keys():
            if k.startswith('spk_proj.'):state[k[9:]]=f.get_tensor(k)
hidden=json.loads((a.repo/config['lm_dir']/'config.json').read_text())['hidden_size']
dtype=torch.bfloat16 if config.get('dtype','bf16')=='bf16' else torch.float32
proj=torch.nn.Sequential(torch.nn.Linear(config.get('speaker_embedding_dim',512),hidden),torch.nn.LayerNorm(hidden)).to(dtype).eval()
proj.load_state_dict(state,strict=True)
with torch.inference_mode():
    spk=proj(torch.from_numpy(encoder(a.reference)).reshape(1,-1).to(dtype)).float().flatten()
    cond=codec.reference_conditioning(a.reference)
    ref_tokens=codec.encode_reference(a.reference)
voices[a.name]=dict(transcript=a.transcript.strip(),reference_tokens=ref_tokens,speaker=spk.tolist(),
    prompt_token=cond['prompt_token'].long().tolist(),prompt_feat=cond['prompt_feat'].float().tolist(),embedding=cond['embedding'].float().tolist())
output=a.assets/'voices.json'
temporary=output.with_suffix('.json.tmp')
temporary.write_text(json.dumps(voices));temporary.replace(output)
print('Prepared voice:',a.name)
