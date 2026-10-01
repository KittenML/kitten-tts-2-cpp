#!/usr/bin/env python3
"""Differential checks against the supplied KittenTTS2 Python package."""
import argparse
import json
import pathlib
import subprocess
import sys


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--binary', type=pathlib.Path, required=True)
    p.add_argument('--python-package',type=pathlib.Path,required=True)
    p.add_argument('--assets',type=pathlib.Path)
    p.add_argument('--repo',type=pathlib.Path)
    p.add_argument('--model',type=pathlib.Path)
    p.add_argument('--lm',action='store_true')
    p.add_argument('--decoder',action='store_true')
    a=p.parse_args()
    root=pathlib.Path(__file__).resolve().parents[2]
    sys.path[:0]=[str(a.python_package),str(root/'vendor/kitten-text-processing')]
    from kittenml.kittentts2 import text as reference
    cases=[
        '', 'hello world', 'Dr. Rivera paid $12.50 at 3:05 p.m.',
        'Smith et al. 2024, pp. 31-35', 'Fig. 2', 'May 5, 2026', '10:30 AM',
        '$1,250.00', '9%', 'V0.8 is here!', 'v1.2.3', 'Version 2.4',
        '[joyful] We won $12.50 <laugh> I can (((hardly))) believe it!',
        '[ANGRY] why 123?! <sigh>', 'section [3] and (equation 2), x < 5',
        'He said "42". The TV is 50" wide.', 'Hello.\nWe meet again.',
        'Hello.There!Again?Yes:Now;Fine', 'First -- second — third – fourth.',
        'α² + β₂ = 9, x ≠ π and ∑ x → ∞', '• one\n• two',
        'Wait . . . what… really...', 'hi, ;:', '(((very))) good',
        'Visit https://example.com or email hello@example.com.',
        'café déjà vu. Übermäßig schön!', '  leading\tspace\nnew line  ',
        'a'*600, ('A rather long sentence, with a clause; and another. '*30),
        ('"Hello world!" Next sentence? “Indeed.” Yes! '*30),
        ('🙂 café 世界 and more, '*60), '<unknown> [citation] (aside).',
        '\ue0007\ue001 test', 'Just <pause> wait', 'hello\u00a0world\u2003again', 'line.\u2003Next sentence!', '٣?!', '(((123))) [sad] $20',
    ]
    requests=[]
    for text in cases:
        for normalize in [True,False]:
            for max_chars,min_chars in [(380,130),(60,20),(0,0)]:
                requests.append(dict(text=text,normalize=normalize,chunk_chars=max_chars,chunk_min_chars=min_chars))
    proc=subprocess.run([str(a.binary),'--frontend'],input=''.join(json.dumps(r)+'\n' for r in requests),text=True,capture_output=True,check=True)
    results=[json.loads(x) for x in proc.stdout.splitlines()]
    assert len(results)==len(requests)
    failures=[]
    for request,result in zip(requests,results):
        normalized=reference.normalize_text(request['text'],normalize=request['normalize'])
        expected=dict(normalized=normalized,chunks=reference.split_for_synthesis(normalized,request['chunk_chars'],request['chunk_min_chars']),expression=reference.has_expression_tag(request['text']))
        if expected!=result:failures.append(dict(request=request,expected=expected,actual=result))
    print('Frontend:',len(requests)-len(failures),'/',len(requests),'passed')
    if failures:
        print(json.dumps(failures[:10],ensure_ascii=False,indent=2));raise SystemExit(1)
    check_sampling(a)
    check_join(a,reference)
    if a.lm: check_lm(a,reference)
    if a.decoder: check_decoder(a)


def check_join(a,reference):
    import numpy as np
    rng=np.random.default_rng(3)
    cases=[[[0.1]*30], [[],[0.0]*2000,[0.0]*1300],
           [np.pad(rng.normal(0,.1,3000).astype(np.float32),(900,1200)).tolist(),
            np.pad(rng.normal(0,.1,3200).astype(np.float32),(1000,2000)).tolist()]]
    requests=[dict(operation='join',waves=waves,gap=.16) for waves in cases]
    out=subprocess.run([str(a.binary),'--frontend'],input=''.join(json.dumps(r)+'\n' for r in requests),text=True,capture_output=True,check=True)
    for line,waves in zip(out.stdout.splitlines(),cases):
        expected=reference.join_chunks([np.asarray(w,np.float32) for w in waves],24000)
        np.testing.assert_allclose(json.loads(line)['audio'],expected,atol=1e-7,rtol=1e-6)
    assert len(out.stdout.splitlines())==len(cases)
    print('Audio joins:',len(cases),'cases passed')


def check_sampling(a):
    import numpy as np
    import torch
    from kittenml.kittentts2.tokens import TokenMap
    from kittenml.kittentts2.logits import build_logits_processors
    cfg=json.loads((a.repo/'config.json').read_text()) if a.repo else {
        'token_map': dict(audio_id_base=20,num_audio_tokens=6561,speech_start_id=1,
                          speech_end_id=2,text_start_id=3,text_end_id=4,start_id=5,stop_id=6)}
    tm=TokenMap.from_config(cfg)
    n=tm.audio_id_end+10
    rng=np.random.default_rng(42)
    requests=[]; expected=[]
    for window in [0,5,50]:
        for top_k,top_p,min_p in [(0,1.0,0),(50,.8,0),(50,.9,.05),(0,.95,.1)]:
            history=[tm.start_id,tm.audio_id_base+2,tm.audio_id_base+4299]*5+[tm.audio_id_base+4299]*12
            opts=dict(temperature=.8,top_k=top_k,top_p=top_p,min_p=min_p,repetition_penalty=1.1,
                      repetition_window=window,token_run_penalty=1.3,token_run_grace=10)
            scores=rng.normal(size=n).astype(np.float32)
            proc=build_logits_processors(tm,prompt_length=15,**opts)
            expected.append(proc(torch.tensor([history]),torch.from_numpy(scores.copy())[None])[0].numpy())
            requests.append(dict(operation='sampling',scores=scores.tolist(),history=history,
                                 token_map=cfg['token_map'],settings=opts,prompt_length=15))
    out=subprocess.run([str(a.binary),'--frontend'],input=''.join(json.dumps(r)+'\n' for r in requests),text=True,capture_output=True,check=True)
    for line,ref in zip(out.stdout.splitlines(),expected):
        actual=np.array([float('-inf') if x is None else x for x in json.loads(line)['scores']],np.float32)
        np.testing.assert_array_equal(np.isfinite(actual),np.isfinite(ref))
        np.testing.assert_allclose(actual[np.isfinite(ref)],ref[np.isfinite(ref)],atol=1e-6,rtol=1e-6)
    assert len(out.stdout.splitlines())==len(expected)
    print('Sampling:',len(expected),'cases passed')


def check_decoder(a):
    import numpy as np
    import torch
    import soundfile as sf
    from kittenml.kittentts2.vocoder import S3Codec
    torch.set_num_threads(8)
    cfg=json.loads((a.assets/'config.json').read_text())
    name=cfg.get('cpp_decoder','default')
    student=None if name=='default' else a.repo/cfg['decoders'][name]
    codec=S3Codec(device='cpu',student_weights=student)
    voices=json.loads((a.assets/'voices.json').read_text())
    results=[]
    for voice,n in [('Bruno',7),('Kiki',51),('Bruno',113)]:
        ids=[4299,123,80,100,9,332,2000]*(n//7+1);ids=ids[:n]
        ref=voices[voice]
        cond=dict(prompt_token=torch.tensor(ref['prompt_token']),prompt_token_len=torch.tensor([len(ref['prompt_token'][0])]),
                  prompt_feat=torch.tensor(ref['prompt_feat']),embedding=torch.tensor(ref['embedding']))
        if student is None: cond['prompt_feat_len']=None
        torch.manual_seed(1234)
        expected=codec.decode(cond,ids)
        stem=a.assets/('decoder-check-'+voice+str(n))
        stem.with_suffix('.json').write_text(json.dumps(ids))
        subprocess.run([str(a.binary),'--assets',str(a.assets),'--voice',voice,'--seed','1234',
                        '--decode-tokens',str(stem.with_suffix('.json')),'--output',str(stem.with_suffix('.wav'))],check=True,capture_output=True)
        actual,sr=sf.read(stem.with_suffix('.wav'),dtype='float32')
        assert sr==24000
        np.testing.assert_allclose(actual,expected,atol=2e-4,rtol=2e-4)
        results.append(dict(voice=voice,tokens=n,samples=len(actual),max_error=float(np.max(np.abs(actual-expected)))))
        print(results[-1])
    (a.assets/'native-decoder-parity.json').write_text(json.dumps(results,indent=2))


def check_lm(a,text_reference):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from kittenml.kittentts2.prompt import build_generation_prompt,build_reference_prefix
    from kittenml.kittentts2.tokens import TokenMap
    from kittenml.kittentts2.logits import build_logits_processors
    torch.set_num_threads(8)
    cfg=json.loads((a.repo/'config.json').read_text())
    voices=json.loads((a.assets/'voices.json').read_text())
    tm=TokenMap.from_config(cfg)
    tok=AutoTokenizer.from_pretrained(a.repo/cfg['lm_dir'])
    model=AutoModelForCausalLM.from_pretrained(a.repo/cfg['lm_dir'],dtype=torch.float32,attn_implementation='sdpa').eval()
    checks=[]
    for voice,text in [('Bruno','Hello there. This is a CPU parity test.'),('Kiki','[joyful] We won $12.50 <laugh> Amazing!')]:
        ref=voices[voice]
        text=text_reference.normalize_text(text)
        prefix=build_reference_prefix(tm,tok.encode(ref['transcript'].strip(),add_special_tokens=False),ref['reference_tokens'])
        emotion=tok.encode(cfg['generation']['emotion_control'],add_special_tokens=False) if text_reference.has_expression_tag(text) else None
        prompt=build_generation_prompt(tm,tok.encode(text,add_special_tokens=False),prefix,emotion)
        forced=[tm.audio_id_base+x for x in [4299,1,12,200,4299,4299,4299,4299,4299,4299,4299,4299,4299,4299,4299]]
        path=a.assets/('parity-'+voice)
        path.with_suffix('.tokens.json').write_text(json.dumps(forced))
        cmd=[str(a.binary),'--assets',str(a.assets),'--text',text,'--voice',voice,'--tokens-only','--force-tokens',str(path.with_suffix('.tokens.json')),'--logits',str(path.with_suffix('.f32')),'--report',str(path.with_suffix('.json'))]
        if a.model:cmd+=['--model',str(a.model)]
        subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        report=json.loads(path.with_suffix('.json').read_text())
        assert report['chunks'][0]['prompt']==prompt, 'prompt/tokenizer mismatch'
        with torch.inference_mode():
            inp=torch.tensor([prompt+forced])
            emb=model.get_input_embeddings()(inp)
            emb[:,0]=torch.tensor(ref['speaker'])
            logits=model(inputs_embeds=emb).logits[0,len(prompt)-1:].float().numpy()
        actual=np.fromfile(path.with_suffix('.f32'),dtype=np.float32).reshape(-1,logits.shape[1])
        error=actual-logits
        cosine=np.sum(actual*logits,axis=1)/(np.linalg.norm(actual,axis=1)*np.linalg.norm(logits,axis=1))
        # CPU GGML uses different accumulation and quantized activation kernels.
        assert cosine.min()>0.999,cosine
        assert np.sqrt(np.mean(error**2))<0.15,np.sqrt(np.mean(error**2))
        result=dict(voice=voice,prompt_tokens=len(prompt),steps=len(actual),max_abs=float(abs(error).max()),rmse=float(np.sqrt(np.mean(error**2))),min_cosine=float(cosine.min()),argmax_agreement=float(np.mean(actual.argmax(1)==logits.argmax(1))))
        model.to(torch.bfloat16)
        with torch.inference_mode():
            emb=model.get_input_embeddings()(inp)
            emb[:,0]=torch.tensor(ref['speaker'],dtype=torch.bfloat16)
            bf16=model(inputs_embeds=emb).logits[0,len(prompt)-1:].float().numpy()
        model.to(torch.float32)
        bf_cos=np.sum(actual*bf16,axis=1)/(np.linalg.norm(actual,axis=1)*np.linalg.norm(bf16,axis=1))
        assert bf_cos.min()>0.999
        result['bf16_rmse']=float(np.sqrt(np.mean((actual-bf16)**2)))
        result['bf16_min_cosine']=float(bf_cos.min())
        result['bf16_argmax_agreement']=float(np.mean(actual.argmax(1)==bf16.argmax(1)))
        checks.append(result);print(result)
    (a.assets/('lm-parity-'+(a.model.stem if a.model else 'default')+'.json')).write_text(json.dumps(checks,indent=2))

if __name__=='__main__':main()
