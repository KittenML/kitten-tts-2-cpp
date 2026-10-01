#!/usr/bin/env python3
"""Verify every packed tensor against the FP16 reference GGUF."""
import argparse
import json
import pathlib
import sys
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]/'gguf-py'))
import gguf
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--reference',required=True)
p.add_argument('--packed',required=True)
p.add_argument('--report',type=pathlib.Path)
a=p.parse_args()
reference=gguf.GGUFReader(a.reference)
packed=gguf.GGUFReader(a.packed)
expected={t.name:t for t in reference.tensors}
assert {t.name for t in packed.tensors}==set(expected)
counts={}
for t in packed.tensors:
    r=expected[t.name]
    actual=gguf.quants.dequantize(t.data,t.tensor_type)
    wanted=gguf.quants.dequantize(r.data,r.tensor_type)
    np.testing.assert_array_equal(actual,wanted,err_msg=t.name)
    name=t.tensor_type.name
    counts[name]=counts.get(name,0)+1
result=dict(tensors=len(packed.tensors),types=counts,max_error=0.0)
print(json.dumps(result,indent=2))
if a.report:a.report.write_text(json.dumps(result,indent=2))
