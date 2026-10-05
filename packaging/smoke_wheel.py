"""Exercise an installed wheel without downloading model weights or importing source code."""
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
import kitten_tts_cpp
from kitten_tts_cpp import _native

package = Path(kitten_tts_cpp.__file__).parent
info = json.loads((package / '_build_info.json').read_text())
assert info['portable'], info
assert info['torch_version'] == torch.__version__.split('+')[0]
assert (package / 'data/en/tagger.fst').is_file()
assert (package / 'licenses/vendor_kitten-text-processing_LICENSE').is_file()


class Decoder(torch.nn.Module):
    def forward(self, tokens, reference, features, embedding):
        return tokens.float().flatten() / 10000 + features.sum() * 0 + embedding.sum() * 0 + reference.sum() * 0


with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    inputs = (torch.tensor([[1, 2]]), torch.tensor([[0]]), torch.zeros(1, 1, 80), torch.zeros(1, 2))
    torch.jit.trace(Decoder(), inputs).save(str(root / 'decoder.pt'))
    (root / 'config.json').write_text(json.dumps({
        'type': 'KITTEN2', 'default_voice': 'Test', 'token_map': {
            'audio_id_base': 10, 'num_audio_tokens': 6561,
            'speech_start_id': 1, 'speech_end_id': 2, 'text_start_id': 3,
            'start_id': 4, 'stop_id': 5,
        },
    }))
    (root / 'voices.json').write_text(json.dumps({'Test': {
        'prompt_token': [[0]], 'prompt_feat': [[[0.0] * 80]], 'embedding': [[0.0, 0.0]],
    }}))
    tokens = root / 'tokens.json'
    tokens.write_text('[1, 2]')
    model = _native.Engine({'--assets': str(root), '--decode-tokens': str(tokens),
                            '--data': str(package / 'data'), '--threads': '1'})
    assert 'twelve' in model.normalize('It costs $12.').lower()
    audio, report = model.generate({'--seed': '1234'})
    assert audio.dtype == np.float32 and audio.shape == (5,)
    np.testing.assert_allclose(audio, np.array([1, 2, 4299, 4299, 4299], dtype=np.float32) / 10000)
    assert json.loads(report)['samples'] == 5
    del model
print('Installed wheel: import, ABI, grammar, TorchScript decoding, and NumPy output passed')

if os.environ.get('KITTEN_WHEEL_NETWORK_TEST') == '1':
    with tempfile.TemporaryDirectory() as cache:
        try:
            kitten_tts_cpp.KittenTTS('gpt2', cache_dir=cache)
        except RuntimeError:
            pass
        config = Path(cache) / 'gpt2/main/config.json'
        assert config.is_file(), 'Native HTTPS config download failed'
        assert json.loads(config.read_text())['model_type'] == 'gpt2'
    print('Native HTTPS config download passed')

if os.environ.get('KITTEN_WHEEL_MODEL_TEST') == '1':
    with kitten_tts_cpp.KittenTTS(decoder='student_w4', threads=2) as model:
        audio = model.generate('Hello from Kitten T T S.', max_new_tokens=100, advanced={'seed': 1234})
        assert audio.dtype == np.float32 and audio.size > 0 and np.isfinite(audio).all()
    print('Real W4 model synthesis passed')
