"""Run with PYTHONPATH=python and KITTEN_TEST_ASSETS pointing to prepared assets."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import unittest

import numpy as np
import soundfile as sf

from kitten_tts_cpp import KittenTTS, KittenTTS2, get_model


class InterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assets = os.environ.get('KITTEN_TEST_ASSETS')
        if not assets:
            raise unittest.SkipTest('set KITTEN_TEST_ASSETS for real model tests')
        cls.assets = str(Path(assets).resolve())
        cls.model_path = os.environ.get('KITTEN_TEST_GGUF')
        cls.model = KittenTTS(cls.assets, model_path=cls.model_path, decoder='student_w4')

    @classmethod
    def tearDownClass(cls):
        cls.model.close()

    def test_metadata_and_validation(self):
        self.assertIs(KittenTTS, KittenTTS2)
        self.assertEqual(self.model.sample_rate, 24000)
        self.assertIn('Bruno', self.model.available_voices)
        self.assertIn('student_w4', self.model.available_decoders)
        for kwargs in [dict(voice='missing'), dict(advanced={'typo': 1}), dict(advanced={'chunk_chars': 0}),
                       dict(temperature=float('nan')), dict(top_k=1.5), dict(reference_text='Hello'),
                       dict(max_new_tokens=0), dict(advanced={'use_emotion': 'false'})]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.model.generate('Hello.', **kwargs)
        with self.assertRaises(NotImplementedError):
            self.model.generate('Hello.', reference='recording.wav')
        with self.assertRaises(ValueError):
            get_model(self.assets, device='cuda')
        with self.assertRaises(ValueError):
            get_model(self.assets, weights='emb4')
        self.assertIn('twelve', self.model.normalize_text('It costs $12.'))
        self.assertEqual(self.model._options('Hello.', advanced={'repetition_window': None})['--repetition-window'],
                         str(self.model.advanced_options['repetition_window']))

    def test_cli_parity_and_repeat(self):
        audio = self.model.generate('Hello again.', advanced={'seed': 1234})
        ids = self.model.last_report['chunks'][0]['generated']
        self.assertEqual(audio.dtype, np.float32)
        self.assertEqual(audio.ndim, 1)
        self.assertTrue(np.isfinite(audio).all())
        self.assertGreater(audio.size, 0)
        np.testing.assert_array_equal(audio, self.model.generate('Hello again.', advanced={'seed': 1234}))
        with tempfile.TemporaryDirectory() as d:
            wav = Path(d) / 'cli.wav'
            report = Path(d) / 'cli.json'
            cmd = [os.environ.get('KITTEN_TEST_BINARY', 'build/bin/kitten-tts'), '--assets', self.assets,
                   '--decoder', 'student_w4', '--text', 'Hello again.', '--seed', '1234', '--threads', '8',
                   '--output', str(wav), '--report', str(report)]
            if self.model_path:
                cmd += ['--model', self.model_path]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            actual, rate = sf.read(wav, dtype='float32')
            self.assertEqual(rate, self.model.sample_rate)
            self.assertEqual(ids, json.loads(report.read_text())['chunks'][0]['generated'])
            np.testing.assert_array_equal(audio, actual)
            output = self.model.generate_to_file('Hello again.', Path(d) / 'python.wav', advanced={'seed': 1234})
            written, rate = sf.read(output, dtype='float32')
            self.assertEqual(rate, 24000)
            np.testing.assert_allclose(audio, written, atol=1/32768)

    def test_stream_and_cancellation(self):
        kwargs = {'advanced': {'seed': 1234}}
        whole = self.model.generate('Hello again.', **kwargs)
        chunks = list(self.model.generate_stream('Hello again.', **kwargs))
        self.assertEqual(len(chunks), 1)
        np.testing.assert_array_equal(whole, chunks[0])
        stream = self.model.generate_stream('Hello again. Welcome home. Nice to meet you.',
                    advanced={'seed': 1234, 'chunk_chars': 15, 'chunk_min_chars': 0})
        self.assertGreater(next(stream).size, 0)
        stream.close()
        np.testing.assert_array_equal(whole, self.model.generate('Hello again.', **kwargs))

    def test_z_gil_and_close(self):
        ticks = []
        stopped = threading.Event()

        def heartbeat():
            while not stopped.wait(0.005):
                ticks.append(time.monotonic())

        thread = threading.Thread(target=heartbeat)
        thread.start()
        try:
            self.model.generate('Hello again.', advanced={'seed': 1234})
        finally:
            stopped.set()
            thread.join()
        self.assertGreater(len(ticks), 10)
        stream = self.model.generate_stream('Hello again. Welcome home. Nice to meet you.',
                    advanced={'seed': 1234, 'chunk_chars': 15, 'chunk_min_chars': 0})
        next(stream)
        self.model.close()
        stream.close()
        with self.assertRaises(RuntimeError):
            self.model.generate('Hello.')


if __name__ == '__main__':
    unittest.main()
