"""Check lossless TQ2_1 export and the native GGML byte layout."""
import ctypes
import pathlib
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'gguf-py'))
import gguf


class TestTQ2(unittest.TestCase):
    def test_lossless_native_layout(self):
        lib = ctypes.CDLL(str(ROOT / 'build/bin/libggml-base.so'))
        quantize = lib.quantize_row_tq2_1_ref
        dequantize = lib.dequantize_row_tq2_1
        for fn in (quantize, dequantize):
            fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
            fn.restype = None
        rng = np.random.default_rng(123)
        scales = np.array([0, .03125, .5, 2, 2**-24, 65504], np.float32)
        weights = rng.integers(-1, 2, (12, 128)).astype(np.float32) * np.tile(scales, 2)[:, None]
        weights[:, :3] = np.array([-1, 0, 1]) * np.tile(scales, 2)[:, None]
        weights = weights.reshape(3, 512)
        packed = gguf.quantize(weights, gguf.GGMLQuantizationType.TQ2_1)
        self.assertEqual(packed.shape, (3, 136))
        np.testing.assert_array_equal(gguf.dequantize(packed, gguf.GGMLQuantizationType.TQ2_1), weights)
        native = np.empty_like(packed)
        quantize(weights.ctypes.data, native.ctypes.data, weights.size)
        np.testing.assert_array_equal(native, packed)
        restored = np.empty_like(weights)
        dequantize(packed.ctypes.data, restored.ctypes.data, weights.size)
        np.testing.assert_array_equal(restored, weights)

    def test_reject_lossy_input(self):
        for value in [.25, float('nan'), float('inf')]:
            weights = np.ones((1, 256), np.float32)
            weights[0, 1] = value
            with self.assertRaises(ValueError):
                gguf.quantize(weights, gguf.GGMLQuantizationType.TQ2_1)
        weights = np.full((1, 256), .1234567, np.float32)
        with self.assertRaises(ValueError):
            gguf.quantize(weights, gguf.GGMLQuantizationType.TQ2_1)


if __name__ == '__main__':
    unittest.main()
