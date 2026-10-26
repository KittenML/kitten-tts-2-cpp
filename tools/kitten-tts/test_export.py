"""Run in the environment from requirements-export.txt."""
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from export import full_precision_tensors
from conversion.base import LazyTorchTensor


class CheckpointTests(unittest.TestCase):
    def test_full_checkpoint_ignores_packed_alternatives(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = {'model.embed_tokens.weight': torch.arange(6).reshape(2, 3).float(),
                       'model.layers.0.mlp.down_proj.weight': torch.ones(2, 3)}
            save_file(weights, root / 'model.safetensors')
            save_file({'model.layers.0.mlp.down_proj.weight::codes': torch.zeros(1, dtype=torch.uint8)},
                      root / 'model-ternary.safetensors')
            save_file({'model.embed_tokens.weight': torch.zeros(2, 3)}, root / 'model-tl2-emb4.safetensors')
            tensors = full_precision_tensors(root)
            self.assertEqual(set(tensors), set(weights))
            for name, expected in weights.items():
                torch.testing.assert_close(LazyTorchTensor.to_eager(tensors[name]()), expected)

    def test_defers_sharded_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_file({'weight': torch.ones(1)}, root / 'model-00001-of-00001.safetensors')
            self.assertIsNone(full_precision_tensors(root))


if __name__ == '__main__':
    unittest.main()
