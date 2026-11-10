"""CPU regressions for channel conversion, causal history, and checkpoint replay."""

import tempfile
import unittest
from functools import partial
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import nn
from torch.utils.checkpoint import checkpoint

from recosplat.model.layers.block import BlockRope
from scripts.train_rgb import causal_layer, load_rgb_initialization, sample_indices


class TrainingRGBTests(unittest.TestCase):
    def test_causal_history_and_checkpoint_gradients(self):
        torch.manual_seed(1)
        for mode in ("default", "ignore_kv_cache", "global_prev_last_view_plus_current"):
            with self.subTest(mode=mode):
                block = BlockRope(dim=16, num_heads=2, mlp_ratio=2, qkv_bias=True,
                                  block_layer_type=mode)
                x = torch.randn(1, 12, 16, requires_grad=True)
                pos = torch.zeros(1, 12, 2, dtype=torch.long)
                fn = partial(causal_layer, block, pos=pos, intervals=[(0, 2), (2, 4), (4, 6)], tokens_per_view=2)
                reference = fn(x)
                altered = x.detach().clone()
                altered[:, 8:] += 10
                torch.testing.assert_close(reference[:, :8], fn(altered)[:, :8])
                reference[:, 8:].square().mean().backward()
                grads = [p.grad.clone() for p in block.parameters() if p.grad is not None]
                input_grad = x.grad.clone()
                if mode == "ignore_kv_cache":
                    self.assertEqual(input_grad[:, :8].abs().sum(), 0)
                else:
                    self.assertGreater(input_grad[:, :4].abs().sum(), 0)
                block.zero_grad(set_to_none=True)
                x.grad = None
                replayed = checkpoint(fn, x, use_reentrant=False)
                replayed[:, 8:].square().mean().backward()
                torch.testing.assert_close(reference, replayed)
                torch.testing.assert_close(x.grad, input_grad)
                for expected, actual in zip(grads, [p.grad for p in block.parameters() if p.grad is not None]):
                    torch.testing.assert_close(expected, actual)

    def test_rgb_conversion_preserves_observed_channels_and_is_strict(self):
        class TinyEncoder(nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.first_chunk_guidance_embed = nn.Parameter(torch.randn(channels - 3, 1, 1))
                self.guidance_to_tokens = nn.Module()
                self.guidance_to_tokens.conv_block = nn.Sequential(nn.Conv2d(channels, 2, 3))
                self.other = nn.Linear(2, 2)
                if channels == 15:
                    self.guidance_feature_head = nn.Linear(2, 9)

        full, rgb = TinyEncoder(15), TinyEncoder(6)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            save_file(full.state_dict(), path)
            load_rgb_initialization(rgb, path)
            torch.testing.assert_close(rgb.guidance_to_tokens.conv_block[0].weight,
                                       full.guidance_to_tokens.conv_block[0].weight[:, [0, 1, 2, 12, 13, 14]])
            torch.testing.assert_close(rgb.first_chunk_guidance_embed, full.first_chunk_guidance_embed[:3])
            save_file(rgb.state_dict(), path)
            load_rgb_initialization(TinyEncoder(6), path)
            broken = full.state_dict()
            del broken["other.weight"]
            save_file(broken, path)
            with self.assertRaises(RuntimeError):
                load_rgb_initialization(rgb, path)

    def test_targets_are_held_out(self):
        generator = torch.Generator().manual_seed(12)
        for _ in range(50):
            context, target = sample_indices(300, 16, 8, 128, generator)
            self.assertEqual(len(context.unique()), 16)
            self.assertEqual(len(target.unique()), 8)
            self.assertFalse(set(context.tolist()) & set(target.tolist()))
            self.assertTrue(torch.all(context[1:] > context[:-1]))
            self.assertGreater(target.min(), context.min())
            self.assertLess(target.max(), context.max())
            self.assertLessEqual(int(context[-1] - context[0] + 1), 128)
        self.assertIsNone(sample_indices(10, 16, 8, 128, generator))


if __name__ == "__main__":
    unittest.main()
