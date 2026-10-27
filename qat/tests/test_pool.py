import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "qat"))
sys.path.insert(0, str(ROOT / "scripts"))

from flashocc_qat import differentiable_bev_pool
from pool_reference import pool


class PoolTest(unittest.TestCase):
    def test_matches_numpy_and_backpropagates(self):
        generator = torch.Generator().manual_seed(7)
        depth = torch.rand(6, 44, 16, 44, generator=generator, requires_grad=True)
        features = torch.rand(6, 64, 16, 44, generator=generator, requires_grad=True)
        ranks_depth = torch.tensor([0, 4, 1, 6, 100, 200])
        ranks_feat = torch.tensor([0, 0, 1, 2, 10, 20])
        ranks_bev = torch.tensor([0, 0, 1, 1, 12, 400])
        actual = differentiable_bev_pool(
            depth, features, ranks_depth, ranks_feat, ranks_bev
        )
        expected = pool(
            depth.detach().numpy(),
            features.detach().numpy(),
            ranks_depth.numpy(),
            ranks_feat.numpy(),
            ranks_bev.numpy(),
        )[0]
        np.testing.assert_allclose(actual.detach().numpy(), expected, rtol=1e-6, atol=1e-6)
        actual.sum().backward()
        self.assertIsNotNone(depth.grad)
        self.assertIsNotNone(features.grad)
        self.assertGreater(depth.grad.abs().sum().item(), 0)
        self.assertGreater(features.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()

