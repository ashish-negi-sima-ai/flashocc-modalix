"""Portable FP32 reference for upstream BEVPoolv2, outside the MLA graphs.

Depth is [N,D,H,W]; features are [N,C,H,W]. Rank arrays must come from
upstream voxel_pooling_prepare_v2 with the real calibration and image transforms.
This is a correctness reference, not an optimized Modalix runtime kernel.
"""
import numpy as np


def pool(depth, features, ranks_depth, ranks_feat, ranks_bev, height=200, width=200):
    depth = np.asarray(depth, dtype=np.float32).reshape(-1)
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 4:
        raise ValueError('features must be NCHW')
    channels = features.shape[1]
    features = features.transpose(0, 2, 3, 1).reshape(-1, channels)
    ranks = [np.asarray(r) for r in (ranks_depth, ranks_feat, ranks_bev)]
    for r, limit in zip(ranks, [depth.size, len(features), height*width]):
        if r.ndim != 1 or not np.issubdtype(r.dtype, np.integer):
            raise ValueError('ranks must be one-dimensional integer arrays')
        if len(r) and (r.min() < 0 or r.max() >= limit):
            raise ValueError('rank outside tensor bounds')
    if not len(ranks[0]) == len(ranks[1]) == len(ranks[2]):
        raise ValueError('rank lengths differ')
    rd, rf, rb = ranks
    out = np.zeros((height*width, channels), dtype=np.float32)
    for start in range(0, len(rd), 16384):
        sl = slice(start, start+16384)
        np.add.at(out, rb[sl], depth[rd[sl], None]*features[rf[sl]])
    return out.reshape(height, width, channels).transpose(2, 0, 1)[None]


if __name__ == '__main__':
    # Published upstream BEVPoolv2 fixture: expected total is 4.4.
    depth = np.array([.3, .4, .2, .1, .7, .6, .8, .9], np.float32).reshape(1, 2, 2, 2)
    features = np.ones((1, 2, 2, 2), np.float32)
    result = pool(depth, features, [0, 4, 1, 6], [0, 0, 1, 2], [0, 0, 1, 1], 2, 2)
    np.testing.assert_allclose(result[0, :, 0, :], [[1., 1.2], [1., 1.2]])
    np.testing.assert_allclose(result.sum(), 4.4)
    print('Upstream pooling fixture passed.')
