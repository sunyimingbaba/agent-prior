"""Simultaneously exchange training credit between dominated nonwinning candidates."""
import numpy as np


def residual_scores(strict, hard_margins, visible_rr):
    scores = np.asarray(strict, dtype=np.float64)
    margins = np.asarray(hard_margins)
    captions = np.asarray(visible_rr)
    lower = scores < scores.max()
    count_lower = int(lower.sum())
    if count_lower < 2:
        return scores.copy()

    harder = (margins > 0).sum(axis=1)
    weaker_margin = margins.min(axis=1)
    gap = scores[None, :] - scores[:, None]
    eligible = lower[:, None] & lower[None, :] & (gap > 0)
    eligible &= harder[:, None] > harder[None, :]
    eligible &= weaker_margin[:, None] > weaker_margin[None, :]
    eligible &= (captions[:, None, :] >= captions[None, :, :]).all(axis=2)

    transfer = np.where(eligible, gap / (count_lower - 1), 0.0)
    revised = scores + (transfer.sum(axis=1) - transfer.sum(axis=0))
    assert np.allclose(revised.sum(), scores.sum(), atol=1e-12, rtol=0)
    assert np.array_equal(revised == revised.max(), ~lower)
    return revised
