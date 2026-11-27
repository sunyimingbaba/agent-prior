"""Bounded detail bonus added to the original R2 score for every candidate."""
import numpy as np


def graded_bonus(base_retrieval, hard_margins):
    """Score one eight-candidate group using its native-R2 range and two annotated margins."""
    retrieval = np.asarray(base_retrieval, dtype=np.float64)
    margins = np.asarray(hard_margins, dtype=np.float64)
    if retrieval.shape != (8,) or margins.shape != (8, 2):
        raise ValueError("expected eight R2 scores and two signed detail margins per candidate")
    if not (np.isfinite(retrieval).all() and np.isfinite(margins).all()):
        raise ValueError("group contains nonfinite scores")
    weakest = np.maximum(0., np.min(margins, axis=1))
    width = np.ptp(retrieval) / 7.
    if width == 0. or np.ptp(weakest) == 0.:
        return np.zeros_like(retrieval)

    _, inverse, counts = np.unique(weakest, return_inverse=True, return_counts=True)
    # Midranks of tied blocks. Shifting and scaling maps the weakest to zero,
    # even when every candidate got both directions correct.
    midranks = np.cumsum(counts) - (counts + 1.) / 2.
    return width * ((midranks[inverse] - midranks[0]) / (midranks[-1] - midranks[0]))
