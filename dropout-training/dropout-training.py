import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    x = np.asarray(x, float)
    k = 1.0 - p
    mask = (rng.random(x.shape) < k).astype(float) / k
    return x * mask, mask