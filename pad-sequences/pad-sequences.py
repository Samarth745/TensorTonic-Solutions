import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    if seqs == []:
        return  np.empty((0, 0), dtype=int)
    if max_len is None:
        max_len = 0
        for seq in seqs:
            max_len = max(max_len, len(seq))
    for i in range(len(seqs)):
        n = max_len - len(seqs[i])
        if n > 0:
            seqs[i] = seqs[i] + [pad_value]*n
        elif n < 0:
            seqs[i] = seqs[i][:n]
        else:
            pass
    return np.array(seqs)
        