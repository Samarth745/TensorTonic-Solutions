import numpy as np
import math
def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """
    Returns a NumPy array of shape (seq_len, d_model).
    """

    def embed_position(pos: int, dm: int):
        num = pos
        res = []
        n = int(dm/2)
        for i in range(0, n,1):
            den = base**(2*i/dm)
            res.append(np.sin(num/den))
            res.append(np.cos(num/den))
        if dm%2!=0:
            i = (dm-1)/2
            den = base**(2*i/dm)
            res.append(np.sin(num/den))
           
        return res
        
    result = []
    for k in range(seq_len):
        result.append(embed_position(k, d_model))
    return np.array(result)