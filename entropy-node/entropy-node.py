import numpy as np

def entropy_node(y: list[int]) -> float:
    """
    Returns the Shannon entropy as a Python float.
    """
    # Write code here
    x = np.array(y)
    y = np.unique(x, return_counts=True)
    pi = (y[1]/len(x))
    entropy = np.sum(pi*np.log2(pi))
    print(entropy)
    return -entropy