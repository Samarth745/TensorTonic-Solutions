import numpy as np

def sample_var_std(x: list) -> dict:
    """
    Returns a dictionary with variance and standard_deviation.
    """
    # Write code here
    x = np.array(x, dtype=float)
    var = float(np.sum((x - np.mean(x))**2)/(len(x)-1))
    return {"variance" : var, 
     "standard_deviation" : float(np.sqrt(var))}