import numpy as np

def adam_step(
    param: list,
    grad: list,
    m: list,
    v: list,
    t: int,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (param_new, m_new, v_new) as NumPy arrays.
    """
    # Write code here
    m_ = beta1*np.array(m) + (1-beta1)*np.array(grad)
    v_ = beta2*np.array(v) + (1-beta2)*np.power(np.array(grad),2)
    num = m_/(1-beta1**t)
    denom = np.sqrt(v_/(1-beta2**t)) + eps
    param-=lr*(num/denom)
    return (param, m_, v_)
    