import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))
def B_loss(y, y_pred):
    pass
    
def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    # Write code here
    m, n = X.shape
    w = np.zeros(n)
    b = 0
    for step in range(0, steps):
        z = X@w + b # ()
        y_pred = _sigmoid(z)
        loss = -np.mean((y*np.log(y_pred) + (1-y)*np.log(1-y_pred)))
        dw = (X.T @ (y_pred-y))/m
        db = np.sum(y_pred-y)/m
        w-=lr*dw
        b-=lr*db
        if step%100==0:
            print(f"For epoch {step} w is {w}, b is {b}, loss is {loss}")
    return w, b
        
            