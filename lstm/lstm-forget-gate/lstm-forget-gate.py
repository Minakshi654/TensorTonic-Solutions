import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def forget_gate(h_prev: np.ndarray, x_t: np.ndarray,
                W_f: np.ndarray, b_f: np.ndarray) -> np.ndarray:
    single = np.ndim(h_prev) == 1
    h_prev = np.atleast_2d(h_prev)
    x_t = np.atleast_2d(x_t)
    concat = np.concatenate([h_prev, x_t], axis=-1)
    z = concat @ W_f.T + b_f
    f_t = 1.0/ (1.0 + np.exp(-z))
    if single:
        return f_t[0]  # shape (H,)
    return f_t 

    
    
    
    