import numpy as np

def hidden_update(h_prev: np.ndarray, h_tilde: np.ndarray,
                  z_t: np.ndarray) -> np.ndarray:
    term = z_t * h_prev
    new_use = (1 - z_t) * h_tilde
    return term + new_use