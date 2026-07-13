import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def input_gate(h_prev: np.ndarray, x_t: np.ndarray,
               W_i: np.ndarray, b_i: np.ndarray,
               W_c: np.ndarray, b_c: np.ndarray) -> tuple:
    single = np.ndim(h_prev) == 1
    h_prev = np.atleast_2d(h_prev)
    x_t = np.atleast_2d(x_t)
    concat = np.concatenate([h_prev, x_t], axis=-1)
    
# Input gate
    i_t = 1.0 / (1.0 + np.exp(-(concat @ W_i.T + b_i)))

    # Candidate memory
    c_tilde = np.tanh(concat @ W_c.T + b_c)

    # Return 1D arrays for single examples
    if single:
        return i_t[0], c_tilde[0]

    return i_t, c_tilde
