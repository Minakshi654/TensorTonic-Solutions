import numpy as np

def rnn_cell(x_t: np.ndarray, h_prev: np.ndarray, 
             W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> np.ndarray:
    h_t = np.tanh(W_xh @ x_t + W_hh @ h_prev + b_h)
    return h_t