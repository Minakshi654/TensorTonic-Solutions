import numpy as np

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    hidden = np.dot(x,W1) + b1
    f1 = np.maximum(0, hidden)
    f2 = np.dot(f1,W2) + b2
    return f2