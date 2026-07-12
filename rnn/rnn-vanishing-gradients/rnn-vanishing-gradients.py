import numpy as np

def compute_gradient_norm_decay(T: int, W_hh: np.ndarray) -> list:
    Norm = np.linalg.norm(W_hh, ord=2)
    gradients = [1.0]
    for _ in range(1, T):
        gradients.append(gradients[-1]* Norm)
    return gradients
        