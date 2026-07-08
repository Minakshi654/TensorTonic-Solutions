import numpy as np

def kl_divergence(mu: np.ndarray, log_var: np.ndarray) -> float:
    var = np.exp(log_var)
    output = -0.5 * np.sum(1+ log_var - var - mu**2, axis = 1 )
    return float(np.mean(output))
