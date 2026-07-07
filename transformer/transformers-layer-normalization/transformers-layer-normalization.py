import numpy as np

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mea = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    layer1 = x- mea
    layer2 = np.sqrt(var + eps)
    Norm = gamma * layer1 / layer2 + beta
    return Norm