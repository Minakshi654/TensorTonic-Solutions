import numpy as np

def vgg_maxpool(x: np.ndarray) -> np.ndarray:
    if len(x.shape) == 4:
        N, H, W, C = x.shape
        x_reshaped = x.reshape(N, H //2, 2, W//2, 2, C)
        out = x_reshaped.max(axis = 2).max(axis=3)
        return out
    return x