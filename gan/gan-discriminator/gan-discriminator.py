import numpy as np

def discriminator(x, W):
    x = np.asarray(x)
    W = np.asarray(W)
    batch_size, dim = x.shape
    if W is None:
        W = np.random.randn(dim, 1)
    logits = np.dot(x, W)
    sigmoid = 1 / (1+ np.exp(-logits))
    return sigmoid