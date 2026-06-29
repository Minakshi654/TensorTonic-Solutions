import numpy as np

def generator(z, W, b):
    z= np.asarray(z)
    W = np.asarray(W)
    generate = np.tanh(z@W+b)
    return generate