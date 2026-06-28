import numpy as np

def adagrad_step(w, g, G, lr=0.01, eps=1e-8):
    w = np.array(w, dtype=float)
    g = np.array(g, dtype=float)
    G = np.array(G, dtype=float)
    new_G = G + g**2
    new_w = w - lr * g / (np.sqrt(new_G + eps))
    return new_w, new_G
    new_w, new_G = adagrad_step(w, g, G, lr, eps)
    print(new_w, new_G)