import numpy as np

def init_hidden(batch_size: int, hidden_dim: int) -> np.ndarray:
    result = np.zeros((batch_size, hidden_dim)) 
    return result