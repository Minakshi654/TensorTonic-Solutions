import numpy as np

def vae_encoder(x: np.ndarray, W_mu: np.ndarray, b_mu: np.ndarray, W_logvar: np.ndarray, b_logvar: np.ndarray) -> dict:
    N, D = x.shape
    mu = np.dot(x, W_mu) + b_mu
    log_var = np.dot(x, W_logvar) +b_logvar
    return {
        "mu": np.round(mu, 4).tolist(),
        "log_var": np.round(log_var, 4).tolist()
    }
    
