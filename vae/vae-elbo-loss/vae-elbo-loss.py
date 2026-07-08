import numpy as np

def vae_loss(x: np.ndarray, x_recon: np.ndarray, mu: np.ndarray, log_var: np.ndarray) -> dict:
    MSE = np.sum((x - x_recon)**2, axis=1).mean()
    KL_Divergence = -0.5 * np.sum(1 + log_var - mu**2 - np.exp(log_var), axis=1).mean()
    total = MSE + KL_Divergence
    return {
        "total": round(float(total), 4),
        "recon": round(float(MSE), 4),
        "kl": round(float(KL_Divergence), 4)
    }

