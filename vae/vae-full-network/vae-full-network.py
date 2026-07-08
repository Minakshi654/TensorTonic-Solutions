import numpy as np

class VAE:
    def __init__(self, W_mu: np.ndarray, b_mu: np.ndarray, W_logvar: np.ndarray, b_logvar: np.ndarray, W_dec: np.ndarray, b_dec: np.ndarray):
        self.W_mu = np.array(W_mu, dtype=float)
        self.b_mu = np.array(b_mu, dtype=float)
        self.W_logvar = np.array(W_logvar, dtype=float)
        self.b_logvar = np.array(b_logvar, dtype=float)
        self.W_dec = np.array(W_dec, dtype=float)
        self.b_dec = np.array(b_dec, dtype=float)
        
        
    
    def forward(self, x: np.ndarray, epsilon: np.ndarray) -> dict:
        x= np.array(x, dtype = float)
        epsilon = np.array(epsilon, dtype = float)
        mu = np.dot(x, self.W_mu)+ self.b_mu
        log_var = np.dot(x, self.W_logvar)+ self.b_logvar
        std = np.exp(0.5 * log_var)
        z = mu + std * epsilon
        recon = np.dot(z, self.W_dec) + self.b_dec
        return {
            "recon": np.round(recon, 4),
            "mu": np.round(mu, 4),
            "log_var": np.round(log_var, 4)
        }
    
    def generate(self, z: np.ndarray) -> np.ndarray:
        z= np.array(z, dtype = float)
        recon = np.dot(z, self.W_dec) + self.b_dec
        return recon
        
