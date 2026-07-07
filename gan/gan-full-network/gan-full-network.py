import numpy as np

class GAN:
    def __init__(self, G_W, D_W):
        """
        Initialize GAN with concrete weights.
        """
        self.G_W = np.array(G_W, dtype=float)
        self.D_W = np.array(D_W, dtype=float)
    
    def generate(self, z):
        z= np.asarray(z, dtype = float)
        fake_data = np.tanh(np.dot(z, self.G_W))
        return np.round(fake_data, 4).tolist()
    
    def discriminate(self, x):
        x = np.asarray(x, dtype = float)
        logits = np.dot(x, self.D_W)
        sigmoid = 1.0 / (1.0 + np.exp(-logits))
        return np.round(sigmoid, 4).tolist()
    
    def train_step(self, real_data, z):
        EPS = 1e-8

        real_data = np.array(real_data, dtype=float)

        z= np.asarray(z, dtype = float)
        fake_data = np.tanh(np.dot(z, self.G_W))
        real_logits = np.dot(real_data, self.D_W)
        fake_logits = np.dot(fake_data, self.D_W)

        real_probs = 1.0 / (1.0 + np.exp(-real_logits))
        fake_probs = 1.0 / (1.0 + np.exp(-fake_logits))

        # Clip probabilities
        real_probs = np.clip(real_probs, EPS, 1 - EPS)
        fake_probs = np.clip(fake_probs, EPS, 1 - EPS)
        d_loss = -np.mean(
            np.log(real_probs) + np.log(1 - fake_probs)
        )

        # Generator loss
        g_loss = -np.mean(
            np.log(fake_probs)
        )

        return {
            "d_loss": round(float(d_loss), 4),
            "g_loss": round(float(g_loss), 4)}


        
        