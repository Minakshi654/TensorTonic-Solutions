import numpy as np

def train_gan_step(real_data, fake_data, D_W):
    real_data = np.array(real_data, dtype=np.float64)
    fake_data = np.array(fake_data, dtype=np.float64)
    D_W = np.array(D_W, dtype=np.float64)

    EPS = 1e-8
    real_logits = np.dot(real_data, D_W)
    fake_logits = np.dot(fake_data, D_W)
    real_probs = 1 / (1 + np.exp(-real_logits))
    fake_probs = 1 / (1 + np.exp(-fake_logits))

    real_probs = np.clip(real_probs, EPS, 1-EPS)
    fake_probs = np.clip(fake_probs, EPS, 1-EPS)
    d_loss = -np.mean(np.log(real_probs) + np.log(1- fake_probs))
    G_loss = -np.mean(np.log(fake_probs))
    return {
        "d_loss": round(float(d_loss), 4),
        "g_loss": round(float(G_loss), 4)
    }
    