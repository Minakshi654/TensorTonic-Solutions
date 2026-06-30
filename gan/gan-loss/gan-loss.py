import numpy as np
EPS = 1e-8

def discriminator_loss(real_probs, fake_probs):
    real_probs = np.clip(real_probs, EPS, 1-EPS)
    fake_probs = np.clip(fake_probs, EPS, 1-EPS)
    loss = -np.mean(np.log(real_probs) + np.log(1-fake_probs))
    return float(loss)
    

def generator_loss(fake_probs):
    fake_probs = np.clip(fake_probs, EPS, 1-EPS)
    loss = -np.mean(np.log(fake_probs))
    return float(loss)
 