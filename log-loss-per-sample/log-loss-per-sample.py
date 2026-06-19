import math

def log_loss(y_true, y_pred, eps=1e-15):
    losses = []
    for y,p in zip(y_true, y_pred):
        p_hat = max(min(p, 1-eps), eps)
        loss = -(y* math.log(p_hat) + (1-y)* math.log(1- p_hat))
        losses.append(loss)
    return losses
    
    