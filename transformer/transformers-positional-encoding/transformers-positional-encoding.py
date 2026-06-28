import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    positions = np.arange(seq_length)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1/ np.power(10000, (dims// 2*2)/ d_model)
    angle_rads = positions * angle_rates
    pos_encodeing = np.zeros_like(angle_rads)
    pos_encodeing[:, 0::2]= np.sin(angle_rads[:, 0::2])
    pos_encodeing[:, 1::2]= np.cos(angle_rads[:, 1::2])
    return pos_encodeing.astype(np.float64)
    