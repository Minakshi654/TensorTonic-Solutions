import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mea = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    layer1 = x- mea
    layer2 = np.sqrt(var + eps)
    Norm = gamma * layer1 / layer2 + beta
    return Norm

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    batch_size, seq_len, d_model = Q.shape
    d_k = d_model // num_heads
    
    Q_Proj = Q @ W_q
    K_Proj = K @ W_k
    V_Proj = V @ W_v
    Q_heads = Q_Proj.reshape(batch_size, seq_len, num_heads, d_k).transpose(0,2,1,3)
    K_heads = K_Proj.reshape(batch_size, seq_len, num_heads, d_k).transpose(0,2,1,3)
    V_heads = V_Proj.reshape(batch_size, seq_len, num_heads, d_k).transpose(0,2,1,3)
    scores = (Q_heads @ K_heads.transpose(0,1,3,2))/ np.sqrt(d_k)
    attention_weights = softmax(scores, axis = -1)
    head_outputs = attention_weights @ V_heads
    concat_heads = head_outputs.transpose(0,2,1,3).reshape(batch_size, seq_len, d_model)
    output = concat_heads @ W_o
    return output

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    hidden = np.dot(x,W1) + b1
    f1 = np.maximum(0, hidden)
    f2 = np.dot(f1,W2) + b2
    return f2

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    attention_output = multi_head_attention(x,x,x, W_q, W_k, W_v, W_o, num_heads)
    result = layer_norm(x + attention_output, gamma1, beta1)
    ff_result = feed_forward(result, W1, b1, W2, b2)
    output = layer_norm(result+ ff_result, gamma2,beta2)
    return output
    
    