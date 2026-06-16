import numpy as np

def compute_advantage(states, rewards, V, gamma):
    states = np.asarray(states, dtype = int)
    rewards = np.asarray(rewards, dtype = float)
    V = np.asarray(V, dtype = float)
    N = len(rewards)
    if N < 1:
        return None
    G = np.zeros(N)
    running_return = 0.0
    for t in reversed(range(N)):
        running_return = rewards[t]+ gamma* running_return 
        G[t] = running_return
    A = G- V[states]
    return A
    
