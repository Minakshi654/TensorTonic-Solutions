import torch

def subsample_keep_probs(counts: torch.Tensor, t: float = 1e-5) -> torch.Tensor:
    counts = counts.float()
    freqs = counts / counts.sum()
    keep_probs = torch.sqrt(t/ freqs)
    return torch.clamp(keep_probs, max=1.0)
    
    
    
