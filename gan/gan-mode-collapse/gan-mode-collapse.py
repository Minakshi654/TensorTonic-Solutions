import numpy as np

def detect_mode_collapse(generated_samples, threshold=0.1):
    standard = np.std(generated_samples, axis=0)
    Diversity_score = np.mean(standard)
    is_collapsed = Diversity_score < threshold

    return {
        "diversity_score": round(float(Diversity_score), 4),
        "is_collapsed": bool(is_collapsed)
    }
