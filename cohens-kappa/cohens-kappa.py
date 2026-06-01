import numpy as np
from collections import Counter
def cohens_kappa(rater1, rater2):
    if len(rater1) != len(rater2):
        raise ValueError("Both are same")
    n = len(rater1)
    matches = sum(a==b for a,b in zip(rater1, rater2))
    p_O = matches/ n
    labels = set(rater1)| set(rater2)
    p_e = sum((rater1.count(l)/n * (rater2.count(l)/n)) for l in labels)
    if p_e == 1:
        return 1.0
    return (p_O - p_e) / (1- p_e)