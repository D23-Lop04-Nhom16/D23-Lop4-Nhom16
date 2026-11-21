import numpy as np 
from d23_lop4_nhom16.metrics import histogram_intersection 

def hsv_similarity(q_features: np.ndarray, c_features: np.ndarray) -> float:
    q_features = np.asarray(q_features, dtype=np.float32) 
    c_features = np.asarray(c_features, dtype=np.float32) 

    if len(q_features) == 0 or len(c_features) == 0: return 0.0 

    sim = [] 

    for q in q_features: 
        b_match = max(
            histogram_intersection(q, c) for c in c_features 
        )
        sim.append(b_match) 

    return float(np.mean(sim)) 

