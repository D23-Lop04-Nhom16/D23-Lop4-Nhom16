import numpy as np 
from d23_lop4_nhom16.metrics import dtw_distance 


def mhi_similarity(
    q_features: np.ndarray, 
    c_features: np.ndarray
) -> float: 
    q_features = np.asarray(q_features, dtype=np.float32) 
    c_features = np.asarray(c_features, dtype=np.float32) 

    if len(q_features) == 0 or len(c_features) == 0: return float("inf") 

    return float(dtw_distance(q_features, c_features))
