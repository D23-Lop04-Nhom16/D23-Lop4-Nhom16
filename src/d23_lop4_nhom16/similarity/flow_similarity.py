import numpy as np
from d23_lop4_nhom16.metrics import cosine_similarity
def flow_similarity(
    q_features: np.ndarray, 
    c_features: np.ndarray
) -> float:
    
    q_features = np.asarray(q_features, dtype=np.float32).flatten()
    c_features = np.asarray(c_features, dtype=np.float32).flatten()
    
    if len(q_features) != 13 or len(c_features) != 13:
        return 0.0
    
    return float(cosine_similarity(q_features, c_features))
