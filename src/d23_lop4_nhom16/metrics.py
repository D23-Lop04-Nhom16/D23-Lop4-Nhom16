#type: ignore 
import numpy as np 

def histogram_intersection(h1: np.ndarray, h2: np.ndarray) -> float:
    return float(np.sum(np.minimum(h1, h2))) 

def cosine_similarity(v1: np.ndarray, v2: np.ndarray, eps: float = 1e-10): 
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2) 
    if norm1 < eps or norm2 < eps: return 0.0 
    return float(np.dot(v1, v2) / (norm2 * norm1)) 

def l2_distance(v1: np.ndarray, v2: np.ndarray) -> float: 
    return float(np.linalg.norm(v1 - v2)) 

def dtw_distance(seq1: np.ndarray, seq2: np.ndarray) -> float: 
    n, m = len(seq1), len(seq2) 
    if n == 0 or m == 0: 
        return float("inf") 
    diff = seq1[:,np.newaxis,:] - seq2[np.newaxis,:,:] 
    cost_matrix = np.linalg.norm(diff, axis=2) 

    dp = np.full((n+1, m+1), float("inf")) 
    dp[0,0]=0.0 
    for i in range(1, n+1): 
        for j in range(1, m+1): 
            cost = cost_matrix[i-1, j-1] 
            dp[i,j] = cost + min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1]) 
    return float(dp[n,m] / (n+m)) 

