import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler 
from sqlalchemy import create_engine 
from d23_lop4_nhom16.config import get_db_url 
from d23_lop4_nhom16.metrics import l2_distance 

_GLOBAL_GLCM_SCALER = None 

def glcm_scaler(engine=None) -> StandardScaler: 
    global _GLOBAL_GLCM_SCALER 
    if _GLOBAL_GLCM_SCALER is not None: 
        return _GLOBAL_GLCM_SCALER 
    if engine is None: 
        engine = create_engine(get_db_url()) 

    query = "SELECT glcm_vector FROM keyframe_features;" 
    df = pd.read_sql(query, con=engine) 
    if df.empty: raise ValueError("Not data")
    X = np.array(df["glcm_vector"].tolist(), dtype=np.float32) 

    scaler = StandardScaler() 
    scaler.fit(X) 

    print("Done fit") 

    _GLOBAL_GLCM_SCALER = scaler 
    return _GLOBAL_GLCM_SCALER 

def glcm_similarity(
    q_features: np.ndarray, 
    c_features: np.ndarray, 
    scaler: StandardScaler = None, 
    engine = None 
) -> float: 
    q_features = np.asarray(q_features, dtype=np.float32) 
    c_features = np.asarray(c_features, dtype=np.float32) 

    if len(q_features) == 0 or len(c_features) == 0: return 0.0 

    if scaler is None: 
        scaler = glcm_scaler(engine=engine) 

    q_norm = scaler.transform(q_features) 
    c_norm = scaler.transform(c_features) 

    dis = [] 
    for q in q_norm: 
        min_dis = min(l2_distance(q, c) for c in c_norm) 
        dis.append(min_dis) 
    avg = float(np.mean(dis)) 
    return float(1 / (1 + avg)) 


