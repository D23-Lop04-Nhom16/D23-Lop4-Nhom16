from typing import Dict 
import numpy as np 
import pandas as pd 
from d23_lop4_nhom16.config import DEFAULT_WEIGHTS 


def normalize_and_ranking(
    raw_scores: pd.DataFrame, 
    df_videos: pd.DataFrame, 
    weights: Dict[str, float]=None, 
    top_k: int=5 
) -> pd.DataFrame: 
    if weights is None:
        weights = DEFAULT_WEIGHTS
    df = raw_scores.copy() 
    metric_is_distance = {
        "hsv": False,
        "glcm": False,
        "hog": False,
        "flow": False,
        "mhi": True
    }

    for feat, is_dist in metric_is_distance.items():
        if feat not in df.columns:
            continue
        vals = df[feat].values.astype(np.float64)
        min_v = np.min(vals)
        max_v = np.max(vals)
        diff = max_v - min_v if max_v > min_v else 1e-10
        if not is_dist: df[f"norm_{feat}"] = (vals - min_v) / diff
        else: df[f"norm_{feat}"] = (max_v - vals) / diff

    total_w = sum(weights.values())
    df["final_score"] = 0.0
    for feat, w in weights.items():
        norm_col = f"norm_{feat}"
        if norm_col in df.columns:
            df["final_score"] += (w / total_w) * df[norm_col]

    df_ranked = df.merge(df_videos, on="video_id").sort_values(by="final_score", ascending=False) 

    display_cols = [
        "video_id", 
        "filepath", 
        "category", 
        "final_score", 
        "norm_hsv", 
        "norm_glcm", 
        "norm_hog", 
        "norm_flow", 
        "norm_mhi"
    ]
    existing_cols = [c for c in display_cols if c in df_ranked.columns]  
    return df_ranked[existing_cols].head(top_k).reset_index(drop=True) 
