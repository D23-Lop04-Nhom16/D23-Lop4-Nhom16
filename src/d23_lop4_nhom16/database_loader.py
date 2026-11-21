from sqlalchemy import create_engine 
from d23_lop4_nhom16.config import get_db_url 
from typing import Tuple, Dict, Any
import pandas as pd 
import numpy as np 
def load_database_features(engine) -> Tuple[pd.DataFrame, Dict[int, Dict[str, Any]]]: 

    df_videos = pd.read_sql(
        "SELECT video_id, filename, filepath, category FROM videos;", 
        con=engine 
    )
    df_kff = pd.read_sql(
        """
        SELECT k.video_id, f.hsv_vector, f.glcm_vector, f.hog_vector 
        FROM keyframes k 
        JOIN keyframe_features f ON k.keyframe_id = f.keyframe_id
        ORDER BY k.video_id, k.timestamp_sec; 
        """, 
        con=engine 
    )
    df_flow = pd.read_sql(
        """
        SELECT video_id, flow_vector 
        FROM flow_features; 
        """, 
        con=engine

    )
    df_mhi = pd.read_sql(
        """
        SELECT video_id, mhi_window_index, mhi_vector
        FROM mhi_features 
        ORDER BY video_id, mhi_window_index; 
        """, 
        con=engine 
    )

    db_store = {}
    
    for vid in df_videos["video_id"]: 
        kfs = df_kff[df_kff["video_id"] == vid] 
        hsv_arr = np.array(kfs["hsv_vector"].tolist(), dtype=np.float32) 
        glcm_arr = np.array(kfs["glcm_vector"].tolist(), dtype=np.float32) 
        hog_arr = np.array(kfs["hog_vector"].tolist(), dtype=np.float32) 
        flow_row = df_flow[df_flow["video_id"] == vid] 
        flow_arr = (
            np.array(flow_row["flow_vector"].iloc[0], dtype=np.float32) 
            if not flow_row.empty else np.zeros(13) 
        )
        mhi_rows = df_mhi[df_mhi["video_id"] == vid]
        mhi_arr = (
            np.array(mhi_rows["mhi_vector"].tolist(), dtype=np.float32)
            if not mhi_rows.empty
            else np.zeros((1, 256))
        )
        db_store[vid] = {
            "hsv": hsv_arr, 
            "glcm": glcm_arr, 
            "hog": hog_arr, 
            "flow": flow_arr, 
            "mhi": mhi_arr 
        }
    return df_videos, db_store 





