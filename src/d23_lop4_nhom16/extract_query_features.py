from d23_lop4_nhom16.feature_extraction.color_histogram import extract_video_hsv_histogram 
from d23_lop4_nhom16.feature_extraction.glcm import extract_glcm_from_video 
from d23_lop4_nhom16.feature_extraction.flow_extracter import extract_flow 
from d23_lop4_nhom16.feature_extraction.hog_extracter import extract_video_hog 
from d23_lop4_nhom16.feature_extraction.mhi_extracter import extract_mhi 
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple
import cv2
import joblib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

def extract_query_features(video_path: str | Path) -> Dict[str, Any]: 
    
    video_path = Path(video_path) 
    if not video_path.is_file(): 
        raise FileNotFoundError("File not found") 

    print("Stating extract...") 
    hsv_data = extract_video_hsv_histogram(str(video_path)) 
    hsv_vec = np.array([x["feature"] for x in hsv_data], dtype=np.float32) 

    glcm_data = extract_glcm_from_video(str(video_path)) 
    glcm_vec = np.array([x["feature"] for x in glcm_data], dtype=np.float32) 

    hog_data = extract_video_hog(video_path)
    hog_vec = np.array([x["vector"] for x in hog_data], dtype=np.float32)
    
    flow_data = extract_flow(video_path)
    flow_vec = np.array(flow_data["flow_vector"], dtype=np.float32)
    
    mhi_data = extract_mhi(video_path)
    mhi_vec = np.array(mhi_data["mhi_vectors"], dtype=np.float32)

    return {
        "hsv": hsv_vec,  # (N_kf, 72)
        "glcm": glcm_vec,  # (N_kf, 4)
        "hog": hog_vec,  # (N_kf, 8100)
        "flow": flow_vec,  # (13,)
        "mhi": mhi_vec,  # (T_win, 256)
    }










