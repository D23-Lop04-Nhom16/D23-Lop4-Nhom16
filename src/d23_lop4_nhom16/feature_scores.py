from d23_lop4_nhom16.similarity.hsv_similarity import hsv_similarity 
from d23_lop4_nhom16.similarity.glcm_similarity import glcm_similarity 
from d23_lop4_nhom16.similarity.flow_similarity import flow_similarity 
from d23_lop4_nhom16.similarity.hog_similarity import hog_similarity 
from d23_lop4_nhom16.similarity.mhi_similarity import mhi_similarity 

def feature_scores(
    q_feat: dict, 
    c_feat: dict, 
    scaler_glcm=None 
) -> dict: 
    return {
        "hsv": hsv_similarity(q_feat["hsv"], c_feat["hsv"]), 
        "glcm": glcm_similarity(q_feat["glcm"], c_feat["glcm"], scaler=scaler_glcm), 
        "hog": hog_similarity(q_feat["hog"], c_feat["hog"]), 
        "flow": flow_similarity(q_feat["flow"], c_feat["flow"]), 
        "mhi": mhi_similarity(q_feat["mhi"], c_feat["mhi"]) 
    }

