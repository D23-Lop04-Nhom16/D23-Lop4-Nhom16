import cv2 
from d23_lop4_nhom16.dataset import VideoDogDataset 
from d23_lop4_nhom16.config import RAW_DATASET_DIR, HSV_FEATURE_PATH
import numpy as np 
from pathlib import Path
import pickle 

def extract_hsv_histogram(frame, h_bins=8, s_bins=3, v_bins=3): 
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.int32) 

    h = hsv[:,:,0] 
    s = hsv[:,:,1] 
    v = hsv[:,:,2] 

    h_idx = (h * h_bins / 180).astype(int) 
    s_idx = (s * s_bins / 256).astype(int) 
    v_idx = (v * v_bins / 256).astype(int) 

    bin_idx = h_idx * s_bins * v_bins + s_idx * v_bins + v_idx 

    hist = np.bincount(bin_idx.ravel(), minlength=h_bins * s_bins * v_bins) 

    hist = hist.astype(float) 

    if hist.sum() > 0: hist = hist / hist.sum() 

    return hist 

def extract_video_hsv_histogram(video_path): 
    cap = cv2.VideoCapture(video_path) 
    if not cap.isOpened(): 
        print(f"Không mở được video: {video_path}") 
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) 
    if fps <= 0: 
        fps = 25  
    frame_interval = int(round(fps)) 
    keyframe_features = [] 
    # keyframe_images = [] 
    frame_idx = 0 
    # keyframe_count = 0 

    while True: 
        ret, frame = cap.read() 
        if not ret: 
            break 
        if frame_idx % frame_interval == 0: 
            hist = extract_hsv_histogram(frame) 
            keyframe_features.append({
                "video_path": str(video_path), 
                "frame_idx": frame_idx, 
                "timestamp": frame_idx / fps, 
                "feature": hist 
            }) 
            # keyframe_images.append(frame) 
            # keyframe_count += 1
        frame_idx += 1
    cap.release() 
    # features = np.array(keyframe_features, dtype=np.float32) 
    return keyframe_features 

def extract_dataset_hsv_histogram(dataset): 
    all_features = [] 

    for video_path in dataset.video_paths: 
        video_features = extract_video_hsv_histogram(video_path) 
        all_features.extend(video_features) 
        print(
            f"Video: {video_path} | "
            f"Total extracted frames: {len(video_features)}"
        )
    return all_features 

if __name__ == '__main__': 
    dataset = VideoDogDataset(RAW_DATASET_DIR) 

    all_features = extract_dataset_hsv_histogram(dataset) 
    with open(HSV_FEATURE_PATH, "wb") as f: 
        pickle.dump(all_features, f, protocol=pickle.HIGHEST_PROTOCOL)  


