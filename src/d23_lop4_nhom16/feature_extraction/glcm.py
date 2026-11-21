import cv2 
import numpy as np 
from math import log2 
from d23_lop4_nhom16.config import RAW_DATASET_DIR, GLCM_FEATURE_PATH
from d23_lop4_nhom16.dataset import VideoDogDataset 
import os 
import pickle

def quantize_gray(gray, levels=8): 
    return (gray.astype(np.float32) / 256 * levels).astype(np.int32) 

def compute_glcm(gray_q, levels=8, dx=1, dy=0): 
    h, w = gray_q.shape 
    glcm = np.zeros((levels, levels), dtype=np.float64) 
    for i in range(h): 
        for j in range(w): 
            x2=i+dy 
            y2=j+dx 
            if 0<=x2<h and 0<=y2<w: 
                g1 = gray_q[i,j] 
                g2 = gray_q[x2, y2] 
                glcm[g1, g2] += 1 
    total = glcm.sum() 
    if total > 0: glcm /= total 
    return glcm 

def glcm_features(glcm): 
    levels = glcm.shape[0] 
    energy = 0.0 
    contrast = 0.0 
    entropy = 0.0 
    homogeneity = 0.0 

    for i in range(levels): 
        for j in range(levels): 
            p = glcm[i,j] 
            energy += p**2 
            contrast += ((i - j) ** 2) * p 
            if p > 0: 
                entropy -= p*log2(p) 
            homogeneity += p / (1 + abs(i-j)) 

    return np.array([energy, contrast, entropy, homogeneity], dtype=np.float32) 

def extract_glcm_feature(frame, levels=8): 

    if len(frame.shape) == 3: 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    else: 
        gray = frame 

    gray_q = quantize_gray(gray, levels=levels) 

    directions = [
        (1,0), 
        (1,1), 
        (0,1), 
        (-1,1) 
    ]
    features_list = [] 
    for dx, dy in directions: 
        glcm = compute_glcm(gray_q, levels=levels, dx=dx, dy=dy) 
        feat = glcm_features(glcm) 
        features_list.append(feat) 

    feature_vector = np.mean(features_list, axis=0) 
    return feature_vector 

def extract_glcm_from_video(video_path): 
    cap = cv2.VideoCapture(video_path) 
    if not cap.isOpened(): 
        raise ValueError(f"Khong mo duoc video: {video_path}") 
    fps = cap.get(cv2.CAP_PROP_FPS) 
    if fps <= 0: fps = 25 
    video_name = os.path.basename(video_path)
    frame_interval = int(round(fps)) 
    records = []
    frame_idx = 0 
    # frame_images = [] 
    while True: 
        ret, frame = cap.read() 
        if not ret: 
            break 
        if frame_idx % frame_interval == 0: 
            vec = extract_glcm_feature(frame)
            # frame_images.append(frame) 
            # features.append(vec)
            records.append({
                "video_path": video_path, 
                "video_name": video_name, 
                "frame_idx": frame_idx, 
                "timestamp_sec": round(frame_idx / fps, 3), 
                "feature": vec
            }) 
        frame_idx += 1 
    cap.release() 

    return records 
def load_existing_records(pkl_path): 
    if not os.path.exists(pkl_path):
        return []
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
            if isinstance(data, list):
                return data
            return []
    except Exception as e:
        print(f"[Warning] Khong doc duoc file pkl cu ({e}). Bat dau lai tu dau.")
        return []
    
def save_records(pkl_path, records):
    with open(pkl_path, "wb") as f:
        pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)




def main(): 
    dataset = VideoDogDataset(RAW_DATASET_DIR)
    all_records = load_existing_records(GLCM_FEATURE_PATH) 
    processed_video_paths = set(r["video_path"] for r in all_records)

    if all_records:
        last_video_id = max(r.get("video_id", -1) for r in all_records)
    else:
        last_video_id = -1

    total_videos = len(dataset.video_paths)
    for idx, video_path in enumerate(dataset.video_paths):
        if video_path in processed_video_paths:
            print(f"[Skip {idx + 1}/{total_videos}] Da co: {video_path}")
            continue

        print(f"[Process {idx + 1}/{total_videos}] Dang trich xuat: {video_path}")
        try:
            records = extract_glcm_from_video(video_path)
        except Exception as e:
            print(f"  -> Loi khi xu ly video: {e}. Bo qua.")
            continue

        last_video_id += 1
        for r in records:
            r["video_id"] = last_video_id

        # Cộng dồn vào danh sách tổng và ghi ngay xuống file
        all_records.extend(records)
        save_records(GLCM_FEATURE_PATH, all_records)

        print(f"Da luu {len(records)} frame. "
              f"Tong cong: {len(all_records)} frame (video_id={last_video_id}).")

    print(f"\nHoan tat. Tong so video: {last_video_id + 1}, "
          f"tong so frame: {len(all_records)}.")        
  
if __name__ == "__main__": 
    main() 