"""HOG round-fps-v1: keyframe cách nhau round(fps) frame, mỗi vector có 8100 chiều."""

import csv
import json
from pathlib import Path
from d23_lop4_nhom16.dataset import VideoDogDataset 
from d23_lop4_nhom16.config import RAW_DATASET_DIR
import cv2
import numpy as np
from skimage.feature import hog
import pandas as pd 

def extract_video_keyframes(video_path: str | Path) -> list[dict]:
    """Lấy frame 0, frame_interval, 2 * frame_interval... như thuật toán đã chọn."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Không mở được video: {video_path}")

    keyframes = []
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            fps = 25

        frame_interval = max(1, int(round(fps)))
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                keyframes.append({
                    "frame_idx": frame_idx,
                    "timestamp": frame_idx / fps,
                    "frame": frame,
                    "fps": fps,
                    "frame_interval": frame_interval,
                })
            frame_idx += 1
    finally:
        cap.release()

    if not keyframes:
        raise ValueError(f"Video không có frame đọc được: {video_path}")
    return keyframes


def extract_hog_keyframe(frame: np.ndarray) -> np.ndarray:
    """Resize toàn frame về 128x128, chuyển xám và trích HOG L2-Hys."""
    resized = cv2.resize(frame, (128, 128), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=False,
        feature_vector=True,
        channel_axis=None,
    ).astype(np.float32)


def extract_video_hog(video_path: str | Path) -> list[dict]:
    """Trả về một bản ghi cho mỗi keyframe, giữ riêng từng vector HOG."""
    video_path = Path(video_path).resolve()
    return [
        {
            "filename": video_path.name,
            "filepath": str(video_path),
            "label": video_path.parent.name,
            "keyframe_idx": keyframe_idx,
            "frame_idx": keyframe["frame_idx"],
            "timestamp_sec": keyframe["timestamp"],
            "fps": keyframe["fps"],
            "frame_interval": keyframe["frame_interval"],
            "feature_dim": 8100,
            "vector": extract_hog_keyframe(keyframe["frame"]),
        }
        for keyframe_idx, keyframe in enumerate(extract_video_keyframes(video_path))
    ]


def write_hog_csv(records, csv_path: str | Path) -> None:
    """Ghi mỗi keyframe thành một dòng CSV; vector là mảng số theo định dạng JSON."""
    fields = [
        "filename", "filepath", "label", "keyframe_idx", "frame_idx",
        "timestamp_sec", "fps", "frame_interval", "feature_dim", "vector",
    ]
    with Path(csv_path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({
                **record,
                "vector": json.dumps(
                    record["vector"].tolist(), separators=(",", ":"), allow_nan=False
                ),
            })

if __name__ == "__main__": 
    dataset = VideoDogDataset(RAW_DATASET_DIR) 
    all_features = [] 
    
    for video_path in dataset.video_paths: 
        video_features = extract_video_hog(video_path) 
        all_features.extend(video_features) 
        print(
            f"Video: {video_path} | "
            f"Total extracted frames: {len(video_features)}"
        )
    df = pd.DataFrame(all_features) 
    df.to_csv("/home/ninhdao/D23_Lop4_Nhom16/src/d23_lop4_nhom16/features/new_hog.csv", index=False) 