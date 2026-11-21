#type:ignore 

import pandas as pd 
import cv2 
from sqlalchemy import create_engine 
from pathlib import Path 
from d23_lop4_nhom16.dataset import VideoDogDataset 
from d23_lop4_nhom16.config import RAW_DATASET_DIR, METADATA_CSV_PATH

class MetadataExtractor: 
    def __init__(self, raw_data_dir=RAW_DATASET_DIR): 
        self.raw_data_dir = Path(raw_data_dir) 
        self.dataset = VideoDogDataset(self.raw_data_dir) 
        self.metadata_list = [] 

    def extract_all(self): 
        for video_path in self.dataset.video_paths: 
            metadata = self.extract_metadata(video_path) 
            self.metadata_list.append(metadata) 

    def save_metadata_to_csv(self, output_csv_path): 
        df = pd.DataFrame(self.metadata_list) 
        df.to_csv(output_csv_path, index=False)

    def extract_metadata(self, video_path) -> dict: 
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path)) 
        if not cap.isOpened(): 
            print(f"[Error]: {video_path}")
            return None  
        metadata = {} 
        metadata["uuid"] = video_path.stem 
        metadata["filename"] = video_path.name 
        metadata["filepath"] = str(video_path) 
        metadata["category"] = video_path.parent.name 
        metadata["fps"] = cap.get(cv2.CAP_PROP_FPS) 
        metadata["total_frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
        metadata["duration"] = metadata["total_frames"] / metadata["fps"]
        metadata["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        metadata["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release() 
        return metadata 

if __name__ == "__main__": 
    extractor = MetadataExtractor() 
    extractor.extract_all() 
    extractor.save_metadata_to_csv(METADATA_CSV_PATH)































