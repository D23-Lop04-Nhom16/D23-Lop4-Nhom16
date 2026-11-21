import torch.nn 
from torch.utils.data import Dataset 
from pathlib import Path 

class VideoDogDataset(Dataset): 
    def __init__(self, raw_data_dir):
        self.root = Path(raw_data_dir)  
        self.video_paths = [] 
        for video_path in self.root.rglob("*.mp4"): 
            if video_path.is_file(): 
                self.video_paths.append(video_path) 

    def __len__(self): 
        return len(self.video_paths) 
    def __getitem__(self, index):
        pass 


