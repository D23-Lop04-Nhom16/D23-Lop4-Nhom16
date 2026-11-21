from pathlib import Path 
from collections import defaultdict
import hashlib

DATASET_DIR = Path(__file__).resolve().parent.parent.parent.parent/ "data_collection"

# print(DATASET_DIR)

def calculate_sha256(file_path: Path) -> str:
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            sha256.update(chunk)

    return sha256.hexdigest()

def find_duplicate_videos(dataset_dir: Path):
    # Tìm tất cả MP4 kể cả trong thư mục con
    videos = list(dataset_dir.rglob("*.mp4"))

    print(f"Total videos: {len(videos)}")

    hash_groups = defaultdict(list)

    for i, video in enumerate(videos, 1):
        # print(f"[{i}/{len(videos)}] {video}")

        file_hash = calculate_sha256(video)

        hash_groups[file_hash].append(video)

    duplicates = {
        file_hash: paths
        for file_hash, paths in hash_groups.items()
        if len(paths) > 1
    }

    return duplicates

# videos = list(DATASET_DIR.rglob("*.mp4"))
# print(f"Total videos: {len(videos)}")

if __name__ == "__main__":

    duplicates = find_duplicate_videos(DATASET_DIR)

    if duplicates:
        print("Duplicate videos found:")
        for file_hash, paths in duplicates.items():
            print(f"Hash: {file_hash}")
            for path in paths:
                print(f" - {path}")
    else:
        print("No duplicate videos found.")








