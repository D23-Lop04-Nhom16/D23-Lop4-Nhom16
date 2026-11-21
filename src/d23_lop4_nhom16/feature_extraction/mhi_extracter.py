#!/usr/bin/env python3
"""Trích Motion History Image -> ma trận (n_window, 256) cho 1 video."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np


MHI_DIM = 256


@dataclass(frozen=True)
class MHIConfig:
    sample_fps: float = 5.0
    frame_width: int = 320
    frame_height: int = 240
    mhi_window_seconds: float = 1.0
    mhi_threshold: int = 25
    mhi_width: int = 16
    mhi_height: int = 16


CONFIG = MHIConfig()


def decode_sampled_video(
    video_path: Path,
    consume: Callable[[np.ndarray], None],
    config: MHIConfig = CONFIG,
) -> dict:
    """Sample đều 5 FPS, resize 320x240, chuyển gray, gọi consume(gray)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Không mở được video: {video_path}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / source_fps if source_fps > 0 else 0.0
    if source_fps <= 0:
        source_fps = config.sample_fps

    next_sample_time = 0.0
    sample_step = 1.0 / config.sample_fps
    frame_index = 0
    sampled_frames = 0
    last_timestamp = -1.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            nominal_time = frame_index / source_fps
            raw_time = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            if frame_index == 0:
                current_time = 0.0
            elif math.isfinite(raw_time) and raw_time > last_timestamp + 1e-6:
                current_time = raw_time
            else:
                current_time = nominal_time
            if math.isfinite(raw_time) and raw_time >= 0:
                last_timestamp = max(last_timestamp, raw_time)

            if current_time + 1e-9 >= next_sample_time:
                frame = cv2.resize(
                    frame, (config.frame_width, config.frame_height),
                    interpolation=cv2.INTER_AREA,
                )
                consume(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                sampled_frames += 1
                while next_sample_time <= current_time + 1e-9:
                    next_sample_time += sample_step
            frame_index += 1
    finally:
        cap.release()

    if sampled_frames == 0:
        raise ValueError("Video không có frame giải mã được")
    if duration <= 0:
        duration = frame_index / source_fps
    return {
        "path": str(video_path.resolve()),
        "duration": duration,
        "source_fps": source_fps,
        "frame_count": frame_count or frame_index,
        "width": width,
        "height": height,
        "sampled_frames": sampled_frames,
    }


def mhi_window_descriptor(
    frames: Sequence[np.ndarray], config: MHIConfig = CONFIG
) -> np.ndarray:
    """MHI 16x16 = 256 chiều cho 1 cửa sổ."""
    if len(frames) < 2:
        return np.zeros(MHI_DIM, dtype=np.float32)

    duration = len(frames) - 1
    mhi = np.zeros(frames[0].shape, dtype=np.float32)
    previous = frames[0]
    for current in frames[1:]:
        motion = cv2.absdiff(current, previous) > config.mhi_threshold
        np.subtract(mhi, 1.0, out=mhi)
        np.maximum(mhi, 0.0, out=mhi)
        mhi[motion] = float(duration)
        previous = current

    mhi /= float(duration)
    return cv2.resize(
        mhi, (config.mhi_width, config.mhi_height),
        interpolation=cv2.INTER_AREA,
    ).reshape(-1).astype(np.float32)


class MHIExtractor:
    """MHI theo cửa sổ 1s không chồng lấp."""

    def __init__(self, config: MHIConfig = CONFIG):
        self.config = config
        self.window_size = max(
            2, int(round(config.sample_fps * config.mhi_window_seconds))
        )
        self.current: list[np.ndarray] = []
        self.last_full: list[np.ndarray] | None = None
        self.descriptors: list[np.ndarray] = []

    def add(self, frame: np.ndarray) -> None:
        self.current.append(frame)
        if len(self.current) == self.window_size:
            if self.last_full is not None:
                self.descriptors.append(mhi_window_descriptor(self.last_full, self.config))
            self.last_full = self.current
            self.current = []

    def finish(self) -> np.ndarray:
        if self.last_full is None:
            if len(self.current) == 1:
                self.descriptors.append(np.zeros(MHI_DIM, dtype=np.float32))
            elif self.current:
                self.descriptors.append(mhi_window_descriptor(self.current, self.config))
        elif len(self.current) == 1:
            self.descriptors.append(mhi_window_descriptor(self.last_full + self.current, self.config))
        else:
            self.descriptors.append(mhi_window_descriptor(self.last_full, self.config))
            if self.current:
                self.descriptors.append(mhi_window_descriptor(self.current, self.config))
        return np.stack(self.descriptors)


def extract_mhi(video_path: Path, config: MHIConfig = CONFIG) -> dict:
    """Trả về {'mhi_vectors': (n_win, 256), 'metadata': dict}."""
    extractor = MHIExtractor(config)
    metadata = decode_sampled_video(video_path, extractor.add, config)
    return {"mhi_vectors": extractor.finish(), "metadata": metadata}


def main() -> int:
    parser = argparse.ArgumentParser(description="Trích MHI (n_win, 256) cho 1 video")
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.video.with_name(f"{args.video.stem}_mhi.npz")
    result = extract_mhi(args.video)
    np.savez_compressed(
        output,
        mhi_vectors=result["mhi_vectors"],
        metadata_json=np.array(json.dumps(result["metadata"], ensure_ascii=False)),
        config_json=np.array(json.dumps(asdict(CONFIG), ensure_ascii=False)),
    )
    print(f"MHI shape: {result['mhi_vectors'].shape}")
    print(f"Đã lưu: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())