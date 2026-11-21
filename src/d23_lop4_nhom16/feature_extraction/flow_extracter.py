#!/usr/bin/env python3
"""Trích xuất Farnebäck Optical Flow -> vector 13 chiều cho 1 video."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np


FLOW_DIM = 13


@dataclass(frozen=True)
class FlowConfig:
    sample_fps: float = 5.0
    frame_width: int = 320
    frame_height: int = 240
    direction_bins: int = 8
    magnitude_threshold: float = 1.0
    pyr_scale: float = 0.5
    levels: int = 3
    winsize: int = 15
    iterations: int = 3
    poly_n: int = 5
    poly_sigma: float = 1.2
    flags: int = 0


CONFIG = FlowConfig()


def decode_sampled_video(
    video_path: Path,
    consume: Callable[[np.ndarray], None],
    config: FlowConfig = CONFIG,
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


class FarnebackFlowExtractor:
    """Tích lũy Farnebäck flow toàn video -> vector 13 chiều."""

    def __init__(self, config: FlowConfig = CONFIG):
        self.config = config
        self.previous: np.ndarray | None = None
        self.histogram = np.zeros(config.direction_bins, dtype=np.float64)
        self.magnitude_sum = 0.0
        self.magnitude_square_sum = 0.0
        self.u_sum = 0.0
        self.v_sum = 0.0
        self.pixel_count = 0
        self.valid_weight = 0.0
        self.cosine_sum = 0.0
        self.sine_sum = 0.0

    def add(self, current: np.ndarray) -> None:
        if self.previous is None:
            self.previous = current
            return
        c = self.config
        flow = cv2.calcOpticalFlowFarneback(
            self.previous, current, None,
            c.pyr_scale, c.levels, c.winsize,
            c.iterations, c.poly_n, c.poly_sigma, c.flags,
        )
        u = flow[..., 0].astype(np.float64)
        v = flow[..., 1].astype(np.float64)
        magnitude, angle = cv2.cartToPolar(u, v, angleInDegrees=False)

        self.magnitude_sum += float(np.sum(magnitude, dtype=np.float64))
        self.magnitude_square_sum += float(np.sum(np.square(magnitude), dtype=np.float64))
        self.u_sum += float(np.sum(u, dtype=np.float64))
        self.v_sum += float(np.sum(v, dtype=np.float64))
        self.pixel_count += magnitude.size

        valid = magnitude >= c.magnitude_threshold
        if np.any(valid):
            weights = magnitude[valid]
            theta = angle[valid] % (2.0 * np.pi)
            self.histogram += np.histogram(
                theta, bins=c.direction_bins,
                range=(0.0, 2.0 * np.pi), weights=weights,
            )[0]
            weight_sum = float(np.sum(weights, dtype=np.float64))
            self.valid_weight += weight_sum
            self.cosine_sum += float(np.sum(weights * np.cos(theta), dtype=np.float64))
            self.sine_sum += float(np.sum(weights * np.sin(theta), dtype=np.float64))
        self.previous = current

    def finish(self) -> np.ndarray:
        """8 bin hướng (mirror + L1) + 5 thống kê = 13 chiều."""
        if self.pixel_count == 0:
            return np.zeros(FLOW_DIM, dtype=np.float32)
        c = self.config
        if self.valid_weight > 0:
            histogram = self.histogram / self.valid_weight
            bin_width = 2.0 * np.pi / c.direction_bins
            centers = (np.arange(c.direction_bins) + 0.5) * bin_width
            mirrored_bins = np.floor(
                ((np.pi - centers) % (2.0 * np.pi)) / bin_width
            ).astype(int)
            mirrored = np.zeros_like(histogram)
            mirrored[mirrored_bins] = histogram
            histogram = 0.5 * (histogram + mirrored)

            cosine = self.cosine_sum / self.valid_weight
            sine = self.sine_sum / self.valid_weight
            resultant = min(1.0, math.hypot(cosine, sine))
            circular_std = math.sqrt(-2.0 * math.log(max(resultant, 1e-12)))
        else:
            histogram = np.zeros(c.direction_bins, dtype=np.float64)
            circular_std = 0.0

        mean_magnitude = self.magnitude_sum / self.pixel_count
        variance = max(
            0.0,
            self.magnitude_square_sum / self.pixel_count
            - mean_magnitude * mean_magnitude,
        )
        statistics = np.array([
            mean_magnitude,
            math.sqrt(variance),
            circular_std,
            abs(self.u_sum / self.pixel_count),
            self.v_sum / self.pixel_count,
        ], dtype=np.float64)
        return np.concatenate([histogram, statistics]).astype(np.float32)


def extract_flow(video_path: Path, config: FlowConfig = CONFIG) -> dict:
    """Trả về {'flow_vector': (13,), 'metadata': dict}."""
    extractor = FarnebackFlowExtractor(config)
    metadata = decode_sampled_video(video_path, extractor.add, config)
    return {"flow_vector": extractor.finish(), "metadata": metadata}


def main() -> int:
    parser = argparse.ArgumentParser(description="Trích Flow 13 chiều cho 1 video")
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.video.with_name(f"{args.video.stem}_flow.npz")
    result = extract_flow(args.video)
    np.savez_compressed(
        output,
        flow_vector=result["flow_vector"],
        metadata_json=np.array(json.dumps(result["metadata"], ensure_ascii=False)),
        config_json=np.array(json.dumps(asdict(CONFIG), ensure_ascii=False)),
    )
    print(f"Flow shape: {result['flow_vector'].shape}")
    print(f"Đã lưu: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())