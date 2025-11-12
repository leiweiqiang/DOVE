#!/usr/bin/env python3
"""
Utility script to convert an HQ video into its Canny edge counterpart.

Usage example:
    python generate_canny_video.py \
        --input_video /path/to/hq.mp4 \
        --output_video /path/to/hq_canny.mp4 \
        --threshold1 50 \
        --threshold2 150
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import cv2
import numpy as np
import imageio.v3 as iio

from finetune.data_modules.utils import generate_canny_edge_map


def read_video_frames(video_path: Path) -> tuple[List[np.ndarray], float]:
    """Read all frames from a video file.

    Returns:
        frames: list of frames in RGB uint8 format [H, W, 3]
        fps: frames per second of the input video
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames: List[np.ndarray] = []

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()
    if len(frames) == 0:
        raise RuntimeError(f"No frames found in video: {video_path}")

    return frames, fps


def write_video(frames: np.ndarray, output_path: Path, fps: float) -> None:
    """Write frames [F, H, W, 3] as an mp4 video."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(
        output_path,
        frames,
        fps=fps,
        codec="libx264",
        pixelformat="yuv444p",
        macro_block_size=None,
        ffmpeg_params=["-crf", "0"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Canny edge video from HQ video.")
    parser.add_argument("--input_video", type=str, required=True, help="Path to the HQ input video.")
    parser.add_argument(
        "--output_video",
        type=str,
        default=None,
        help="Path to save the Canny edge video. Defaults to <input>_canny.mp4",
    )
    parser.add_argument("--threshold1", type=float, default=50.0, help="Lower threshold for Canny.")
    parser.add_argument("--threshold2", type=float, default=150.0, help="Upper threshold for Canny.")
    parser.add_argument(
        "--kernel_size",
        type=int,
        default=5,
        help="Gaussian blur kernel size (must be odd, set 1 to disable).",
    )
    parser.add_argument(
        "--sigma", type=float, default=1.4, help="Gaussian blur sigma before Canny edge detection."
    )

    args = parser.parse_args()

    input_path = Path(args.input_video).expanduser().resolve()
    if args.output_video is None:
        output_path = input_path.with_name(f"{input_path.stem}_canny.mp4")
    else:
        output_path = Path(args.output_video).expanduser().resolve()

    frames_rgb, fps = read_video_frames(input_path)

    edge_maps = generate_canny_edge_map(
        frames_rgb,
        threshold1=args.threshold1,
        threshold2=args.threshold2,
        kernel_size=args.kernel_size,
        sigma=args.sigma,
    )  # [F, H, W, 1]

    # Convert to 3-channel RGB for video writing
    edge_maps_rgb = np.repeat(edge_maps, repeats=3, axis=-1)

    write_video(edge_maps_rgb.astype(np.uint8), output_path, fps)
    print(f"Saved Canny edge video to {output_path}")


if __name__ == "__main__":
    main()



