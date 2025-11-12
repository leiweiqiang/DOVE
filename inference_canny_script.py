#!/usr/bin/env python3
"""
Inference helper that accepts an LQ video (or directory) together with a pre-computed
HQ Canny edge video, merges the edges into the LQ frames, and runs the standard
CogVideoX-based super-resolution pipeline.

The implementation reuses the core inference utilities from `inference_script.py`
while inserting an additive Canny edge merge step that mirrors the training logic.

Memory Optimization Tips:
- Use --is_vae_st to enable VAE slicing and tiling (recommended for large videos)
- Use --is_cpu_offload for CPU offloading (slower but uses less GPU memory)
- Use --enable_model_cpu_offload for more aggressive memory management
- Use --tile_size_hw to enable spatial tiling for large resolutions
- Use --chunk_len to enable temporal chunking for long videos
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

import pyiqa
from diffusers import CogVideoXDPMScheduler, CogVideoXPipeline
from safetensors.torch import load_file
from transformers import set_seed

import inference_script as base_infer
from finetune.data_modules.utils import merge_edge_map_with_lq


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def collect_video_files(directory: Path) -> List[Path]:
    """Return all supported video files (non-recursive) sorted by name."""
    return sorted(
        [
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in base_infer.video_exts
        ]
    )


def resolve_input_pairs(
    lq_input: Path,
    canny_input: Path,
    canny_suffix: str = "",
) -> List[Tuple[Path, Path]]:
    """
    Resolve all (LQ, Canny) video pairs based on the provided inputs.

    Supports both file and directory inputs. When directories are provided,
    the function looks for matching filenames in `canny_input`. If `canny_suffix`
    is supplied, it is appended to the LQ stem before searching.
    """
    if lq_input.is_file():
        if not canny_input.is_file():
            raise ValueError(
                f"Canny input ({canny_input}) must be a file when LQ input is a file."
            )
        return [(lq_input, canny_input)]

    if not lq_input.is_dir():
        raise ValueError(f"LQ input path {lq_input} is neither a file nor a directory.")

    if not canny_input.is_dir():
        raise ValueError(
            f"Canny input ({canny_input}) must be a directory when LQ input is a directory."
        )

    lq_videos = collect_video_files(lq_input)
    if not lq_videos:
        raise ValueError(f"No supported video files found in {lq_input}.")

    pairs: List[Tuple[Path, Path]] = []
    for video_path in lq_videos:
        candidates: Sequence[Path]
        if canny_suffix:
            candidates = (
                canny_input / f"{video_path.stem}{canny_suffix}{video_path.suffix}",
                canny_input / video_path.name,
            )
        else:
            candidates = (canny_input / video_path.name,)

        canny_path = next((cand for cand in candidates if cand.exists()), None)
        if canny_path is None:
            suffix_hint = f" with suffix '{canny_suffix}'" if canny_suffix else ""
            raise FileNotFoundError(
                f"Could not find matching Canny video for {video_path.name} in "
                f"{canny_input}{suffix_hint}."
            )
        pairs.append((video_path, canny_path))

    return pairs


def load_and_prepare_canny_video(
    canny_path: Path,
    original_shape: Tuple[int, int, int, int],
    pad_f: int,
    pad_h: int,
    pad_w: int,
    upscale: int,
    target_height: int,
    target_width: int,
) -> torch.Tensor:
    """
    Load the HQ Canny video, convert it to a single-channel tensor, and align it with
    the (possibly padded & upscaled) LQ tensor so it can be merged directly.

    Args:
        canny_path: Path to the HQ Canny edge video.
        original_shape: (frames, height, width, channels) before padding.
        pad_f: Additional frames that were appended to the LQ sequence.
        pad_h: Spatial padding (pre-upscale) applied to the LQ sequence.
        pad_w: Spatial padding (pre-upscale) applied to the LQ sequence.
        upscale: Spatial upscale factor applied to the LQ sequence.
        target_height: Final LQ height after upscaling and padding.
        target_width: Final LQ width after upscaling and padding.

    Returns:
        torch.Tensor: Edge tensor with shape [F, 1, target_height, target_width]
        in the [0, 255] range (float32).
    """
    reader = base_infer.decord.VideoReader(uri=canny_path.as_posix())
    canny_frames = reader.get_batch(list(range(len(reader)))).to(torch.float32)  # [F, H, W, C]

    if canny_frames.ndim != 4:
        raise ValueError(f"Unexpected Canny tensor shape {canny_frames.shape} for {canny_path}")

    # Ensure single channel by taking the first channel (Canny export often replicates RGB)
    if canny_frames.shape[-1] == 3:
        canny_frames = canny_frames[..., 0:1]
    elif canny_frames.shape[-1] == 1:
        pass
    else:
        raise ValueError(
            f"Canny video {canny_path} has unsupported channel count {canny_frames.shape[-1]}"
        )

    canny_frames = canny_frames.permute(0, 3, 1, 2).contiguous()  # [F, 1, H, W]

    original_frames = original_shape[0]
    if canny_frames.shape[0] < original_frames:
        last = canny_frames[-1:].repeat(original_frames - canny_frames.shape[0], 1, 1, 1)
        canny_frames = torch.cat([canny_frames, last], dim=0)
    elif canny_frames.shape[0] > original_frames:
        canny_frames = canny_frames[:original_frames]

    expected_height = original_shape[1] * upscale
    expected_width = original_shape[2] * upscale
    if canny_frames.shape[2] != expected_height or canny_frames.shape[3] != expected_width:
        canny_frames = F.interpolate(
            canny_frames,
            size=(expected_height, expected_width),
            mode="bilinear",
            align_corners=False,
        )

    # Pad spatial dimensions to match the padded (pre-upscale) LQ tensor
    if pad_h > 0 or pad_w > 0:
        canny_frames = F.pad(
            canny_frames,
            pad=(0, pad_w * upscale, 0, pad_h * upscale),
            mode="constant",
            value=0.0,
        )

    # Pad temporal dimension if the LQ tensor added extra frames
    if pad_f > 0:
        last = canny_frames[-1:].repeat(pad_f, 1, 1, 1)
        canny_frames = torch.cat([canny_frames, last], dim=0)

    # Final safety resize to ensure perfect alignment (e.g., rounding effects)
    if canny_frames.shape[2] != target_height or canny_frames.shape[3] != target_width:
        canny_frames = F.interpolate(
            canny_frames,
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        )

    if canny_frames.shape[0] != original_frames + pad_f:
        raise ValueError(
            f"Temporal dimension mismatch after padding for {canny_path}: "
            f"expected {original_frames + pad_f}, got {canny_frames.shape[0]}"
        )

    return canny_frames.clamp(0.0, 255.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference with Canny edge guidance.")

    parser.add_argument("--lq_input", type=str, required=True, help="LQ video path or directory.")
    parser.add_argument("--canny_input", type=str, required=True, help="HQ Canny video path or directory.")
    parser.add_argument(
        "--canny_suffix",
        type=str,
        default="",
        help="Optional suffix appended to LQ stems when resolving Canny filenames.",
    )

    parser.add_argument("--input_json", type=str, default=None)
    parser.add_argument("--gt_dir", type=str, default=None)
    parser.add_argument("--eval_metrics", type=str, default="")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=None, help="Optional LoRA weights path.")
    parser.add_argument("--output_path", type=str, default="./results", help="Output directory.")
    parser.add_argument("--fps", type=int, default=16, help="Output video FPS.")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Computation dtype.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--upscale_mode", type=str, default="bilinear")
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--noise_step", type=int, default=0)
    parser.add_argument("--sr_noise_step", type=int, default=399)
    parser.add_argument("--is_cpu_offload", action="store_true")
    parser.add_argument("--is_vae_st", action="store_true")
    parser.add_argument("--png_save", action="store_true")
    parser.add_argument("--save_format", type=str, default="yuv444p")
    parser.add_argument("--tile_size_hw", type=int, nargs=2, default=(0, 0))
    parser.add_argument("--overlap_hw", type=int, nargs=2, default=(32, 32))
    parser.add_argument("--chunk_len", type=int, default=0)
    parser.add_argument("--overlap_t", type=int, default=8)
    parser.add_argument(
        "--enable_model_cpu_offload",
        action="store_true",
        help="Use enable_model_cpu_offload instead of enable_sequential_cpu_offload for more aggressive memory management",
    )
    parser.add_argument(
        "--no_subtract_edge",
        action="store_true",
        help="Disable subtraction of Canny edges from output. By default, edges are subtracted to remove artifacts.",
    )
    parser.add_argument(
        "--edge_subtraction_scale",
        type=float,
        default=1.0,
        help="Scale factor for edge subtraction (default: 1.0). Lower values reduce edge removal strength.",
    )

    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if args.dtype not in dtype_map:
        raise ValueError("Invalid dtype. Choose from 'float16', 'bfloat16', or 'float32'.")
    dtype = dtype_map[args.dtype]

    overlap_t = args.overlap_t if args.chunk_len > 0 else 0
    overlap_hw = tuple(args.overlap_hw) if args.tile_size_hw != (0, 0) else (0, 0)

    set_seed(args.seed)

    empty_prompt_embedding = None
    empty_prompt_path = Path(
        "pretrained_models/prompt_embeddings/"
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855.safetensors"
    )
    if empty_prompt_path.exists():
        try:
            empty_prompt_embedding = load_file(str(empty_prompt_path))["prompt_embedding"]
            logger.info("Loaded empty prompt embedding from %s", empty_prompt_path)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Failed to load empty prompt embedding: %s", exc)
            empty_prompt_embedding = None
    else:
        logger.warning("Empty prompt embedding not found at %s", empty_prompt_path)

    if args.input_json is not None:
        with open(args.input_json, "r", encoding="utf-8") as handle:
            video_prompt_dict = json.load(handle)
    else:
        video_prompt_dict = {}

    lq_input_path = Path(args.lq_input).expanduser().resolve()
    canny_input_path = Path(args.canny_input).expanduser().resolve()

    input_pairs = resolve_input_pairs(lq_input_path, canny_input_path, args.canny_suffix)
    logger.info("Found %d input pair(s) for inference.", len(input_pairs))

    os.makedirs(args.output_path, exist_ok=True)

    pipe = CogVideoXPipeline.from_pretrained(args.model_path, torch_dtype=dtype)

    if args.lora_path:
        logger.info("Loading LoRA weights from %s", args.lora_path)
        pipe.load_lora_weights(
            args.lora_path,
            weight_name="pytorch_lora_weights.safetensors",
            adapter_name="test_1",
        )
        pipe.fuse_lora(components=["transformer"], lora_scale=1.0)

    pipe.scheduler = CogVideoXDPMScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="trailing",
    )

    if args.is_cpu_offload:
        if args.enable_model_cpu_offload:
            pipe.enable_model_cpu_offload()
            logger.info("Using enable_model_cpu_offload for aggressive memory management")
        else:
            pipe.enable_sequential_cpu_offload()
            logger.info("Using enable_sequential_cpu_offload")
    else:
        pipe.to("cuda")
        # Clear cache after moving to GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Always enable VAE slicing/tiling for memory efficiency if not already enabled
    if args.is_vae_st:
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
        logger.info("VAE slicing and tiling enabled for memory efficiency")
    else:
        logger.warning(
            "VAE slicing/tiling not enabled. Consider using --is_vae_st to reduce memory usage."
        )

    if args.eval_metrics:
        metrics_list = [metric.strip().lower() for metric in args.eval_metrics.split(",")]
        metrics_models: Dict[str, pyiqa.BaseMetric] = {}
        for name in metrics_list:
            try:
                metrics_models[name] = pyiqa.create_metric(name).to(pipe.device).eval()
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("Failed to initialize metric '%s': %s", name, exc)
        metric_accumulator: Dict[str, List[float]] = {name: [] for name in metrics_models}
    else:
        metrics_list = []
        metrics_models = {}
        metric_accumulator = {}

    frame_transform = transforms.Compose(
        [
            transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0),
        ]
    )

    for lq_path, canny_path in tqdm(input_pairs, desc="Processing videos"):
        video_name = lq_path.name
        prompt = video_prompt_dict.get(video_name, "")

        video, pad_f, pad_h, pad_w, original_shape = base_infer.preprocess_video_match(
            lq_path, is_match=True
        )
        H_raw, W_raw = video.shape[2], video.shape[3]

        video = F.interpolate(
            video,
            size=(H_raw * args.upscale, W_raw * args.upscale),
            mode=args.upscale_mode,
            align_corners=False,
        )

        edge_tensor = load_and_prepare_canny_video(
            canny_path=canny_path,
            original_shape=original_shape,
            pad_f=pad_f,
            pad_h=pad_h,
            pad_w=pad_w,
            upscale=args.upscale,
            target_height=video.shape[2],
            target_width=video.shape[3],
        )

        # Save edge tensor for subtraction from output later (if enabled)
        subtract_edge = not args.no_subtract_edge
        if subtract_edge:
            # Convert edge tensor to RGB format: [F, 1, H, W] -> [F, 3, H, W]
            edge_tensor_rgb = edge_tensor.repeat(1, 3, 1, 1)  # [F, 3, H, W] in [0, 255]
            
            # Apply same transform as video to match input format
            edge_tensor_transformed = torch.stack([frame_transform(frame) for frame in edge_tensor_rgb], dim=0)
            # Store in [B, C, F, H, W] format for later subtraction
            edge_tensor_for_subtraction = edge_tensor_transformed.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()
            # edge_tensor_for_subtraction is now in [-1, 1] range, will be converted to [0, 1] later
        else:
            edge_tensor_for_subtraction = None

        video = merge_edge_map_with_lq(
            lq_frames=video,
            edge_maps=edge_tensor,
            method="additive_clip",
        )
        
        # Clear original edge tensor from memory after merging
        del edge_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        video = torch.stack([frame_transform(frame) for frame in video], dim=0)
        video = video.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()

        _, _, F_total, H_total, W_total = video.shape
        time_chunks = base_infer.make_temporal_chunks(F_total, args.chunk_len, overlap_t)
        spatial_tiles = base_infer.make_spatial_tiles(H_total, W_total, tuple(args.tile_size_hw), overlap_hw)

        output_video = torch.zeros_like(video)
        write_count = torch.zeros_like(video, dtype=torch.int)

        logger.info(
            (
                "Process video: %s | Prompt: %s | Frames: %d (orig: %d, pad: %d) | "
                "Resolution: %dx%d (orig: %dx%d; pad: %d x %d) | Chunks: %d"
            ),
            video_name,
            prompt,
            F_total,
            original_shape[0],
            pad_f,
            H_total,
            W_total,
            original_shape[1] * args.upscale,
            original_shape[2] * args.upscale,
            pad_h * args.upscale,
            pad_w * args.upscale,
            len(time_chunks) * len(spatial_tiles),
        )

        for t_start, t_end in time_chunks:
            for h_start, h_end, w_start, w_end in spatial_tiles:
                video_chunk = video[:, :, t_start:t_end, h_start:h_end, w_start:w_end]
                
                # Log chunk size for debugging
                chunk_size_gb = video_chunk.numel() * video_chunk.element_size() / (1024**3)
                if chunk_size_gb > 5.0:  # Warn if chunk is larger than 5GB
                    logger.warning(
                        "Large chunk detected: %.2f GB (t: %d-%d, h: %d-%d, w: %d-%d). "
                        "Consider using smaller tile_size_hw or chunk_len.",
                        chunk_size_gb,
                        t_start,
                        t_end,
                        h_start,
                        h_end,
                        w_start,
                        w_end,
                    )

                generated_chunk = base_infer.process_video(
                    pipe=pipe,
                    video=video_chunk,
                    prompt=prompt,
                    noise_step=args.noise_step,
                    sr_noise_step=args.sr_noise_step,
                    empty_prompt_embedding=empty_prompt_embedding,
                )

                region = base_infer.get_valid_tile_region(
                    t_start,
                    t_end,
                    h_start,
                    h_end,
                    w_start,
                    w_end,
                    video_shape=video.shape,
                    overlap_t=overlap_t,
                    overlap_h=overlap_hw[0],
                    overlap_w=overlap_hw[1],
                )

                output_video[
                    :,
                    :,
                    region["out_t_start"]: region["out_t_end"],
                    region["out_h_start"]: region["out_h_end"],
                    region["out_w_start"]: region["out_w_end"],
                ] = generated_chunk[
                    :,
                    :,
                    region["valid_t_start"]: region["valid_t_end"],
                    region["valid_h_start"]: region["valid_h_end"],
                    region["valid_w_start"]: region["valid_w_end"],
                ]

                write_count[
                    :,
                    :,
                    region["out_t_start"]: region["out_t_end"],
                    region["out_h_start"]: region["out_h_end"],
                    region["out_w_start"]: region["out_w_end"],
                ] += 1
                
                # Clear GPU cache after each chunk to prevent memory accumulation
                del generated_chunk, video_chunk
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if (write_count == 0).any():
            raise RuntimeError("Some regions were never written during tiling.")
        if (write_count > 1).any():
            raise RuntimeError("Some regions were written multiple times during tiling.")

        video_generate = output_video
        
        # Subtract Canny edges from output to remove edge artifacts if enabled
        subtract_edge = not args.no_subtract_edge
        if subtract_edge:
            # Convert edge tensor from [-1, 1] to [0, 1] range to match output format
            edge_normalized = (edge_tensor_for_subtraction * 0.5 + 0.5).clamp(0.0, 1.0)
            # Apply scaling factor for edge subtraction
            edge_scaled = edge_normalized * args.edge_subtraction_scale
            # Subtract edges from output
            video_generate = torch.clamp(video_generate - edge_scaled, 0.0, 1.0)
            logger.info(
                "Subtracted Canny edges from output (scale: %.2f) to remove edge artifacts.",
                args.edge_subtraction_scale,
            )
        
        # Clear edge tensor from memory
        if subtract_edge:
            del edge_normalized, edge_scaled
        if edge_tensor_for_subtraction is not None:
            del edge_tensor_for_subtraction
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        video_generate = base_infer.remove_padding_and_extra_frames(
            video_generate,
            pad_f,
            pad_h * args.upscale,
            pad_w * args.upscale,
        )

        output_path = os.path.join(args.output_path, video_name)

        if metrics_models:
            pred_frames = video_generate[0].permute(1, 0, 2, 3).contiguous()
            if args.gt_dir is not None:
                gt_frames = base_infer.load_sequence(os.path.join(args.gt_dir, video_name))
            else:
                gt_frames = None

            print(f"\n\n[{video_name}] Metrics:", end=" ")
            for name, model in metrics_models.items():
                scores = []
                for idx in range(pred_frames.shape[0]):
                    pred = pred_frames[idx].unsqueeze(0)
                    if gt_frames is not None:
                        gt = gt_frames[idx].unsqueeze(0)
                    else:
                        gt = None
                    if name in base_infer.fr_metrics:
                        score = model(pred, gt).item()
                    else:
                        score = model(pred).item()
                    scores.append(score)
                val = sum(scores) / len(scores)
                metric_accumulator[name].append(val)
                print(f"{name.upper()}={val:.4f}", end="  ")
            print()

        if args.png_save:
            output_dir = output_path.rsplit(".", 1)[0]
            base_infer.save_frames_as_png(video_generate, output_dir, fps=args.fps)
        else:
            output_path = output_path.replace(".mkv", ".mp4")
            base_infer.save_video_with_imageio(
                video_generate,
                output_path,
                fps=args.fps,
                format=args.save_format,
            )

    if metrics_models:
        print("\n=== Overall Average Metrics ===")
        overall_avg: Dict[str, float] = defaultdict(float)
        for metric in metrics_models:
            scores = metric_accumulator.get(metric, [])
            if scores:
                avg = sum(scores) / len(scores)
                overall_avg[metric] = avg
                print(f"{metric.upper()}: {avg:.4f}")

        out_name = "metrics_" + "_".join(metrics_models.keys()) + ".json"
        out_path = os.path.join(args.output_path, out_name)
        output = {
            "per_sample": metric_accumulator,
            "average": overall_avg,
            "count": len(next(iter(metric_accumulator.values()), [])),
        }
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(output, handle, indent=2)

    print("All videos processed.")


if __name__ == "__main__":
    main()

