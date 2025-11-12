# `inference_canny_script.py`

This helper script runs CogVideoX super-resolution while fusing a low-quality (LQ) input video with a pre-computed high-quality Canny edge video. It mirrors the Canny merge logic used during training, making it easy to benchmark or demo the edge-guided inference pipeline.

## Key Workflow

1. Resolve matching pairs between LQ inputs and Canny videos (supports files or directories, optional suffix).
2. Load each LQ clip, pad it for chunked inference, and upscale it by the desired factor.
3. Load the corresponding Canny video, align it (temporal padding, spatial padding, scaling), and expand to match the upscaled LQ tensor.
4. Merge edges via additive clipping to highlight structural details before running the CogVideoX pipeline.
5. Decode the SR output, remove any padding, and optionally compute QA metrics or save as PNG frames.

## Two-Step Workflow

### Step 1: Generate Canny Edge Videos

First, generate Canny edge videos from your HQ videos using `generate_canny_video.py`:

```bash
python generate_canny_video.py \
  --input_video /path/to/hq_video.mp4 \
  --output_video /path/to/hq_canny.mp4 \
  --threshold1 50 \
  --threshold2 150
```

**Note:** `generate_canny_video.py` does NOT use the CogVideoX model and does NOT accept inference-related flags like `--is_vae_st`.

### Step 2: Run Inference with Canny Guidance

Then, run the inference script with the generated Canny videos:

```bash
python inference_canny_script.py \
  --lq_input /path/to/lq_videos \
  --canny_input /path/to/hq_canny_videos \
  --model_path pretrained_models/CogVideoX1.5-5B \
  --output_path ./results/canny_guided \
  --is_vae_st \  # Enable VAE slicing/tiling for memory efficiency
  --is_cpu_offload  # Optional: for memory-constrained GPUs
```

If your Canny files follow a naming convention like `foo_canny.mp4`, add `--canny_suffix _canny` so that the script matches `foo.mp4` ↔ `foo_canny.mp4`.

## Notable Arguments

### Input/Output
- `--lq_input` / `--canny_input`: Accept a single video file or a directory of videos. When directories are supplied, only flat (non-recursive) matching is performed.
- `--canny_suffix`: Optional suffix appended to LQ stems when resolving Canny filenames (e.g., `_canny`).

### Processing
- `--upscale` and `--upscale_mode`: Control the spatial resize that is applied to the LQ tensor before merging edges.
- `--noise_step`, `--sr_noise_step`: Control the noise injection and denoising steps.

### Edge Processing
- `--no_subtract_edge`: Disable subtraction of Canny edges from output. **By default, edges are automatically subtracted from the output** to remove edge artifacts that may appear in the generated video.
- `--edge_subtraction_scale`: Scale factor for edge subtraction (default: 1.0). Lower values (e.g., 0.5) reduce edge removal strength if edges are over-subtracted.

### Memory Optimization (Important for Large Videos)
- `--is_vae_st`: **Highly recommended** - Enables VAE slicing and tiling to reduce GPU memory usage.
- `--is_cpu_offload`: Enables CPU offloading (slower but uses less GPU memory).
- `--enable_model_cpu_offload`: More aggressive CPU offloading (use with `--is_cpu_offload`).
- `--chunk_len`, `--tile_size_hw`, `--overlap_t`, `--overlap_hw`: Enable temporal chunking and spatial tiling for memory-constrained inference.

### Evaluation
- `--eval_metrics`: Comma-separated list of supported PyIQA metrics (e.g., `psnr,ssim`) if reference HQ videos are available in `--gt_dir`.

### Other
- Other parameters (`--seed`, `--png_save`, `--save_format`, etc.) are identical to the base `inference_script.py`.

Run `python inference_canny_script.py --help` for the full list of options.

## Output

- Enhanced videos are saved to `--output_path` using the original LQ filenames (with `.mp4` by default).
- If metrics are enabled, per-sample and average scores are emitted to `metrics_<metric_list>.json`.
- When `--png_save` is toggled, each video produces a PNG sequence alongside the video export.

## Requirements

- Pretrained CogVideoX weights and optional LoRA adapters (see `pretrained_models/`).
- Matching Canny edge videos, which can be generated with `generate_canny_video.py`.
- GPU capable of running the CogVideoX pipeline or use `--is_cpu_offload` for CPU-assisted inference.

## Memory Optimization Tips

If you encounter CUDA out-of-memory errors:

1. **Always use `--is_vae_st`** - This enables VAE slicing and tiling, which significantly reduces memory usage.
2. **Enable CPU offloading**: Use `--is_cpu_offload` or `--enable_model_cpu_offload` for more aggressive memory management.
3. **Use spatial tiling**: For large resolutions, use `--tile_size_hw 512 512` (or smaller) to process the video in tiles.
4. **Use temporal chunking**: For long videos, use `--chunk_len 16` (or smaller) to process frames in chunks.
5. **Monitor chunk sizes**: The script will warn if chunks exceed 5GB - reduce tile/chunk sizes if you see warnings.

Example with all memory optimizations:
```bash
python inference_canny_script.py \
  --lq_input /path/to/lq_videos \
  --canny_input /path/to/hq_canny_videos \
  --model_path pretrained_models/CogVideoX1.5-5B \
  --output_path ./results \
  --is_vae_st \
  --is_cpu_offload \
  --tile_size_hw 512 512 \
  --chunk_len 16 \
  --overlap_hw 32 32 \
  --overlap_t 8
```

## Edge Subtraction

By default, the script automatically subtracts Canny edges from the output video to remove edge artifacts. This is because the Canny edges are added to the input (via `additive_clip`), and the model may not completely remove them from the output.

- **Default behavior**: Edges are subtracted from output (enabled by default)
- **To disable**: Use `--no_subtract_edge` flag
- **To adjust strength**: Use `--edge_subtraction_scale` (default: 1.0). Lower values (e.g., 0.5-0.8) reduce subtraction strength if the output appears too dark.

Example with adjusted edge subtraction:
```bash
python inference_canny_script.py \
  --lq_input /path/to/lq_videos \
  --canny_input /path/to/hq_canny_videos \
  --model_path pretrained_models/CogVideoX1.5-5B \
  --output_path ./results \
  --edge_subtraction_scale 0.8  # Reduce subtraction strength
```

## Other Tips

- Ensure the Canny videos were derived from the ground-truth HQ clips so that the edge information aligns well with the LQ inputs.
- For bulk inference, verify that frame counts are consistent; the script pads both LQ and Canny sequences to prevent misalignment.
- Adjust `--save_format` (`yuv444p` vs `yuv420p`) depending on downstream playback requirements.
- If the output appears too dark after edge subtraction, try reducing `--edge_subtraction_scale` or disabling edge subtraction with `--no_subtract_edge`.


