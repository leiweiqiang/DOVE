# Modification Plan: Add Canny Edge Map to Training Pipeline

## Overview
Modify the training data pipeline to generate Canny edge maps from HQ images/videos and merge them with LQ inputs before training.

## New Data Flow

```
1. Read file list (existing)
   └─> Load video/image paths from text files
   
2. Load video/image (existing)
   └─> Read video frames using decord
   └─> Apply random cropping
   
3. Generate Canny edge map from HQ (NEW)
   └─> Extract edges from HQ frames using cv2.Canny()
   └─> Parameters: threshold1, threshold2 (configurable)
   └─> Output: Single-channel edge map [F, 1, H, W]
   
4. Apply on-the-fly degradation (existing)
   └─> Two-stage degradation pipeline
   └─> Output: LQ frames [F, C, h, w] where h,w < H,W
   
5. Resize LQ and merge edge map (NEW)
   └─> Resize LQ to match HQ resolution [F, C, H, W]
   └─> Merge canny edge map into LQ
   └─> Merge methods: concatenate, blend, replace_channel
   
6. Create paired HQ/LQ samples (modified)
   └─> HQ: [F, 3, H, W] (unchanged)
   └─> LQ: [F, 3+k, H, W] where k=0 or 1 depending on merge method
```

## Files to Modify

### 1. **finetune/schemas/args.py**
**Purpose:** Add new configuration parameters for Canny edge detection

**Changes:**
```python
# Add to Args class around line 103:

########## Canny Edge Detection ##########
enable_canny: bool = False  # Whether to use Canny edge detection
canny_threshold1: float = 50.0  # Lower threshold for Canny
canny_threshold2: float = 150.0  # Upper threshold for Canny
canny_kernel_size: int = 5  # Kernel size for Gaussian blur (odd number)
canny_sigma: float = 1.4  # Gaussian kernel standard deviation
canny_merge_method: Literal["additive_clip"] = "additive_clip"  # Method to merge edge with LQ
```

**Details:**
- `enable_canny`: Master switch for the feature
- `canny_threshold1`: Lower threshold for Canny (50.0 default)
- `canny_threshold2`: Upper threshold for Canny (150.0 default)
- `canny_kernel_size`: Gaussian blur kernel size before edge detection (5 default, must be odd)
- `canny_sigma`: Gaussian kernel standard deviation (1.4 default)
  - Controls blur strength: smaller = less blur, larger = more blur
  - 1.4 is a good balance for 5×5 kernel
- `canny_merge_method`: "additive_clip" - adds edge map to LQ, then clips to [0, 255]
  - Formula: `merged = clamp(LQ_resized + Edge_map, 0, 255)`
  - Preserves both LQ texture and edge structure
  - Keeps 3 channels (no model modification needed)

---

### 2. **finetune/datasets/utils.py**
**Purpose:** Add utility functions for Canny detection and merging

**Changes:**
Add two new functions:

```python
def generate_canny_edge_map(
    frames: np.ndarray,
    threshold1: float = 50.0,
    threshold2: float = 150.0,
    kernel_size: int = 5,
    sigma: float = 1.4
) -> np.ndarray:
    """
    Generate Canny edge map from video frames.
    
    Args:
        frames: numpy array of shape [F, H, W, C] or list of frames
        threshold1: Lower threshold for Canny edge detection
        threshold2: Upper threshold for Canny edge detection
        kernel_size: Gaussian blur kernel size (must be odd)
        sigma: Gaussian kernel standard deviation
        
    Returns:
        edge_maps: numpy array [F, H, W, 1] with edge maps in range [0, 255]
    """
    import cv2
    import numpy as np
    
    edge_maps = []
    for frame in frames:
        # Convert to numpy if needed
        if isinstance(frame, torch.Tensor):
            frame_np = frame.cpu().numpy()
        else:
            frame_np = frame
            
        # Ensure [H, W, C] format and uint8
        if frame_np.ndim == 3 and frame_np.shape[0] == 3:  # [C, H, W]
            frame_np = frame_np.transpose(1, 2, 0)
        frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
        
        # Convert to grayscale
        if frame_np.shape[-1] == 3:
            gray = cv2.cvtColor(frame_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame_np.squeeze()
        
        # Apply Gaussian blur to grayscale image FIRST (reduce noise before edge detection)
        if kernel_size > 1:
            gray_blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigma)
        else:
            gray_blurred = gray
        
        # Then apply Canny edge detection on the blurred grayscale
        edges = cv2.Canny(gray_blurred, threshold1, threshold2, 
                         apertureSize=3, L2gradient=True)
        
        edge_maps.append(edges[..., np.newaxis])  # Add channel dim
    
    return np.array(edge_maps)  # [F, H, W, 1]

def merge_edge_map_with_lq(
    lq_frames: torch.Tensor,
    edge_maps: torch.Tensor,
    method: str = "additive_clip"
) -> torch.Tensor:
    """
    Merge Canny edge maps with LQ frames using additive clipping.
    
    Args:
        lq_frames: LQ frames [F, C, H, W] in range [0, 255]
        edge_maps: Edge maps [F, 1, H, W] in range [0, 255]
        method: "additive_clip" (only supported method)
        
    Returns:
        merged_frames: [F, C, H, W] in range [0, 255]
    """
    if method != "additive_clip":
        raise ValueError(f"Only 'additive_clip' method is supported, got {method}")
    
    # Expand edge map to 3 channels: [F, 1, H, W] -> [F, 3, H, W]
    edge_maps_rgb = edge_maps.repeat(1, 3, 1, 1)
    
    # Add LQ and edge map, then clip to [0, 255]
    merged = torch.clamp(lq_frames + edge_maps_rgb, 0.0, 255.0)
    
    return merged
```

**Implementation Details:**
- Convert frames to grayscale first
- Apply Gaussian blur (kernel_size × kernel_size) to grayscale BEFORE edge detection (reduces noise)
- Use `cv2.Canny()` for edge detection on blurred grayscale with specified thresholds
- Keep edge maps in [0, 255] range (same as input frames)
- Merge method: element-wise addition with clipping
- Final normalization to [-1, 1] happens in `video_transform()`

---

### 3. **finetune/datasets/real_sr_dataset.py**
**Purpose:** Integrate Canny edge detection into Stage-1 training

**Changes:**

#### 3a. Modify `__init__` method (around line 80):
```python
# Add after line 102:
self.enable_canny = trainer.args.enable_canny
self.canny_threshold1 = trainer.args.canny_threshold1
self.canny_threshold2 = trainer.args.canny_threshold2
self.canny_kernel_size = trainer.args.canny_kernel_size
self.canny_sigma = trainer.args.canny_sigma
self.canny_merge_method = trainer.args.canny_merge_method
```

#### 3b. Modify `preprocess` method (around line 232):
```python
def preprocess(self, video_path: Path) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        hq_video_tensor: [F, C, H, W]
        lq_video_tensor: [F, C', h, w] where C'=3 or 4
        edge_maps: [F, 1, H, W] or None
    """
    if self.crop_mode == 'random_crop':
        # ... existing code for frame loading and degradation ...
        
        # Convert to tensors
        hq_tensor_list = [self.to_tensor(f) for f in hq_frame_list]
        hq_video_tensor = torch.stack(hq_tensor_list, dim=0)  # [F, C, H, W]
        
        # NEW: Generate Canny edge maps from HQ (BEFORE degradation)
        if self.enable_canny:
            from finetune.datasets.utils import generate_canny_edge_map
            edge_maps = generate_canny_edge_map(
                hq_frame_list,  # Use original HQ frames [F, H, W, C]
                threshold1=self.canny_threshold1,
                threshold2=self.canny_threshold2,
                kernel_size=self.canny_kernel_size,
                sigma=self.canny_sigma
            )
            edge_maps = torch.from_numpy(edge_maps).float()  # [F, H, W, 1]
            edge_maps = edge_maps.permute(0, 3, 1, 2).contiguous()  # [F, 1, H, W]
        else:
            edge_maps = None
        
        # Convert LQ to tensor
        lq_tensor_list = [self.to_tensor(f) for f in lq_frame_list]
        lq_video_tensor = torch.stack(lq_tensor_list, dim=0)  # [F, C, h, w]
        
        return hq_video_tensor, lq_video_tensor, edge_maps
```

#### 3c. Modify `__getitem__` method (around line 132):
```python
# Replace line 132:
hq_frames, lq_frames, edge_maps = self.preprocess(video)

# After line 134 (resize LQ to HQ resolution):
lq_frames_resize = F.interpolate(lq_frames, size=(H_, W_), mode="bilinear", align_corners=False)

# NEW: Merge edge maps with LQ if enabled (BEFORE normalization)
# Note: Both lq_frames_resize and edge_maps are in [0, 255] range at this point
if self.enable_canny and edge_maps is not None:
    from finetune.datasets.utils import merge_edge_map_with_lq
    lq_frames_resize = merge_edge_map_with_lq(
        lq_frames_resize,  # [F, 3, H, W] in [0, 255]
        edge_maps,         # [F, 1, H, W] in [0, 255]
        method=self.canny_merge_method
    )
    # Result: [F, 3, H, W] in [0, 255], ready for video_transform
```

---

### 4. **finetune/datasets/real_sr_image_video_dataset.py**
**Purpose:** Integrate Canny edge detection into Stage-2 training (handles both video and images)

**Changes:** Similar to `real_sr_dataset.py` but with additional logic for handling images

#### 4a. Modify `__init__` (around line 108):
Same as 3a above

#### 4b. Modify `preprocess` and `preprocess_image` methods:
```python
def preprocess(self, video_path: Path) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Similar to RealSRDataset.preprocess
    # Add edge map generation logic
    pass

def preprocess_image(self, image_path: Path) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Preprocess a single image (for DIV2K dataset in Stage 2)
    
    Returns:
        hq_image: [1, C, H, W] (treat as 1-frame video)
        lq_image: [1, C', h, w]
        edge_map: [1, 1, H, W] or None
    """
    # Load image, apply degradation, generate edge map
    pass
```

#### 4c. Modify `__getitem__` method:
Handle both video and image cases with edge map merging

---

### 5. **finetune/train_ddp_one_s1.sh**
**Purpose:** Add Canny parameters to Stage-1 training script

**Changes:**
```bash
# Add new section around line 85:

# Canny Edge Detection parameters
CANNY_ARGS=(
    --enable_canny true
    --canny_threshold1 50.0
    --canny_threshold2 150.0
    --canny_kernel_size 5
    --canny_sigma 1.4
    --canny_merge_method "additive_clip"
)

# Update launch command (line 87):
accelerate launch --config_file accelerate_config.yaml train.py \
    "${MODEL_ARGS[@]}" \
    "${LORA_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${SYSTEM_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${VALIDATION_ARGS[@]}" \
    "${SR_ARGS[@]}" \
    "${CANNY_ARGS[@]}"  # NEW
```

---

### 6. **finetune/train_ddp_one_s2.sh**
**Purpose:** Add Canny parameters to Stage-2 training script

**Changes:** Same as file 5 above

---

## Model Input Changes

### Current Model Input
```python
# LQ input to model: [B, C, F, H, W] where C=3 (RGB)
lq_frames_resize: torch.Tensor  # Shape: [B, 3, F, H, W] in [-1, 1]
```

### New Model Input (with Canny - Additive Clip Method)
```python
# LQ input with edge map merged via additive clipping
lq_frames_resize: torch.Tensor  # Shape: [B, 3, F, H, W] in [-1, 1]

# Process:
# 1. LQ_resized [F, 3, H, W] in [0, 255]
# 2. Edge_map [F, 1, H, W] in [0, 255] -> expanded to [F, 3, H, W]
# 3. Merged = clamp(LQ_resized + Edge_map, 0, 255)
# 4. Normalized to [-1, 1] via video_transform
# 5. Result: Brighter pixels where edges exist
```

**Advantage of Additive Clip Method:**
- ✅ No model architecture changes needed (still 3 channels)
- ✅ Edge information naturally enhances textures
- ✅ Simple, intuitive operation
- ✅ Backward compatible when enable_canny=False

---

## Testing Considerations

1. **Backward Compatibility:** When `enable_canny=false`, the code should work exactly as before
2. **Memory Usage:** Storing edge maps will slightly increase memory usage
3. **Speed:** Canny edge detection adds ~5-10ms per frame
4. **Validation:** Edge maps should also be generated for validation videos

---

## Implementation Steps

1. ✅ Create this plan document
2. Add parameters to Args schema
3. Implement utility functions (Canny detection + merging)
4. Modify RealSRDataset
5. Modify RealSRImageVideoDataset  
6. Update training scripts
7. Test with small dataset
8. Verify backward compatibility (enable_canny=false)
9. Run full training

---

## Parameters to Tune

| Parameter | Current Value | Range | Description |
|-----------|---------------|-------|-------------|
| `canny_threshold1` | 50.0 | 10-100 | Lower threshold (more sensitive) |
| `canny_threshold2` | 150.0 | 100-300 | Upper threshold (less noise) |
| `canny_kernel_size` | 5 | 3, 5, 7 | Gaussian blur kernel (odd) |
| `canny_sigma` | 1.4 | 0.5-3.0 | Gaussian standard deviation |
| `canny_merge_method` | "additive_clip" | - | Fixed method |

**Tuning Tips:**
- For stronger edges: Increase threshold1 → 70-100
- For smoother/cleaner edges: Increase sigma → 2.0-3.0
- For sharper edges: Decrease sigma → 0.8-1.0
- For subtle edges: Decrease threshold1 → 30-40
- The additive method naturally weights edges (bright edges = more influence)

---

## Example Usage

```bash
# Stage 1 with Canny edges (concatenate method)
bash train_ddp_one_s1.sh

# Stage 2 with Canny edges (blend method)
# Edit train_ddp_one_s2.sh to set:
# --canny_merge_method "blend"
# --canny_blend_alpha 0.3
bash train_ddp_one_s2.sh
```

---

## Design Decisions (FINALIZED)

1. ✅ **Merge method:** Additive Clip
   - Formula: `merged = clamp(LQ + Edge, 0, 255)`
   - Keeps 3 channels (no model changes)
   - Edge brightens LQ pixels naturally
   
2. ✅ **Canny parameters:**
   - Kernel size: 5×5 Gaussian blur
   - Lower threshold: 50
   - Upper threshold: 150
   
3. ✅ **Edge map normalization:**
   - Keep in [0, 255] during processing
   - Normalize to [-1, 1] after merging (via video_transform)

4. ✅ **Validation edges:** Yes, apply same edge generation/merging

5. ⚠️ **Caching:** Optional - implement if training is slow

---

## Next Steps

Please review this plan and let me know:
1. Your preferred merge method
2. Any parameter adjustments
3. Whether to proceed with implementation

I'm ready to implement these modifications once you approve the plan!

