# Data Flow Diagram: Before vs After Modification

## Current Data Flow (Without Canny Edge)

```
┌─────────────────────────────────────────────────────────────┐
│ 1. READ FILE LIST                                           │
│    - Load video paths from HQ-VSR.txt                       │
│    - Load image paths from DIV2K_train_HR.txt               │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. LOAD VIDEO/IMAGE                                         │
│    - Read frames using decord                               │
│    - Shape: [F, H, W, C] in numpy                          │
│    - Random crop to inter_height × inter_width              │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. APPLY DEGRADATION (Two-stage pipeline)                  │
│                                                             │
│    HQ Frames [F, H, W, C]                                  │
│         │                                                    │
│         ├─► Stage 1: Blur → Resize → Noise → JPEG → MPEG  │
│         │                                                    │
│         └─► Stage 2: Blur → Resize → Noise → JPEG → Video │
│                                                             │
│    LQ Frames [F, h, w, C]  (h,w = H/4, W/4)               │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. CROP & CONVERT                                           │
│    - Paired random crop: HQ and LQ                         │
│    - HQ: [F, C, H, W]  (target_h, target_w)               │
│    - LQ: [F, C, h, w]  (h/4, w/4)                         │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. RESIZE LQ TO HQ RESOLUTION                               │
│    - Bilinear interpolation                                 │
│    - LQ_resized: [F, 3, H, W]                              │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. NORMALIZE & PREPARE FOR TRAINING                        │
│    - Transform: [0, 255] → [-1, 1]                         │
│    - Reshape: [F, C, H, W] → [B, C, F, H, W]              │
│                                                             │
│    Final Output:                                            │
│    ├─ HQ: [B, 3, F, H, W]  (Ground Truth)                 │
│    └─ LQ: [B, 3, F, H, W]  (Model Input)                  │
└─────────────────────────────────────────────────────────────┘
```

---

## New Data Flow (With Canny Edge Map)

```
┌─────────────────────────────────────────────────────────────┐
│ 1. READ FILE LIST                                           │
│    - Load video paths from HQ-VSR.txt                       │
│    - Load image paths from DIV2K_train_HR.txt               │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. LOAD VIDEO/IMAGE                                         │
│    - Read frames using decord                               │
│    - Shape: [F, H, W, C] in numpy                          │
│    - Random crop to inter_height × inter_width              │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. ★NEW★ GENERATE CANNY EDGE MAP FROM HQ                   │
│                                                             │
│    HQ Frames [F, H, W, C]                                  │
│         │                                                    │
│         ├─► Convert to grayscale                            │
│         ├─► Apply Gaussian blur (5×5, sigma=1.4)            │
│         ├─► Apply cv2.Canny(threshold1=50, threshold2=150)  │
│         └─► Output in [0, 255] range                        │
│                                                             │
│    Edge Maps [F, H, W, 1]                                  │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. APPLY DEGRADATION (Two-stage pipeline)                  │
│                                                             │
│    HQ Frames [F, H, W, C]                                  │
│         │                                                    │
│         ├─► Stage 1: Blur → Resize → Noise → JPEG → MPEG  │
│         │                                                    │
│         └─► Stage 2: Blur → Resize → Noise → JPEG → Video │
│                                                             │
│    LQ Frames [F, h, w, C]  (h,w = H/4, W/4)               │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. CROP & CONVERT                                           │
│    - Paired random crop: HQ, LQ, and Edge Maps            │
│    - HQ: [F, C, H, W]  (target_h, target_w)               │
│    - LQ: [F, C, h, w]  (h/4, w/4)                         │
│    - Edge: [F, 1, H, W] (same as HQ)                      │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. RESIZE LQ TO HQ RESOLUTION                               │
│    - Bilinear interpolation                                 │
│    - LQ_resized: [F, 3, H, W]                              │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. ★NEW★ MERGE EDGE MAP WITH LQ (ADDITIVE CLIP)            │
│                                                             │
│    Selected Method: ADDITIVE CLIP                           │
│    ┌──────────────────────────────────────┐               │
│    │ LQ_resized [F, 3, H, W] (0-255)      │               │
│    │      +                                │               │
│    │ Edge_maps [F, 1, H, W] (0-255)       │               │
│    │      │                                │               │
│    │      ├─► Expand: [F, 1, H, W]        │               │
│    │      │          → [F, 3, H, W]       │               │
│    │      │                                │               │
│    │      └─► Add & Clip:                 │               │
│    │          clamp(LQ + Edge, 0, 255)    │               │
│    │      ↓                                │               │
│    │ LQ_merged [F, 3, H, W] (0-255)       │               │
│    │ Channels: [R+E, G+E, B+E]            │               │
│    │ (Edges brighten all RGB equally)     │               │
│    └──────────────────────────────────────┘               │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 8. NORMALIZE & PREPARE FOR TRAINING                        │
│    - Transform: [0, 255] → [-1, 1]                         │
│    - Reshape: [F, C, H, W] → [B, C, F, H, W]              │
│                                                             │
│    Final Output:                                            │
│    ├─ HQ: [B, 3, F, H, W]  (Ground Truth)                 │
│    └─ LQ: [B, C, F, H, W]  (Model Input)                  │
│         where C = 4 (concatenate)                           │
│            or C = 3 (blend/replace)                         │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Differences

| Aspect | Before | After |
|--------|--------|-------|
| **HQ Processing** | Load → Degrade → Crop | Load → **Generate Edge Map** → Degrade → Crop |
| **Edge Map** | Not used | **Generated from HQ before degradation** |
| **LQ Channels** | 3 (RGB only) | **3 or 4** (depending on merge method) |
| **Merge Step** | N/A | **Edge map merged with upsampled LQ** |
| **Model Input** | [B, 3, F, H, W] | **[B, 3/4, F, H, W]** |

---

## Selected Merge Method: Additive Clip

### Visual Explanation
```
┌─────────┐     ┌─────────┐
│   LQ    │     │  Edge   │
│ [F,3,H,W]│     │ [F,1,H,W]│
│ [0-255] │     │ [0-255] │
└────┬────┘     └────┬────┘
     │                │
     │                ├─► Repeat to 3 channels
     │                │   [F,1,H,W] → [F,3,H,W]
     │                │
     └────────┬───────┘
              │ Element-wise add
              ▼
        ┌─────────┐
        │  Added  │
        │[F,3,H,W]│
        │ [0-510] │ ← May exceed 255!
        └────┬────┘
             │ Clip to valid range
             ▼
        ┌─────────┐
        │  Final  │
        │[F,3,H,W]│
        │ [0-255] │
        └─────────┘
```

**Formula:** `merged = torch.clamp(LQ + Edge.repeat(1,3,1,1), 0, 255)`

### Example with Pixel Values

```
Pixel Before Merge:
  LQ_R = 120,  LQ_G = 100,  LQ_B = 80
  Edge = 50 (edge detected)

After Merge:
  Merged_R = clamp(120 + 50, 0, 255) = 170
  Merged_G = clamp(100 + 50, 0, 255) = 150
  Merged_B = clamp( 80 + 50, 0, 255) = 130
  
  → All RGB channels brightened by 50
  → Edge enhances texture uniformly


Pixel Before Merge (strong edge):
  LQ_R = 220,  LQ_G = 200,  LQ_B = 180
  Edge = 100 (strong edge)

After Merge:
  Merged_R = clamp(220 + 100, 0, 255) = 255 ← clipped!
  Merged_G = clamp(200 + 100, 0, 255) = 255 ← clipped!
  Merged_B = clamp(180 + 100, 0, 255) = 255 ← clipped!
  
  → Strong edges create bright highlights
```

### Advantages
- ✅ **No model changes:** Still 3 channels (RGB)
- ✅ **Simple operation:** Addition + clipping
- ✅ **Natural highlighting:** Edges brighten existing textures
- ✅ **Preserves color ratios:** All RGB enhanced equally
- ✅ **Backward compatible:** Works when enable_canny=False

---

## Validation Data Flow

For validation, the same edge generation and merging should be applied:

```
Validation LQ Video
      │
      ├─► Resize to target resolution
      │
      ├─► Generate edge map from GT (if available)
      │   or from upsampled LQ (if GT not available)
      │
      └─► Merge edge map
          │
          ▼
    Model Input [B, C, F, H, W]
```

---

## Performance Considerations

| Operation | Time per Frame | Memory Overhead |
|-----------|----------------|-----------------|
| Canny Edge Detection | ~5-10 ms | +H×W×1 bytes |
| Edge Map Storage | 0 ms | +F×H×W×1 bytes |
| Merge Operation | ~1-2 ms | 0 bytes |
| **Total Added** | **~6-12 ms** | **~F×H×W bytes** |

For 25 frames at 320×640:
- Memory: 25 × 320 × 640 × 1 byte = **~5 MB per video**
- Time: 25 × 10 ms = **~250 ms per video**

**Impact:** Minimal (~5-10% increase in data loading time)

---

## Why Additive Clip Method?

### Comparison with Alternatives

| Aspect | Additive Clip ✅ | Concatenate | Alpha Blend |
|--------|------------------|-------------|-------------|
| **Channels** | 3 (no change) | 4 (requires model mod) | 3 (no change) |
| **Edge Strength** | Full (0-255) | Full | Reduced by α |
| **Color Preserved** | Yes, enhanced | Yes, separate | Mixed |
| **Implementation** | Simple | Complex | Medium |
| **Clipping Behavior** | Natural highlights | N/A | No clipping |

### Visual Effect

```
Original LQ Frame:     Edge Map:          Merged Result:
[Dark texture]         [White edges]      [Brightened edges]

  ░░░░░░░░              ┌─────┐              ░░█████░
  ░░░░░░░░              │     │              ░░█░░░█░
  ░░░░░░░░       +      │     │       =      ░░█░░░█░
  ░░░░░░░░              │     │              ░░█░░░█░
  ░░░░░░░░              └─────┘              ░░█████░

(Darker pixels)      (Edge pixels)      (Edges pop out!)
```

**Result:** Edges become bright highlights that guide the model's attention during super-resolution.

**Selected:** Additive Clip is the optimal method for this use case! 🎯

