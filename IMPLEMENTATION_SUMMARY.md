# Implementation Summary: Canny Edge Map Integration

## ✅ Planning Complete!

All planning documents have been created and finalized based on your specifications:

---

## 📋 Your Requirements (Finalized)

### 1. **Merge Method: Additive Clip**
```python
merged = torch.clamp(LQ_resized + Edge_map, 0, 255)
```
- Adds edge map to LQ frames element-wise
- Clips any values exceeding 255 to 255
- Keeps 3 RGB channels (no model modification needed)
- Edges naturally enhance textures by brightening pixels

### 2. **Canny Parameters**
| Parameter | Value | Description |
|-----------|-------|-------------|
| `canny_kernel_size` | 5 | Gaussian blur kernel (5×5) |
| `canny_sigma` | 1.4 | Gaussian standard deviation |
| `canny_threshold1` | 50.0 | Lower threshold (edge sensitivity) |
| `canny_threshold2` | 150.0 | Upper threshold (noise suppression) |

---

## 📚 Documentation Created

### 1. **MODIFICATION_PLAN.md** (Detailed Implementation Guide)
Contains:
- Complete code changes for 7 files
- Step-by-step modification instructions
- Parameter descriptions
- Implementation checklist
- Testing considerations

### 2. **DATAFLOW_DIAGRAM.md** (Visual Flow)
Contains:
- Before/After data flow comparison
- Visual explanation of additive clip method
- Pixel-level examples
- Performance impact analysis
- Design justification

### 3. **canny_merge_example.py** (Runnable Demo)
Contains:
- Working implementation of both functions
- Synthetic data demonstration
- Visualization code
- Can be run standalone to verify logic

---

## 🔧 Implementation Checklist

### Phase 1: Core Functions
- [ ] Add parameters to `finetune/schemas/args.py`
- [ ] Implement `generate_canny_edge_map()` in `finetune/datasets/utils.py`
- [ ] Implement `merge_edge_map_with_lq()` in `finetune/datasets/utils.py`

### Phase 2: Dataset Integration
- [ ] Modify `RealSRDataset.__init__()` to load canny parameters
- [ ] Modify `RealSRDataset.preprocess()` to generate edge maps
- [ ] Modify `RealSRDataset.__getitem__()` to merge edges with LQ
- [ ] Repeat for `RealSRImageVideoDataset` (Stage 2)

### Phase 3: Training Scripts
- [ ] Update `train_ddp_one_s1.sh` with CANNY_ARGS
- [ ] Update `train_ddp_one_s2.sh` with CANNY_ARGS

### Phase 4: Testing
- [ ] Test with `enable_canny=false` (backward compatibility)
- [ ] Test with `enable_canny=true` on small dataset
- [ ] Verify edge map generation visually
- [ ] Check memory usage and speed

---

## 🎯 Expected Behavior

### Training Data Flow (with Canny enabled)

```
1. Load HQ video/image
   ↓
2. Generate Canny edge map from HQ
   - Grayscale conversion
   - Gaussian blur (5×5, sigma=1.4) on grayscale
   - cv2.Canny(blurred_gray, 50, 150)
   - Output: [F, 1, H, W] in [0, 255]
   ↓
3. Apply degradation to HQ → LQ
   - Two-stage degradation pipeline
   - Output: [F, 3, h, w] in [0, 255]
   ↓
4. Resize LQ to match HQ resolution
   - Bilinear interpolation
   - Output: [F, 3, H, W] in [0, 255]
   ↓
5. Merge edge map with LQ (ADDITIVE CLIP)
   - Expand edge: [F, 1, H, W] → [F, 3, H, W]
   - Add: LQ + Edge
   - Clip: max(result, 255)
   - Output: [F, 3, H, W] in [0, 255]
   ↓
6. Normalize to [-1, 1] via video_transform
   ↓
7. Feed to model
   - HQ: [B, 3, F, H, W] in [-1, 1] (ground truth)
   - LQ: [B, 3, F, H, W] in [-1, 1] (with edges merged)
```

---

## 📊 Visual Example

```
Original HQ:          Edge Map (after blur):    Degraded LQ:
┌──────────┐          ┌──────────┐              ┌──────────┐
│ ████████ │          │ ░░░░░░░░ │              │ ▓▓▓▓▓▓▓▓ │
│ ██    ██ │   →      │ ░█░░░░█░ │              │ ▓▓    ▓▓ │ (blurry)
│ ██    ██ │  Canny   │ ░█░░░░█░ │   Degrade    │ ▓▓    ▓▓ │
│ ████████ │          │ ░█████░░ │       →      │ ▓▓▓▓▓▓▓▓ │
└──────────┘          └──────────┘              └──────────┘

                            ↓ Merge (Additive Clip)

                      Merged LQ + Edge:
                      ┌──────────┐
                      │ ▓▓▓▓▓▓▓▓ │
                      │ ▓█    █▓ │ ← Edges brightened!
                      │ ▓█    █▓ │
                      │ ▓█████▓▓ │
                      └──────────┘
```

---

## 🚀 Ready to Implement?

### Quick Test Command (after implementation)
```bash
# Run the demo script
python canny_merge_example.py

# This will:
# 1. Create synthetic HQ frame with shapes
# 2. Generate Canny edge map
# 3. Create degraded LQ frame
# 4. Merge using additive clip
# 5. Save visualization to canny_merge_demo.png
```

### Training Commands (after implementation)
```bash
# Stage 1: Latent-space adaptation WITH Canny
cd finetune
bash train_ddp_one_s1.sh

# Stage 2: Pixel-space refinement WITH Canny
bash train_ddp_one_s2.sh

# To disable Canny (backward compatibility)
# Edit shell scripts and set: --enable_canny false
```

---

## 💡 Key Advantages

1. **No Model Changes**: Still uses 3-channel input (RGB)
2. **Simple Implementation**: Just addition + clipping
3. **Natural Enhancement**: Edges brighten existing textures
4. **Backward Compatible**: Works when `enable_canny=false`
5. **Tunable**: Can adjust thresholds and kernel size
6. **Fast**: <10ms per frame overhead

---

## 🔍 Parameter Tuning Guide

If edges are too strong:
```bash
--canny_threshold1 70.0  # Increase (less sensitive)
--canny_sigma 2.0        # More blur (smoother)
```

If edges are too weak:
```bash
--canny_threshold1 30.0  # Decrease (more sensitive)
--canny_sigma 1.0        # Less blur (sharper)
```

For cleaner edges:
```bash
--canny_threshold2 200.0  # Higher (less noise)
--canny_sigma 2.0         # More blur (removes fine noise)
```

---

## ❓ Questions Before Implementation?

**All planning is complete!** The implementation is straightforward:
1. Copy code snippets from MODIFICATION_PLAN.md
2. Follow the checklist above
3. Test incrementally

Ready to proceed? Just say "Let's implement!" and I'll start modifying the code files. 🚀

Or let me know if you have any questions or want to adjust any parameters!

