"""
Example: Canny Edge Map Generation and Additive Clip Merging
This demonstrates the exact implementation that will be added to the training pipeline.
"""

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt


def generate_canny_edge_map(
    frames,
    threshold1: float = 50.0,
    threshold2: float = 150.0,
    kernel_size: int = 5,
    sigma: float = 1.4
) -> np.ndarray:
    """
    Generate Canny edge map from video frames.
    
    Args:
        frames: list of numpy arrays [H, W, C] or [H, W]
        threshold1: Lower threshold for Canny edge detection
        threshold2: Upper threshold for Canny edge detection
        kernel_size: Gaussian blur kernel size (must be odd)
        sigma: Gaussian kernel standard deviation
        
    Returns:
        edge_maps: numpy array [F, H, W, 1] with edge maps in range [0, 255]
    """
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
        
        # Apply Gaussian blur to grayscale image FIRST (to reduce noise)
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


def demo_with_synthetic_data():
    """
    Demonstrate the edge detection and merging with synthetic data.
    """
    print("=" * 70)
    print("Canny Edge Map + Additive Clip Demo")
    print("=" * 70)
    
    # Create synthetic HQ frame (320x320)
    print("\n1. Creating synthetic HQ frame (320x320)...")
    hq_frame = np.ones((320, 320, 3), dtype=np.uint8) * 128  # Gray background
    
    # Add some shapes to create edges
    cv2.rectangle(hq_frame, (50, 50), (150, 150), (200, 100, 50), -1)  # Rectangle
    cv2.circle(hq_frame, (220, 220), 50, (50, 150, 200), -1)  # Circle
    cv2.line(hq_frame, (100, 200), (250, 280), (255, 200, 100), 5)  # Line
    
    print(f"   HQ frame shape: {hq_frame.shape}")
    print(f"   HQ frame range: [{hq_frame.min()}, {hq_frame.max()}]")
    
    # Generate Canny edge map
    print("\n2. Generating Canny edge map...")
    print(f"   Parameters: threshold1=50, threshold2=150, kernel_size=5, sigma=1.4")
    edge_maps = generate_canny_edge_map(
        [hq_frame],
        threshold1=50.0,
        threshold2=150.0,
        kernel_size=5,
        sigma=1.4
    )
    print(f"   Edge map shape: {edge_maps.shape}")
    print(f"   Edge map range: [{edge_maps.min()}, {edge_maps.max()}]")
    print(f"   Non-zero edge pixels: {np.count_nonzero(edge_maps)} / {edge_maps.size}")
    
    # Create degraded LQ frame (simulate blur and noise)
    print("\n3. Creating degraded LQ frame (simulating SR input)...")
    lq_frame = cv2.GaussianBlur(hq_frame, (7, 7), 0)
    noise = np.random.normal(0, 10, lq_frame.shape).astype(np.int16)
    lq_frame = np.clip(lq_frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    print(f"   LQ frame shape: {lq_frame.shape}")
    print(f"   LQ frame range: [{lq_frame.min()}, {lq_frame.max()}]")
    
    # Convert to torch tensors [F, C, H, W]
    print("\n4. Converting to torch tensors...")
    lq_tensor = torch.from_numpy(lq_frame).permute(2, 0, 1).unsqueeze(0).float()  # [1, 3, H, W]
    edge_tensor = torch.from_numpy(edge_maps).permute(0, 3, 1, 2).float()  # [1, 1, H, W]
    print(f"   LQ tensor shape: {lq_tensor.shape}")
    print(f"   Edge tensor shape: {edge_tensor.shape}")
    
    # Merge using additive clip
    print("\n5. Merging edge map with LQ (additive clip)...")
    merged_tensor = merge_edge_map_with_lq(lq_tensor, edge_tensor, method="additive_clip")
    print(f"   Merged tensor shape: {merged_tensor.shape}")
    print(f"   Merged tensor range: [{merged_tensor.min():.1f}, {merged_tensor.max():.1f}]")
    
    # Statistics
    print("\n6. Analyzing differences...")
    lq_np = lq_tensor[0].permute(1, 2, 0).numpy()
    merged_np = merged_tensor[0].permute(1, 2, 0).numpy()
    diff = merged_np - lq_np
    print(f"   Mean difference: {diff.mean():.2f}")
    print(f"   Max difference: {diff.max():.2f}")
    print(f"   Pixels enhanced: {(diff > 0).sum()} / {diff.size}")
    
    # Show pixel-level example
    print("\n7. Pixel-level example:")
    y, x = 100, 100  # Sample pixel near an edge
    print(f"   Position: ({y}, {x})")
    print(f"   LQ RGB:     [{lq_np[y, x, 0]:.0f}, {lq_np[y, x, 1]:.0f}, {lq_np[y, x, 2]:.0f}]")
    print(f"   Edge value: {edge_maps[0, y, x, 0]:.0f}")
    print(f"   Merged RGB: [{merged_np[y, x, 0]:.0f}, {merged_np[y, x, 1]:.0f}, {merged_np[y, x, 2]:.0f}]")
    print(f"   Difference: [{diff[y, x, 0]:.0f}, {diff[y, x, 1]:.0f}, {diff[y, x, 2]:.0f}]")
    
    # Visualize (optional, requires matplotlib display)
    print("\n8. Visualization saved to 'canny_merge_demo.png'")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(hq_frame)
    axes[0, 0].set_title("Original HQ Frame")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(edge_maps[0, :, :, 0], cmap='gray')
    axes[0, 1].set_title("Canny Edge Map (Blurred)")
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(lq_np.astype(np.uint8))
    axes[0, 2].set_title("Degraded LQ Frame")
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(merged_np.astype(np.uint8))
    axes[1, 0].set_title("Merged (LQ + Edge)")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(diff.mean(axis=2), cmap='hot', vmin=0, vmax=100)
    axes[1, 1].set_title("Difference Map (Enhancement)")
    axes[1, 1].axis('off')
    
    # Side-by-side comparison
    axes[1, 2].imshow(np.concatenate([lq_np.astype(np.uint8), merged_np.astype(np.uint8)], axis=1))
    axes[1, 2].set_title("LQ | Merged (Side-by-side)")
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('canny_merge_demo.png', dpi=150, bbox_inches='tight')
    print("   Saved!")
    
    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print("\nKey Observations:")
    print("  • Edge pixels are brightened in all RGB channels equally")
    print("  • Strong edges may clip to 255 (white highlights)")
    print("  • Smooth Gaussian blur reduces noise in edge map")
    print("  • Original LQ texture is preserved, just enhanced")
    print("  • No model architecture changes needed (still 3 channels)")
    

if __name__ == "__main__":
    demo_with_synthetic_data()

