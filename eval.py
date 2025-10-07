import tifffile
import math
import torch
import numpy as np

def to_uint8(img, is_mask=False):
    """Convert to uint8, preserving global range for images, binary for masks."""
    img = img.astype(np.float32)
    if is_mask:
        # For binary masks, map 0/1 to 0/255
        return (img > 0.5).astype(np.uint8) * 255
    else:
        # For images, assume already normalized to [0,1] in data_loader
        return (img * 255).astype(np.uint8)

def export_predictions_tiff(model, loader, save_path="predictions.tif", device="cuda", crop_size=1024, patch_size=128):
    model.eval()
    n_h = n_w = crop_size // patch_size  # e.g., 1024 / 128 = 8
    Z = math.ceil(len(loader.dataset) / (n_h * n_w)) # Number of Slices
    out_stack = np.zeros((len(loader.dataset), 128, 128 * 3), dtype=np.uint8)  # Individual patches
    full_images = np.zeros((Z, crop_size, crop_size), dtype=np.uint8)
    full_masks = np.zeros((Z, crop_size, crop_size), dtype=np.uint8)
    full_preds = np.zeros((Z, crop_size, crop_size), dtype=np.uint8)

    with torch.no_grad():
        idx = 0
        for x, y in loader:  # x: (B,1,128,128), y: (B,1,128,128)
            x, y = x.to(device), y.to(device)
            preds = torch.sigmoid(model(x))  # (B,1,128,128)
            preds = (preds > 0.5).float()

            # Move to CPU numpy
            x_np = x[:,0].cpu().numpy()  # (B,128,128)
            y_np = y.squeeze(1).cpu().numpy()  # (B,128,128)
            p_np = preds.squeeze(1).cpu().numpy()  # (B,128,128)

            # Convert to uint8
            x_np = np.stack([to_uint8(img, is_mask=False) for img in x_np])
            y_np = np.stack([to_uint8(img, is_mask=True) for img in y_np])
            p_np = np.stack([to_uint8(img, is_mask=True) for img in p_np])

            # Store individual patches
            combined = np.concatenate([x_np, y_np, p_np], axis=-1)
            batch_size = x.shape[0]
            out_stack[idx:idx+batch_size] = combined

            # Reconstruct full image
            for b in range(batch_size):
                slice_idx = (idx + b) // (n_h * n_w)
                patch_idx = (idx + b) % (n_h * n_w)
                i = patch_idx // n_w
                j = patch_idx % n_w
                h_start = i * patch_size
                w_start = j * patch_size
                full_images[slice_idx, h_start:h_start+patch_size, w_start:w_start+patch_size] = x_np[b]
                full_masks[slice_idx, h_start:h_start+patch_size, w_start:w_start+patch_size] = y_np[b]
                full_preds[slice_idx, h_start:h_start+patch_size, w_start:w_start+patch_size] = p_np[b]
            idx += batch_size

    # Save individual patches and full reconstructed images
    tifffile.imwrite(save_path, out_stack)
    tifffile.imwrite(save_path.replace(".tif", "_full.tif"), 
                     np.concatenate([full_images, full_masks, full_preds], axis=2))

    print(f"✅ Saved patches to {save_path}, shape={out_stack.shape}, dtype={out_stack.dtype}")
    print(f"✅ Saved full images to {save_path.replace('.tif', '_full.tif')}, shape={(Z, crop_size, crop_size*3)}")