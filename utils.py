import tifffile
import numpy as np
import torch

def to_uint8(img):
    """Normalize per image to [0,255] and convert to uint8."""
    img = img.astype(np.float32)
    if img.max() > img.min():  # avoid division by zero
        img = (img - img.min()) / (img.max() - img.min())
    return (img * 255).astype(np.uint8)

def export_predictions_tiff(model, loader, save_path="predictions.tif", device="cuda"):
    model.eval()
    out_list = []

    with torch.no_grad():
        for x, y in loader:  # iterate batches
            x, y = x.to(device), y.to(device)  # x: (B,3,128,128), y: (B,1,128,128)
            preds = torch.sigmoid(model(x))    # (B,1,128,128)
            preds = (preds > 0.5).float()

            # Move to CPU numpy
            x_np = x[:,0].cpu().numpy()        # OG image channel (B,128,128)
            y_np = y.squeeze(1).cpu().numpy()  # Ground truth mask (B,128,128)
            p_np = preds.squeeze(1).cpu().numpy()  # Predicted mask (B,128,128)

            # Normalize to uint8
            x_np = np.stack([to_uint8(img) for img in x_np])
            y_np = np.stack([to_uint8(img) for img in y_np])
            p_np = np.stack([to_uint8(img) for img in p_np])

            # Concatenate horizontally: (B, 128, 128*3)
            combined = np.concatenate([x_np, y_np, p_np], axis=-1)
            out_list.append(combined)

    # Stack into (N, 128, 128*3)
    out_stack = np.concatenate(out_list, axis=0)

    # Save as multipage TIFF
    tifffile.imwrite(save_path, out_stack, photometric="minisblack")

    print(f"✅ Saved predictions to {save_path}, shape={out_stack.shape}, dtype={out_stack.dtype}")