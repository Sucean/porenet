import tifffile
import math
import torch
import numpy as np

def compute_segmentation_metrics(preds, targets, apply_sigmoid=False, threshold=0.5, eps=1e-7):
    """
    Compute segmentation metrics (Precision, Recall, Dice, IoU) for binary masks.

    Args:
        preds (torch.Tensor): Model outputs, shape (B, 1, H, W) or (B, H, W).
        targets (torch.Tensor): Ground-truth masks, same shape as preds.
        apply_sigmoid (bool): Whether to apply sigmoid to preds (set True if passing raw logits).
        threshold (float): Threshold for binarizing predictions if apply_sigmoid=True.
        eps (float): Small epsilon to avoid division by zero.

    Returns:
        Tuple of (precision, recall, dice, iou) as Python floats (averaged over batch).
    """
    if apply_sigmoid:
        preds = torch.sigmoid(preds)
        preds = (preds > threshold).float()
    else:
        preds = (preds > 0.5).float()  # assume already thresholded

    # Flatten all dimensions except batch
    preds = preds.view(preds.shape[0], -1)
    targets = targets.view(targets.shape[0], -1)

    # True positives, false positives, false negatives
    TP = (preds * targets).sum(dim=1)
    FP = (preds * (1 - targets)).sum(dim=1)
    FN = ((1 - preds) * targets).sum(dim=1)

    precision = TP / (TP + FP + eps)
    recall    = TP / (TP + FN + eps)
    dice      = 2 * TP / (2 * TP + FP + FN + eps)
    iou       = TP / (TP + FP + FN + eps)

    # Average across batch
    precision_mean = precision.mean().item()
    recall_mean    = recall.mean().item()
    dice_mean      = dice.mean().item()
    iou_mean       = iou.mean().item()

    return precision_mean, recall_mean, dice_mean, iou_mean

def to_uint8(img, is_mask=False):
    """Convert to uint8, preserving global range for images, binary for masks."""
    img = img.astype(np.float32)
    if is_mask:
        return (img > 0.5).astype(np.uint8) * 255
    else:
        return img

def export_predictions_tiff(model, loader, save_path="predictions.tif", device="cuda", crop_size=1024, patch_size=128):
    model.eval()
    n_h = n_w = crop_size // patch_size  # e.g., 1024 / 128 = 8
    Z = math.ceil(len(loader.dataset) / (n_h * n_w))  # Number of Slices
    out_stack = np.zeros((len(loader.dataset), 128, 128 * 3), dtype=np.uint8)  # Individual patches
    full_images = np.zeros((Z, crop_size, crop_size), dtype=np.uint8)
    full_masks = np.zeros((Z, crop_size, crop_size), dtype=np.uint8)
    full_preds = np.zeros((Z, crop_size, crop_size), dtype=np.uint8)
    
    history = {
        "eval_precision_epoch": [],
        "eval_recall_epoch": [],
        "eval_dice_epoch": [],
        "eval_iou_epoch": []
    }

    # 🔧 Initialize metric accumulators
    precision_sum = 0.0
    recall_sum = 0.0
    dice_sum = 0.0
    iou_sum = 0.0
    num_batches = 0

    with torch.no_grad():
        idx = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = torch.sigmoid(model(x))
            
            # Compute metrics
            precision, recall, dice, iou = compute_segmentation_metrics(preds, y, apply_sigmoid=False)


            preds = (preds > 0.5).float()


            # Convert tensors to scalars if needed
            if torch.is_tensor(precision): precision = precision.item()
            if torch.is_tensor(recall):    recall = recall.item()
            if torch.is_tensor(dice):      dice = dice.item()
            if torch.is_tensor(iou):       iou = iou.item()

            precision_sum += precision
            recall_sum += recall
            dice_sum += dice
            iou_sum += iou
            num_batches += 1

            # Rest of your code remains unchanged
            x_np = x[:,0].cpu().numpy()
            y_np = y.squeeze(1).cpu().numpy()
            p_np = preds.squeeze(1).cpu().numpy()

            x_np = np.stack([to_uint8(img, is_mask=False) for img in x_np])
            y_np = np.stack([to_uint8(img, is_mask=True) for img in y_np])
            p_np = np.stack([to_uint8(img, is_mask=True) for img in p_np])

            combined = np.concatenate([x_np, y_np, p_np], axis=-1)
            batch_size = x.shape[0]
            out_stack[idx:idx+batch_size] = combined

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

        # 🔧 Average metrics after loop
        if num_batches > 0:
            eval_precision_epoch = precision_sum / num_batches
            eval_recall_epoch = recall_sum / num_batches
            eval_dice_epoch = dice_sum / num_batches
            eval_iou_epoch = iou_sum / num_batches
        else:
            eval_precision_epoch = eval_recall_epoch = eval_dice_epoch = eval_iou_epoch = 0.0

        history["eval_precision_epoch"].append(eval_precision_epoch)
        history["eval_recall_epoch"].append(eval_recall_epoch)
        history["eval_dice_epoch"].append(eval_dice_epoch)
        history["eval_iou_epoch"].append(eval_iou_epoch)
        
    print(f"\nFor Evaluation: "
          f"Precision: {eval_precision_epoch:.4f}, Recall: {eval_recall_epoch:.4f}, "
          f"Dice: {eval_dice_epoch:.4f}, IoU: {eval_iou_epoch:.4f}")

    # Save TIFFs (unchanged)
    tifffile.imwrite(save_path, out_stack)
    
    combined_stack = np.concatenate([full_images, full_masks, full_preds], axis=2)
    tifffile.imwrite(save_path.replace(".tif", "_full.tif"), 
                    combined_stack)

    print(f"✅ Saved patches to {save_path}, shape={out_stack.shape}, dtype={out_stack.dtype}")
    print(f"✅ Saved full images to {save_path.replace('.tif', '_full.tif')}, shape={(Z, crop_size, crop_size*3)}")
    
    return history, combined_stack, out_stack