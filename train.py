import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet import combined_loss, dice_loss

def train_model(model, train_loader, val_loader, device, epochs=10, lr=1e-3, weights=None):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss_epoch": [],
        "val_loss_epoch": [],
        "train_precision_epoch": [],
        "train_recall_epoch": [],
        "train_dice_epoch": [],
        "train_iou_epoch": [],
        "val_precision_epoch": [],
        "val_recall_epoch": [],
        "val_dice_epoch": [],
        "val_iou_epoch": []
    }

    for epoch in range(epochs):
        # -----------------------------
        # Training
        # -----------------------------
        model.train()
        running_loss = 0.0
        precision_sum = 0.0
        recall_sum = 0.0
        dice_sum = 0.0
        iou_sum = 0.0

        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = combined_loss(pred, y, weights)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Generate Metrics
            precision, recall, dice, iou = segmentation_metrics(pred, y)
            precision_sum += precision
            recall_sum += recall
            dice_sum += dice
            iou_sum += iou
   

            avg_batch_loss = running_loss / (i + 1)
            print(f"\rEpoch {epoch+1}/{epochs} Step {i+1}/{len(train_loader)} - "
                  f"Loss: {avg_batch_loss:.4f}", end="")
        
        
        # Calculate epoch averages for training
        train_loss_epoch = running_loss / len(train_loader)
        train_precision_epoch = precision_sum / len(train_loader)
        train_recall_epoch = recall_sum / len(train_loader)
        train_dice_epoch = dice_sum / len(train_loader)
        train_iou_epoch = iou_sum / len(train_loader)
        
        history["train_loss_epoch"].append(train_loss_epoch)
        history["train_precision_epoch"].append(train_precision_epoch)
        history["train_recall_epoch"].append(train_recall_epoch)
        history["train_dice_epoch"].append(train_dice_epoch)
        history["train_iou_epoch"].append(train_iou_epoch)
        
        print(f"\nEpoch {epoch+1} Training - Loss: {train_loss_epoch:.4f}, "
              f"Precision: {train_precision_epoch:.4f}, Recall: {train_recall_epoch:.4f}, "
              f"Dice: {train_dice_epoch:.4f}, IoU: {train_iou_epoch:.4f}")

        # -----------------------------
        # Validation
        # -----------------------------
        model.eval()
        val_running_loss = 0.0
        val_precision_sum = 0.0
        val_recall_sum = 0.0
        val_dice_sum = 0.0
        val_iou_sum = 0.0

        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                pred_val = model(x_val)
                val_loss = combined_loss(pred_val, y_val, weights)
                val_running_loss += val_loss.item()

                # Calculate metrics for validation batch
                precision, recall, dice, iou = segmentation_metrics(pred_val, y_val)
                val_precision_sum += precision
                val_recall_sum += recall
                val_dice_sum += dice
                val_iou_sum += iou

                
        # Calculate epoch averages for validation
        val_loss_epoch = val_running_loss / len(val_loader)
        val_precision_epoch = val_precision_sum / len(val_loader)
        val_recall_epoch = val_recall_sum / len(val_loader)
        val_dice_epoch = val_dice_sum / len(val_loader)
        val_iou_epoch = val_iou_sum / len(val_loader)
        
        history["val_loss_epoch"].append(val_loss_epoch)
        history["val_precision_epoch"].append(val_precision_epoch)
        history["val_recall_epoch"].append(val_recall_epoch)
        history["val_dice_epoch"].append(val_dice_epoch)
        history["val_iou_epoch"].append(val_iou_epoch)
        
        print(f"Epoch {epoch+1} Validation - Loss: {val_loss_epoch:.4f}, "
              f"Precision: {val_precision_epoch:.4f}, Recall: {val_recall_epoch:.4f}, "
              f"Dice: {val_dice_epoch:.4f}, IoU: {val_iou_epoch:.4f}\n")

    return model, history

def segmentation_metrics(pred, target, threshold=0.5, eps=1e-7):
    # Binarize predictions
    pred_bin = (torch.sigmoid(pred) > threshold).float()

    # Flatten for simplicity (assuming binary segmentation)
    pred_bin = pred_bin.view(-1)
    target = target.view(-1)

    # True positives, false positives, false negatives
    TP = (pred_bin * target).sum()
    FP = (pred_bin * (1 - target)).sum()
    FN = ((1 - pred_bin) * target).sum()
    TN = ((1 - pred_bin) * (1 - target)).sum()

    # Calculate metrics
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    dice = 2 * TP / (2 * TP + FP + FN + eps)
    iou = TP / (TP + FP + FN + eps)

    return precision.item(), recall.item(), dice.item(), iou.item()