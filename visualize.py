import matplotlib.pyplot as plt

    
import matplotlib.pyplot as plt

def visualize_history(history):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Loss plot
    axes[0, 0].plot(history["train_loss_epoch"], label="Train Loss", linewidth=2)
    axes[0, 0].plot(history["val_loss_epoch"], label="Val Loss", linewidth=2)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Training & Validation Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Dice Score plot
    axes[0, 1].plot(history["train_dice_epoch"], label="Train Dice", linewidth=2)
    axes[0, 1].plot(history["val_dice_epoch"], label="Val Dice", linewidth=2)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Dice Score")
    axes[0, 1].set_title("Dice Score")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1)
    
    # IoU Score plot
    axes[0, 2].plot(history["train_iou_epoch"], label="Train IoU", linewidth=2)
    axes[0, 2].plot(history["val_iou_epoch"], label="Val IoU", linewidth=2)
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].set_ylabel("IoU Score")
    axes[0, 2].set_title("Intersection over Union (IoU)")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_ylim(0, 1)
    
    # Precision plot
    axes[1, 0].plot(history["train_precision_epoch"], label="Train Precision", linewidth=2)
    axes[1, 0].plot(history["val_precision_epoch"], label="Val Precision", linewidth=2)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Precision")
    axes[1, 0].set_title("Precision")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1)
    
    # Recall plot
    axes[1, 1].plot(history["train_recall_epoch"], label="Train Recall", linewidth=2)
    axes[1, 1].plot(history["val_recall_epoch"], label="Val Recall", linewidth=2)
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Recall")
    axes[1, 1].set_title("Recall")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(0, 1)
    
    # F1-Score (Dice) vs IoU comparison
    axes[1, 2].plot(history["val_dice_epoch"], label="Val Dice (F1)", linewidth=2, linestyle='-')
    axes[1, 2].plot(history["val_iou_epoch"], label="Val IoU", linewidth=2, linestyle='-')
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].set_ylabel("Score")
    axes[1, 2].set_title("Validation: Dice (F1) vs IoU Comparison")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.show()
    
    
def visualize_history_compact(history):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss plot
    axes[0, 0].plot(history["train_loss_epoch"], label="Train Loss", linewidth=2)
    axes[0, 0].plot(history["val_loss_epoch"], label="Val Loss", linewidth=2)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Training & Validation Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Main segmentation metrics
    axes[0, 1].plot(history["train_dice_epoch"], label="Train Dice", linewidth=2)
    axes[0, 1].plot(history["val_dice_epoch"], label="Val Dice", linewidth=2)
    axes[0, 1].plot(history["train_iou_epoch"], label="Train IoU", linewidth=2, linestyle='--')
    axes[0, 1].plot(history["val_iou_epoch"], label="Val IoU", linewidth=2, linestyle='--')
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Score")
    axes[0, 1].set_title("Dice & IoU Scores")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1)
    
    # Precision-Recall
    axes[1, 0].plot(history["train_precision_epoch"], label="Train Precision", linewidth=2)
    axes[1, 0].plot(history["val_precision_epoch"], label="Val Precision", linewidth=2)
    axes[1, 0].plot(history["train_recall_epoch"], label="Train Recall", linewidth=2, linestyle='--')
    axes[1, 0].plot(history["val_recall_epoch"], label="Val Recall", linewidth=2, linestyle='--')
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Score")
    axes[1, 0].set_title("Precision & Recall")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1)
    
    # Validation metrics summary
    epochs = range(1, len(history["val_dice_epoch"]) + 1)
    axes[1, 1].plot(epochs, history["val_dice_epoch"], label="Dice", marker='o', markersize=3)
    axes[1, 1].plot(epochs, history["val_iou_epoch"], label="IoU", marker='s', markersize=3)
    axes[1, 1].plot(epochs, history["val_precision_epoch"], label="Precision", marker='^', markersize=3)
    axes[1, 1].plot(epochs, history["val_recall_epoch"], label="Recall", marker='d', markersize=3)
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Score")
    axes[1, 1].set_title("All Validation Metrics")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.show()
    
    # Print final metrics
    print("\nFinal Validation Metrics:")
    print(f"Dice: {history['val_dice_epoch'][-1]:.4f}")
    print(f"IoU: {history['val_iou_epoch'][-1]:.4f}")
    print(f"Precision: {history['val_precision_epoch'][-1]:.4f}")
    print(f"Recall: {history['val_recall_epoch'][-1]:.4f}")