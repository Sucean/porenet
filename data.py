import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.data import Dataset
import torchvision.transforms as T


def data_loader(path):
    data = np.load(path)
    return crop_and_slice(data)

def crop_and_slice(stack, crop_size=1024, patch_size=128):
    # Handle 4D input shapes
    if len(stack.shape) == 4:
        # For (175, 1128, 1128, 2), take first channel
        if stack.shape[-1] == 2:
            stack = stack[:, :, :, 0]  # Shape becomes (175, 1128, 1128)
        # For (175, 1128, 1128, 1), remove singleton dimension
        elif stack.shape[-1] == 1:
            stack = stack.squeeze(-1)  # Shape becomes (175, 1128, 1128)
        else:
            raise ValueError(f"Unsupported channel dimension: {stack.shape[-1]}")
    
    # Now stack should be (175, 1128, 1128)
    Z, H, W = stack.shape
    
    # Crop to 175x1024x1024 from center
    start_h = (H - crop_size) // 2  # (1128 - 1024) // 2 = 52
    start_w = (W - crop_size) // 2  # (1128 - 1024) // 2 = 52
    stack_crop = stack[:, start_h:start_h+crop_size, start_w:start_w+crop_size]
    
    # Verify cropped shape: (175, 1024, 1024)
    Z, H, W = stack_crop.shape
    
    # Calculate number of patches
    n_h = crop_size // patch_size  # 1024 ÷ 128 = 8
    n_w = crop_size // patch_size  # 1024 ÷ 128 = 8
    
    # Reshape to (175, 8, 128, 8, 128)
    stack = stack_crop.reshape(Z, n_h, patch_size, n_w, patch_size)
    
    # Transpose to (175, 8, 8, 128, 128)
    stack = stack.transpose(0, 1, 3, 2, 4)
    
    # Reshape to (175 * 8 * 8, 128, 128) = (11200, 128, 128)
    stack = stack.reshape(-1, patch_size, patch_size)
    
    return stack

# -----------------------------------
# Define transforms
# -----------------------------------
train_transform = T.Compose([
    # Convert to tensor (if not already)
    T.ConvertImageDtype(torch.float32),

    # Intensity-based augmentations
    T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5),
    T.RandomApply([T.GaussianBlur(kernel_size=(3, 3))], p=0.3),

    # Geometric augmentations (only if orientation doesn’t matter!)
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomRotation(degrees=90),

    # Make sure dtype stays float32
    lambda x: x.float()
])

val_transform = T.Compose([
    T.ConvertImageDtype(torch.float32)
])

# -----------------------------------
# Custom dataset
# -----------------------------------
class PoreDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]  # (C, H, W)
        mask = self.masks[idx]    # (1, H, W)

        # Ensure tensor type
        image = image.float()
        mask = mask.float()

        if self.transform:
            # torchvision expects PIL or Tensor in (C,H,W)
            # We need same transform for image and mask → handle geometric manually
            # For intensity transforms, only apply to image
            seed = torch.randint(0, 2**32, (1,)).item()
            
            # Set random seed so both get the same geometric transform
            torch.manual_seed(seed)
            image = self.transform(image)

            torch.manual_seed(seed)
            mask = self.transform(mask)

        return image, mask
    
# -----------------------------
# Train / val / test split
# ----------------------------
def data_split(inputs, targets, patches_per_slice = 64, train_frac = 0.7, val_frac = 0.2):
        
    N = inputs.shape[0]
    N_slices = N // patches_per_slice

    train_slices, val_slices = int(N_slices * train_frac), int(N_slices * val_frac)
    test_slices = N_slices - train_slices - val_slices

    train_size = train_slices * patches_per_slice
    val_size = val_slices * patches_per_slice
    test_size = test_slices * patches_per_slice

    # Split by slice indices
    train_dataset = PoreDataset(inputs[:train_size], targets[:train_size])
    val_dataset = PoreDataset(inputs[train_size:train_size+val_size], targets[train_size:train_size+val_size])
    test_dataset = PoreDataset(inputs[train_size+val_size:], targets[train_size+val_size:])

    return train_dataset, val_dataset, test_dataset