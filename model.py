"""
Binary Segmentation of Petal Veins using Deep Learning

This script implements a complete pipeline for training and evaluating a binary segmentation model
to detect petal veins in plant images. The implementation uses PyTorch and segmentation_models_pytorch
for model architecture, and includes data loading, augmentation, training, evaluation, and visualization.

Key Features:
- Supports multiple model architectures (Unet, FPN, etc.)
- Configurable encoders (resnet34, resnet50, etc.)
- Multiple loss functions (Dice, BCE, Focal)
- Data augmentation pipeline
- Early stopping and learning rate scheduling
- Comprehensive metrics tracking
- Visualization of results

Usage:
python train.py --arch Unet --encoder_name resnet34 --loss dice --output_dir results
"""

import logging
import os
import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import json
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from tqdm import tqdm
import pandas as pd
import albumentations as A
import segmentation_models_pytorch as smp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%d:%m:%Y %H:%M:%S",
)

# ----------------------------
# Hardware Configuration
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")
if device == "cpu":
    os.system("export OMP_NUM_THREADS=64")
    torch.set_num_threads(os.cpu_count())

# ----------------------------
# Hyperparameters
# ----------------------------
epochs_max = 250        # Maximum number of training epochs
adam_lr = 2e-4          # Learning rate for Adam optimizer
eta_min = 1e-5          # Minimum learning rate for scheduler
batch_size = 8          # Batch size for training
input_image_reshape = (640, 640)  # Target image size
foreground_class = 1    # Class to consider as foreground in binary segmentation

# ----------------------------
# Command Line Arguments
# ----------------------------
def parse_args():
    """Parse command line arguments for model configuration."""
    parser = argparse.ArgumentParser(description='Petal Vein Segmentation Training')
    parser.add_argument('--arch', type=str, default='Unet', 
                      help='Model architecture (e.g., Unet, FPN)')
    parser.add_argument('--encoder_name', type=str, default='resnet34',
                      help='Encoder backbone (e.g., resnet34, resnet50)')
    parser.add_argument('--loss_fn', type=str, default='dice',
                      help='Loss function (dice, bce, focal)')
    parser.add_argument('--output_dir', type=str, default='output',
                      help='Directory to save results')
    parser.add_argument('--data_split', type=str, 
                      default='../experiments/experiment_70_15_15',
                      help='Path to dataset split')
    return parser.parse_args()

# ----------------------------
# Dataset Classes
# ----------------------------
class PetalVeinDataset(BaseDataset):
    """
    Custom dataset for petal vein segmentation with TIFF images and PNG masks.
    
    Args:
        images_dir (str): Directory containing TIFF images
        masks_dir (str): Directory containing PNG masks
        input_image_reshape (tuple): Target image size (height, width)
        augmentation (albumentations.Compose): Augmentation pipeline
        normalize (bool): Whether to normalize images to [0,1]
    """
    
    def __init__(
        self,
        images_dir,
        masks_dir,
        input_image_reshape=(320, 320),
        augmentation=None,
        normalize=True
    ):
        # Get all TIFF image files
        self.ids = [f for f in os.listdir(images_dir) if f.lower().endswith('.tif')]
        
        if not self.ids:
            raise ValueError(f"No TIFF images found in {images_dir}")
        
        # Create full paths
        self.images_filepaths = [
            os.path.join(images_dir, img_id) for img_id in self.ids
        ]
        self.masks_filepaths = [
            os.path.join(masks_dir, img_id.replace('.tif', '.png').replace('.TIF', '.png'))
            for img_id in self.ids
        ]

        self.input_image_reshape = input_image_reshape
        self.augmentation = augmentation
        self.normalize = normalize
        
    def __getitem__(self, i):
        """Load and preprocess a single image-mask pair."""
        # Read and preprocess image
        image = cv2.imread(self.images_filepaths[i], cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image at {self.images_filepaths[i]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.input_image_reshape)
        
        # Read and preprocess mask
        mask = cv2.imread(self.masks_filepaths[i], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not read mask at {self.masks_filepaths[i]}")
        mask = (mask > 0).astype(np.uint8)
        mask = cv2.resize(mask, self.input_image_reshape, interpolation=cv2.INTER_NEAREST)
        
        # Apply augmentations
        if self.augmentation:
            augmented = self.augmentation(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']
        
        # Convert to tensors
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        if self.normalize:
            image = image / 255.0
            
        mask = torch.from_numpy(mask).long()
        
        return image, mask
    
    def __len__(self):
        return len(self.ids)

# ----------------------------
# Data Augmentation
# ----------------------------
augmentation_train = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])

augmentation_val_test = A.Compose([])

# ----------------------------
# Model Definition
# ----------------------------
class PetalVeinModel(torch.nn.Module):
    """
    Segmentation model with configurable architecture and encoder.
    
    Args:
        arch (str): Model architecture (Unet, FPN, etc.)
        encoder_name (str): Encoder backbone (resnet34, etc.)
        in_channels (int): Number of input channels
        out_classes (int): Number of output classes
        **kwargs: Additional model arguments
    """
    
    def __init__(self, arch, encoder_name, in_channels=3, out_classes=1, **kwargs):
        super().__init__()
        # Initialize normalization parameters
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        
        # Load dataset-specific stats if available
        if os.path.exists('dataset_mean.npy'):
            self.mean = torch.tensor(np.load('dataset_mean.npy')).view(1, 3, 1, 1).to(device)
            
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=out_classes,
            **kwargs,
        )

    def forward(self, image):
        """Forward pass with input normalization."""
        image = (image - self.mean) / self.std
        return self.model(image)

# ----------------------------
# Visualization Utilities
# ----------------------------
def visualize_samples(images, masks, output_dir, prefix="train", num_samples=3):
    """Save sample images with their masks for visualization."""
    samples_dir = os.path.join(output_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    
    for i in range(min(num_samples, len(images))):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        # Display image
        img = images[i].transpose(1, 2, 0) if images[i].shape[0] == 3 else images[i][0]
        ax1.imshow(img, cmap='gray')
        ax1.set_title(f'Image {i}')
        ax1.axis('off')
        
        # Display mask
        ax2.imshow(masks[i], cmap='gray')
        ax2.set_title(f'Mask {i}')
        ax2.axis('off')
        
        plt.savefig(os.path.join(samples_dir, f"{prefix}_sample_{i}.png"))
        plt.close()

def compute_dataset_statistics(images_dir, input_shape=(640, 640), batch_size=32):
    """
    Compute mean and std of dataset images.
    
    Args:
        images_dir (str): Directory containing images
        input_shape (tuple): Target image size
        batch_size (int): Batch size for computation
        
    Returns:
        tuple: (mean, std) per channel
    """
    image_paths = [os.path.join(images_dir, f) for f in os.listdir(images_dir) 
                  if f.lower().endswith('.tif')]
    
    pixel_sum = np.zeros(3)
    pixel_sq_sum = np.zeros(3)
    num_pixels = 0
    
    for i in tqdm(range(0, len(image_paths), batch_size)):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        
        for path in batch_paths:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, input_shape)
            batch_images.append(img)
            
        batch_images = np.stack(batch_images)
        
        pixel_sum += np.sum(batch_images, axis=(0, 1, 2))
        pixel_sq_sum += np.sum(batch_images**2, axis=(0, 1, 2))
        num_pixels += batch_images.shape[0] * batch_images.shape[1] * batch_images.shape[2]
    
    mean = pixel_sum / num_pixels
    std = np.sqrt((pixel_sq_sum / num_pixels) - mean**2)
    
    return mean, std

# ----------------------------
# Training Utilities
# ----------------------------
def train_and_evaluate_one_epoch(
    model, train_dataloader, valid_dataloader, optimizer, scheduler, loss_fn, device
):
    """Train and validate model for one epoch."""
    model.train()
    train_loss = 0
    for batch in tqdm(train_dataloader, desc="Training"):
        images, masks = batch
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        # Prepare masks for loss calculation
        if isinstance(loss_fn, (torch.nn.BCEWithLogitsLoss, smp.losses.SoftBCEWithLogitsLoss)):
            masks = masks.unsqueeze(1).float()
        else:
            masks = masks.float()

        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    scheduler.step()
    avg_train_loss = train_loss / len(train_dataloader)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(valid_dataloader, desc="Evaluating"):
            images, masks = batch
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            if isinstance(loss_fn, (torch.nn.BCEWithLogitsLoss, smp.losses.SoftBCEWithLogitsLoss)):
                masks = masks.unsqueeze(1).float()
            else:
                masks = masks.float()
            loss = loss_fn(outputs, masks)

            val_loss += loss.item()

    avg_val_loss = val_loss / len(valid_dataloader)
    return avg_train_loss, avg_val_loss

# Early stopping parameters
early_stop_patience = 10
early_stop_min_delta = 0.001

def train_model(
    model,
    train_dataloader,
    valid_dataloader,
    optimizer,
    scheduler,
    loss_fn,
    device,
    epochs,
    output_dir=None,
    patience=early_stop_patience,
    min_delta=early_stop_min_delta,
    args=None
):
    """Train model with early stopping and save best weights."""
    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_weights = None

    for epoch in range(epochs):
        avg_train_loss, avg_val_loss = train_and_evaluate_one_epoch(
            model,
            train_dataloader,
            valid_dataloader,
            optimizer,
            scheduler,
            loss_fn,
            device,
        )
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        logging.info(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        )
        
        # Early stopping logic
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_weights = model.state_dict().copy()
            
            if output_dir:
                torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
        else:
            epochs_no_improve += 1
            logging.info(f"No improvement for {epochs_no_improve} epochs")
            
            if epochs_no_improve >= patience:
                logging.info(f"Early stopping at epoch {epoch + 1}")
                if best_model_weights is not None:
                    model.load_state_dict(best_model_weights)
                break

    # Save training history and plots
    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
    }

    if output_dir:
        # Save loss curve
        plt.figure()
        plt.plot(history['train_losses'], label='Train Loss')
        plt.plot(history['val_losses'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
        plt.close()
        
        # Save configuration
        if args:
            args_dict = vars(args)
            with open(os.path.join(output_dir, 'config.json'), 'w') as f:
                json.dump(args_dict, f, indent=4)
        
        # Save history to CSV
        pd.DataFrame(history).to_csv(os.path.join(output_dir, "train_history.csv"), index=False)

    return history

# ----------------------------
# Evaluation Utilities
# ----------------------------
def test_model(model, output_dir, test_dataloader, loss_fn, device):
    """Evaluate model on test set and save metrics."""
    model.eval()
    test_loss = 0
    image_metrics = []
    
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="Testing")):
            images, masks = batch
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            # Get probabilities and predictions
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            # Prepare masks for loss calculation
            if isinstance(loss_fn, (torch.nn.BCEWithLogitsLoss, smp.losses.SoftBCEWithLogitsLoss)):
                loss_input = outputs
                target = masks.unsqueeze(1).float()
            else:
                loss_input = probs
                target = masks.float()

            loss = loss_fn(loss_input, target)
            test_loss += loss.item()

            # Calculate metrics for each image
            for i in range(images.shape[0]):
                pred_mask = preds[i].squeeze().cpu().numpy()
                true_mask = masks[i].squeeze().cpu().numpy()

                tp, fp, fn, tn = smp.metrics.get_stats(
                    torch.tensor(pred_mask).unsqueeze(0).unsqueeze(0),
                    torch.tensor(true_mask).unsqueeze(0).unsqueeze(0),
                    mode="binary",
                    threshold=0.5,
                )

                iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
                precision = smp.metrics.precision(tp, fp, fn, tn, reduction="micro")
                recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro")
                f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")

                image_metrics.append({
                    "image_id": f"batch_{batch_idx}_image_{i}",
                    "test_loss": loss.item(),
                    "iou_score": iou_score.item(),
                    "precision": precision.item(),
                    "recall": recall.item(),
                    "f1_score": f1_score.item(),
                })

    # Calculate aggregate metrics
    test_loss_mean = test_loss / len(test_dataloader)
    
    aggregate_metrics = {
        "test_loss": test_loss_mean,
        "iou_score": np.mean([m["iou_score"] for m in image_metrics]),
        "precision": np.mean([m["precision"] for m in image_metrics]),
        "recall": np.mean([m["recall"] for m in image_metrics]),
        "f1_score": np.mean([m["f1_score"] for m in image_metrics]),
    }

    # Save metrics
    pd.DataFrame(image_metrics).to_csv(
        os.path.join(output_dir, "individual_metrics.csv"), index=False)
    pd.DataFrame([aggregate_metrics]).to_csv(
        os.path.join(output_dir, "aggregate_metrics.csv"), index=False)

    return aggregate_metrics

# ----------------------------
# Main Execution
# ----------------------------
if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configure loss function
    if args.loss_fn.lower() == 'bce':
        loss_fn = smp.losses.SoftBCEWithLogitsLoss()
    elif args.loss_fn.lower() == 'focal':
        loss_fn = smp.losses.FocalLoss(smp.losses.BINARY_MODE)
    else:  # Default to Dice
        loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
    
    # Clear GPU cache
    torch.cuda.empty_cache()

    print("\n=== Starting Training ===")

    # Set up data paths
    train_images_dir = os.path.join(args.data_split, 'train', 'images')
    train_masks_dir = os.path.join(args.data_split, 'train', 'masks')
    val_images_dir = os.path.join(args.data_split, 'val', 'images')
    val_masks_dir = os.path.join(args.data_split, 'val', 'masks')
    test_images_dir = os.path.join(args.data_split, 'test', 'images')
    test_masks_dir = os.path.join(args.data_split, 'test', 'masks')

    # Create datasets and dataloaders
    train_dataset = PetalVeinDataset(
        images_dir=train_images_dir,
        masks_dir=train_masks_dir,
        input_image_reshape=input_image_reshape,
        augmentation=augmentation_train
    )
    val_dataset = PetalVeinDataset(
        images_dir=val_images_dir,
        masks_dir=val_masks_dir,
        input_image_reshape=input_image_reshape,
        augmentation=augmentation_val_test
    )
    test_dataset = PetalVeinDataset(
        images_dir=test_images_dir,
        masks_dir=test_masks_dir,
        input_image_reshape=input_image_reshape,
        augmentation=augmentation_val_test
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, optimizer, and scheduler
    model = PetalVeinModel(args.arch, args.encoder_name, in_channels=3, out_classes=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=adam_lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_max, eta_min=eta_min)

    # Compute and save dataset statistics
    train_mean, train_std = compute_dataset_statistics(train_images_dir)
    print(f"Dataset mean: {train_mean}")
    print(f"Dataset std: {train_std}")
    np.save(os.path.join(args.output_dir, 'dataset_mean.npy'), train_mean)
    np.save(os.path.join(args.output_dir, 'dataset_std.npy'), train_std)

    # Train the model
    history = train_model(
        model, 
        train_loader, 
        valid_loader, 
        optimizer, 
        scheduler, 
        loss_fn, 
        device, 
        epochs_max,
        output_dir=args.output_dir,
        patience=early_stop_patience,
        min_delta=early_stop_min_delta,
        args=args
    )

    # Evaluate on test set
    metrics = test_model(
        model, 
        args.output_dir, 
        test_loader, 
        loss_fn, 
        device
    )

    # Save final metrics
    pd.DataFrame([metrics]).to_csv(
        os.path.join(args.output_dir, "final_metrics.csv"), index=False)
    
    print("\n=== Training Complete ===")
    print(f"Results saved to: {args.output_dir}")