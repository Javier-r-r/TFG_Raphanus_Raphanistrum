import os
from glob import glob
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torch.utils.data import random_split

# Config
IMAGE_DIR = '../database_petals'
MASK_DIR = '../masks_petals'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 500
BATCH_SIZE = 4
LR = 1e-4
SAVE_PATH = 'unet_petals.pth'
RANDOM_SEED = 42  # Semilla para mantener los conjuntos de validacion y prueba

# Dataset personalizado
class PetalVeinDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_transform=None, mask_transform=None):
        self.images = sorted(glob(os.path.join(image_dir, '*.tif')))
        self.masks = sorted(glob(os.path.join(mask_dir, '*.png')))
        self.mask_transform = mask_transform
        self.image_transform = image_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        mask = Image.open(self.masks[idx]).convert('L')

        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        mask = (mask > 0).float()  # binarizar

        return image, mask

# Transformaciones
image_transform = T.Compose([
    T.Resize((320, 448)),
    T.RandomAffine(degrees=1, translate=(0.01, 0.01), shear=2),
    T.ColorJitter(brightness=0.05, contrast=0.05),
    T.ToTensor()
])
mask_transform = T.Compose([
    T.Resize((320, 448)),
    T.ToTensor()
])
# Dataset y DataLoader
dataset = PetalVeinDataset(IMAGE_DIR, MASK_DIR, image_transform, mask_transform)
# Especificar los tamaños de los diferentes conjuntos
train_size = int(0.7*len(dataset))
val_size = int(0.15*len(dataset))
test_size = len(dataset) - train_size - val_size
#Dividir el dataset
train_data, val_data, test_data = random_split(dataset, [0.7, 0.15, 0.15],
						generator=torch.Generator().manual_seed(RANDOM_SEED))
# Guardar nombres de archivos de val y test
def save_split_files(dataset, output_file):
    with open(output_file, 'w') as f:
        for idx in dataset.indices:
            f.write(f"{os.path.basename(dataset.dataset.images[idx])}\n")

save_split_files(test_data, 'test_files.txt')
#Crear los DataLoaders para cada conjunto
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size = BATCH_SIZE)

# Reemplazar la clase UNet original por la nueva versión mejorada
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        self.bottleneck = DoubleConv(256, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv1 = DoubleConv(128, 64)
        
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        b = self.bottleneck(self.pool3(d3))
        up3 = self.conv3(torch.cat([self.up3(b), d3], dim=1))
        up2 = self.conv2(torch.cat([self.up2(up3), d2], dim=1))
        up1 = self.conv1(torch.cat([self.up1(up2), d1], dim=1))
        return self.final(up1)

class VeinLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.2, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.sobel = SobelFilter()

    def forward(self, preds, targets):
        # --- Dice Loss ---
        smooth = 1.0
        preds_sigmoid = torch.sigmoid(preds)
        preds_flat = preds_sigmoid.view(-1)
        targets_flat = targets.view(-1)

        intersection = (preds_flat * targets_flat).sum()
        dice_coeff = (2. * intersection + smooth) / (preds_flat.sum() + targets_flat.sum() + smooth)
        dice_loss = 1 - dice_coeff

        # --- Edge Loss ---
        edge_targets = self.sobel(targets)
        edge_preds = self.sobel(preds_sigmoid)
        edge_loss = F.mse_loss(edge_preds, edge_targets)

        # --- Focal Loss ---
        bce_loss = self.bce(preds, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = ((1 - pt) ** self.gamma * bce_loss).mean()

        # --- Total loss ---
        total_loss = self.alpha * dice_loss + self.beta * edge_loss + (1 - self.alpha - self.beta) * focal_loss

        return total_loss, dice_coeff.item()

class SobelFilter(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel_x = torch.tensor([[[[1, 0, -1],
                                        [2, 0, -2],
                                        [1, 0, -1]]]], dtype=torch.float32)
        self.kernel_y = torch.tensor([[[[1, 2, 1],
                                        [0, 0, 0],
                                        [-1, -2, -1]]]], dtype=torch.float32)

    def forward(self, x):
        # x: [B, 1, H, W]
        if x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)  # Convertir RGB a gris si hace falta
        grad_x = F.conv2d(x, self.kernel_x.to(x.device), padding=1)
        grad_y = F.conv2d(x, self.kernel_y.to(x.device), padding=1)
        return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)  # evitar sqrt(0)

# Dice + BCE Loss
class ImprovedDiceBCELoss(nn.Module):
    def __init__(self, pos_weight=100.0, alpha=0.5, smooth=0.1):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(DEVICE))
        self.alpha = alpha  # Peso para combinar losses
        self.smooth = smooth  # Suavizado para Dice

    def forward(self, preds, targets):
        preds_sigmoid = torch.sigmoid(preds)  # Para Dice
        preds_flat = preds_sigmoid.view(-1)
        targets_flat = targets.view(-1)

        # Dice Loss
        intersection = (preds_flat * targets_flat).sum()
        dice_coeff = (2. * intersection + self.smooth) / (preds_flat.sum() + targets_flat.sum() + self.smooth)
        dice_loss = 1 - dice_coeff

        # BCE Loss (ya maneja logits)
        bce_loss = self.bce(preds, targets)

        # Combinación
        total_loss = self.alpha * dice_loss + (1 - self.alpha) * bce_loss

        return total_loss, dice_coeff.item()

class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss(reduction='none')  # Cambiado a BCEWithLogitsLoss
        
    def forward(self, preds, targets):
        # --- Focal Loss ---
        bce_loss = self.bce(preds, targets)  # Ahora acepta logits directamente
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        focal_loss = focal_loss.mean()
        
        # --- Dice Loss --- (ahora con sigmoid aplicado solo para el cálculo Dice)
        preds_sigmoid = torch.sigmoid(preds)
        preds_flat = preds_sigmoid.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (preds_flat * targets_flat).sum()
        dice_coeff = (2. * intersection + self.smooth) / (preds_flat.sum() + targets_flat.sum() + self.smooth)
        dice_loss = 1 - dice_coeff
        
        # Pérdida total y coeficiente Dice
        total_loss = focal_loss + dice_loss
        
        return total_loss, dice_coeff.item()

# Entrenamiento
model = UNet(
    in_channels=3,          # Canales de entrada (RGB)
    out_channels=1,         # Segmentación binaria
).to(DEVICE)
criterion = ImprovedDiceBCELoss()
optimizer = torch.optim.AdamW(model.parameters(), LR, weight_decay=2e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', 
patience=20, factor=0.5
)
patience = 20
best_loss = float('inf')
counter = 0

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    epoch_dice = 0
    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Training"):
        images, masks = images.to(DEVICE), masks.to(DEVICE)

        preds = model(images)
        loss, dice = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_dice += dice

    avg_loss = epoch_loss/len(train_loader)
    avg_dice = epoch_dice/len(train_loader)

    model.eval()
    val_loss = 0.0
    val_dice = 0.0
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Validation"):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            preds = model(images)
            loss, dice = criterion(preds, masks)
            val_loss += loss.item()
            val_dice += dice

    avg_val_loss = val_loss / len(val_loader)
    avg_val_dice = val_dice / len(val_loader)
    #print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Train Dice: {avg_dice:.4f} | Val Dice: {avg_val_dice:.4f}")
    
    scheduler.step(avg_val_loss)

    for i, param_group in enumerate(optimizer.param_groups):
        print(f"Learning rate actual (grupo {i}): {param_group['lr']}")

    # Guardar el mejor modelo
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        counter = 0
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"Modelo guardado en {SAVE_PATH}")
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping at epoch {epoch}")
            break

print("Entrenamiento finalizado.")
