import os
from glob import glob
from PIL import Image
import numpy as np
from tqdm import tqdm

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
    def __init__(self, image_dir, mask_dir, transform=None):
        self.images = sorted(glob(os.path.join(image_dir, '*.tif')))
        self.masks = sorted(glob(os.path.join(mask_dir, '*.png')))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        mask = Image.open(self.masks[idx]).convert('L')

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        mask = (mask > 0).float()  # binarizar

        return image, mask

# Transformaciones
transform = T.Compose([
    T.Resize((320, 448)),  # tamaño fijo según XML
    T.ToTensor()
])

# Dataset y DataLoader
dataset = PetalVeinDataset(IMAGE_DIR, MASK_DIR, transform)
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

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        def CBR(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        # Encoder con menos canales (para reducir overfitting)
        self.enc1 = CBR(3, 32)  # Mantenemos 32 canales como en tu versión original
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = CBR(32, 64)  # Reducido de 128 a 64
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = CBR(64, 128)  # Reducido de 256 a 128
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck también reducido
        self.bottleneck = CBR(128, 256)  # Reducido de 512 a 256

        # Decoder - ajustamos las dimensiones para que coincidan
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = CBR(256, 128)  # 128*2 por la concatenación
        
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = CBR(128, 64)  # 64*2 por la concatenación
        
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = CBR(64, 32)   # 32*2 por la concatenación

        self.final = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)       # 32 canales
        e2 = self.enc2(self.pool1(e1))  # 64 canales
        e3 = self.enc3(self.pool2(e2))  # 128 canales
        b = self.bottleneck(self.pool3(e3))  # 256 canales
        
        # Decoder
        d3 = self.up3(b)        # 128 canales
        d3 = torch.cat([d3, e3], dim=1)  # 128 + 128 = 256
        d3 = self.dec3(d3)      # 128 canales
        
        d2 = self.up2(d3)       # 64 canales
        d2 = torch.cat([d2, e2], dim=1)  # 64 + 64 = 128
        d2 = self.dec2(d2)      # 64 canales
        
        d1 = self.up1(d2)       # 32 canales
        d1 = torch.cat([d1, e1], dim=1)  # 32 + 32 = 64
        d1 = self.dec1(d1)      # 32 canales
        
        return torch.sigmoid(self.final(d1))

# Dice + BCE Loss
class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, preds, targets):
        smooth = 1.
        preds = preds.view(-1)
        targets = targets.view(-1)
        intersection = (preds * targets).sum()
        dice = (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
        return 1 - dice + self.bce(preds, targets)

class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.bce = nn.BCELoss(reduction='none')
        
        # Variables para almacenar los valores actuales (opcional, para debug)
        self.current_focal = 0.0
        self.current_dice = 0.0

    def forward(self, preds, targets):
        # --- Focal Loss ---
        bce_loss = self.bce(preds, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        focal_loss = focal_loss.mean()
        self.current_focal = focal_loss.item()  # Guardar para acceso externo

        # --- Dice Loss ---
        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)
        intersection = (preds_flat * targets_flat).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (preds_flat.sum() + targets_flat.sum() + self.smooth)
        self.current_dice = dice_loss.item()  # Guardar para acceso externo

        return focal_loss + dice_loss  # Pérdida total

# Entrenamiento
model = UNet().to(DEVICE)
criterion = FocalDiceLoss(alpha=0.8, gamma=2.0).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
patience = 10
best_loss = float('inf')
counter = 0

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    epoch_focal = 0
    epoch_dice = 0
    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Training"):
        images, masks = images.to(DEVICE), masks.to(DEVICE)

        preds = model(images)
        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_focal += criterion.current_focal
        epoch_dice += criterion.current_dice

    avg_loss = epoch_loss/len(train_loader)
    avg_focal = epoch_focal/len(train_loader)
    avg_dice = epoch_dice/len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Validation"):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            preds = model(images)
            val_loss += criterion(preds, masks).item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} (Focal: {avg_focal:.4f}, Dice: {avg_dice:.4f})")


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
