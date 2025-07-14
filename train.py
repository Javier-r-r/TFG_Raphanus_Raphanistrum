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
    T.Resize((320, 448)),
    T.RandomAffine(degrees=1, translate=(0.01, 0.01), shear=2),
    T.ElasticTransform(alpha=5.0, sigma=1.0),  # Más suave
    T.ColorJitter(brightness=0.05, contrast=0.05),  # Menor variación
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

# Reemplazar la clase UNet original por la nueva versión mejorada
class UNet(nn.Module):
    def __init__(self, input_ch=3, output_ch=1, base_ch=32, use_bn=True, dropout_p=0.5):
        """
        U-Net mejorada y configurable
        
        Args:
            input_ch (int): Canales de entrada (3 para RGB)
            output_ch (int): Canales de salida (1 para segmentación binaria)
            base_ch (int): Canales base (default: 32)
            use_bn (bool): Usar BatchNorm (True por defecto)
            dropout_p (float): Probabilidad de Dropout (0.5 por defecto)
        """
        super().__init__()
        
        def CBR(in_channels, out_channels):
            layers = [
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True)
            ]
            
            if use_bn:
                # Insertar BatchNorm después de cada Conv2d
                layers.insert(1, nn.BatchNorm2d(out_channels))
                layers.insert(4, nn.BatchNorm2d(out_channels))
            
            return nn.Sequential(*layers)
        
        # Encoder
        self.enc1 = CBR(input_ch, base_ch)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = CBR(base_ch, base_ch*2)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = CBR(base_ch*2, base_ch*4)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = CBR(base_ch*4, base_ch*8)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck con Dropout
        self.bottleneck = CBR(base_ch*8, base_ch*16)
        if dropout_p > 0:
            self.bottleneck = nn.Sequential(self.bottleneck, nn.Dropout(dropout_p))
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(base_ch*16, base_ch*8, 2, stride=2)
        self.dec4 = CBR(base_ch*16, base_ch*8)
        
        self.up3 = nn.ConvTranspose2d(base_ch*8, base_ch*4, 2, stride=2)
        self.dec3 = CBR(base_ch*8, base_ch*4)
        
        self.up2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, stride=2)
        self.dec2 = CBR(base_ch*4, base_ch*2)
        
        self.up1 = nn.ConvTranspose2d(base_ch*2, base_ch, 2, stride=2)
        self.dec1 = CBR(base_ch*2, base_ch)
        
        # Capa final
        self.final = nn.Conv2d(base_ch, output_ch, kernel_size=1)
        
        # Inicialización de pesos
        self._initialize_weights()
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool4(e4))
        
        # Decoder con skip connections
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        #return torch.sigmoid(self.final(d1))
        return self.final(d1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class VeinLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # Peso para pérdida estructural
        self.beta = beta    # Peso para pérdida de bordes
        self.gamma = gamma  # Factor focal
        
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.sobel = SobelFilter()
        
    def forward(self, preds, targets):
        # Pérdida estructural (Dice modificado)
        smooth = 1.0
        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (preds_flat * targets_flat).sum()
        structural_loss = 1 - (2. * intersection + smooth) / (preds_flat.sum() + targets_flat.sum() + smooth)
        
        # Pérdida de bordes (énfasis en venas)
        edge_targets = self.sobel(targets)
        edge_preds = self.sobel(preds)
        edge_loss = F.mse_loss(edge_preds, edge_targets)
        
        # Pérdida focal para clase minoritaria (venas)
        bce_loss = self.bce(preds, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = ((1 - pt) ** self.gamma * bce_loss).mean()
        
        return (self.alpha * structural_loss + 
                self.beta * edge_loss + 
                (1-self.alpha-self.beta) * focal_loss)

class SobelFilter(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel_x = torch.tensor([[[[1, 0, -1], [2, 0, -2], [1, 0, -1]]]]).float()
        self.kernel_y = torch.tensor([[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]]).float()
    
    def forward(self, x):
        # x: [B, 1, H, W]
        if x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)
            
        grad_x = F.conv2d(x, self.kernel_x.to(x.device), padding=1)
        grad_y = F.conv2d(x, self.kernel_y.to(x.device), padding=1)
        return torch.sqrt(grad_x**2 + grad_y**2)

# Dice + BCE Loss
class ImprovedDiceBCELoss(nn.Module):
    def __init__(self, pos_weight=10.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(DEVICE))
        
    def forward(self, preds, targets):
        smooth = 1.0
        preds = preds.view(-1)
        targets = targets.view(-1)
        
        # Calcular Dice
        intersection = (preds * targets).sum()
        dice_coeff = (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
        dice_loss = 1 - dice_coeff
        
        # Calcular BCE
        bce_loss = self.bce(preds, targets)
        
        # Pérdida combinada (puedes ajustar los pesos)
        total_loss = 0.5 * dice_loss + 0.5 * bce_loss
        
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
    input_ch=3,          # Canales de entrada (RGB)
    output_ch=1,         # Segmentación binaria
    base_ch=32,          # Canales base (puedes ajustar según necesidades)
    use_bn=True,         # Batch Normalization activado
    dropout_p=0.5        # Dropout para regularización
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
