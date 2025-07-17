import os
from glob import glob
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2

# Modelo UNet
import torch.nn as nn
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

# Dataset para visualizar las imágenes de test
class PetalVeinDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir, test_files, transform=None):
        self.images = []
        self.masks = []
        self.transform = transform

        # Cargar solo los archivos listados en test_files.txt
        with open(test_files, 'r') as f:
            test_names = [line.strip() for line in f]

        for name in test_names:
            img_path = os.path.join(image_dir, name)
            mask_name = name.replace('.tif', '.png')
            mask_path = os.path.join(mask_dir, mask_name)
            
            if os.path.exists(img_path) and os.path.exists(mask_path):
                self.images.append(img_path)
                self.masks.append(mask_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask, os.path.basename(img_path)

# Configuración
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_DIR = '../database_petals'
MASK_DIR = '../masks_petals'
MODEL_PATH = 'unet_petals.pth'
OUTPUT_DIR = './resultados_predicciones'
FILES_TO_USE = 'test_files.txt'

os.makedirs(OUTPUT_DIR, exist_ok=True)

transform = T.Compose([
    T.Resize((320, 448)),
    T.ToTensor()
])

# Modelo
model = UNet(
    in_channels=3,          # Canales de entrada (RGB)
    out_channels=1,         # Segmentación binaria
).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Dataset
dataset = PetalVeinDataset(IMAGE_DIR, MASK_DIR, FILES_TO_USE, transform)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

def postprocess_veins(pred, threshold=0.7, min_vein_width=2):
    """
    pred: Tensor [1, H, W] con valores 0-1
    """
    # Convertir a numpy
    pred_np = pred.squeeze().cpu().numpy()
    
    # Umbralización adaptativa
    binary = cv2.adaptiveThreshold(
        (pred_np*255).astype(np.uint8), 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        101, 
        -20
    )
    
    # Operaciones morfológicas
    kernel = np.ones((min_vein_width, min_vein_width), np.uint8)
    processed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Eliminar pequeños componentes
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        processed, connectivity=8)
    
    # Mantener solo componentes con cierta longitud
    result = np.zeros_like(processed)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] > 10:  # Área mínima
            result[labels == i] = 255
    
    return torch.from_numpy(result/255.0).float()

# Guardar predicciones
def save_predictions():
    with torch.no_grad():
        for img, mask, name in loader:
            img, mask = img.to(DEVICE), mask.to(DEVICE)
            pred = model(img)
            pred_bin = (pred > 0.5).float()

            img_np = img.squeeze().permute(1, 2, 0).cpu().numpy() * 255
            mask_np = mask.squeeze().cpu().numpy() * 255
            pred_np = pred_bin.squeeze().cpu().numpy() * 255

            # Convertir a imagen
            img_pil = Image.fromarray(img_np.astype(np.uint8))
            mask_pil = Image.fromarray(mask_np.astype(np.uint8), mode='L')
            pred_pil = Image.fromarray(pred_np.astype(np.uint8), mode='L')

            base_name = os.path.splitext(name[0])[0]
            img_pil.save(os.path.join(OUTPUT_DIR, f"{base_name}_input.png"))
            mask_pil.save(os.path.join(OUTPUT_DIR, f"{base_name}_gt.png"))
            pred_pil.save(os.path.join(OUTPUT_DIR, f"{base_name}_pred.png"))

            print(f"Guardadas predicciones para {name[0]}")

# Guardar las imagenes
save_predictions()

