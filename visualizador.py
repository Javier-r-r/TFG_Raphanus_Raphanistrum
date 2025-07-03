import os
from glob import glob
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from PIL import Image
import numpy as np

# Modelo UNet
import torch.nn as nn
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
model = UNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Dataset
dataset = PetalVeinDataset(IMAGE_DIR, MASK_DIR, FILES_TO_USE, transform)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

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

