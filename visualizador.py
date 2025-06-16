import os
from glob import glob
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from PIL import Image
import numpy as np

# Modelo UNet (igual que antes)
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
        self.enc1 = CBR(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = CBR(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = CBR(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = CBR(256, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = CBR(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = CBR(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = CBR(128, 64)
        self.final = nn.Conv2d(64, 1, kernel_size=1)
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.final(d1))

# Dataset
class PetalVeinDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.images = sorted(glob(os.path.join(image_dir, '*.tif')))
        self.masks = sorted(glob(os.path.join(mask_dir, '*.png')))
        self.transform = transform
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
dataset = PetalVeinDataset(IMAGE_DIR, MASK_DIR, transform)
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

save_predictions()

