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

        return torch.sigmoid(self.final(d1))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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
    input_ch=3,          # Canales de entrada (RGB)
    output_ch=1,         # Segmentación binaria
    base_ch=32,          # Canales base (puedes ajustar según necesidades)
    use_bn=True,         # Batch Normalization activado
    dropout_p=0.5        # Dropout para regularización
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
            
            # Aplicar postprocesamiento específico para venas
            pred_bin = postprocess_veins(pred)  # Usamos nuestra función de postprocesamiento
            
            # Convertir tensores a numpy arrays
            img_np = img.squeeze().permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)  # Escalar a 0-255
            
            mask_np = mask.squeeze().cpu().numpy()
            mask_np = (mask_np * 255).astype(np.uint8)  # Las máscaras ya deberían estar binarizadas
            
            pred_np = pred_bin.squeeze().cpu().numpy()
            pred_np = (pred_np * 255).astype(np.uint8)  # Nuestro postprocesamiento ya devuelve 0/1

            # Crear imágenes PIL
            img_pil = Image.fromarray(img_np)
            
            # Máscara de verdad terreno (ground truth)
            mask_pil = Image.fromarray(mask_np, mode='L')
            
            # Predicción postprocesada
            pred_pil = Image.fromarray(pred_np, mode='L')

            # Guardar con nombres descriptivos
            base_name = os.path.splitext(name[0])[0]
            img_pil.save(os.path.join(OUTPUT_DIR, f"{base_name}_input.png"))
            mask_pil.save(os.path.join(OUTPUT_DIR, f"{base_name}_gt.png"))
            pred_pil.save(os.path.join(OUTPUT_DIR, f"{base_name}_pred.png"))

            # Opcional: Guardar una versión superpuesta para visualización
            overlay = Image.blend(
                img_pil.convert("RGBA"),
                Image.fromarray(np.stack([pred_np]*3, axis=-1)).convert("RGBA"),
                alpha=0.4
            )
            overlay.save(os.path.join(OUTPUT_DIR, f"{base_name}_overlay.png"))

            print(f"Imágenes guardadas para {name[0]}")

# Guardar las imagenes
save_predictions()

