"""
This script demonstrates how to train a binary segmentation model using the
CamVid dataset and segmentation_models_pytorch. The CamVid dataset is a
collection of videos with pixel-level annotations for semantic segmentation.
The dataset includes 367 training images, 101 validation images, and 233 test.
Each training image has a corresponding mask that labels each pixel as belonging
to these classes with the numerical labels as follows:
- Sky: 0
- Building: 1
- Pole: 2
- Road: 3
- Pavement: 4
- Tree: 5
- SignSymbol: 6
- Fence: 7
- Car: 8
- Pedestrian: 9
- Bicyclist: 10
- Unlabelled: 11

In this script, we focus on binary segmentation, where the goal is to classify
each pixel as whether belonging to a certain class (Foregorund) or
not (Background).

Class Labels:
- 0: Background
- 1: Foreground

The script includes the following steps:
1. Set the device to GPU if available, otherwise use CPU.
2. Download the CamVid dataset if it is not already present.
3. Define hyperparameters for training.
4. Define a custom dataset class for loading and preprocessing the CamVid
     dataset.
5. Define a function to visualize images and masks.
6. Create datasets and dataloaders for training, validation, and testing.
7. Define a model class for the segmentation task.
8. Train the model using the training and validation datasets.
9. Evaluate the model using the test dataset and save the output masks and
     metrics.
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%d:%m:%Y %H:%M:%S",
)

# ----------------------------
# Set the device to GPU if available
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")
if device == "cpu":
    os.system("export OMP_NUM_THREADS=64")
    torch.set_num_threads(os.cpu_count())

# ----------------------------
# Define the hyperparameters
# ----------------------------
epochs_max = 250 # Number of epochs to train the model
adam_lr = 2e-4  # Learning rate for the Adam optimizer
eta_min = 1e-5  # Minimum learning rate for the scheduler
batch_size = 8  # Batch size for training
input_image_reshape = (640, 640)  # Desired shape for the input images and masks
foreground_class = 1  # 1 for binary segmentation

# Añade esto al inicio del script, antes de las definiciones de hiperparámetros
def parse_args():
    parser = argparse.ArgumentParser(description='Entrenamiento de modelo de segmentación')
    parser.add_argument('-arquitectura', '--arch', type=str, default='Unet', 
                        help='Arquitectura del modelo (e.g., Unet, FPN)')
    parser.add_argument('-encoder', '--encoder_name', type=str, default='resnet34', 
                        help='Nombre del encoder (e.g., resnet34, resnet50)')
    parser.add_argument('-loss', '--loss_fn', type=str, default='dice', 
                        help='Función de pérdida (e.g., dice, bce, focal)')
    parser.add_argument('-output', '--output_dir', type=str, default='output', 
                        help='Nombre del directorio donde almacenar los resultados')
    parser.add_argument('--data_split', type=str, default='../experiments/experiment_70_15_15',
                      help='Ruta a la división del dataset a usar')
    return parser.parse_args()

# ----------------------------
# Define a custom dataset class for the CamVid dataset
# ----------------------------
class Dataset(BaseDataset):
    """
    A custom dataset class for binary segmentation tasks.

    Parameters:
    ----------

    - images_dir (str): Directory containing the input images.
    - masks_dir (str): Directory containing the corresponding masks.
    - input_image_reshape (tuple, optional): Desired shape for the input
      images and masks. Default is (320, 320).
    - foreground_class (int, optional): The class value in the mask to be
      considered as the foreground. Default is 1.
    - augmentation (callable, optional): A function/transform to apply to the
      images and masks for data augmentation.
    """

    def __init__(
        self,
        images_dir,
        masks_dir,
        input_image_reshape=(320, 320),
        foreground_class=1,
        augmentation=None,
    ):
        self.images_ids = os.listdir(images_dir)
        self.masks_ids = os.listdir(masks_dir)
        self.images_filepaths = [
            os.path.join(images_dir, image_id) for image_id in self.images_ids
        ]
        self.masks_filepaths = [
            os.path.join(masks_dir, image_id) for image_id in self.masks_ids
        ]

        self.input_image_reshape = input_image_reshape
        self.foreground_class = foreground_class
        self.augmentation = augmentation

    def __getitem__(self, i):
        """
        Retrieves the image and corresponding mask at index `i`.

        Parameters:
        ----------

        - i (int): Index of the image and mask to retrieve.
        Returns:
        - A tuple containing:
            - image (torch.Tensor): The preprocessed image tensor of shape
            (1, input_image_reshape) - e.g., (1, 320, 320) - normalized to [0, 1].
            - mask_remap (torch.Tensor): The preprocessed mask tensor of
            shape input_image_reshape with values 0 or 1.
        """
        # Read the image
        image = cv2.imread(
            self.images_filepaths[i], cv2.IMREAD_GRAYSCALE
        )  # Read image as grayscale
        image = np.expand_dims(image, axis=-1)  # Add channel dimension

        # resize image to input_image_reshape
        image = cv2.resize(image, self.input_image_reshape)

        # Read the mask in grayscale mode
        mask = cv2.imread(self.masks_filepaths[i], 0)

        # Update the mask: Set foreground_class to 1 and the rest to 0
        mask_remap = np.where(mask == self.foreground_class, 1, 0).astype(np.uint8)

        # resize mask to input_image_reshape
        mask_remap = cv2.resize(mask_remap, self.input_image_reshape)

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask_remap)
            image, mask_remap = sample["image"], sample["mask"]

        # Convert to PyTorch tensors
        # Add channel dimension if missing
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)

        # HWC -> CHW and normalize to [0, 1]
        image = torch.tensor(image).float().permute(2, 0, 1) / 255.0

        # Ensure mask is LongTensor
        mask_remap = torch.tensor(mask_remap).long()

        return image, mask_remap

    def __len__(self):
        return len(self.images_ids)

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class PetalVeinDataset(Dataset):
    """
    Custom dataset class for petal vein segmentation with TIFF images and PNG masks.
    
    Parameters:
    ----------
    - images_dir (str): Directory containing the .tif petal images.
    - masks_dir (str): Directory containing the .png vein masks.
    - input_image_reshape (tuple, optional): Desired shape for resizing. Default (320, 320).
    - augmentation (callable, optional): Augmentation transforms to apply.
    - normalize (bool, optional): Whether to normalize images to [0,1]. Default True.
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
        
        # Validate that we found images
        if not self.ids:
            raise ValueError(f"No TIFF images found in {images_dir}")
        
        # Create full paths for images and corresponding masks
        self.images_filepaths = [
            os.path.join(images_dir, img_id) for img_id in self.ids
        ]
        
        # Masks are PNG files with same base name
        self.masks_filepaths = [
            os.path.join(masks_dir, img_id.replace('.tif', '.png').replace('.TIF', '.png'))
            for img_id in self.ids
        ]

        self.input_image_reshape = input_image_reshape
        self.augmentation = augmentation
        self.normalize = normalize
        
    def __getitem__(self, i):
        """
        Retrieves the petal image (TIFF) and corresponding vein mask (PNG).
        
        Returns:
        -------
        - image (torch.Tensor): Image tensor (3, H, W) normalized to [0,1] if normalize=True
        - mask (torch.Tensor): Binary mask tensor (H, W) with values 0 or 1
        """
        # Read TIFF image
        image = cv2.imread(self.images_filepaths[i], cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image at {self.images_filepaths[i]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        
        # Read PNG mask (grayscale)
        mask = cv2.imread(self.masks_filepaths[i], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not read mask at {self.masks_filepaths[i]}")
        
        # Convert mask to binary (veins=1, background=0)
        mask = (mask > 0).astype(np.uint8)
        
        # Resize both image and mask
        image = cv2.resize(image, self.input_image_reshape)
        mask = cv2.resize(mask, self.input_image_reshape, interpolation=cv2.INTER_NEAREST)
        
        # Apply augmentations if specified
        if self.augmentation:
            augmented = self.augmentation(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']
        
        # Convert to PyTorch tensors
        # Image: HWC -> CHW
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        if self.normalize:
            image = image / 255.0  # Normalize to [0,1]
        
        # Mask: ensure it's LongTensor
        mask = torch.from_numpy(mask).long()
        
        return image, mask
    
    def __len__(self):
        return len(self.ids)

# Transformaciones para imágenes y máscaras (sincronizadas)
augmentation_train = A.Compose([
    A.HorizontalFlip(p=0.5),  # Volteo horizontal con 50% de probabilidad
    A.VerticalFlip(p=0.5),    # Volteo vertical con 50% de probabilidad
    A.RandomRotate90(p=0.5),  # Rotación 90 grados
    A.RandomBrightnessContrast(p=0.2),  # Ajuste de brillo/contraste
])

# Transformaciones para validación/test (solo normalización)
augmentation_val_test = A.Compose([])  # Sin aumentos, solo paso de normalización

mask_augmentation = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
])

# Define a class for the CamVid model
class CamVidModel(torch.nn.Module):
    """
    A PyTorch model for binary segmentation using the Segmentation Models
    PyTorch library.

    Parameters:
    ----------

    - arch (str): The architecture name of the segmentation model
       (e.g., 'Unet', 'FPN').
    - encoder_name (str): The name of the encoder to use
       (e.g., 'resnet34', 'vgg16').
    - in_channels (int, optional): Number of input channels (e.g., 3 for RGB).
    - out_classes (int, optional): Number of output classes (e.g., 1 for binary)
    **kwargs: Additional keyword arguments to pass to the model
    creation function.
    """

    def __init__(self, arch, encoder_name, in_channels=3, out_classes=1, **kwargs):
        super().__init__()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        # Si tienes archivos guardados con tus estadísticos:
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
        # Normalize image
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

def visualize_samples(images, masks, output_dir, prefix="train", num_samples=3):
    """Visualiza y guarda muestras de imágenes y máscaras."""
    samples_dir = os.path.join(output_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    
    for i in range(min(num_samples, len(images))):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        # Mostrar imagen
        img = images[i].transpose(1, 2, 0) if images[i].shape[0] == 3 else images[i][0]
        ax1.imshow(img, cmap='gray')
        ax1.set_title(f'Imagen {i}')
        ax1.axis('off')
        
        # Mostrar máscara
        ax2.imshow(masks[i], cmap='gray')
        ax2.set_title(f'Máscara {i}')
        ax2.axis('off')
        
        plt.savefig(os.path.join(samples_dir, f"{prefix}_sample_{i}.png"))
        plt.close()

def compute_dataset_statistics(images_dir, input_shape=(640, 640), batch_size=32):
    """
    Calcula la media y desviación estándar por canal para un conjunto de imágenes.
    
    Args:
        images_dir (str): Directorio con las imágenes (.tif)
        input_shape (tuple): Tamaño al que se redimensionarán las imágenes
        batch_size (int): Tamaño del lote para el cálculo
        
    Returns:
        mean (np.array): Media por canal [R, G, B]
        std (np.array): Desviación estándar por canal [R, G, B]
    """
    # Obtener lista de imágenes
    image_paths = [os.path.join(images_dir, f) for f in os.listdir(images_dir) 
                  if f.lower().endswith('.tif')]
    
    # Variables para acumular estadísticos
    pixel_sum = np.zeros(3)
    pixel_sq_sum = np.zeros(3)
    num_pixels = 0
    
    # Procesar imágenes por lotes
    for i in tqdm(range(0, len(image_paths), batch_size)):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        
        for path in batch_paths:
            # Leer imagen y redimensionar
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir a RGB
            img = cv2.resize(img, input_shape)
            batch_images.append(img)
            
        batch_images = np.stack(batch_images)
        
        # Acumular estadísticos
        pixel_sum += np.sum(batch_images, axis=(0, 1, 2))
        pixel_sq_sum += np.sum(batch_images**2, axis=(0, 1, 2))
        num_pixels += batch_images.shape[0] * batch_images.shape[1] * batch_images.shape[2]
    
    # Calcular media y desviación estándar
    mean = pixel_sum / num_pixels
    std = np.sqrt((pixel_sq_sum / num_pixels) - mean**2)
    
    return mean, std

def visualize(output_dir, image_filename, **images):
    """Save each image separately without plotting."""
    os.makedirs(output_dir, exist_ok=True)

    for name, image in images.items():
        output_path = os.path.join(output_dir, f"{name}_{image_filename}")
        
        if name in ['output_mask', 'binary_mask']:
            # Máscara binaria (0-1) → Guardar como PNG (valores 0 y 255)
            mask = (image * 255).astype(np.uint8)  # Convertir a 0-255
            cv2.imwrite(output_path, mask)
        else:
            # Imagen de entrada (RGB o escala de grises)
            if image.ndim == 3 and image.shape[-1] in [3, 4]:
                plt.imsave(output_path, image)
            else:
                plt.imsave(output_path, image.squeeze(), cmap='gray')

# Use multiple CPUs in parallel
def train_and_evaluate_one_epoch(
    model, train_dataloader, valid_dataloader, optimizer, scheduler, loss_fn, device
):
    # Set the model to training mode
    model.train()
    train_loss = 0
    for batch in tqdm(train_dataloader, desc="Training"):
        images, masks = batch
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        # Asegurar que las máscaras tengan la forma correcta
        if isinstance(loss_fn, (torch.nn.BCEWithLogitsLoss, smp.losses.SoftBCEWithLogitsLoss)):
            masks = masks.unsqueeze(1).float()  # Añade dimensión de canal si es necesario
        else:
            masks = masks.float()

        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    scheduler.step()
    avg_train_loss = train_loss / len(train_dataloader)

    # Set the model to evaluation mode
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
            f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.2f}, Validation Loss: {avg_val_loss:.2f}"
        )
        # Early stopping logic
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # Guardar los pesos del mejor modelo
            best_model_weights = model.state_dict().copy()
            
            # Guardar el modelo si hay directorio de salida
            if output_dir:
                torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
        else:
            epochs_no_improve += 1
            logging.info(f"No improvement for {epochs_no_improve} epochs")
            
            if epochs_no_improve >= patience:
                logging.info(f"Early stopping triggered after {epoch + 1} epochs")
                # Cargar los mejores pesos antes de terminar
                if best_model_weights is not None:
                    model.load_state_dict(best_model_weights)
                break
    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
    }

    if output_dir:
        # Guardar gráficas de pérdida
        plt.figure()
        plt.plot(history['train_losses'], label='Train Loss')
        plt.plot(history['val_losses'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
        plt.close()
    
    if output_dir and args:
        args_dict = vars(args)
        with open(os.path.join(output_dir, 'config.json'), 'w') as f:
            json.dump(args_dict, f, indent=4)
    
    if output_dir:
        pd.DataFrame(history).to_csv(os.path.join(output_dir, "train_history.csv"), index=False)

    return history

def test_model(model, output_dir, test_dataloader, loss_fn, device):
    model.eval()
    test_loss = 0
    image_metrics = []
    all_outputs = []
    all_targets = []

    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="Evaluando")):
            images, masks = batch
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            # Aplicar sigmoid a las salidas para obtener probabilidades
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()  # Umbralizar a 0.5

            # Preparar máscaras para cálculo de pérdida
            if isinstance(loss_fn, (torch.nn.BCEWithLogitsLoss, smp.losses.SoftBCEWithLogitsLoss)):
                loss_input = outputs
                target = masks.unsqueeze(1).float()  # Añadir dimensión de canal
            else:
                loss_input = probs  # Usar probabilidades ya normalizadas
                target = masks.float()  # Mantener forma (B, H, W)

            loss = loss_fn(loss_input, target)
            test_loss += loss.item()

            # Calcular métricas para cada imagen en el batch
            for i in range(images.shape[0]):
                # Convertir a numpy para cálculo de métricas
                pred_mask = preds[i].squeeze().cpu().numpy()
                true_mask = masks[i].squeeze().cpu().numpy()

                # Calcular métricas
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

    # Calcular métricas agregadas
    test_loss_mean = test_loss / len(test_dataloader)
    
    aggregate_metrics = {
        "test_loss": test_loss_mean,
        "iou_score": np.mean([m["iou_score"] for m in image_metrics]),
        "precision": np.mean([m["precision"] for m in image_metrics]),
        "recall": np.mean([m["recall"] for m in image_metrics]),
        "f1_score": np.mean([m["f1_score"] for m in image_metrics]),
    }

    # Guardar métricas
    pd.DataFrame(image_metrics).to_csv(os.path.join(output_dir, "individual_metrics.csv"), index=False)
    pd.DataFrame([aggregate_metrics]).to_csv(os.path.join(output_dir, "aggregate_metrics.csv"), index=False)

    return aggregate_metrics

# ----------------------------
# Create and train the model
# ----------------------------
max_iter = epochs_max

args = parse_args()

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

test_losses = []
iou_scores = []
all_metrics = []

# Configura la función de pérdida según el argumento
if args.loss_fn.lower() == 'bce':
    loss_fn = smp.losses.SoftBCEWithLogitsLoss()
elif args.loss_fn.lower() == 'focal':
    loss_fn = smp.losses.FocalLoss(smp.losses.BINARY_MODE)
else:  # default es Dice
    loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
torch.cuda.empty_cache()

print(f"\n=== Entrenamiento ===")

# Configuración de paths usando el argumento data_split
train_images_dir = os.path.join(args.data_split, 'train', 'images')
train_masks_dir = os.path.join(args.data_split, 'train', 'masks')
val_images_dir = os.path.join(args.data_split, 'val', 'images')
val_masks_dir = os.path.join(args.data_split, 'val', 'masks')
test_images_dir = os.path.join(args.data_split, 'test', 'images')
test_masks_dir = os.path.join(args.data_split, 'test', 'masks')

# Crear DataLoaders (igual que antes)
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

# Reiniciar el modelo y optimizador para cada entrenamiento
model = CamVidModel(args.arch, args.encoder_name, in_channels=3, out_classes=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=adam_lr)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iter, eta_min=eta_min)

# Calcular estadísticos del conjunto de entrenamiento
train_mean, train_std = compute_dataset_statistics(train_images_dir)

print(f"Media del dataset: {train_mean}")
print(f"Desviación estándar: {train_std}")

# Guardar estos valores para uso futuro
np.save(os.path.join(output_dir, 'dataset_mean.npy'), train_mean)
np.save(os.path.join(output_dir, 'dataset_std.npy'), train_std)

# Entrenar
history = train_model(
    model, 
    train_loader, 
    valid_loader, 
    optimizer, 
    scheduler, 
    loss_fn, 
    device, 
    epochs_max,
    output_dir=args.output_dir,  # Pasa el directorio de salida
    patience=early_stop_patience,
    min_delta=early_stop_min_delta,
    args=args
)

# Evaluar
metrics = test_model(
    model, 
    args.output_dir, 
    test_loader, 
    loss_fn, 
    device
)

all_metrics.append(metrics)

# Convertir a DataFrame y guardar en CSV
df_metrics = pd.DataFrame(all_metrics)
df_metrics.to_csv(f"{output_dir}/metricas_detalladas.csv", index=False)
