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
# Download the CamVid dataset, if needed
# ----------------------------
# Change this to your desired directory
#main_dir = "./examples/binary_segmentation_data/"

#data_dir = os.path.join(
#if not os.path.exists(data_dir):
    #logging.info("Loading data...")
    #os.system(f"git clone https://github.com/alexgkendall/SegNet-Tutorial {data_dir}")
    #logging.info("Done!")

# ----------------------------
# Define the hyperparameters
# ----------------------------
epochs_max = 250 # Number of epochs to train the model
adam_lr = 2e-4  # Learning rate for the Adam optimizer
eta_min = 1e-5  # Minimum learning rate for the scheduler
batch_size = 8  # Batch size for training
input_image_reshape = (320, 320)  # Desired shape for the input images and masks
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
    return parser.parse_args()

# ----------------------------
# Define a custom dataset class for the CamVid dataset
# ----------------------------
# class Dataset(BaseDataset):
#     """
#     A custom dataset class for binary segmentation tasks.

#     Parameters:
#     ----------

#     - images_dir (str): Directory containing the input images.
#     - masks_dir (str): Directory containing the corresponding masks.
#     - input_image_reshape (tuple, optional): Desired shape for the input
#       images and masks. Default is (320, 320).
#     - foreground_class (int, optional): The class value in the mask to be
#       considered as the foreground. Default is 1.
#     - augmentation (callable, optional): A function/transform to apply to the
#       images and masks for data augmentation.
#     """

#     def __init__(
#         self,
#         images_dir,
#         masks_dir,
#         input_image_reshape=(320, 320),
#         foreground_class=1,
#         augmentation=None,
#     ):
#         self.ids = os.listdir(images_dir)
#         self.images_filepaths = [
#             os.path.join(images_dir, image_id) for image_id in self.ids
#         ]
#         self.masks_filepaths = [
#             os.path.join(masks_dir, image_id) for image_id in self.ids
#         ]

#         self.input_image_reshape = input_image_reshape
#         self.foreground_class = foreground_class
#         self.augmentation = augmentation

#     def __getitem__(self, i):
#         """
#         Retrieves the image and corresponding mask at index `i`.

#         Parameters:
#         ----------

#         - i (int): Index of the image and mask to retrieve.
#         Returns:
#         - A tuple containing:
#             - image (torch.Tensor): The preprocessed image tensor of shape
#             (1, input_image_reshape) - e.g., (1, 320, 320) - normalized to [0, 1].
#             - mask_remap (torch.Tensor): The preprocessed mask tensor of
#             shape input_image_reshape with values 0 or 1.
#         """
#         # Read the image
#         image = cv2.imread(
#             self.images_filepaths[i], cv2.IMREAD_GRAYSCALE
#         )  # Read image as grayscale
#         image = np.expand_dims(image, axis=-1)  # Add channel dimension

#         # resize image to input_image_reshape
#         image = cv2.resize(image, self.input_image_reshape)

#         # Read the mask in grayscale mode
#         mask = cv2.imread(self.masks_filepaths[i], 0)

#         # Update the mask: Set foreground_class to 1 and the rest to 0
#         mask_remap = np.where(mask == self.foreground_class, 1, 0).astype(np.uint8)

#         # resize mask to input_image_reshape
#         mask_remap = cv2.resize(mask_remap, self.input_image_reshape)

#         if self.augmentation:
#             sample = self.augmentation(image=image, mask=mask_remap)
#             image, mask_remap = sample["image"], sample["mask"]

#         # Convert to PyTorch tensors
#         # Add channel dimension if missing
#         if image.ndim == 2:
#             image = np.expand_dims(image, axis=-1)

#         # HWC -> CHW and normalize to [0, 1]
#         image = torch.tensor(image).float().permute(2, 0, 1) / 255.0

#         # Ensure mask is LongTensor
#         mask_remap = torch.tensor(mask_remap).long()

#         return image, mask_remap

#     def __len__(self):
#         return len(self.ids)


# Transformaciones para imágenes y máscaras (sincronizadas)
augmentation_train = A.Compose([
    A.HorizontalFlip(p=0.5),  # Volteo horizontal con 50% de probabilidad
    A.VerticalFlip(p=0.5),    # Volteo vertical con 50% de probabilidad
    A.RandomRotate90(p=0.5),  # Rotación 90 grados
    A.GaussianBlur(p=0.3),    # Desenfoque gaussiano
    A.RandomBrightnessContrast(p=0.2),  # Ajuste de brillo/contraste
    # Añade más transformaciones según necesites
])

# Transformaciones para validación/test (solo normalización)
augmentation_val_test = A.Compose([])  # Sin aumentos, solo paso de normalización

class CustomDatasetFromArrays(BaseDataset):
    def __init__(self, X, y, augmentation=None):
        self.X = X
        self.y = y
        self.augmentation = augmentation  # Objeto albumentations.Compose

    def __getitem__(self, i):
        image = self.X[i]
        mask = self.y[i].squeeze()  # Asegura máscara con forma (H, W)

        if self.augmentation:
            transformed = self.augmentation(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        # Convertir a tensores y normalizar
        image = torch.tensor(image).float().permute(2, 0, 1) / 255.0  # CHW, [0, 1]
        mask = torch.tensor(mask).long()

        return image, mask

    def __len__(self):
        return len(self.X)

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


def visualize(output_dir, image_filename, **images):
    """Save each image separately without plotting."""
    os.makedirs(output_dir, exist_ok=True)
    
    for name, image in images.items():
        # Guarda cada imagen individualmente
        output_path = os.path.join(output_dir, f"{name}_{image_filename}")
        plt.imsave(output_path, image)

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
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)  # Añade dimensión de canal si es necesario

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
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)
            masks = masks.float()
            loss = loss_fn(outputs, masks)

            val_loss += loss.item()

    avg_val_loss = val_loss / len(valid_dataloader)
    return avg_train_loss, avg_val_loss


def train_model(
    model,
    train_dataloader,
    valid_dataloader,
    optimizer,
    scheduler,
    loss_fn,
    device,
    epochs,
):
    train_losses = []
    val_losses = []

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

    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
    return history


def test_model(model, output_dir, test_dataloader, loss_fn, device):
    # Set the model to evaluation mode
    model.eval()
    test_loss = 0
    tp, fp, fn, tn = 0, 0, 0, 0
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            images, masks = batch
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, masks)
            test_loss += loss.item()

            prob_mask = outputs.sigmoid().squeeze(1)
            pred_mask = (prob_mask > 0.5).long()  # Umbral 0.5 para clasificación binaria

            batch_tp, batch_fp, batch_fn, batch_tn = smp.metrics.get_stats(
                pred_mask, masks, mode="binary"
            )
            tp += batch_tp.sum().item()
            fp += batch_fp.sum().item()
            fn += batch_fn.sum().item()
            tn += batch_tn.sum().item()

            for i, output in enumerate(outputs):
                input = images[i].cpu().numpy().transpose(1, 2, 0)
                output = output.squeeze().cpu().numpy()

                visualize(
                    output_dir,
                    f"output_{i}.png",
                    input_image=input,
                    output_mask=output,
                    binary_mask=output > 0.5,
                )

        test_loss_mean = test_loss / len(test_dataloader)
        # Calcular métricas adicionales
        precision = tp / (tp + fp + 1e-10)  # Evitar división por cero
        recall = tp / (tp + fn + 1e-10)     # Sensibilidad (Recall)
        specificity = tn / (tn + fp + 1e-10) # Especificidad
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)

        logging.info(f"Test Loss: {test_loss_mean:.2f}")

    iou_score = smp.metrics.iou_score(
        torch.tensor([tp]),
        torch.tensor([fp]),
        torch.tensor([fn]),
        torch.tensor([tn]),
        reduction="micro",
    )

        # Devolver métricas en un diccionario
    metrics = {
        "test_loss": test_loss_mean,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": f1_score,
        "iou_score": iou_score,
    }

    return metrics


# ----------------------------
# Define the data directories and create the datasets
# ----------------------------
# x_train_dir = os.path.join(data_dir, "CamVid", "train")
# y_train_dir = os.path.join(data_dir, "CamVid", "trainannot")

# x_val_dir = os.path.join(data_dir, "CamVid", "val")
# y_val_dir = os.path.join(data_dir, "CamVid", "valannot")

# x_test_dir = os.path.join(data_dir, "CamVid", "test")
# y_test_dir = os.path.join(data_dir, "CamVid", "testannot")

# train_dataset = Dataset(
#     x_train_dir,
#     y_train_dir,
#     input_image_reshape=input_image_reshape,
#     foreground_class=foreground_class,
# )
# valid_dataset = Dataset(
#     x_val_dir,
#     y_val_dir,
#     input_image_reshape=input_image_reshape,
#     foreground_class=foreground_class,
# )
# test_dataset = Dataset(
#     x_test_dir,
#     y_test_dir,
#     input_image_reshape=input_image_reshape,
#     foreground_class=foreground_class,
# )

# image, mask = train_dataset[0]
# logging.info(f"Unique values in mask: {np.unique(mask)}")
# logging.info(f"Image shape: {image.shape}")
# logging.info(f"Mask shape: {mask.shape}")

# ----------------------------
# Cargar los conjuntos pre-generados (tu código)
# ----------------------------
conjuntos = []
for i in range(3):
    conjuntos.append((
        np.load(f"X_train_set{i+1}.npy"),
        np.load(f"X_val_set{i+1}.npy"),
        np.load(f"X_test_set{i+1}.npy"),
        np.load(f"y_train_set{i+1}.npy"),
        np.load(f"y_val_set{i+1}.npy"),
        np.load(f"y_test_set{i+1}.npy")
    ))
# ----------------------------
# Create the dataloaders using the datasets
# ----------------------------
# logging.info(f"Train size: {len(train_dataset)}")
# logging.info(f"Valid size: {len(valid_dataset)}")
# logging.info(f"Test size: {len(test_dataset)}")

# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
# test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ----------------------------
# Lets look at some samples
# ----------------------------
# Visualize and save train sample
# sample = train_dataset[0]
# visualize(
#     output_dir,
#     "train_sample.png",
#     train_image=sample[0].numpy().transpose(1, 2, 0),
#     train_mask=sample[1].squeeze(),
# )

# # Visualize and save validation sample
# sample = valid_dataset[0]
# visualize(
#     output_dir,
#     "validation_sample.png",
#     validation_image=sample[0].numpy().transpose(1, 2, 0),
#     validation_mask=sample[1].squeeze(),
# )

# # Visualize and save test sample
# sample = test_dataset[0]
# visualize(
#     output_dir,
#     "test_sample.png",
#     test_image=sample[0].numpy().transpose(1, 2, 0),
#     test_mask=sample[1].squeeze(),
# )

# ----------------------------
# Create and train the model
# ----------------------------
max_iter = epochs_max

args = parse_args()

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

model = CamVidModel(args.arch, args.encoder_name, in_channels=3, out_classes=1)

# Training loop
model = model.to(device)

# Define the Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=adam_lr)

# Define the learning rate scheduler
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iter, eta_min=eta_min)

# Train the model
# history = train_model(
#     model,
#     train_loader,
#     valid_loader,
#     optimizer,
#     scheduler,
#     loss_fn,
#     device,
#     epochs_max,
# )

# Visualize the training and validation losses
# plt.figure(figsize=(10, 5))
# plt.plot(history["train_losses"], label="Train Loss")
# plt.plot(history["val_losses"], label="Validation Loss")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.title("Training and Validation Losses")
# plt.legend()
# plt.savefig(os.path.join(output_dir, "train_val_losses.png"))
# plt.close()
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

for i in range(3):
    print(f"\n=== Entrenamiento {i+1} ===")
    
    # Cargar el conjunto correspondiente desde archivo .npz
    conjunto = np.load(f"conjunto_{i+1}.npz")
    
    # Extraer los arrays
    X_train = conjunto['X_train']
    X_val = conjunto['X_val']
    X_test = conjunto['X_test']
    y_train = conjunto['y_train']
    y_val = conjunto['y_val']
    y_test = conjunto['y_test']
    
    # Verificar formas (opcional, para debug)
    print(f"Formas - Train: {X_train.shape}, {y_train.shape}")
    print(f"Formas - Val: {X_val.shape}, {y_val.shape}")
    print(f"Formas - Test: {X_test.shape}, {y_test.shape}")

    # Crear DataLoaders (igual que antes)
    train_dataset = CustomDatasetFromArrays(
        X_train, y_train, augmentation=augmentation_train
    )
    val_dataset = CustomDatasetFromArrays(
        X_val, y_val, augmentation=augmentation_val_test
    )
    test_dataset = CustomDatasetFromArrays(
        X_test, y_test, augmentation=augmentation_val_test
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Reiniciar el modelo y optimizador para cada entrenamiento
    model = CamVidModel(args.arch, args.encoder_name, in_channels=3, out_classes=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=adam_lr)

    # Entrenar
    history = train_model(model, train_loader, valid_loader, optimizer, scheduler, loss_fn, device, epochs_max)
    
    # Evaluar
    metrics = test_model(model, f"{output_dir}/exp_{i+1}", test_loader, loss_fn, device)
    print(f"Test Loss (Conjunto {i+1}): {metrics['test_loss']:.4f}, IoU: {metrics['iou_score']:.4f}")

    test_losses.append(metrics['test_loss'])
    iou_scores.append(metrics['iou_score'])
    all_metrics.append(metrics)
    

# Convertir a DataFrame y guardar en CSV
df_metrics = pd.DataFrame(all_metrics)
df_metrics.to_csv(f"{output_dir}/metricas_detalladas.csv", index=False)

# Convertir a arrays de NumPy
test_losses = np.array(test_losses)
iou_scores = np.array(iou_scores)

# Calcular estadísticas
mean_loss = np.mean(test_losses)
std_loss = np.std(test_losses)
mean_iou = np.mean(iou_scores)
std_iou = np.std(iou_scores)

print("\n=== Resumen Estadístico ===")
print(f"Loss en Test: Media = {mean_loss:.4f}, Desviación Típica = {std_loss:.4f}")
print(f"IoU: Media = {mean_iou:.4f}, Desviación Típica = {std_iou:.4f}")

df = pd.DataFrame({
    "Entrenamiento": [1, 2, 3],
    "Test_Loss": test_losses,
    "IoU": iou_scores
})
df.to_csv("{args.output_dir}/metricas_entrenamientos.csv", index=False)

# logging.info(f"Test Loss: {test_loss[0]}, IoU Score: {test_loss[1]}")
# logging.info(f"The output masks are saved in {output_dir}.")
