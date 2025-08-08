import os
import argparse
import torch
import cv2
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp

class RealTimeSegmenter:
    def __init__(self, arch='Unet', encoder='resnet34', loss='dice', experiments_dir='experiment_results'):
        """
        Inicializa el segmentador en tiempo real.
        
        Args:
            arch (str): Arquitectura del modelo (Unet, FPN, PSPNet, DeepLabV3)
            encoder (str): Nombre del encoder (resnet34, resnet50, etc.)
            loss (str): Función de pérdida (dice, bce, focal)
            experiments_dir (str): Directorio base donde se guardaron los modelos
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.arch = arch
        self.encoder = encoder
        self.loss = loss
        
        # Buscar automáticamente la ruta al modelo
        self.model_dir = self._find_model_dir(experiments_dir)
        if not self.model_dir:
            raise FileNotFoundError(f"No se encontró modelo para {arch}/{encoder}/{loss}")
        
        self.model_path = os.path.join(self.model_dir, 'best_model.pth')
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"No se encontró el modelo en {self.model_path}")
        
        # Inicializar modelo
        self.model = self._load_model()
        print(f"Modelo cargado: {self.model_path}")

    def _find_model_dir(self, base_dir):
        """Busca recursivamente el directorio que coincida con los parámetros."""
        for root, dirs, files in os.walk(base_dir):
            for dir_name in dirs:
                if (self.arch in dir_name and 
                    self.encoder in dir_name and 
                    self.loss in dir_name):
                    return os.path.join(root, dir_name)
        return None

    def _load_model(self):
        """Carga el modelo con los pesos entrenados."""
        # Cargar estadísticos de normalización si existen
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        
        mean_file = os.path.join(self.model_dir, 'dataset_mean.npy')
        std_file = os.path.join(self.model_dir, 'dataset_std.npy')
        
        if os.path.exists(mean_file):
            mean = torch.tensor(np.load(mean_file)).view(1, 3, 1, 1).to(self.device)
        if os.path.exists(std_file):
            std = torch.tensor(np.load(std_file)).view(1, 3, 1, 1).to(self.device)
        
        self.mean = mean
        self.std = std
        
        # Crear modelo
        model = smp.create_model(
            self.arch,
            encoder_name=self.encoder,
            in_channels=3,
            classes=1,
        ).to(self.device)
        
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.eval()
        return model

    def process_frame(self, frame):
        """
        Procesa un frame (numpy array BGR) y devuelve la máscara segmentada.
        
        Args:
            frame (np.array): Imagen en formato BGR (OpenCV)
            
        Returns:
            np.array: Máscara binaria (0-255) con la segmentación
        """
        # Preprocesamiento
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (640, 640))
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        image_tensor = (image_tensor.unsqueeze(0).to(self.device) - self.mean) / self.std
        
        # Predicción
        with torch.no_grad():
            output = self.model(image_tensor)
            probs = torch.sigmoid(output)
            mask = (probs > 0.5).float().cpu().numpy().squeeze()
        
        # Postprocesamiento
        mask = (mask * 255).astype(np.uint8)
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        return mask

    def process_image_file(self, image_path):
        """Procesa un archivo de imagen y devuelve la máscara."""
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"No se pudo leer la imagen: {image_path}")
        return self.process_frame(frame)

    def run_realtime(self, source=0):
        """
        Ejecuta el segmentador en tiempo real desde una fuente de video.
        
        Args:
            source: Puede ser un número (webcam), archivo de video o IP cam
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la fuente de video {source}")

        print("\nModo tiempo real activado. Presiona:")
        print(" - 'q' para salir")
        print(" - 's' para guardar el frame actual\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Procesar frame
            mask = self.process_frame(frame)
            
            # Mostrar resultados
            display = cv2.hconcat([
                frame,
                cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            ])
            
            cv2.imshow('Segmentación en tiempo real', display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self._save_result(frame, mask)
        
        cap.release()
        cv2.destroyAllWindows()

    def _save_result(self, frame, mask):
        """Guarda el frame y la máscara actual."""
        os.makedirs('output', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Guardar frame original
        frame_path = f"output/{timestamp}_frame.png"
        cv2.imwrite(frame_path, frame)
        
        # Guardar máscara
        mask_path = f"output/{timestamp}_mask.png"
        cv2.imwrite(mask_path, mask)
        
        print(f"Resultados guardados en {frame_path} y {mask_path}")

def main():
    parser = argparse.ArgumentParser(description='Segmentador de venas en pétalos en tiempo real')
    parser.add_argument('-a', '--arch', default='Unet', help='Arquitectura del modelo')
    parser.add_argument('-e', '--encoder', default='resnet34', help='Encoder usado')
    parser.add_argument('-l', '--loss', default='dice', help='Función de pérdida')
    parser.add_argument('-i', '--input', help='Ruta a imagen para procesar (opcional)')
    parser.add_argument('-c', '--camera', type=int, default=0, 
                       help='Índice de cámara para tiempo real (0 por defecto)')
    args = parser.parse_args()

    # Inicializar segmentador
    segmenter = RealTimeSegmenter(arch=args.arch, encoder=args.encoder, loss=args.loss)

    if args.input:
        # Procesar imagen única
        mask = segmenter.process_image_file(args.input)
        
        # Mostrar resultados
        frame = cv2.imread(args.input)
        display = cv2.hconcat([
            frame,
            cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        ])
        
        cv2.imshow('Resultado de segmentación', display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Guardar resultados
        filename = os.path.splitext(os.path.basename(args.input))[0]
        cv2.imwrite(f"{filename}_mask.png", mask)
        print(f"Máscara guardada como {filename}_mask.png")
    else:
        # Modo tiempo real
        segmenter.run_realtime(source=args.camera)

if __name__ == "__main__":
    from datetime import datetime
    main()
