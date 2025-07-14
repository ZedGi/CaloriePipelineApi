# pipeline.py
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class FoodSegmentationClassificationPipeline:
    """Pipeline complet YOLO segmentation + MobileNetV2 classification + estimation des calories"""

# pipeline.py
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class FoodSegmentationClassificationPipeline:
    """Pipeline complet YOLO segmentation + MobileNetV2 classification + estimation des calories"""

    def __init__(self, yolo_weights, mobilenet_weights, class_names_path, weight_csv, calorie_csv):
        # Détection de l'appareil (GPU ou CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Utilisation de: {self.device}")

        # Chargement du modèle YOLOv8 pour la segmentation
        logger.info("Chargement de YOLO...")
        self.yolo_model = YOLO(yolo_weights)

        # Chargement des noms de classes depuis le fichier texte
        with open(class_names_path, 'r', encoding='utf-8') as f:
            self.class_names = [line.strip() for line in f.readlines() if line.strip()]
        logger.info(f"Classes détectées: {len(self.class_names)} classes")

        # Chargement du modèle MobileNetV2 pour la classification
        logger.info("Chargement de MobileNetV2...")
        self.mobilenet = models.mobilenet_v2(weights=None)
        self.mobilenet.classifier[1] = nn.Linear(self.mobilenet.last_channel, len(self.class_names))
        self.mobilenet.load_state_dict(torch.load(mobilenet_weights, map_location=self.device))
        self.mobilenet.eval()
        self.mobilenet.to(self.device)

        # Transformation d'image pour MobileNetV2
        self.transform = transforms.Compose([
            transforms.Resize((288, 288)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Charger les données nutritionnelles
        self._load_nutrition_data(weight_csv, calorie_csv)

        # Transformation d'image pour MobileNetV2
        self.transform = transforms.Compose([
            transforms.Resize((288, 288)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Charger les données nutritionnelles
        self._load_nutrition_data(weight_csv, calorie_csv)

    def _load_nutrition_data(self, weight_csv, calorie_csv):
        """Charge les fichiers CSV de poids moyens et calories"""
        try:
            # Charger les CSV
            weight_table = pd.read_csv(weight_csv)
            calorie_table = pd.read_csv(calorie_csv)

            # Normaliser les colonnes
            weight_table.columns = weight_table.columns.str.strip().str.lower()
            calorie_table.columns = calorie_table.columns.str.strip().str.lower()

            # Créer les dictionnaires
            self.avg_weights = dict(zip(weight_table.iloc[:, 0], weight_table.iloc[:, 1]))
            self.calories_100g = dict(zip(calorie_table.iloc[:, 0], calorie_table.iloc[:, 1]))
            
            logger.info(f"✅ Données nutritionnelles chargées: {len(self.avg_weights)} aliments")
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement données nutritionnelles: {e}")
            self.avg_weights = {}
            self.calories_100g = {}

    def extract_segmented_crop(self, image, mask, bbox):
        """Extrait le crop de l'objet détecté en n'affichant que les pixels masqués"""
        x1, y1, x2, y2 = bbox
        crop = image[y1:y2, x1:x2]
        crop_mask = mask[y1:y2, x1:x2]
        result = np.zeros_like(crop)
        result[crop_mask > 0] = crop[crop_mask > 0]
        return result

    def classify_crop(self, crop_image):
        """Classifie un crop d'aliment avec MobileNetV2"""
        pil_image = Image.fromarray(cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB))
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.mobilenet(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
        predicted_class = self.class_names[predicted.item()]
        confidence_score = confidence.item()
        
        # Top 3 pour debug
        top3_prob, top3_idx = torch.topk(probabilities, 3)
        top3_classes = [(self.class_names[idx], prob.item()) for idx, prob in zip(top3_idx[0], top3_prob[0])]
        
        return predicted_class, confidence_score, top3_classes

    def estimate_calories(self, class_name):
        """Estime les calories d'un aliment en se basant sur sa classe"""
        weight = self.avg_weights.get(class_name, 100)  # Par défaut 100g
        cal_per_100g = self.calories_100g.get(class_name, 100)  # Par défaut 100 kcal/100g
        total_calories = (weight / 100.0) * cal_per_100g
        return total_calories, weight

    def process_image(self, image_array, conf_threshold=0.3):
        """Pipeline complet pour traiter une image numpy array"""
        # Application de YOLO
        results = self.yolo_model(image_array, conf=conf_threshold)[0]
        detections = []
        
        if results.masks is not None and results.boxes is not None:
            masks = results.masks.data.cpu().numpy()
            boxes = results.boxes.xyxy.cpu().numpy().astype(int)
            
            logger.info(f"Nombre d'aliments détectés par YOLO: {len(boxes)}")
            image_area = image_array.shape[0] * image_array.shape[1]

            for i, (mask, box) in enumerate(zip(masks, boxes)):
                x1, y1, x2, y2 = box
                box_area = (x2 - x1) * (y2 - y1)
                area_ratio = box_area / image_area

                # Filtre : ignorer les petites zones
                if area_ratio < 0.02:
                    continue

                mask_resized = cv2.resize(mask, (image_array.shape[1], image_array.shape[0]))
                segmented_crop = self.extract_segmented_crop(image_array, mask_resized, box)
                
                if np.sum(segmented_crop) == 0:
                    continue
                    
                predicted_class, confidence, top3 = self.classify_crop(segmented_crop)
                
                if confidence < 0.25:
                    continue

                estimated_cal, weight = self.estimate_calories(predicted_class)

                detections.append({
                    'bbox': box.tolist(),
                    'class_name': predicted_class,
                    'confidence': float(confidence),
                    'estimated_calories': float(estimated_cal),
                    'weight_grams': float(weight),
                    'top3': top3  # Pour debug
                })
                
                logger.info(f"  - {predicted_class}: {confidence:.2%} confiance, {estimated_cal:.0f} kcal")

        return detections