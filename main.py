# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import logging
from typing import List
from pydantic import BaseModel
import uvicorn
import os
import time

# Import du pipeline
from pipeline import FoodSegmentationClassificationPipeline

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modèles Pydantic pour les réponses
class FoodDetection(BaseModel):
    bbox: List[int]
    class_name: str
    confidence: float
    estimated_calories: float
    weight_grams: float

class DetectionResponse(BaseModel):
    success: bool
    total_calories: float
    detections: List[FoodDetection]
    processing_time_ms: float
    message: str = ""

# Initialisation de l'API
app = FastAPI(
    title="Food Detection API",
    description="API pour la détection et l'estimation des calories des aliments",
    version="1.0.0"
)

# Configuration CORS pour Flutter
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Pour le développement. En production, mettez l'URL de votre app
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variable globale pour le pipeline
pipeline = None

# Configuration des chemins
YOLO_WEIGHTS = "model_weights/best.pt"
MOBILENET_WEIGHTS = "model_weights/mobilenetv2_best.pth"
CLASS_NAMES_FILE = "assets/food_labels.txt"
WEIGHT_CSV = "assets/average_weight.csv"
CALORIE_CSV = "assets/calorie_per_100g.csv"

@app.on_event("startup")
async def startup_event():
    """Initialise le pipeline au démarrage"""
    global pipeline
    
    try:
        logger.info("Initialisation du pipeline...")
        
        # Vérifier que les fichiers existent
        required_files = [
            YOLO_WEIGHTS, MOBILENET_WEIGHTS, 
            CLASS_NAMES_FILE, WEIGHT_CSV, CALORIE_CSV
        ]
        
        for file in required_files:
            if not os.path.exists(file):
                logger.error(f"Fichier manquant: {file}")
                raise FileNotFoundError(f"Fichier requis manquant: {file}")
        
        # Initialiser le pipeline
        pipeline = FoodSegmentationClassificationPipeline(
            yolo_weights=YOLO_WEIGHTS,
            mobilenet_weights=MOBILENET_WEIGHTS,
            class_names_path=CLASS_NAMES_FILE,
            weight_csv=WEIGHT_CSV,
            calorie_csv=CALORIE_CSV
        )
        
        logger.info("✅ Pipeline initialisé avec succès")
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'initialisation: {e}")
        # Ne pas faire crasher l'API, mais le pipeline ne sera pas disponible

@app.get("/")
async def root():
    """Point d'entrée de l'API"""
    return {
        "message": "Food Detection API",
        "status": "ready" if pipeline else "not_initialized",
        "endpoints": {
            "health": "/health",
            "detect": "/detect",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Vérification de l'état de l'API"""
    return {
        "status": "healthy",
        "pipeline_loaded": pipeline is not None,
        "device": str(pipeline.device) if pipeline else None
    }

@app.post("/detect", response_model=DetectionResponse)
async def detect_food(file: UploadFile = File(...)):
    """
    Détecte les aliments dans une image et estime les calories
    
    Args:
        file: Image uploadée (JPEG, PNG)
    
    Returns:
        DetectionResponse avec les détections et calories
    """
    # Vérifier que le pipeline est initialisé
    if not pipeline:
        raise HTTPException(
            status_code=503, 
            detail="Pipeline not initialized. Check server logs."
        )
    
    # Vérifier le type de fichier
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail="Format de fichier invalide. Seuls JPEG et PNG sont acceptés."
        )
    
    try:
        # Lire l'image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Image invalide")
        
        # Limiter la taille de l'image pour économiser la mémoire
        height, width = image.shape[:2]
        max_dimension = 1024
        
        if width > max_dimension or height > max_dimension:
            scale = max_dimension / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
            logger.info(f"Image redimensionnée de {width}x{height} à {new_width}x{new_height}")
        
        # Timer pour mesurer le temps de traitement
        start_time = time.time()
        
        # Traiter l'image avec le pipeline
        detections = pipeline.process_image(image)
        
        # Calculer le temps de traitement
        processing_time = (time.time() - start_time) * 1000  # en millisecondes
        
        # Calculer le total des calories
        total_calories = sum(d['estimated_calories'] for d in detections)
        
        # Formater la réponse
        response = DetectionResponse(
            success=True,
            total_calories=total_calories,
            detections=[
                FoodDetection(
                    bbox=d['bbox'],
                    class_name=d['class_name'],
                    confidence=d['confidence'],
                    estimated_calories=d['estimated_calories'],
                    weight_grams=d['weight_grams']
                ) for d in detections
            ],
            processing_time_ms=processing_time,
            message=f"{len(detections)} aliment(s) détecté(s)"
        )
        
        logger.info(f"✅ Analyse terminée: {len(detections)} aliments détectés, {total_calories:.0f} kcal")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'analyse: {e}")
        return DetectionResponse(
            success=False,
            total_calories=0,
            detections=[],
            processing_time_ms=0,
            message=str(e)
        )

if __name__ == "__main__":
    # Configuration pour Render
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False  # Désactiver le reload en production
    )