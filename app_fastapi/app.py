import os
import uvicorn
import pandas as pd
import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, status, File, UploadFile
from transformers import pipeline
import mlflow
from dotenv import load_dotenv

description = """
# [Détection des fausses informations sur le réchauffement climatique]

## À propos
Les fausses informations et les contenus manipulateurs sur le climat se propagent rapidement,
nuisant à la lutte contre le réchauffement climatique.
Ce projet vise à automatiser la classification des articles en trois catégories : vrai, biaisé ou faux.

## Machine-Learning
Where you can:
* `/predict` : prediction for a single value

Check out documentation for more information on each endpoint.
"""

tags_metadata = [
    {
        "name": "Predictions",
        "description": "Endpoints that uses our Machine Learning model",
    },
]

load_dotenv()

# Variables MLflow : URI de tracking, nom du modèle et stage
MLFLOW_TRACKING_APP_URI = os.getenv("MLFLOW_TRACKING_APP_URI")
MODEL_NAME = os.getenv("MODEL_NAME")
STAGE = os.getenv("STAGE", "production")

# Variables AWS pour accéder au bucket S3 qui contient les artifacts de MLflow
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")

# Variables globales pour stocker le modèle
mlflow.set_tracking_uri(MLFLOW_TRACKING_APP_URI)
model_uri = f"models:/{MODEL_NAME}@{STAGE}"
model = None
model_type = None  # "sklearn" ou "pytorch"
classifier = None  # Pour le pipeline Hugging Face

# Chargement conditionnel du modèle
try:
    # Essayer de charger un modèle scikit-learn
    model = mlflow.sklearn.load_model(model_uri)
    model_type = "sklearn"
    print("Modèle scikit-learn chargé avec succès.")

except mlflow.exceptions.MlflowException:
    try:
        model = mlflow.pytorch.load_model(model_uri)
        model_type = "pytorch"
        print("Modèle PyTorch chargé avec succès.")
        classifier = pipeline(task="text-classification", 
                              model=model, 
                              tokenizer="camembert-base")
        
    except mlflow.exceptions.MlflowException as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors du chargement du modèle : {e}"
    )

app = FastAPI(
    title="Climate Fake News Detector API",
    description="API pour détecter les fake news sur le climat",
    version="1.0",
    openapi_tags=tags_metadata,
)

@app.get("/")
def index():
    """
    Renvoie un message de bienvenue sur l'API ainsi que le lien vers la documentation.
    """
    return "Hello world! Go to /docs to try the API."

class TextInput(BaseModel):
    text: str

@app.post("/predict", tags=["Predictions"])
def predict(features: TextInput):
    """
    Fait une prédiction sur un texte donné en utilisant le modèle chargé.
    """
    try:
        if model_type == "sklearn":
            # Cas scikit-learn : prédiction directe
            df = pd.DataFrame({"Text": [features.text]})
            prediction = model.predict(df)[0]
            return {"prediction": int(prediction)}

        elif model_type == "pytorch":
            # Cas PyTorch (transformers) : utiliser le pipeline
            result = classifier(features.text)
            return {
                "prediction": result[0]["label"],
                "score": result[0]["score"]
            }

        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Aucun modèle chargé ou type de modèle non reconnu."
            )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Erreur lors de la prédiction : {e}"
        )

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)