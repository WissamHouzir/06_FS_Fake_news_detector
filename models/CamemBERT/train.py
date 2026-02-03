import pandas as pd
import numpy as np
import mlflow
from mlflow.models.signature import infer_signature
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    CamembertTokenizer,
    CamembertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import time
from dotenv import load_dotenv

load_dotenv()

# Configuration des variables d'environnement
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_APP_URI"])
MLFLOW_TRACKING_APP_URI = os.environ["MLFLOW_TRACKING_APP_URI"]
AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]

URL_TRAIN_DATA = "hf://datasets/readerbench/fakenews-climate-fr/fake-fr.csv"
EXPERIMENT_NAME = "Climate_Fake_News_Detector_Project"
TARGET_COLUMN = "Label"

if __name__ == "__main__":
    mlflow.set_experiment(EXPERIMENT_NAME)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    client = mlflow.tracking.MlflowClient()
    run = client.create_run(experiment.experiment_id)

    print("Training model...")

    start_time = time.time()

    mlflow.pytorch.autolog(log_models=False)

    df = pd.read_csv(URL_TRAIN_DATA)
    print("Data loaded from Hugging Face")
    label_encoder = LabelEncoder()
    df["Label"] = label_encoder.fit_transform(df["Label"])

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["Label"]
    )

    # Conversion en Dataset Hugging Face
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    tokenizer = CamembertTokenizer.from_pretrained("camembert-base", force_download=True)
    print("Tokenizer loaded")

    def tokenize(batch):
        return tokenizer(
            batch["Text"],
            truncation=True,
            padding="max_length",
            max_length=256
        )

    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    # Renommage des colonnes pour correspondre aux attentes de Hugging Face
    train_dataset = train_dataset.rename_column("Label", "labels")
    test_dataset = test_dataset.rename_column("Label", "labels")

    # Sélection des colonnes utiles
    cols_to_keep = ["input_ids", "attention_mask", "labels"]
    train_dataset = train_dataset.remove_columns(
        [c for c in train_dataset.column_names if c not in cols_to_keep]
    )
    test_dataset = test_dataset.remove_columns(
        [c for c in test_dataset.column_names if c not in cols_to_keep]
    )

    # Format PyTorch
    train_dataset.set_format("torch")
    test_dataset.set_format("torch")

    # Calcul des poids de classe
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(df["Label"]),
        y=df["Label"]
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    # Initialisation du modèle
    model = CamembertForSequenceClassification.from_pretrained("camembert-base",
                                                               num_labels=len(label_encoder.classes_)
                                                               )
    print("Model initialized")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro")
        }

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        weight_decay=0.01,
        learning_rate=5e-5
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Entraînement et logging avec MLflow
    with mlflow.start_run(run_id=run.info.run_id) as run:
        trainer.train()

        predictions = trainer.predict(test_dataset)

        preds = np.argmax(predictions.predictions, axis=1)

        X = test_dataset["input_ids"][:1]

        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="climate-fake-news-detector-model",
            registered_model_name="climate-fake-news-detector-model-CamemBERT",
            signature=infer_signature(X, preds),
        )

    print("...Done!")
    print(f"---Total training time: {time.time() - start_time:.2f}s")