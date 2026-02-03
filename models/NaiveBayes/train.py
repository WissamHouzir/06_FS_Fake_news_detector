import pandas as pd    
import numpy as np
import time
import mlflow
from mlflow.models.signature import infer_signature
import os
import spacy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score
from dotenv import load_dotenv

load_dotenv()
# Tracking URI (HF Space)
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_APP_URI"])
AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]

URL_TRAIN_DATA = "hf://datasets/readerbench/fakenews-climate-fr/fake-fr.csv"
EXPERIMENT_NAME = "Climate_Fake_News_Detector_Project"
TEXT_COLUMN = "Text"
TARGET_COLUMN = "Label"

class LemmatizerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.nlp = spacy.load('fr_core_news_sm')

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        lemmatized_texts = []
        for doc in self.nlp.pipe(X, batch_size=50):
            lemmatized_texts.append(
                ' '.join(
                    [token.lemma_.lower() for token in doc
                     if not token.is_stop
                     and not token.is_punct
                     and token.is_alpha]
                )
            )
        return np.array(lemmatized_texts, dtype=object)

def to_dense(X):
    return X.toarray()

if __name__ == "__main__":

    mlflow.set_experiment(EXPERIMENT_NAME)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    client = mlflow.tracking.MlflowClient()
    run = client.create_run(experiment.experiment_id)

    print("training model...")

    start_time = time.time()

    mlflow.sklearn.autolog(log_models=False)

    df = pd.read_csv(URL_TRAIN_DATA)

    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    nlp = spacy.load('fr_core_news_sm')

    dense_transformer = FunctionTransformer(to_dense, validate=False)

    text_pipeline = Pipeline([
    ('lemmatizer', LemmatizerTransformer()),
    ('tfidf', TfidfVectorizer(
        max_df=0.8,
        min_df=5,
        max_features=10000,
        ngram_range=(1, 1),
        )),
    ('to_dense', dense_transformer)
    ])
    
    preprocessor = ColumnTransformer([
    ('text', text_pipeline, TEXT_COLUMN)
    ])

    hyperparameters = {
    'var_smoothing': 1e-9
    }

    pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', GaussianNB(**hyperparameters))
    ])

    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_test_encoded = encoder.transform(y_test)

    with mlflow.start_run(run_id=run.info.run_id) as run:

        pipeline.fit(X_train, y_train_encoded)
        predictions = pipeline.predict(X_test)

        f1 = f1_score(y_test_encoded, predictions, average='macro')
        accuracy = accuracy_score(y_test_encoded, predictions)

        mlflow.log_params(hyperparameters)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="climate-fake-news-detector-model",
            registered_model_name="climate-fake-news-detector-model-NB",
            signature=infer_signature(X_test, predictions),
        )

    print("...Done!")
    print(f"---Total training time: {time.time()-start_time:.2f}s")