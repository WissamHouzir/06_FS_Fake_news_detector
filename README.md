# Détection des fausses informations sur le réchauffement climatique

**Outils pour classifier automatiquement les articles sur le climat en "vrai", "biaisé" ou "faux".**

---

## 📝 À propos

Les fausses informations et les contenus manipulateurs sur le climat se propagent rapidement, nuisant à la lutte contre le réchauffement climatique. Ce projet vise à automatiser la classification des articles en trois catégories : **vrai**, **biaisé** ou **faux**.

**Objectifs :**
- Améliorer la modération des contenus en ligne.
- Assister les journalistes dans la vérification des informations.
- Augmenter la qualité de l’information disponible pour le public.
- Réduire la diffusion des fausses informations.

**Avantages :**
- Réduction du temps de vérification manuelle.
- Automatisation des tâches répétitives.
- Protection des lecteurs contre la désinformation.

---

## ✨ Fonctionnalités

### Modèles de Machine Learning
Le projet inclut cinq modèles entraînables via des scripts python `train.py` :
- CamemBERT
- Régression logistique
- Naive Bayes
- Support Vector Machine (SVM)
- XGBoost

### API FastAPI
- **Déploiement simplifié** via Docker.
- **Sélection dynamique du modèle** grâce aux variables d’environnement :
  - `MODEL_NAME` : Nom du modèle à utiliser.
  - `STAGE` : Alias pour sélectionner la version du modèle (via MLFlow).
  - Lien API : [https://wissamh-real-time-climate-fake-news.hf.space](https://wissamh-full-stack-api-fake-news-detector.hf.space)

### Interface utilisateur (GUI) avec Streamlit
- **Déploiement facile** via Docker.
- **Intégration avec l’API FastAPI** pour les prédictions.
- https://wissamh-full-stack-try-fake-news-detector.hf.space

---

## 🔧 Prérequis
- **Python 3.10 ou supérieur**
- **Docker**
- **Environnement [MLFlow](https://mlflow.org/docs/latest/genai/getting-started/connect-environment/) (version 2.21.3)**
- **Librairies Python** : Répertoriées dans les `requirements.txt`, et installées automatiquement via les DockerFile.
- **Accès à un [bucket S3 AWS](https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html)** (pour les artefacts MLFlow).

---

## 🛠 Installation

### 1. Cloner le projet
```bash
git clone https://github.com/Olivier-52/Fake_news_detector.git
cd Fake_news_detector
```

### 2. Entraîner les modèles souhaités

Pour chaque modèle, se placer dans le répertoire correspondant :
```Bash
cd models/repertoire_du_model_a_entrainer
```
Utiliser le répertoire corréspondant au modèle à entraîner :

- **CamemBERT** pour entraîner CamemBERT, un modèle d’IA spécialisé dans le traitement du français, inspiré par BERT, qui comprend le sens des mots et des phrases en analysant de grands textes.

- **LogisticRegression** pour entraîner un modèle de Régression Logistique simple et efficace pour classer des données en catégories (ex. : spam ou non-spam) en calculant des probabilités.

- **NaiveBayes** pour entraîner un modèle Naive Bayes rapide et intuitif qui classe des éléments (comme des emails) en se basant sur des probabilités et des hypothèses simplificatrices.

- **SVM** pour entraîner un modèle SVM (Support Vector Machine), une technique qui trace des frontières entre des groupes de données pour les séparer au mieux, utile pour la classification.

- **XGBoost** pour entraîner un modèle XGBoost puissant et précis qui combine plusieurs "arbres de décision" pour améliorer ses prédictions, souvent utilisé en compétition.


Copiez le fichier .env.exemple à la racine en .env et remplissez les valeurs. Positionner celui-ci au niveau du repertoire du model pour être copié dans l'image.

Puis lancer l’entraînement depuis un conteneur Docker:
```Bash
docker build -t your_image_name .
docker run your_image_name
```
### 3. Déployer l’API FastAPI (Backend)

Construire l’image Docker depuis le répertoire app_fastapi.

L'application comporte les variables d'environnement suivantes :
```
MLFLOW_TRACKING_APP_URI=Endpoint_du_serveur_MLFlow
MODEL_NAME=Nom_exact_du_modèle
STAGE=Alias_du_modèle (=production)
AWS_ACCESS_KEY_ID=Votre_ID_clé_AWS
AWS_SECRET_ACCESS_KEY=Votre_clé_secrète_AWS
```
**IMPORTANT** : Ajouter l'alias 'production' au modèle dans mlflow.

### 4. Déployer l'application Streamlit (Frontend)

Construire l’image Docker depuis le répertoire app_streamlit.
Variable d’environnement requise :

```
API_URL=URL_du_endpoint
```

---

## 📂 Utilisation

Accéder à l’interface Streamlit via un navigateur.
Saisir le texte de l’article dans la zone prévue.
Cliquer sur "Vérifier la nouvelle" pour obtenir la prédiction.

Résultats possibles :
- Probablement vrai
- Probablement faux
- Biaisé (si l’article contient un biais identifiable)

Capture d’écran :

Page d’accueil :

![Page d’accueil](/images/FakeNews_app_homepage.png)

Résultat de prédiction :

![Résultat de prédiction](/images/FakeNews_app_utilisation.png)

---
## 🤝 Contributeur

[madamanastasia](https://github.com/madamanastasia), [WissamHouzir](https://github.com/WissamHouzir), [Olivier-52](https://github.com/Olivier-52)

---

## 📜 Licence
Ce projet est sous licence MIT.
