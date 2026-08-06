# 🚀 P8 - Industrialisation & Monitoring d'un Modèle Machine Learning (MLOps)

Bienvenue dans le dépôt du projet d'industrialisation MLOps. Ce projet met en place une infrastructure complète pour le déploiement, le test automatique, le benchmark sous charge et le monitoring continu d'un modèle de prédiction du scoring client (**XGBoost / ONNX Runtime**).


## ⚡ Rapport d'Optimisation & Performance MLOps

Afin de répondre aux contraintes de production et de garantir une haute disponibilité (SLA stricts), le moteur d'inférence de l'API a été migré de l'exécuteur Python natif (`Joblib` / `Scikit-Learn`) vers **ONNX Runtime** avec le provider `CPUExecutionProvider`.

### 📊 Résultats des Benchmarks (Avant vs Après)

| Indicateur / Métrique | Modèle Natif (`.joblib` / `.pkl`) | Modèle Optimisé (`.onnx`) | Gain / Impact |
| :--- | :---: | :---: | :---: |
| **Taille de l'artefact** | **804 Ko** | **279 Ko** | **~65% de réduction** |
| **Latence d'inférence (p95)** | ~15ms - 50ms | **< 5 ms** | **Accélération > 3x** |
| **Moteur d'inférence** | Python / Scikit-Learn | ONNX Runtime (C++ Core) | Portabilité cross-platform |
| **SLA Latence CI/CD** | Non garanti | **Validé par Pytest (`< 5ms`)** | Déploiement sécurisé |

*(Ce tableau sera enrichi au fil des futures phases d'optimisation infrastructure : pools de connexions, async, etc.)*

### 🛠️ Leviers d'Optimisation Techniques
1. **Normalisation du Graphe de Calcul (ONNX Opset) :** Conversion du pipeline XGBoost au format ONNX pour éliminer l'overhead de l'interpréteur Python lors du passage de matrices.
2. **Gestion du Lifespan FastAPI :** Chargement unique du graphe ONNX au démarrage du serveur Uvicorn via `asynccontextmanager`, évitant toute ré-allocation mémoire par requête.
3. **Zero-Padding Dynamique & Types NumPy :** Vectorisation des entrées en `float32` et ajustement automatique des dimensions d'entrée dans FastAPI pour prévenir tout crash de schéma.
4. **Middleware de Latence :** Injection automatique du header HTTP `X-Process-Time-Ms` sur chaque requête pour une observabilité en temps réel.

## 📌 Architecture du Projet

├── .github/workflows/      # Pipelines CI/CD (GitHub Actions)
├── data/
│   └── processed/          # Données de référence (X_train.csv)
├── logs/                   # Traçabilité des requêtes d'inférence (.jsonl)
│   └── .gitkeep
├── models/                 # Artefacts du pipeline (.joblib & .onnx)
├── reports/                # Rapports d'analyse et métriques d'optimisation
│   ├── historique_optimisations.csv
│   └── .gitkeep
├── src/
│   └── main.py             # API FastAPI (ONNX Runtime, /predict, /health)
├── tests/                  # Suite de tests unitaires et de latence (pytest)
├── Dockerfile              # Conteneurisation de l'application
├── generate_drift_report.py# Calcul des métriques & rapport Evidently AI / Trafic
├── pyproject.toml          # Gestion des dépendances du projet
└── README.md

## 🛠️ Fonctionnalités Principales

    Inférence Haute Performance (ONNX Runtime) : Exposition d'un endpoint HTTP POST /predict renvoyant le score et la probabilité avec une latence inférieure à 5 ms.

    Validation des données (Pydantic) : Contrôle strict des types d'entrée et gestion automatique des alias/colonnes manquantes.

    Traçabilité & Logging : Enregistrement local structuré au format JSON Lines (predictions.jsonl) incluant les paramètres d'entrée, les résultats, le statut et la latence (latency_ms).

    Intégration & Déploiement Continus (CI/CD) :

        CI : Validation du code, tests unitaires et assertion de latence sous Pytest sur chaque Push/PR.

        CD : Déploiement automatique du conteneur sur Render dès validation du pipeline.

    Monitoring du Data Drift & Santé Opérationnelle (Evidently AI) : Analyse statistique automatisée des dérives entre le jeu de référence (X_train.csv) et le flux de production.

    Tracking des Performances API (MLflow) : Mesure et archivage des latences séquentielles/concurrentes et des taux de succès avant et après chaque optimisation.

## ⚙️ Installation & Utilisation en Local

### 1. Installation

# Cloner le dépôt
git clone [https://github.com/votre-compte/p8-mlops.git](https://github.com/votre-compte/p8-mlops.git)
cd p8-mlops

# Créer et activer l'environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Sur Windows: .venv\Scripts\Activate.ps1

# Installer les dépendances du projet (Mode Éditable)
pip install -e .

## 📈 Historique des Optimisations (MLflow)

Le suivi fin des performances de l'API (latence séquentielle, latence sous charge concurrentielle, débit req/s) est journalisé à l'aide de MLflow.
### 1. Consulter l'interface MLflow en local
mlflow ui

Accédez ensuite à http://127.0.0.1:5000 pour visualiser et comparer les runs de performance (avant_xxx vs apres_xxx).
### 2. Export des résultats

Un récapitulatif historisé des métriques clés est automatiquement synchronisé dans :

reports/historique_optimisations.csv

## 📊 Monitoring & Analyse du Drift

Exécutez le script d'analyse pour simuler du trafic et générer le rapport d'analyse de dérive des données :
Bash

python generate_drift_report.py

    Console : Bilan des métriques opérationnelles (volume, latence p95, taux de succès).
    Rapport HTML : Rapport interactif généré dans reports/data_drift_report.html.

## 🚀 Déploiement & Conteneurisation

### Docker Local :

docker build -t mlops-scoring-api .
docker run -p 8000:8000 mlops-scoring-api

### Production : Déploiement automatisé sur Render orchestré via GitHub Actions.