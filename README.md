# 🚀 P8 - Industrialisation & Monitoring d'un Modèle Machine Learning (MLOps)

Bienvenue dans le dépôt du projet d'industrialisation MLOps. Ce projet met en place une infrastructure complète pour le déploiement, le test automatique et le monitoring continu d'un modèle de prédiction du scoring client (**XGBoost**).

---

## 📌 Architecture du Projet

```text
├── .github/workflows/      # Pipelines CI/CD (GitHub Actions)
├── data/
│   └── processed/          # Données de référence (X_train.csv)
├── logs/                   # Traçabilité des requêtes d'inférence (.jsonl)
│   └── .gitkeep
├── models/                 # Artefacts du pipeline de modèle entraîné (.pkl / .joblib)
├── reports/                # Rapports d'analyse de drift générés
│   └── .gitkeep
├── scripts/
│   ├── analyze_drift.py    # Calcul des métriques & rapport Evidently AI
│   └── simulate_traffic.py # Simulation de requêtes de production (avec drift)
├── src/
│   ├── main.py             # API FastAPI (endpoints /predict, /health)
│   └── schemas.py          # Modèles Pydantic de validation des données
├── tests/                  # Suite de tests unitaires et d'intégration (pytest)
├── Dockerfile              # Conteneurisation de l'application
├── requirements.txt        # Dépendances du projet
└── README.md

🛠️ Fonctionnalités Principales

    Serving d'API FastAPI : Exposition d'un endpoint HTTP POST /predict acceptant les données clients au format JSON et renvoyant la prédiction ainsi que la probabilité associée.

    Validation des données (Pydantic) : Contrôle strict des types d'entrée et gestion robuste des erreurs (422 Unprocessable Entity).

    Traçabilité & Logging : Enregistrement local structuré au format JSON Lines (predictions.jsonl) incluant les paramètres d'entrée, les résultats, le statut HTTP et la latence d'exécution (latency_ms).

    Intégration & Déploiement Continus (CI/CD) :

        CI : Exécution automatique de pytest à chaque push ou pull request sur la branche main.

        CD : Déploiement automatique de l'API conteneurisée sur la plateforme Render.

    Monitoring du Data Drift & Santé Opérationnelle (Evidently AI) : Analyse statistique des dérives entre le jeu d'entraînement (X_train.csv) et les données reçues en production (logs/predictions.jsonl).

⚙️ Installation & Utilisation en Local
1. Prérequis & Installation
Bash

# Cloner le dépôt
git clone [https://github.com/votre-compte/p8-mlops.git](https://github.com/votre-compte/p8-mlops.git)
cd p8-mlops

# Créer et activer un environnement virtuel
python -m venv .venv
# Sur Windows (PowerShell) :
.venv\Scripts\Activate.ps1
# Sur Linux/macOS :
source .venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt

2. Lancer l'API FastAPI
Bash

uvicorn src.main:app --reload --port 8000

    Interface OpenAPI (Swagger UI) : http://127.0.0.1:8000/docs

    Vérification de santé (Healthcheck) : http://127.0.0.1:8000/health

3. Exécuter les tests automatiques
Bash

pytest tests/ -v

📈 Monitoring & Analyse du Drift

Le projet inclut un système complet pour évaluer la santé opérationnelle de l'API et détecter la dérive des données (Data Drift / Target Drift).
1. Simuler du trafic de production

Pour générer des logs d'inférence (incluant un scénario simulé de dérive statistique) :
Bash

python scripts/simulate_traffic.py

2. Analyser les métriques et générer le rapport

Pour afficher le volume, le taux de succès, les latences p95 et générer le rapport Evidently AI :
Bash

python scripts/analyze_drift.py

    Résultats :

        Console : Bilan des métriques opérationnelles (latence moyenne, p95, taux de succès).

        Rapport HTML : Un rapport interactif complet est généré dans reports/data_drift_report.html.

🚀 Déploiement & Conteneurisation

    Docker : L'image de l'API peut être construite et exécutée localement via :
    Bash

    docker build -t mlops-scoring-api .
    docker run -p 8000:8000 mlops-scoring-api

    Production : L'application est automatiquement déployée sur Render via la CI/CD dès que les tests d'intégration valident le build.