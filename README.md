# 🚀 P8 - Industrialisation & Monitoring d'un Modèle Machine Learning (MLOps)

Bienvenue dans le dépôt du projet d'industrialisation MLOps. Ce projet met en place une infrastructure complète pour le déploiement, le test automatique, le benchmark sous charge et le monitoring continu d'un modèle de prédiction du scoring client (**XGBoost / ONNX Runtime**).

## 🎯 Choix Techniques & Justifications

| Composant                | Outil retenu          | Pourquoi ce choix ? |
|--------------------------|-----------------------|----------------------|
| **API de scoring**       | FastAPI               | Framework moderne, performant (async/await), validation automatique des données via Pydantic, documentation OpenAPI générée automatiquement, et léger. Sa maturité et sa simplicité d’utilisation en font un standard pour les API ML. |
| **Dashboard de monitoring** | Streamlit          | Permet de créer un tableau de bord interactif en quelques lignes de Python, d’intégrer facilement des rapports HTML (Evidently) et de visualiser l’historique des métriques. Un seul fichier (`app.py`) suffit pour centraliser toutes les analyses. |
| **Moteur d’inférence**   | ONNX Runtime          | Réduction de 65 % de la taille de l’artefact, accélération > 3x de la latence d’inférence, exécution en C++ indépendante de Python, garantissant une latence P95 < 5 ms et une meilleure portabilité. |
| **Analyse du Data Drift**| Evidently AI          | Bibliothèque spécialisée dans le suivi de la dérive des données et des cibles, avec des rapports HTML interactifs prêts à être intégrés dans un dashboard. |
| **Tracking des métriques** | MLflow              | Suivi fin des performances de l’API (latence, débit), archivage des artefacts, comparaison des runs avant/après optimisation. |

## ⚡ Rapport d'Optimisation & Performance MLOps

Afin de répondre aux contraintes de production et de garantir une haute disponibilité (SLA stricts), le moteur d'inférence de l'API a été migré de l'exécuteur Python natif (`Joblib` / `Scikit-Learn`) vers **ONNX Runtime** avec le provider `CPUExecutionProvider`.

### 📊 Résultats des Benchmarks (Avant vs Après)

| Indicateur / Métrique            | Modèle Natif (`.joblib` / `.pkl`) | Modèle Optimisé (`.onnx`) | Gain / Impact               |
|----------------------------------|:---------------------------------:|:-------------------------:|-----------------------------|
| **Taille de l'artefact**         | 804 Ko                           | 279 Ko                    | **~65 % de réduction**      |
| **Latence d'inférence (p95)**    | ~15ms - 50ms                     | **< 5 ms**                | **Accélération > 3x**       |
| **Moteur d'inférence**           | Python / Scikit-Learn             | ONNX Runtime (C++ Core)   | Portabilité cross-platform  |
| **SLA Latence CI/CD**            | Non garanti                      | **Validé par Pytest (`< 5ms`)** | Déploiement sécurisé   |

### 🛠️ Leviers d'Optimisation Techniques

1. **Normalisation du Graphe de Calcul (ONNX Opset) :** Conversion du pipeline XGBoost au format ONNX pour éliminer l'overhead de l'interpréteur Python.
2. **Gestion du Lifespan FastAPI :** Chargement unique du graphe ONNX au démarrage via `asynccontextmanager`, évitant toute ré-allocation mémoire par requête.
3. **Zero-Padding Dynamique & Types NumPy :** Vectorisation des entrées en `float32` et ajustement automatique des dimensions pour prévenir tout crash de schéma.
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

- **Inférence Haute Performance (ONNX Runtime)** : endpoint HTTP `POST /predict` renvoyant score et probabilité avec une latence < 5 ms.
- **Validation des données (Pydantic)** : contrôle strict des types d’entrée et gestion des colonnes manquantes.
- **Traçabilité & Logging** : enregistrement structuré au format JSON Lines (`predictions.jsonl`) incluant entrées, résultats, statut et latence.
- **CI/CD** :
  - *CI* : validation du code, tests unitaires et assertion de latence sous Pytest sur chaque Push/PR.
  - *CD* : déploiement automatique du conteneur sur Render dès validation du pipeline.
- **Monitoring du Data Drift & Santé Opérationnelle (Evidently AI)** : analyse statistique des dérives entre le jeu de référence et le flux de production.
- **Tracking des Performances API (MLflow)** : mesure et archivage des latences séquentielles/concurrentes et des taux de succès avant/après optimisation.

## ⚙️ Installation & Utilisation en Local

### 1. Installation

# Cloner le dépôt
git clone https://github.com/Lavin2110120/P8_OC_MLOps_2-2/

# Créer et activer l'environnement virtuel
python -m venv .venv
.venv\Scripts\Activate.ps1

# Installer les dépendances du projet (Mode Éditable)
pip install -e .

2. Lancer l'API en local

uvicorn src.main:app --reload

L’API est accessible sur http://127.0.0.1:8000.
Documentation interactive : http://127.0.0.1:8000/docs.
3. Exécuter les tests unitaires et de performance

pytest tests/ -v --cov=src --cov-report=term-missing

📊 Dashboard de Monitoring et Interprétation du Drift

Un tableau de bord Streamlit (app.py) permet de visualiser en un seul endroit :

    Le dernier rapport de Data Drift (généré par Evidently AI) au format HTML.
    L’historique des métriques de performance (latence, taux de succès) archivées via MLflow.

Lancer le dashboard

streamlit run app.py

Ouvrez l’URL affichée (par défaut http://localhost:8501).
Interpréter les indicateurs de Data Drift (Evidently)

Le rapport Evidently compare les distributions des 20 features de production entre la période de référence (données d’entraînement) et la période courante (logs de production). Voici les points clés à surveiller :

    Drift score global (en haut du rapport) : une valeur élevée (ex. > 0.5) indique qu’une ou plusieurs features ont significativement dérivé.
    Drift par feature : chaque feature est classée par importance. Une feature en rouge signale une dérive statistique (test KS, Wasserstein, etc.) ; elle mérite une investigation métier (changement de comportement client ?).
    Target Drift (optionnel) : compare la distribution des scores prédits entre les deux périodes. Un décalage important peut indiquer que le modèle ne capture plus le profil des demandes actuelles.
    Recommandation : Si une feature critique dérive, il peut être nécessaire de ré-entraîner le modèle ou de vérifier la qualité des données en amont.

Indicateurs de santé opérationnelle

    Latence moyenne / P95 / P99 : la P95 doit rester sous la SLA (ici < 5 ms pour l’inférence). Une augmentation brutale peut signaler une contention CPU ou un modèle mal chargé.
    Taux de succès (> 99 % attendu) : un taux d’erreur élevé peut indiquer des changements de format des payloads ou des problèmes réseau.
    Volume de requêtes : un pic ou une chute inhabituelle peut aider à corréler les variations de drift avec l’activité métier.

📈 Historique des Optimisations (MLflow)

Le suivi fin des performances de l'API est journalisé avec MLflow.
Consulter l'interface MLflow

mlflow ui

Accédez à http://127.0.0.1:5000 pour comparer les runs avant_xxx vs apres_xxx.
Un récapitulatif CSV est exporté automatiquement dans reports/historique_optimisations.csv.
🔍 Génération du rapport de Data Drift

Pour créer le rapport que le dashboard affichera :

python generate_drift_report.py

    La console affiche un bilan des métriques opérationnelles (volume, latence P95, taux de succès).
    Un rapport HTML interactif est généré dans reports/data_drift_report.html.

🚀 Déploiement & Conteneurisation
Docker en local

docker build -t mlops-scoring-api .
docker run -p 8000:8000 mlops-scoring-api

Production

Déploiement automatisé sur Render orchestré via GitHub Actions. L’image est reconstruite à chaque push sur main après validation des tests.
📚 Ressources et Documentation

    Documentation FastAPI
    ONNX Runtime
    Evidently AI
    MLflow Tracking
