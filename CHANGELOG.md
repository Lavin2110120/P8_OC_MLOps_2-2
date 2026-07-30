# Changelog

Toutes les modifications notables apportées à ce projet sont documentées dans ce fichier.

Le format est basé sur [Keep a Changelog](https://keepachangelog.com/fr/1.0.0/)
et ce projet adhère au versionnage sémantique.

---

## [1.0.0] - 2026-07-30

### 🚀 Added
* **Monitoring & Data Drift (Étape 3) :** 
  * Création des scripts de monitoring et intégration de la détection de dérive des données (`data_drift_report`).
  * Conversion et fusion du script `analyze_datadrift.py` dans le notebook combiné `01_data_drift_analysis.ipynb` (`b4e4b2b`).
  * Mise en place de la capture de la latence de l'API via un middleware dédié dans FastAPI (`4bcf26b`).
* **Optimisation & Inférence (Étape 4) :**
  * Profiling de code, conversion des modèles au format ONNX et optimisation avec génération de rapports (`756e3ec`).
* **CI/CD & DevOps :**
  * Configuration du pipeline GitHub Actions (`.github/workflows/ci-cd.yml`) avec gestion sécurisée des secrets (`4bcf26b`, `e391243`).
  * Déploiement et conteneurisation de l'API avec Docker (`Dockerfile`).
* **Documentation :**
  * Rédaction complète du `README.md` enrichi avec des captures d'écran explicatives (`b4e4b2b`, `405ff91`).

### 🛠️ Fixed & Improved
* **API FastAPI (`main.py`) :** Correctifs multiples de stabilité et gestion des exceptions sur l'API (`d645eb7`, `851d6da`, `4bcf26b`).
* **Configuration :** 
  * Ajustements et résolutions de bugs dans les fichiers de configuration `.toml` et `.yaml` (`0f896ca`, `24d5a5c`).
  * Correction du `Dockerfile` pour la génération d'images légères et stables (`3d1686d`).
  * Stabilisation et résolutions successives des échecs du pipeline CI/CD (`b004c8d`, `b888587`).
* **Nettoyage & Git :**
  * Mise à jour régulière et affinement du `.gitignore` pour exclure les artefacts de logs, virtenvs et dossiers inutiles (`11658b3`, `7d021a7`, `9ae3011`).
  * Suppression des scripts obsolètes et centralisation dans l'architecture finale.

### 🏁 Initial (Étape 1 & 2)
* Initialisation du dépôt Git (`56d0ca4`).
* Construction initiale des notebooks d'analyse, préparation des données et premier squelette de l'API REST (`73406d8`).