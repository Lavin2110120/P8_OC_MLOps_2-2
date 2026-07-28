# 1. Image de base légère et stable
FROM python:3.12-slim

# Évite la génération de fichiers .pyc et force l'affichage direct des logs Python dans la console
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=10000

# Répertoire de travail dans le conteneur
WORKDIR /app

# Installation des dépendances système nécessaires
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Copie préalable des fichiers de dépendances pour optimiser le cache Docker
COPY requirements.txt* pyproject.toml* setup.py* ./

# Installation/Mise à niveau de pip
RUN pip install --no-cache-dir --upgrade pip

# Installation depuis requirements.txt si présent, sinon installation du projet
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# 3. Copie de l'intégralité du projet (src, models, artifacts, etc.)
COPY . .

# Installation du projet en mode éditable si pyproject.toml ou setup.py existe
RUN if [ -f pyproject.toml ] || [ -f setup.py ]; then pip install --no-cache-dir -e .; fi

# 4. Exposition du port (Render utilise par défaut 10000)
EXPOSE 10000

# 5. Commande de démarrage d'Uvicorn s'adaptant dynamiquement au port fourni par Render
CMD uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-10000}