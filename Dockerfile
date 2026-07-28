# 1. Image de base légère et stable
FROM python:3.12-slim

# Evite la génération de fichiers .pyc et force l'affichage direct des logs Python dans la console
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Working directory dans le conteneur
WORKDIR /app

# Installation des dépendances système nécessaires (ex: compilation basique si besoin)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /lib/apt/lists/*

# 2. Gestion du cache des dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 3. Copie des fichiers du projet dans le conteneur
COPY src/ ./src/
COPY models/ ./models/
# Si tu utilises aussi un dossier artifacts/, décommente la ligne ci-dessous :
# COPY artifacts/ ./artifacts/

# 4. Exposition du port (par défaut 7860 sur Hugging Face Spaces)
EXPOSE 7860

# 5. Commande de démarrage d'Uvicorn
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "7860"]