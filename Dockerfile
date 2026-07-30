# # 1. Image de base légère et stable
# FROM python:3.12-slim

# # Évite la génération de fichiers .pyc et force l'affichage direct des logs Python dans la console
# ENV PYTHONUNBUFFERED=1 \
#     PYTHONDONTWRITEBYTECODE=1 \
#     PORT=10000

# # Répertoire de travail dans le conteneur
# WORKDIR /app

# # Installation des dépendances système nécessaires
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     && rm -rf /var/lib/apt/lists/*

# # 2. Copie préalable des fichiers de dépendances pour optimiser le cache Docker
# COPY requirements.txt* pyproject.toml* setup.py* ./

# # Installation/Mise à niveau de pip
# RUN pip install --no-cache-dir --upgrade pip

# # Installation depuis requirements.txt si présent, sinon installation du projet
# RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# # 3. Copie de l'intégralité du projet (src, models, artifacts, etc.)
# COPY . .

# # Installation du projet en mode éditable si pyproject.toml ou setup.py existe
# RUN if [ -f pyproject.toml ] || [ -f setup.py ]; then pip install --no-cache-dir -e .; fi

# # 4. Exposition du port (Render utilise par défaut 10000)
# EXPOSE 10000

# # 5. Commande de démarrage d'Uvicorn s'adaptant dynamiquement au port fourni par Render
# CMD uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-10000}

# ==========================================
# STAGE 1 : Builder (Installation des roues)
# ==========================================
FROM python:3.11-slim AS builder

WORKDIR /app

# Empêche Python d'écrire des fichiers .pyc
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Installation des dépendances de build minimales
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Installation des packages dans un dossier wheels local
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ==========================================
# STAGE 2 : Runtime Final (Image ultra-légère)
# ==========================================
FROM python:3.11-slim AS runner

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Création d'un utilisateur non-root pour des raisons de sécurité
RUN adduser --disabled-password --gecos "" appuser

# Copie des packages Python installés depuis le builder
COPY --from=builder /install /usr/local

# Copie du code source et des modèles ONNX
COPY ./src /app/src
COPY ./models /app/models

# Gestion des dossiers de logs et permissions
RUN mkdir -p /app/logs && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Lancement de l'API avec Uvicorn
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]