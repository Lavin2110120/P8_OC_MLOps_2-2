# ==========================================
# STAGE 1 : Builder (Compilation & dépendances)
# ==========================================
FROM python:3.11-slim AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Dépendances système de compilation (libpq-dev requis pour PostgreSQL / asyncpg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copie préalable des fichiers de dépendances (optimisation du cache Docker)
COPY pyproject.toml .
COPY requirements.txt* ./

# Installation du projet et de ses dépendances dans /install
RUN pip install --no-cache-dir --upgrade pip && \
    if [ -f requirements.txt ]; then \
        pip install --no-cache-dir --prefix=/install -r requirements.txt ; \
    fi && \
    pip install --no-cache-dir --prefix=/install .


# ==========================================
# STAGE 2 : Runtime Final (Production ultra-légère)
# ==========================================
FROM python:3.11-slim AS runner

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
# 5. Commande de démarrage d'Uvicorn s'adaptant dynamiquement au port fourni par Render
CMD uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-10000}
=======
=======
>>>>>>> 95a751e (adding performance tests and reports)
# Dépendance runtime minimale pour PostgreSQL
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Création d'un utilisateur non-root pour la sécurité
<<<<<<< HEAD
RUN adduser --disabled-password --gecos "" appuser

# Copie des bibliothèques installées du builder vers le runtime
COPY --from=builder /install /usr/local

# Copie du code source (dossier src/) et des modèles ONNX
COPY ./src /app/src
COPY ./models /app/models

# Structure des logs et attribution des permissions à appuser
=======
# Création d'un utilisateur non-root pour des raisons de sécurité
=======
>>>>>>> 95a751e (adding performance tests and reports)
RUN adduser --disabled-password --gecos "" appuser

# Copie des bibliothèques installées du builder vers le runtime
COPY --from=builder /install /usr/local

# Copie du code source (dossier app/) et du dossier models/
COPY ./app /app/app
COPY ./models /app/models

<<<<<<< HEAD
# Gestion des dossiers de logs et permissions
>>>>>>> 4c2801d (step 4: profiling and onxx then model optimization with reports)
=======
# Structure des logs et attribution des permissions à appuser
>>>>>>> 95a751e (adding performance tests and reports)
RUN mkdir -p /app/logs && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

<<<<<<< HEAD
<<<<<<< HEAD
# Démarrage de l'API avec Uvicorn pointant sur src.main:app
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
>>>>>>> bc6e882 (add storage data with space postgresql)
=======
# Lancement de l'API avec Uvicorn
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
>>>>>>> 4c2801d (step 4: profiling and onxx then model optimization with reports)
=======
# Démarrage de l'API avec Uvicorn pointing sur app.main:app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
>>>>>>> 95a751e (adding performance tests and reports)
