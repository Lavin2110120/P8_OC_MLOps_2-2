import asyncio
import sys
from pathlib import Path
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select, delete

# Ajoute le dossier racine du projet au PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import AsyncSessionLocal
from src.main import app
from src.models import PredictionLog
from test_api import valid_payload
from sqlalchemy.ext.asyncio import create_async_engine
from src.database import DATABASE_URL
from src.models import Base

@pytest.fixture(autouse=True, scope="module")
async def setup_test_database():
    """Crée les tables en base de données avant d'exécuter les tests d'intégration."""
    engine = create_async_engine(DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()

@pytest.mark.asyncio
async def test_postgres_connection():
    """Vérifie que la connexion à PostgreSQL est fonctionnelle."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(1))
        assert result.scalar() == 1


@pytest.mark.asyncio
async def test_predict_and_db_logging(valid_payload):
    """Vérifie qu'un appel /predict insère un log valide dans PostgreSQL via la BackgroundTask."""
    
    # 1. Envoi de la requête de prédiction
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/predict", json=valid_payload)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "prediction" in data

    # 2. Polling dynamique (max 2s) pour attendre l'exécution de la BackgroundTask
    last_log = None
    for _ in range(20):
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(PredictionLog)
                .order_by(PredictionLog.id.desc())
                .limit(1)
            )
            last_log = result.scalar_one_or_none()
            if last_log and last_log.inputs.get("%EC") == 12.5:
                break
        await asyncio.sleep(0.1)

    # 3. Assertions sur les données enregistrées
    assert last_log is not None, "Aucun log n'a été inséré dans la base de données."
    assert last_log.prediction == data["prediction"]
    assert last_log.status == "success"
    assert last_log.engine == "onnxruntime"
    assert last_log.inputs["%EC"] == 12.5