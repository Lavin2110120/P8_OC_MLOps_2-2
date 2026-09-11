import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.exc import SQLAlchemyError

# Ajoute le dossier racine du projet au PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import AsyncSessionLocal, DATABASE_URL
from src.main import app, log_prediction_to_db
from src.models import Base, PredictionLog

# --- FIXTURES GLOBALES ---

@pytest_asyncio.fixture(autouse=True, scope="function")
async def cleanup_database():
    """Nettoie la base de données après chaque test."""
    yield
    async with AsyncSessionLocal() as session:
        await session.execute(delete(PredictionLog))
        await session.commit()

# Marqueur pour les tests qui ont besoin d'utiliser la vraie BDD
REAL_DB = pytest.mark.real_db

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "real_db: test qui utilise la vraie base de données (pas de mock)"
    )

# --- CLASSES DE TEST ---

class TestDatabaseConnection:
    """Tests de connexion et de base de données."""

    @pytest.mark.asyncio
    async def test_postgres_connection(self):
        """Vérifie que la connexion à PostgreSQL est fonctionnelle."""
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(select(1))
                assert result.scalar() == 1
        except Exception as e:
            pytest.fail(f"Connexion à PostgreSQL échouée : {str(e)}")

    @pytest.mark.asyncio
    async def test_table_structure(self):
        """Vérifie que la table prediction_logs a la structure attendue."""
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(PredictionLog.id, PredictionLog.timestamp, PredictionLog.status, PredictionLog.prediction)
            )
            columns = result.keys()
            assert "id" in columns
            assert "timestamp" in columns
            assert "status" in columns
            assert "prediction" in columns

class TestPredictionLogging:
    """Tests d'intégration complète entre l'API et la base de données."""

    @pytest.mark.asyncio
    @pytest.mark.real_db
    async def test_predict_success_logging(self, async_client, valid_payload, real_db):
        """Vérifie qu'une prédiction réussie insère un log valide en base."""
        response = await async_client.post("/predict", json=valid_payload)
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "success"
        assert "prediction" in data
        assert data["prediction"] in [0, 1]

        # Vérification de l'insertion en base
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(PredictionLog)
                .order_by(PredictionLog.id.desc())
                .limit(1)
            )
            log_entry = result.scalar_one_or_none()
        
        assert log_entry is not None, "Aucun log trouvé en base de données"
        assert log_entry.prediction == data["prediction"]
        assert log_entry.status == "success"
        assert log_entry.engine == "onnxruntime"
        assert log_entry.inputs["EC"] == 12.5

    @pytest.mark.asyncio
    async def test_predict_logging_latency(self, async_client, valid_payload):
        """Vérifie que l'insertion en base prend moins de 1000ms."""
        start_time = time.time()
        response = await async_client.post("/predict", json=valid_payload)
        api_latency = (time.time() - start_time) * 1000

        assert response.status_code == 200
        assert api_latency < 3000

        # Mesure du temps d'insertion en base
        start_time = time.time()
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(PredictionLog)
                .order_by(PredictionLog.id.desc())
                .limit(1)
            )
            _ = result.scalar_one_or_none()
        db_latency = (time.time() - start_time) * 1000

        assert db_latency < 1000, f"Insertion en base trop lente : {db_latency:.2f}ms"

    @pytest.mark.asyncio
    @pytest.mark.real_db
    async def test_multiple_predictions_logging(self, async_client, valid_payload, real_db):
        """Vérifie que plusieurs prédictions sont correctement loguées."""
        # Effectue 5 prédictions
        for _ in range(5):
            response = await async_client.post("/predict", json=valid_payload)
            assert response.status_code == 200

        # Vérifie que 5 logs ont été insérés
        async with AsyncSessionLocal() as session:
            result = await session.execute(select(PredictionLog))
            logs = result.scalars().all()
            assert len(logs) == 5

            # Vérifie la cohérence des logs
            for log in logs:
                assert log.status == "success"
                assert log.engine == "onnxruntime"
                assert log.inputs["EC"] == 12.5

class TestErrorHandling:
    """Tests de gestion des erreurs et cas limites."""

    @pytest.mark.asyncio
    async def test_predict_fails_to_log_to_db(self, async_client, valid_payload, mocker):
        """Si la BDD est down, l'API doit répondre quand même."""
        # Simule une BDD down
        mock_session = AsyncMock()
        mock_session.add.side_effect = SQLAlchemyError("Base de données indisponible")
        mock_session.__aenter__.return_value = mock_session
        
        mocker.patch("src.main.AsyncSessionLocal", return_value=mock_session)
        
        response = await async_client.post("/predict", json=valid_payload)
        
        # La prédiction doit réussir malgré l'échec BDD
        assert response.status_code == 200
        assert response.json()["status"] == "success"

    @pytest.mark.asyncio
    @pytest.mark.real_db
    async def test_invalid_payload_does_not_log(self, async_client, valid_payload, real_db):
        """Vérifie qu'un payload invalide ne tente pas d'insérer de log."""
        invalid_payload = valid_payload.copy()
        invalid_payload["annees_depuis_dernier_achat"] = -5.0
        
        response = await async_client.post("/predict", json=invalid_payload)
        assert response.status_code == 422

        # Vérifie qu'aucun log n'a été inséré
        async with AsyncSessionLocal() as session:
            result = await session.execute(select(PredictionLog))
            logs = result.scalars().all()
            assert len(logs) == 0

    @pytest.mark.asyncio
    @pytest.mark.real_db
    async def test_concurrent_predictions_logging(self, async_client, valid_payload, real_db):
        """Teste la tenue de charge avec 20 requêtes concurrentes."""
        tasks = [async_client.post("/predict", json=valid_payload) for _ in range(20)]
        responses = await asyncio.gather(*tasks)

        # Vérifie que toutes les requêtes ont réussi
        assert all(r.status_code == 200 for r in responses)

        # Vérifie que 20 logs ont été insérés
        async with AsyncSessionLocal() as session:
            result = await session.execute(select(PredictionLog))
            logs = result.scalars().all()
            assert len(logs) == 20

            # Vérifie la cohérence des logs
            for log in logs:
                assert log.status == "success"
                assert log.engine == "onnxruntime"

# --- TESTS DE PERFORMANCE ---

@pytest.mark.asyncio
async def test_db_insertion_performance():
    """Test de performance pour l'insertion en base de données."""
    # Insère 100 logs
    train_logs = [
        PredictionLog(
            inputs={"EC": 12.5},
            prediction=1,
            probability=0.95,
            latency_ms=10.5,
            status="success"
        )
        for _ in range(100)
    ]
    
    async with AsyncSessionLocal() as session:
        session.add_all(train_logs)
        await session.commit()

    # Mesure le temps total de lecture
    start_time = time.time()
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(PredictionLog))
        _ = result.scalars().all()
    total_time = (time.time() - start_time) * 1000

    assert total_time < 2000, f"Performance d'insertion trop lente : {total_time:.2f}ms pour 100 logs"