import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.exc import SQLAlchemyError

# Ajoute le dossier racine du projet au PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import AsyncSessionLocal, DATABASE_URL
from src.main import app
from src.models import Base, PredictionLog

# --- FIXTURES GLOBALES ---
@pytest_asyncio.fixture(scope="module")
async def test_db_engine():
    """Crée une engine de test dédiée avec des tables temporaires."""
    engine = create_async_engine(DATABASE_URL.replace("p8_oc_mlops_part2_db", "p8_oc_mlops_part2_db_test"), echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()

@pytest_asyncio.fixture(autouse=True, scope="function")
async def cleanup_database():
    """Nettoie la base de données après chaque test."""
    yield
    async with AsyncSessionLocal() as session:
        await session.execute(delete(PredictionLog))
        await session.commit()

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
                select(PredictionLog.id, PredictionLog.timestamp, PredictionLog.status)
            )
            columns = result.keys()
            assert "id" in columns
            assert "timestamp" in columns
            assert "status" in columns
            assert "prediction" in columns

class TestPredictionLogging:
    """Tests d'intégration complète entre l'API et la base de données."""

    @pytest.mark.asyncio
    async def test_predict_success_logging(self, valid_payload):
        """Vérifie qu'une prédiction réussie insère un log valide en base."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/predict", json=valid_payload)
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
            assert log_entry.inputs["%EC"] == 12.5  # Vérifie le payload attendu

    @pytest.mark.asyncio
    async def test_predict_logging_latency(self, valid_payload):
        """Vérifie que l'insertion en base prend moins de 100ms."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            start_time = time.time()
            response = await client.post("/predict", json=valid_payload)
            api_latency = (time.time() - start_time) * 1000

            assert response.status_code == 200
            assert api_latency < 15  # Latence API attendue

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

        assert db_latency < 100, f"Insertion en base trop lente : {db_latency:.2f}ms"

    @pytest.mark.asyncio
    async def test_multiple_predictions_logging(self, valid_payload):
        """Vérifie que plusieurs prédictions sont correctement loguées."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Effectue 5 prédictions
            for _ in range(5):
                response = await client.post("/predict", json=valid_payload)
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
                assert log.inputs["%EC"] == 12.5

class TestErrorHandling:
    """Tests de gestion des erreurs et cas limites."""

    @pytest.mark.asyncio
    async def test_predict_fails_to_log_to_db(self, mocker, valid_payload):
        """Vérifie que l'API retourne une erreur 500 si l'insertion en base échoue."""
        # Mock de la fonction de logging pour simuler une erreur
        mock_log = mocker.patch(
            "src.main.log_prediction_to_db",
            new_callable=AsyncMock,
            side_effect=SQLAlchemyError("Base de données indisponible")
        )

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/predict", json=valid_payload)

        assert response.status_code == 500
        data = response.json()
        assert "Impossible d'enregistrer le log" in data.get("detail", "")
        mock_log.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalid_payload_does_not_log(self, valid_payload):
        """Vérifie qu'un payload invalide ne tente pas d'insérer de log."""
        invalid_payload = valid_payload.copy()
        invalid_payload["annees_depuis_dernier_achat"] = -5.0  # Valeur invalide

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/predict", json=invalid_payload)

        assert response.status_code == 422  # Validation échoue

        # Vérifie qu'aucun log n'a été inséré
        async with AsyncSessionLocal() as session:
            result = await session.execute(select(PredictionLog))
            logs = result.scalars().all()
            assert len(logs) == 0

    @pytest.mark.asyncio
    async def test_concurrent_predictions_logging(self, valid_payload):
        """Teste la tenue de charge avec 20 requêtes concurrentes."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            tasks = [client.post("/predict", json=valid_payload) for _ in range(20)]
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
    for _ in range(100):
        async with AsyncSessionLocal() as session:
            log = PredictionLog(
                inputs={"%EC": 12.5},
                prediction=1,
                probability=0.95,
                latency_ms=10.5,
                status="success"
            )
            session.add(log)
            await session.commit()

    # Mesure le temps total
    start_time = time.time()
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(PredictionLog))
        _ = result.scalars().all()
    total_time = (time.time() - start_time) * 1000

    # Vérifie que le temps total est raisonnable
    assert total_time < 500, f"Performance d'insertion trop lente : {total_time:.2f}ms pour 100 logs"