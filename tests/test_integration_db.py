import asyncio
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from src.database import AsyncSessionLocal, engine
from src.main import app
from src.models import PredictionLog


@pytest.fixture(scope="session")
def event_loop():
    """Garantit une seule boucle d'événements asyncio pour l'ensemble de la session de test."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.mark.asyncio
async def test_postgres_connection():
    """Vérifie que la connexion à PostgreSQL est fonctionnelle."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(1))
        assert result.scalar() == 1


@pytest.mark.asyncio
async def test_predict_and_db_logging():
    """Vérifie qu'un appel /predict insère bien un enregistrement dans PostgreSQL."""
    payload = {
        "customer_value_score": 50.0,
        "Panier_Moyen_N_signature_3": 120.5,
        "GrandCompte": False,
        "clp_contrat_ap_stat": "STAT_01",
        "annees_depuis_dernier_achat": 1.5,
        "Turnover_N_signature_1": 3500.0,
        "Panier_Moyen_N_signature_1": 150.0,
        "%EC": 12.5,
        "Nb_lignes_N_signature_1": 8.0,
        "Turnover_N_signature_3": 1500.0,
        "Famille_2_N_signature_2": 0.0,
        "Panier_Moyen_N_signature_2": 135.0,
        "act_val_cust_3M": True,
        "annees_depuis_1ere_facture": 4.2,
        "Famille_0_N_signature_1": 0.0,
        "Famille_2_N_signature_1": 0.0,
        "Famille_11_N_signature_1": 0.0,
        "Famille_14_N_signature_1": 0.0,
        "division": "DIV_A",
        "Famille_9_N_signature_3": 0.0,
    }

    # 1. Requête HTTP asynchrone sur l'API FastAPI
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "prediction" in data

    # 2. Pause courte pour laisser la BackgroundTask s'exécuter en BDD
    await asyncio.sleep(0.5)

    # 3. Vérification de l'écriture effective dans PostgreSQL
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(PredictionLog)
            .order_by(PredictionLog.id.desc())
            .limit(1)
        )
        last_log = result.scalar_one_or_none()

        assert last_log is not None
        assert last_log.prediction == data["prediction"]
        assert last_log.status == "success"
        assert last_log.engine == "onnxruntime"
        assert last_log.inputs["%EC"] == 12.5