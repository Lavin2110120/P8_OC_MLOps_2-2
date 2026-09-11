import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.main
from src.main import app, ClientData

VALID_PAYLOAD = {
    "customer_value_score": 50.0,
    "clp_contrat_ap_stat": "BK",
    "Panier_Moyen_N_signature_3": 120.5,
    "GrandCompte": False,
    "act_val_cust_3M": True,
    "Nb_lignes_N_signature_4": 5.0,
    "EC": 12.5,
    "annees_depuis_dernier_achat": 1.5,
    "Panier_Moyen_N_signature_1": 150.0,
    "Nb_lignes_N_signature_1": 8.0,
    "Panier_Moyen_N_signature_2": 135.0,
    "Turnover_N_signature_3": 1500.0,
    "Turnover_N_signature_1": 3500.0,
    "Famille_5_N_signature_1": 0.0,
    "Famille_2_N_signature_2": 0.0,
    "annees_depuis_1ere_facture": 4.2,
    "Famille_11_N_signature_1": 0.0,
    "Famille_14_N_signature_2": 0.0,
    "Famille_1_N_signature_1": 0.0,
    "Famille_2_N_signature_1": 0.0,
}

@pytest.fixture
def valid_payload():
    return VALID_PAYLOAD.copy()

@pytest.fixture
def invalid_payload():
    return {
        "customer_value_score": 50.0,
        "clp_contrat_ap_stat": "BK",
        "Panier_Moyen_N_signature_3": 120.5,
        "GrandCompte": False,
        "act_val_cust_3M": True,
        "Nb_lignes_N_signature_4": 5.0,
        "EC": 12.5,
        "annees_depuis_dernier_achat": 1.5,
        "Panier_Moyen_N_signature_1": 150.0,
        "Nb_lignes_N_signature_1": 8.0,
        "Panier_Moyen_N_signature_2": 135.0,
        "Turnover_N_signature_3": 1500.0,
        "Famille_5_N_signature_1": 0.0,
        "Famille_2_N_signature_2": 0.0,
        "annees_depuis_1ere_facture": 4.2,
        "Famille_11_N_signature_1": 0.0,
        "Famille_14_N_signature_2": 0.0,
        "Famille_1_N_signature_1": 0.0,
        "Famille_2_N_signature_1": 0.0,
    }

@pytest_asyncio.fixture
async def async_client():
    """Client HTTP asynchrone avec lifespan actif."""
    from src.main import lifespan as app_lifespan
    
    async with app_lifespan(app):
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as ac:
            yield ac

@pytest.fixture(autouse=True)
def mock_loaded_onnx_model(tmp_path, monkeypatch):
    """Mock du modèle ONNX pour tous les tests."""
    fake_models_dir = tmp_path / "models"
    fake_models_dir.mkdir(parents=True, exist_ok=True)
    fake_model_file = fake_models_dir / "best_pipeline_xgboost.onnx"
    fake_model_file.write_bytes(b"fake onnx binary content")

    monkeypatch.setattr(src.main, "PROJECT_ROOT", tmp_path)

    input_info = MagicMock()
    input_info.name = "features"
    input_info.type = "tensor(float)"
    input_info.shape = [None, len(ClientData.model_fields)]

    output_prediction = MagicMock()
    output_prediction.name = "label"
    output_prediction.type = "tensor(int64)"
    output_prediction.shape = [None]

    output_probability = MagicMock()
    output_probability.name = "probabilities"
    output_probability.type = "tensor(float)"
    output_probability.shape = [None, 2]

    fake_session = MagicMock(name="fake_onnx_session")
    fake_session.get_inputs.return_value = [input_info]
    fake_session.get_outputs.return_value = [output_prediction, output_probability]
    fake_session.run.return_value = [
        np.asarray([1], dtype=np.int64),
        np.asarray([[0.25, 0.75]], dtype=np.float32),
    ]

    with patch("src.main.ort.InferenceSession", return_value=fake_session):
        yield fake_session

@pytest.fixture
def real_db(mocker):
    """Fixture utilitaire qui restaure les fonctions BDD réelles."""
    mocker.stopall()