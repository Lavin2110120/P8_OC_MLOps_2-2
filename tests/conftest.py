import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.main
from src.main import app, ClientData


@pytest.fixture(autouse=True)
def mock_loaded_onnx_model(tmp_path, monkeypatch):
    """
    Redirige PROJECT_ROOT vers un dossier temporaire contenant un vrai
    fichier .onnx factice, afin que le lifespan réel de src.main trouve
    un artefact "existant" sans jamais mocker pathlib.Path.
    """
    # 1. Arborescence réelle : tmp_path/models/best_pipeline_xgboost.onnx
    fake_models_dir = tmp_path / "models"
    fake_models_dir.mkdir(parents=True, exist_ok=True)
    fake_model_file = fake_models_dir / "best_pipeline_xgboost.onnx"
    fake_model_file.write_bytes(b"fake onnx binary content")

    # 2. Redirige la racine utilisée pour construire candidate_paths.
    monkeypatch.setattr(src.main, "PROJECT_ROOT", tmp_path)

    # 3. Session ONNX factice (le vrai contenu du fichier n'est jamais lu
    #    car InferenceSession est mocké ci-dessous).
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

    # 4. Seul mock : la construction de InferenceSession.
    #    Aucun patch de pathlib.Path -> pytest reste intact.
    with patch("src.main.ort.InferenceSession", return_value=fake_session):
        yield fake_session


@pytest.fixture
def client():
    """TestClient avec lifespan réellement exécutée (startup/shutdown)."""
    with TestClient(app) as c:
        yield c


VALID_PAYLOAD = {
    "customer_value_score": 50.0,
    "clp_contrat_ap_stat": "BK",
    "act_val_cust_3M": True,
    "Panier_Moyen_N_signature_3": 120.5,
    "GrandCompte": False,
    "EC": 12.5,
    "Panier_Moyen_N_signature_2": 135.0,
    "annees_depuis_dernier_achat": 1.5,
    "Turnover_N_signature_3": 1500.0,
    "Turnover_N_signature_1": 3500.0,
    "Famille_0_N_signature_1": 0.0,
    "Famille_10_N_signature_3": 0.0,
    "Famille_1_N_signature_3": 0.0,
    "division": "DIV_A",
    "Famille_2_N_signature_1": 0.0,
    "annees_depuis_1ere_facture": 4.2,
    "Panier_Moyen_N_signature_1": 150.0,
    "Turnover_N_signature_2": 2000.0,
    "Famille_12_N_signature_1": 0.0,
    "Nb_lignes_N_signature_1": 8.0,
}


@pytest.fixture
def valid_payload():
    return VALID_PAYLOAD.copy()