import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient
from src.main import app

@pytest.fixture
def client():
    """Crée un client de test avec la lifespan FastAPI correctement initialisée."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def valid_payload():
    return {
        "customer_value_score": 50.0,
        "Panier_Moyen_N_signature_3": 120.5,
        "GrandCompte": False,
        "clp_contrat_ap_stat": None,
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
        "Famille_1_N_signature_1": 0.0,
        "division": "DIV_A",
        "Famille_12_N_signature_2": 0.0,
    }

@pytest.fixture(scope="function")
def client():
    return TestClient(app)