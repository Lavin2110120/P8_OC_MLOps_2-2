import time
import pytest
from fastapi.testclient import TestClient

# Importe l'instance FastAPI depuis ton module principal
from src.main import app

# Initialisation du client de test FastAPI
client = TestClient(app)


@pytest.fixture
def valid_client_payload():
    """Fixture renvoyant un exemple de payload valide conforme au schéma ClientData."""
    return {
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


def test_health_endpoint_onnx():
    """Vérifie que l'endpoint /health confirme le chargement du moteur ONNX."""
    with TestClient(app) as test_client:
        response = test_client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data.get("engine") == "onnxruntime"
        assert data["model_loaded"] is True


def test_predict_schema_and_types(valid_client_payload):
    """Vérifie la validité des réponses de l'endpoint /predict."""
    with TestClient(app) as test_client:
        response = test_client.post("/predict", json=valid_client_payload)
        assert response.status_code == 200

        data = response.json()
        assert "prediction" in data
        assert "probability" in data
        assert data["prediction"] in [0, 1]
        assert 0.0 <= data["probability"] <= 1.0
        assert data["status"] == "success"


def test_predict_latency_under_5ms(valid_client_payload):
    """
    Test de performance : Vérifie que le temps de traitement de l'inférence
    sur l'endpoint /predict est strictement inférieur à 5 ms.
    """
    with TestClient(app) as test_client:
        # 1. Requête de chauffe (Warmup) pour exclure la latence d'initialisation
        _ = test_client.post("/predict", json=valid_client_payload)

        # 2. Mesure précise du temps d'exécution
        start_time = time.perf_counter()
        response = test_client.post("/predict", json=valid_client_payload)
        execution_time_ms = (time.perf_counter() - start_time) * 1000

        assert response.status_code == 200

        # Vérification via l'en-tête X-Process-Time-Ms calculé par le middleware
        process_time_header = float(response.headers.get("X-Process-Time-Ms", execution_time_ms))

        print(f"\n⏱️ Latence mesurée (Header API) : {process_time_header:.2f} ms")
        print(f"⏱️ Latence mesurée (Client Test) : {execution_time_ms:.2f} ms")

        # Assertion SLA Performance < 5.0 ms
        assert process_time_header < 5.0, (
            f"❌ Viol de SLA : Le temps de réponse de l'API ({process_time_header:.2f} ms) "
            f"dépasse le seuil toléré de 5.0 ms !"
        )