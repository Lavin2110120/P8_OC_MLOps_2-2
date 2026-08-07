import asyncio
import sys
import time
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

# Ajoute le dossier racine du projet au PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.main import app


# --- FIXTURES GLOBALES & LOCALES ---


@pytest.fixture
def valid_payload():
    """Payload Pydantic complet et valide réutilisable."""
    return {
        "customer_value_score": 50.0,
        "Panier_Moyen_N_signature_3": 120.5,
        "GrandCompte": False,
        "clp_contrat_ap_stat": 1.0,
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
        "division": 0.0,
        "Famille_9_N_signature_3": 0.0,
    }


@pytest.fixture(scope="module")
def client():
    """Client synchrone TestClient pour l'API."""
    with TestClient(app) as c:
        yield c


# --- 1. TESTS DES ENDPOINTS DE BASE & HEALTHCHECK ---

def test_read_root(client):
    """Vérifie que la page d'accueil répond 200 OK."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_check_onnx(client):
    """Vérifie le suivi de santé et le chargement du moteur ONNX."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data.get("engine") == "onnxruntime"


# --- 2. TESTS DE PREDICTION & PERFORMANCES (SLA) ---

def test_predict_success(client, valid_payload):
    """Vérifie une prédiction réussie et la structure de la réponse."""
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200, f"Erreur API ({response.status_code}) : {response.json()}"
    
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert data["prediction"] in [0, 1]
    assert data["status"] == "success"
    if data["probability"] is not None:
        assert 0.0 <= data["probability"] <= 1.0


@pytest.mark.performance
def test_predict_latency_under_5ms(client, valid_payload):
    """Vérifie la latence d'inférence (SLA target < 5 ms sur machine dédiée, < 15 ms en CI)."""
    # 1. Requête de chauffe (Warmup)
    _ = client.post("/predict", json=valid_payload)

    # 2. Mesure du temps d'exécution
    start_time = time.perf_counter()
    response = client.post("/predict", json=valid_payload)
    execution_time_ms = (time.perf_counter() - start_time) * 1000

    assert response.status_code == 200

    # Vérification via le header X-Process-Time-Ms du middleware
    process_time_header = float(response.headers.get("X-Process-Time-Ms", execution_time_ms))
    assert process_time_header < 15.0, (
        f"❌ Viol de SLA : Latence de l'API ({process_time_header:.2f} ms) "
        f"supérieure au seuil toléré de 15.0 ms !"
    )


@pytest.mark.asyncio
async def test_async_predict_endpoint(valid_payload):
    """Vérifie l'endpoint /predict via un client asynchrone HTTPX."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as async_client:
        response = await async_client.post("/predict", json=valid_payload)
        assert response.status_code == 200
        assert response.json()["status"] == "success"


# --- 3. TESTS CAS LIMITES ET VALIDATIONS PYDANTIC ---

def test_predict_missing_required_field(client):
    """Vérifie le rejet (422) si un champ obligatoire manque."""
    payload = {"customer_value_score": 50.0, "GrandCompte": False}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_data_types(client, valid_payload):
    """Vérifie le rejet (422) en cas de mauvais types de données."""
    payload = valid_payload.copy()
    payload["customer_value_score"] = "pas_un_nombre"
    payload["GrandCompte"] = "invalide_bool"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_negative_years_validation(client, valid_payload):
    """Vérifie la contrainte ge=0 sur annees_depuis_dernier_achat."""
    payload = valid_payload.copy()
    payload["annees_depuis_dernier_achat"] = -5.0
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_with_none_optional_fields(client, valid_payload):
    """Vérifie la gestion des champs optionnels valant None."""
    payload = valid_payload.copy()
    payload["customer_value_score"] = None
    payload["clp_contrat_ap_stat"] = None
    payload["division"] = None
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["prediction"] in [0, 1]


# --- 4. PERFORMANCES ET CONCURRENCE ---

@pytest.mark.asyncio
async def test_concurrent_requests_performance(valid_payload):
    """Teste la tenue de charge sur 50 requêtes simultanées."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as async_client:
        start_time = time.time()
        tasks = [async_client.post("/predict", json=valid_payload) for _ in range(50)]
        responses = await asyncio.gather(*tasks)
        elapsed = time.time() - start_time

        assert all(r.status_code == 200 for r in responses), "Erreur sur au moins une requête."
        
        # Réhaussé à 10.0s pour accommoder les runners CI à 2 vCPUs
        assert elapsed < 10.0, f"Temps d'exécution trop long ({elapsed:.3f}s)"