import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
import json

# Ajoute le dossier racine du projet au PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src.main
from src.main import app

# --- FIXTURES GLOBALES ---
@pytest.fixture
def valid_payload():
    """Payload valide généré depuis l'exemple du schéma ClientData.

    Garantit la synchronisation test/schéma : si le schéma change,
    le test échoue sur la cohérence, pas sur un payload périmé.
    """
    from src.main import ClientData
    example = ClientData.model_config["json_schema_extra"]["example"]
    return example.copy()

@pytest.fixture(scope="module")
def client():
    """Client synchrone TestClient pour l'API."""
    with TestClient(app) as c:
        yield c

@pytest.fixture(scope="module")
def async_client():
    """Client asynchrone pour les tests asynchrones."""
    transport = ASGITransport(app=app)
    with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c

# --- 1. TESTS DES ENDPOINTS DE BASE & HEALTHCHECK ---
class TestBaseEndpoints:
    """Tests des endpoints de base et de santé."""

    def test_read_root(self, client):
        """Vérifie que la page d'accueil répond 200 OK."""
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()
        assert "Bienvenue sur l'API de Scoring Client" in response.json()["message"]

    def test_health_check_onnx(self, client):
        """Vérifie le suivi de santé et le chargement du moteur ONNX."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data.get("engine") == "onnxruntime"

    def test_health_check_model_not_loaded(self, client):
        """Vérifie le comportement quand le modèle n'est pas chargé."""
        with patch("src.main.ml_models", {}):
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "unhealthy"
            assert data["model_loaded"] is False

# --- 2. TESTS DE PREDICTION & PERFORMANCES ---
class TestPrediction:
    """Tests des endpoints de prédiction et de performance."""

    def test_predict_success(self, client, valid_payload):
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

    def test_predict_with_none_optional_fields(self, client, valid_payload):
        """Vérifie la gestion des champs optionnels valant None."""
        payload = valid_payload.copy()
        payload["customer_value_score"] = None
        payload["clp_contrat_ap_stat"] = None

        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["prediction"] in [0, 1]

    @pytest.mark.performance
    def test_predict_latency_under_sla(self, client, valid_payload):
        """Vérifie la latence d'inférence (SLA target < 15 ms en CI)."""
        # Requête de chauffe (Warmup)
        _ = client.post("/predict", json=valid_payload)

        # Mesure sur 10 requêtes pour éviter les biais
        latencies = []
        for _ in range(10):
            start_time = time.perf_counter()
            response = client.post("/predict", json=valid_payload)
            latencies.append(float(response.headers.get("X-Process-Time-Ms", 0)))

        # Vérification via le header X-Process-Time-Ms du middleware
        assert all(latency < 15.0 for latency in latencies), (
            f"❌ Viol de SLA : Latence moyenne ({sum(latencies)/len(latencies):.2f} ms) "
            f"supérieure au seuil toléré de 15.0 ms !"
        )

    def test_predict_model_not_loaded(self, client):
        """Vérifie le comportement quand le modèle ONNX n'est pas chargé."""
        with patch("src.main.ml_models", {}):
            response = client.post("/predict", json=valid_payload)
            assert response.status_code == 500
            assert "session ONNX n'est pas initialisée" in response.json()["detail"]

# --- 3. TESTS CAS LIMITES ET VALIDATIONS PYDANTIC ---
class TestValidation:
    """Tests de validation des entrées et gestion des erreurs."""

    def test_predict_missing_required_field(self, client):
        """Vérifie le rejet (422) si un champ obligatoire manque."""
        payload = {"customer_value_score": 50.0, "GrandCompte": False}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
        errors = response.json()["detail"]
        assert any(error["loc"] == ["Turnover_N_signature_1"] for error in errors)

    def test_predict_invalid_data_types(self, client, valid_payload):
        """Vérifie le rejet (422) en cas de mauvais types de données."""
        payload = valid_payload.copy()
        payload["customer_value_score"] = "pas_un_nombre"
        payload["GrandCompate"] = "invalide_bool"  # Typo intentionnelle pour tester

        response = client.post("/predict", json=payload)
        assert response.status_code == 422
        errors = response.json()["detail"]
        assert any("value_error.str" in str(error) for error in errors)

    def test_predict_negative_years_validation(self, client, valid_payload):
        """Vérifie la contrainte ge=0 sur annees_depuis_dernier_achat."""
        payload = valid_payload.copy()
        payload["annees_depuis_dernier_achat"] = -5.0
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
        errors = response.json()["detail"]
        assert any(error["loc"] == ["annees_depuis_dernier_achat"] for error in errors)

    def test_predict_negative_annees_depuis_1ere_facture(self, client, valid_payload):
        """Vérifie la contrainte ge=0 sur annees_depuis_1ere_facture."""
        payload = valid_payload.copy()
        payload["annees_depuis_1ere_facture"] = -1.0
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
        errors = response.json()["detail"]
        assert any(error["loc"] == ["annees_depuis_1ere_facture"] for error in errors)

    def test_predict_invalid_categorical_field(self, client, valid_payload):
        """Vérifie la validation des champs catégoriels."""
        payload = valid_payload.copy()
        payload["division"] = "INVALID_DIVISION"  # Valeur non autorisée
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

# --- 4. TESTS ASYNCHRONES ---
@pytest.mark.asyncio
class TestAsyncEndpoints:
    """Tests des endpoints asynchrones."""

    async def test_async_predict_endpoint(self, async_client, valid_payload):
        """Vérifie l'endpoint /predict via un client asynchrone HTTPX."""
        response = await async_client.post("/predict", json=valid_payload)
        assert response.status_code == 200
        assert response.json()["status"] == "success"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_requests_performance(self, async_client, valid_payload):
        """Teste la tenue de charge sur 50 requêtes simultanées."""
        start_time = time.time()
        tasks = [async_client.post("/predict", json=valid_payload) for _ in range(50)]
        responses = await asyncio.gather(*tasks)
        elapsed = time.time() - start_time

        assert all(r.status_code == 200 for r in responses), "Erreur sur au moins une requête."

        # Réhaussé à 10.0s pour accommoder les runners CI à 2 vCPUs
        assert elapsed < 10.0, f"Temps d'exécution trop long ({elapsed:.3f}s)"

# --- 5. TESTS DE SCHÉMA ET COHÉRENCE ---
class TestSchemaConsistency:
    """Tests de cohérence entre les schémas et la configuration."""

    def test_schema_matches_config_production(self):
        """Garde-fou : le schéma API doit être la projection exacte de config_production.json."""
        from src.main import ClientData

        config_path = Path(__file__).resolve().parent.parent / "config_production.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        expected = set(config["features_model"])
        actual = {f.alias or name for name, f in ClientData.model_fields.items()}

        assert actual == expected, (
            f"\nManquants dans l'API : {sorted(expected - actual)}"
            f"\nEn trop dans l'API   : {sorted(actual - expected)}"
        )

    def test_schema_matches_onnx_inputs(self, client):
        """Les inputs nommés du graphe ONNX doivent être couverts par le schéma."""
        response = client.get("/debug/schema")
        assert response.status_code == 200
        onnx_inputs = {inp["name"] for inp in response.json()["inputs"]}

        from src.main import ClientData
        api_fields = {f.alias or name for name, f in ClientData.model_fields.items()}

        # Si le modèle a un input unique (matrice), ce test est non applicable
        if len(onnx_inputs) == 1:
            pytest.skip("Modèle ONNX à matrice unique, pas d'inputs nommés")

        assert onnx_inputs <= api_fields, f"Inputs ONNX absents du schéma : {onnx_inputs - api_fields}"

# --- 6. TESTS DE LOGGING ET MONITORING ---
class TestLogging:
    """Tests des fonctionnalités de logging et monitoring."""

    def test_process_time_header_present(self, client, valid_payload):
        """Vérifie que le header X-Process-Time-Ms est présent."""
        response = client.post("/predict", json=valid_payload)
        assert "X-Process-Time-Ms" in response.headers
        assert float(response.headers["X-Process-Time-Ms"]) >= 0

    def test_logs_stats_endpoint(self, client, valid_payload):
        """Vérifie que l'endpoint /logs/stats fonctionne correctement."""
        client.post("/predict", json=valid_payload)
        response = client.get("/logs/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["n_success"] >= 1
        assert data["n_error"] >= 0
        assert data["latency_ms_mean"] is not None
        assert data["error_rate"] is not None

    def test_prediction_writes_jsonl_log(self, client, valid_payload, tmp_path, monkeypatch):
        """Une prédiction réussie doit produire 1 ligne JSONL avec les clés attendues."""
        log_file = tmp_path / "predictions.jsonl"
        monkeypatch.setattr(src.main, "PREDICTIONS_LOG_FILE", log_file)

        response = client.post("/predict", json=valid_payload)
        assert response.status_code == 200

        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1

        rec = json.loads(lines[0])
        assert rec["status"] == "success"
        assert "timestamp" in rec
        assert "inputs" in rec
        assert "prediction" in rec
        assert "probability" in rec
        assert "latency_ms" in rec
        assert "engine" in rec
        assert rec["engine"] == "onnxruntime"
        assert 0 <= rec["latency_ms"] <= 100  # Plage réaliste

    def test_logs_export_endpoint(self, client, valid_payload):
        """Vérifie que l'export des logs fonctionne."""
        client.post("/predict", json=valid_payload)
        response = client.get("/logs/export")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        assert len(response.text) > 0