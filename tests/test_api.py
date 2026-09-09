import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

# Ajoute le dossier racine du projet au PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src.main


@pytest.fixture
def invalid_payload():
    """Payload invalide ; le conftest ne fournit pas cette fixture."""
    return {"customer_value_score": 50.0, "GrandCompte": False}


class TestBaseEndpoints:
    def test_read_root(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()
        assert "Bienvenue sur l'API de Scoring Client" in response.json()["message"]

    def test_health_check_onnx(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data.get("engine") == "onnxruntime"

    def test_health_check_model_not_loaded(self, client):
        with patch("src.main.ml_models", {}):
            response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["model_loaded"] is False


class TestPrediction:
    def test_predict_success(self, client, valid_payload):
        assert "%EC" in valid_payload
        assert valid_payload["clp_contrat_ap_stat"] is None
        response = client.post("/predict", json=valid_payload)
        assert response.status_code == 200, response.text
        data = response.json()
        assert "prediction" in data
        assert "probability" in data
        assert data["prediction"] in [0, 1]
        assert data["status"] == "success"
        if data["probability"] is not None:
            assert 0.0 <= data["probability"] <= 1.0

    def test_predict_invalid_payload(self, client, invalid_payload):
        response = client.post("/predict", json=invalid_payload)
        assert response.status_code == 422

    def test_predict_with_none_optional_fields(self, client, valid_payload):
        payload = valid_payload.copy()
        payload["customer_value_score"] = None
        payload["clp_contrat_ap_stat"] = None
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        assert response.json()["prediction"] in [0, 1]

    @pytest.mark.performance
    def test_predict_latency_under_sla(self, client, valid_payload):
        client.post("/predict", json=valid_payload)
        latencies = []
        for _ in range(10):
            response = client.post("/predict", json=valid_payload)
            assert response.status_code == 200
            latencies.append(float(response.headers.get("X-Process-Time-Ms", 0)))
        assert all(latency < 15.0 for latency in latencies)

    def test_predict_model_not_loaded(self, client, valid_payload):
        with patch("src.main.ml_models", {}):
            response = client.post("/predict", json=valid_payload)
        assert response.status_code == 500
        assert "session ONNX n'est pas initialisée" in response.json()["detail"]


class TestValidation:
    def test_predict_missing_required_field(self, client, invalid_payload):
        response = client.post("/predict", json=invalid_payload)
        assert response.status_code == 422

    def test_predict_invalid_data_types(self, client, valid_payload):
        payload = valid_payload.copy()
        payload["customer_value_score"] = "pas_un_nombre"
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_negative_years_validation(self, client, valid_payload):
        payload = valid_payload.copy()
        payload["annees_depuis_dernier_achat"] = -5.0
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_negative_annees_depuis_1ere_facture(self, client, valid_payload):
        payload = valid_payload.copy()
        payload["annees_depuis_1ere_facture"] = -1.0
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_invalid_categorical_field(self, client, valid_payload):
        payload = valid_payload.copy()
        payload["division"] = "INVALID_DIVISION"
        response = client.post("/predict", json=payload)
        assert response.status_code == 422


class TestSchemaConsistency:
    def test_schema_matches_config_production(self):
        """Vérifie que le schéma API correspond exactement à config_production.json."""
        from src.main import ClientData
        
        # config_production.json est dans le dossier /models
        config_path = Path(__file__).resolve().parent.parent / "models" / "config_production.json"
        
        if not config_path.exists():
            pytest.skip(f"config_production.json non trouvé à {config_path}")
        
        config = json.loads(config_path.read_text(encoding="utf-8"))
        expected = set(config["features_model"])
        actual = {field.alias or name for name, field in ClientData.model_fields.items()}
        
        assert actual == expected, (
            f"\nManquants dans l'API : {sorted(expected - actual)}"
            f"\nEn trop dans l'API   : {sorted(actual - expected)}"
        )

    def test_schema_matches_onnx_inputs(self, client):
        """Vérifie que les inputs ONNX sont couverts par le schéma API."""
        response = client.get("/debug/schema")
        assert response.status_code == 200, f"Erreur /debug/schema : {response.text}"
        
        onnx_inputs = {item["name"] for item in response.json()["inputs"]}
        
        from src.main import ClientData
        api_fields = {field.alias or name for name, field in ClientData.model_fields.items()}
        

        if len(onnx_inputs) == 1:
            pytest.skip("Modèle ONNX à matrice unique, pas d'inputs nommés")
        
        missing = onnx_inputs - api_fields
        assert not missing, f"Inputs ONNX absents du schéma API : {missing}"


class TestLogging:
    def test_process_time_header_present(self, client, valid_payload):
        response = client.post("/predict", json=valid_payload)
        assert response.status_code == 200
        assert "X-Process-Time-Ms" in response.headers
        assert float(response.headers["X-Process-Time-Ms"]) >= 0

    def test_logs_stats_endpoint(self, client, valid_payload):
        client.post("/predict", json=valid_payload)
        response = client.get("/logs/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["n_success"] >= 1
        assert data["n_error"] >= 0
        assert data["latency_ms_mean"] is not None
        assert data["error_rate"] is not None

    def test_prediction_writes_jsonl_log(self, client, valid_payload, tmp_path, monkeypatch):
        log_file = tmp_path / "predictions.jsonl"
        monkeypatch.setattr(src.main, "PREDICTIONS_LOG_FILE", log_file)
        response = client.post("/predict", json=valid_payload)
        assert response.status_code == 200
        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["status"] == "success"
        assert {"timestamp", "inputs", "prediction", "probability", "latency_ms", "engine"} <= record.keys()
        assert record["engine"] == "onnxruntime"

    def test_logs_export_endpoint(self, client, valid_payload):
        """Vérifie que l'endpoint /logs/export retourne les logs en NDJSON."""
        # Créer au moins une prédiction pour avoir un log
        client.post("/predict", json=valid_payload)
        
        response = client.get("/logs/export")
        assert response.status_code == 200, f"Erreur export logs : {response.text}"
        
        # Vérifier le content-type NDJSON
        assert "application/x-ndjson" in response.headers["content-type"], (
            f"Content-Type inattendu : {response.headers['content-type']}"
        )
        
        # Vérifier que la réponse contient des données
        assert response.text, "Réponse vide"
        
        # Valider que chaque ligne est un JSON valide (NDJSON)
        lines = response.text.strip().splitlines()
        assert len(lines) > 0, "Aucune ligne de log retournée"
        
        for i, line in enumerate(lines):
            record = json.loads(line)  # Vérifie que c'est du JSON valide
            assert record["status"] == "success", f"Ligne {i} : status invalide"
            assert "timestamp" in record, f"Ligne {i} : timestamp manquant"