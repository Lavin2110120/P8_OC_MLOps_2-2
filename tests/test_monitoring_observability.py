import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.main import app

class TestMonitoring:
    """Tests des endpoints de monitoring et observabilité."""

    def test_health_endpoint(self, client):
        """Vérifie que /health retourne les bonnes métriques."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "unhealthy"]
        assert "model_loaded" in data
        assert "engine" in data
        assert "latency_ms" in data

    def test_logs_stats_endpoint(self, client):
        """Vérifie que /logs/stats retourne des statistiques cohérentes."""
        # Insère des logs de test
        for i in range(5):
            client.post("/predict", json={
                "customer_value_score": 50.0 + i,
                "Panier_Moyen_N_signature_3": 120.5,
                # ... autres champs requis
            })

        response = client.get("/logs/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["n_success"] >= 5
        assert data["n_error"] == 0
        assert data["latency_ms_mean"] is not None
        assert 0 <= data["error_rate"] <= 1

    def test_logs_export_format(self, client):
        """Vérifie que l'export des logs est au format attendu."""
        client.post("/predict", json={
            "customer_value_score": 50.0,
            "Panier_Moyen_N_signature_3": 120.5,
            # ... autres champs requis
        })
        response = client.get("/logs/export")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        assert "prediction" in response.text
        assert "timestamp" in response.text

class TestObservability:
    """Tests de visibilité et de debugging."""

    def test_debug_schema_endpoint(self, client):
        """Vérifie que /debug/schema retourne les inputs ONNX."""
        response = client.get("/debug/schema")
        assert response.status_code == 200
        data = response.json()
        assert "inputs" in data
        assert "outputs" in data
        assert all("name" in inp for inp in data["inputs"])

    def test_process_time_header_consistency(self, client):
        """Vérifie que le header X-Process-Time-Ms est cohérent."""
        start_time = time.time()
        response = client.get("/health")
        api_latency = (time.time() - start_time) * 1000

        header_latency = float(response.headers.get("X-Process-Time-Ms", 0))
        assert abs(api_latency - header_latency) < 10  # Tolérance de 10ms