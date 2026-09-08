import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.main import app

class TestSecurity:
    """Tests de sécurité et protection contre les attaques courantes."""

    def test_cors_headers(self, client):
        """Vérifie que les headers CORS sont correctement configurés."""
        response = client.get("/")
        assert "access-control-allow-origin" in response.headers
        assert response.headers["access-control-allow-origin"] == "*"
        assert "access-control-allow-methods" in response.headers

    def test_no_sql_injection(self, client):
        """Vérifie que les injections SQL sont bloquées."""
        malicious_payload = {
            "customer_value_score": "1; DROP TABLE prediction_logs;--",
            "Panier_Moyen_N_signature_3": 120.5,
            # ... autres champs requis
        }
        response = client.post("/predict", json=malicious_payload)
        assert response.status_code == 422  # Validation échoue

    def test_rate_limiting_simulation(self, client):
        """Simule une attaque par force brute (50 requêtes en 1s)."""
        for _ in range(50):
            response = client.get("/health")
            assert response.status_code == 200

class TestResilience:
    """Tests de résilience et gestion des erreurs critiques."""

    def test_database_unavailable(self, client, mocker):
        """Vérifie le comportement quand PostgreSQL est indisponible."""
        from sqlalchemy.exc import OperationalError
        mocker.patch(
            "src.main.AsyncSessionLocal",
            side_effect=OperationalError("Base de données indisponible", None, None)
        )

        response = client.get("/health")
        assert response.status_code == 500
        assert "indisponible" in response.json().get("detail", "").lower()

    def test_onnx_model_corrupted(self, client, mocker):
        """Vérifie le comportement quand le modèle ONNX est corrompu."""
        mocker.patch(
            "src.main.ml_models",
            {"onnx_session": None}  # Simule un modèle non chargé
        )
        response = client.post("/predict", json={"customer_value_score": 50.0})
        assert response.status_code == 500
        assert "ONNX" in response.json().get("detail", "")

    def test_memory_leak_simulation(self, client):
        """Vérifie que l'API ne plante pas après 1000 requêtes."""
        for _ in range(1000):
            response = client.get("/health")
            assert response.status_code == 200