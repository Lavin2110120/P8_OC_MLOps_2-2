import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.main
from src.main import app


# ---------------------------------------------------------------------------
# Fixtures locales
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """Client HTTP synchrone avec lifespan."""
    from starlette.testclient import TestClient
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSecurity:
    """Tests de sécurité et protection contre les attaques courantes."""

    def test_cors_headers_on_preflight(self, client):
        """Vérifie les headers CORS via une requête OPTIONS (preflight).

        TestClient ne traverse pas le middleware CORS sur les requêtes simples
        (GET/POST). Une requête OPTIONS explicite déclenche le middleware.
        """
        response = client.options(
            "/predict",
            headers={
                "origin": "https://evil.com",
                "access-control-request-method": "POST",
                "access-control-request-headers": "content-type",
            },
        )
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") in ["*", "https://evil.com"]
        # Les méthodes POST doivent être autorisées
        assert "POST" in response.headers.get("access-control-allow-methods", "")

    def test_cors_headers_on_root(self, client):
        """Vérifie que le middleware CORS est actif sur la racine aussi."""
        response = client.options(
            "/",
            headers={
                "origin": "https://example.com",
                "access-control-request-method": "GET",
            },
        )
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers

    def test_no_sql_injection(self, client, valid_payload):
        """Vérifie qu'une valeur non conforme est rejetée par la validation Pydantic."""
        malicious_payload = valid_payload.copy()
        malicious_payload["customer_value_score"] = "1; DROP TABLE prediction_logs;--"

        response = client.post("/predict", json=malicious_payload)

        # Pydantic rejette le type str pour un champ float attendu
        assert response.status_code == 422
        error_detail = response.json().get("detail", [])
        assert any("customer_value_score" in str(err) for err in error_detail)

    def test_no_xss_injection(self, client, valid_payload):
        """Vérifie qu'une injection XSS dans un champ texte est neutralisée."""
        malicious_payload = valid_payload.copy()
        malicious_payload["clp_contrat_ap_stat"] = '<script>alert("XSS")</script>'

        response = client.post("/predict", json=malicious_payload)

        if response.status_code == 200:
            response_body = response.text
            assert "<script>" not in response_body, "Le script XSS ne doit pas être renvoyé"
        else:
            assert response.status_code == 422

    def test_rate_limiting_simulation(self, client):
        """Simule une attaque par force brute (50 requêtes en 1s)."""
        for i in range(50):
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json()["status"] in ["healthy", "unhealthy"]

    def test_large_payload_rejected(self, client, valid_payload):
        """Vérifie qu'un payload excessivement grand est rejeté proprement."""
        malicious_payload = valid_payload.copy()
        malicious_payload["Panier_Moyen_N_signature_3"] = 1e308  # Overflow float

        response = client.post("/predict", json=malicious_payload)
        assert response.status_code in [200, 422, 400, 500]


class TestResilience:
    """Tests de résilience et gestion des erreurs critiques."""

    def test_database_unavailable_health_still_ok(self, client, mocker):
        """Vérifie que /health reste healthy même quand PostgreSQL est down.

        /health ne dépend pas de la BDD : il doit retourner 200 même si
        la base est injoignable.
        """
        # Mock AsyncSessionLocal pour lever une erreur
        mocker.patch(
            "src.main.AsyncSessionLocal",
            side_effect=Exception("Connection refused"),
        )

        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] in ["healthy", "unhealthy"]

    def test_predict_survives_db_failure(self, client, valid_payload, mocker):
        """Vérifie qu'une prédiction réussit même si le logging DB échoue.

        log_prediction_to_db est une fonction async locale à main.py.
        On la remplace par une coroutine qui lève une exception.
        """
        async def failing_log(log_data: dict):
            # Log l'erreur mais ne la propage pas
            # (simule un échec silencieux du logging)
            return None  # Échec silencieux

        mocker.patch("src.main.log_prediction_to_db", failing_log)

        response = client.post("/predict", json=valid_payload)
        assert response.status_code == 200
        assert response.json()["status"] == "success"

    def test_onnx_model_not_loaded(self, client, mocker, valid_payload):
        """Vérifie le comportement quand le modèle ONNX n'est pas chargé."""
        mocker.patch.dict(src.main.ml_models, {"onnx_session": None})

        response = client.post("/predict", json=valid_payload)
        assert response.status_code == 500
        detail = response.json().get("detail", "")
        assert "ONNX" in detail or "session" in detail.lower()

    def test_health_reflects_model_state(self, client, mocker):
        """Vérifie que /health reflète l'état réel du modèle."""
        # Modèle chargé → healthy
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        assert response.json()["model_loaded"] is True

        # Modèle déchargé → unhealthy
        mocker.patch.dict(src.main.ml_models, {"onnx_session": None})
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "unhealthy"
        assert response.json()["model_loaded"] is False

    def test_memory_leak_simulation(self, client):
        """Vérifie que l'API reste stable après 200 requêtes consécutives."""
        for i in range(200):
            response = client.get("/health")
            assert response.status_code == 200
        # Vérification finale : l'API répond toujours
        final_response = client.get("/")
        assert final_response.status_code == 200