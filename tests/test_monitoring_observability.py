import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolate_log_file(tmp_path, monkeypatch):
    """Redirige le fichier de logs vers un fichier temporaire isolé.
    
    CRITIQUE : le patch doit être fait AVANT d'importer src.main
    pour que PREDICTIONS_LOG_FILE soit bien positionné."""
    log_file = tmp_path / "predictions_test.jsonl"
    monkeypatch.setenv("PREDICTIONS_LOG_FILE", str(log_file))
    return log_file


@pytest.fixture
def client(isolate_log_file):
    """Client HTTP synchrone avec lifespan.
    
    Import retardé de src.main pour que le patch env soit actif."""
    import src.main
    from src.main import app
    from starlette.testclient import TestClient

    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    """Tests du endpoint /health et du middleware de latence."""

    def test_health_endpoint(self, client):
        """Vérifie que /health retourne les bonnes métriques."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] in ["healthy", "unhealthy"]
        assert "model_loaded" in data
        assert data["engine"] == "onnxruntime"

    def test_process_time_header_present(self, client):
        """Vérifie que le header X-Process-Time-Ms est présent."""
        response = client.get("/health")
        assert "x-process-time-ms" in response.headers
        header_ms = float(response.headers["x-process-time-ms"])
        assert header_ms >= 0
        assert header_ms < 100, f"Latence trop élevée : {header_ms:.2f}ms"


class TestDebugEndpoints:
    """Tests des endpoints de débogage."""

    def test_debug_schema_endpoint(self, client):
        """Vérifie que /debug/schema retourne les entrées/sorties ONNX."""
        response = client.get("/debug/schema")
        assert response.status_code == 200

        data = response.json()
        assert "inputs" in data, "Le schéma doit contenir 'inputs'"
        assert "outputs" in data, "Le schéma doit contenir 'outputs'"

        for inp in data["inputs"]:
            assert "name" in inp, f"Input sans nom : {inp}"
            assert "type" in inp
            assert "shape" in inp


class TestPredictMonitoring:
    """Tests de logging et monitoring post-prédiction."""

    def test_logs_stats_after_predictions(self, client, valid_payload, isolate_log_file):
        """Vérifie que /logs/stats reflète les prédictions effectuées."""
        # Vide le fichier de logs
        isolate_log_file.write_text("")

        for _ in range(5):
            response = client.post("/predict", json=valid_payload)
            assert response.status_code == 200

        # Laisse les BackgroundTasks s'exécuter
        time.sleep(0.5)

        response = client.get("/logs/stats")
        assert response.status_code == 200

        data = response.json()
        assert data["n_success"] >= 5
        assert data["n_error"] == 0
        assert data["latency_ms_mean"] is not None

    def test_logs_export_format(self, client, valid_payload, isolate_log_file):
        """Vérifie que /logs/export retourne du NDJSON valide."""
        isolate_log_file.write_text("")

        client.post("/predict", json=valid_payload)
        time.sleep(0.5)

        response = client.get("/logs/export")
        assert response.status_code == 200

        # Content-Type correct
        assert response.headers["content-type"] == "application/x-ndjson"

        # Vérifie que chaque ligne est un JSON valide
        import json

        lines = response.text.strip().split("\n")
        assert len(lines) >= 1

        for line in lines:
            record = json.loads(line)
            assert "timestamp" in record
            assert "prediction" in record
            assert "status" in record