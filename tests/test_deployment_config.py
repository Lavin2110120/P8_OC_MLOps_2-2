import pytest
from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.main import app, PREDICTIONS_LOG_FILE
from src.database import DATABASE_URL


# ---------------------------------------------------------------------------
# Fixture client synchrone (spécifique à ce fichier)
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

def test_predict(client, valid_payload):
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200


class TestConfiguration:
    """Tests de vérification des variables de configuration."""

    def test_database_url_configured(self):
        """Vérifie que DATABASE_URL est bien configurée."""
        assert DATABASE_URL is not None
        assert "postgresql" in DATABASE_URL
        assert "render.com" in DATABASE_URL or "localhost" in DATABASE_URL or "test_db" in DATABASE_URL

    def test_logs_directory_exists(self):
        """Vérifie que le répertoire de logs existe."""
        assert PREDICTIONS_LOG_FILE.parent.exists()
        assert PREDICTIONS_LOG_FILE.parent.is_dir()

    def test_onnx_model_path_exists(self):
        """Vérifie que le modèle ONNX est accessible."""
        model_path = Path(__file__).resolve().parent.parent / "models" / "best_pipeline_xgboost.onnx"
        assert model_path.exists(), f"Modèle ONNX introuvable : {model_path}"


class TestEnvironmentVariables:
    """Tests de validation des variables d'environnement."""

    def test_env_variable_priority(self, monkeypatch):
        """Vérifie que les variables d'environnement priment sur les valeurs par défaut."""
        # monkeypatch restaure automatiquement l'env après le test
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost/test")

        import src.database
        from importlib import reload
        reload(src.database)

        assert "localhost" in src.database.DATABASE_URL

    def test_missing_env_variable_fallback(self, monkeypatch):
        """Vérifie que l'application fonctionne avec les valeurs par défaut."""
        # Supprime proprement la variable d'env pour ce test uniquement
        monkeypatch.delenv("DATABASE_URL", raising=False)

        import src.database
        from importlib import reload
        reload(src.database)

        assert "render.com" in src.database.DATABASE_URL