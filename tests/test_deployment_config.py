import pytest
from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.main import app, PREDICTIONS_LOG_FILE
from src.database import DATABASE_URL

def test_predict(client, valid_payload):
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200

class TestConfiguration:
    """Tests de vérification des variables de configuration."""

    def test_database_url_configured(self):
        """Vérifie que DATABASE_URL est bien configurée."""
        assert DATABASE_URL is not None
        assert "postgresql" in DATABASE_URL
        assert "render.com" in DATABASE_URL  # Vérifie l'URL de Render

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

    def test_env_variable_priority(self):
        """Vérifie que les variables d'environnement priment sur les valeurs par défaut."""
        os.environ["DATABASE_URL"] = "postgresql://user:pass@localhost/test"
        from importlib import reload
        import src.main
        reload(src.main)
        assert "localhost" in src.main.DATABASE_URL

    def test_missing_env_variable_fallback(self):
        """Vérifie que l'application fonctionne avec les valeurs par défaut."""
        if "DATABASE_URL" in os.environ:
            del os.environ["DATABASE_URL"]
        from importlib import reload
        import src.main
        reload(src.main)
        assert "render.com" in src.main.DATABASE_URL