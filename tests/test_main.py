import pytest
from fastapi.testclient import TestClient
from src.main import app

# Utilisation du TestClient comme context manager pour déclencher l'événement lifespan (chargement du modèle)
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# --- 1. TESTS DES ENDPOINTS DE BASE ---

def test_read_root(client):
    """Vérifie que la page d'accueil répond 200 OK."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_check(client):
    """Vérifie le suivi de santé et le chargement du modèle."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


# --- 2. TEST CAS VALIDE (HAPPY PATH) ---

def test_predict_success(client):
    """Vérifie une prédiction réussie avec un payload complet et valide."""
    payload = {
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
        "Famille_9_N_signature_3": 0.0
    }
    
    response = client.post("/predict", json=payload)
    # Si le status_code n'est pas 200, pytest affichera le contenu de response.json()
    assert response.status_code == 200, f"Erreur de l'API ({response.status_code}) : {response.json()}"
    
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] in [0, 1]
    assert data["status"] == "success"
    if data["probability"] is not None:
        assert 0.0 <= data["probability"] <= 1.0


# --- 3. TESTS DES CAS LIMITES ET ERREURS ---

def test_predict_missing_required_field(client):
    """Vérifie le rejet (422) si un champ obligatoire manque."""
    payload = {
        "customer_value_score": 50.0,
        # 'Panier_Moyen_N_signature_3' est omis volontairement
        "GrandCompte": False,
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
        "Famille_9_N_signature_3": 0.0
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Unprocessable Entity (Erreur Pydantic)


def test_predict_invalid_data_types(client):
    """Vérifie le rejet (422) en cas de mauvais types de données."""
    payload = {
        "customer_value_score": "pas_un_nombre",  # Doit être un float ou int
        "Panier_Moyen_N_signature_3": 120.5,
        "GrandCompte": "invalide_bool",
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
        "Famille_9_N_signature_3": 0.0
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_negative_years_validation(client):
    """Vérifie que la contrainte ge=0 sur annees_depuis_dernier_achat déclenche une erreur."""
    payload = {
        "customer_value_score": 50.0,
        "Panier_Moyen_N_signature_3": 120.5,
        "GrandCompte": False,
        "annees_depuis_dernier_achat": -5.0,  # Invalide car ge=0.0
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
        "Famille_9_N_signature_3": 0.0
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predict_with_none_optional_fields(client):
    """Vérifie que la prédiction fonctionne même quand les champs optionnels sont None."""
    payload = {
        "customer_value_score": None,
        "Panier_Moyen_N_signature_3": 120.5,
        "GrandCompte": False,
        "clp_contrat_ap_stat": None,
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
        "division": None,
        "Famille_9_N_signature_3": 0.0
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["prediction"] in [0, 1]