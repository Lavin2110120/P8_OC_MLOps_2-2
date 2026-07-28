from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

# Dictionnaire global pour stocker le pipeline (chargé une seule fois au démarrage)
ml_models: Dict[str, Any] = {}


# --- GESTION DU CYCLE DE VIE (LIFESPAN) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Charge le pipeline ML au démarrage et libère la mémoire à l'arrêt.
    """
    project_root = Path(__file__).resolve().parent.parent

    # On teste en priorité le fichier généré par le notebook d'entraînement
    candidate_paths = [
        project_root / "models" / "best_pipeline_xgboost_epure.pkl",
        project_root / "models" / "best_pipeline_production.joblib",
        project_root / "artifacts" / "best_model_pipeline.joblib",
    ]

    model_path = None
    for p in candidate_paths:
        if p.exists():
            model_path = p
            break

    if not model_path:
        raise FileNotFoundError(
            f"Aucun artefact de modèle trouvé parmi : {[str(p) for p in candidate_paths]}"
        )

    print(f"🔄 Chargement du pipeline ML depuis : {model_path}")
    ml_models["pipeline"] = joblib.load(model_path)
    print("✅ Pipeline ML chargé avec succès et prêt pour l'inférence !")

    yield

    # Nettoyage à l'arrêt
    ml_models.clear()
    print("🧹 Ressources du modèle libérées.")


# --- INITIALISATION DE L'APPLICATION FASTAPI ---
app = FastAPI(
    title="API de Scoring Client (Projet Morel)",
    description="API MLOps d'inférence basée sur les 20 features réelles du modèle.",
    version="2.0.0",
    lifespan=lifespan,
)


# --- SCHÉMAS PYDANTIC (VALIDATION DES DONNÉES) ---
class ClientData(BaseModel):
    """
    Schéma Pydantic représentant exactement les 20 features d'entrée du modèle.
    """

    customer_value_score: Optional[float] = Field(
        None, description="Score de valeur client", examples=[50.0]
    )
    Panier_Moyen_N_signature_3: float = Field(
        ..., description="Panier moyen signature 3", examples=[120.5]
    )
    GrandCompte: bool = Field(
        ..., description="Indicateur Grand Compte", examples=[False]
    )
    clp_contrat_ap_stat: Optional[str] = Field(
        None, description="Statut contrat AP (Catégorielle)", examples=["STAT_01"]
    )
    annees_depuis_dernier_achat: float = Field(
        ..., ge=0.0, description="Années depuis le dernier achat", examples=[1.5]
    )
    Turnover_N_signature_1: float = Field(
        ..., description="Chiffre d'affaires signature 1", examples=[3500.0]
    )
    Panier_Moyen_N_signature_1: float = Field(
        ..., description="Panier moyen signature 1", examples=[150.0]
    )
    percent_EC: float = Field(
        ...,
        alias="%EC",
        description="Pourcentage EC",
        examples=[12.5],
    )
    Nb_lignes_N_signature_1: float = Field(
        ..., description="Nombre de lignes signature 1", examples=[8.0]
    )
    Turnover_N_signature_3: float = Field(
        ..., description="Chiffre d'affaires signature 3", examples=[1500.0]
    )
    Famille_2_N_signature_2: float = Field(
        ..., description="Famille 2 signature 2", examples=[0.0]
    )
    Panier_Moyen_N_signature_2: float = Field(
        ..., description="Panier moyen signature 2", examples=[135.0]
    )
    act_val_cust_3M: bool = Field(
        ..., description="Valeur client active sur 3 mois", examples=[True]
    )
    annees_depuis_1ere_facture: float = Field(
        ..., ge=0.0, description="Années depuis la première facture", examples=[4.2]
    )
    Famille_0_N_signature_1: float = Field(
        ..., description="Famille 0 signature 1", examples=[0.0]
    )
    Famille_2_N_signature_1: float = Field(
        ..., description="Famille 2 signature 1", examples=[0.0]
    )
    Famille_11_N_signature_1: float = Field(
        ..., description="Famille 11 signature 1", examples=[0.0]
    )
    Famille_14_N_signature_1: float = Field(
        ..., description="Famille 14 signature 1", examples=[0.0]
    )
    division: Optional[str] = Field(
        None, description="Division (Catégorielle)", examples=["DIV_A"]
    )
    Famille_9_N_signature_3: float = Field(
        ..., description="Famille 9 signature 3", examples=[0.0]
    )

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
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
                "Famille_9_N_signature_3": 0.0,
            }
        },
    )


class PredictionResponse(BaseModel):
    """
    Schéma de la réponse de prédiction.
    """

    prediction: int = Field(
        ..., description="Classe prédite par le modèle (0 ou 1)"
    )
    probability: Optional[float] = Field(
        None, description="Probabilité associée à la classe positive (1)"
    )
    status: str = Field("success", description="Statut de la requête")


# --- ENDPOINTS / ROUTES ---
@app.get("/", tags=["Général"])
def read_root():
    """
    Page d'accueil de l'API.
    """
    return {
        "message": "Bienvenue sur l'API de Scoring Client (Projet Morel). Rendez-vous sur /docs pour Swagger UI."
    }


@app.get("/health", tags=["Monitoring"])
def health_check():
    """
    Vérification de la santé de l'API et du chargement du modèle ML.
    """
    is_model_loaded = "pipeline" in ml_models
    if not is_model_loaded:
        raise HTTPException(
            status_code=status.HTTP_530_SERVICE_UNAVAILABLE,
            detail="Le modèle ML n'est pas encore chargé.",
        )
    return {"status": "healthy", "model_loaded": True}


@app.post(
    "/predict", response_model=PredictionResponse, tags=["Machine Learning"]
)
def predict(data: ClientData):
    """
    Reçoit un payload JSON contenant les 20 variables du modèle et renvoie la prédiction.
    """
    pipeline = ml_models.get("pipeline")
    if not pipeline:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Le modèle n'est pas initialisé.",
        )

    try:
        # Conversion Pydantic -> Dict avec les alias de noms de colonnes originaux (%EC)
        input_dict = data.model_dump(by_alias=True)
        input_df = pd.DataFrame([input_dict])

        # Cast des colonnes catégorielles attendues sous forme de catégorie Pandas
        cat_cols = ["clp_contrat_ap_stat", "division"]
        for col in cat_cols:
            if col in input_df.columns:
                input_df[col] = input_df[col].astype("category")

        # Inférence directe sur le DataFrame
        prediction = int(pipeline.predict(input_df)[0])

        probability = None
        if hasattr(pipeline, "predict_proba"):
            proba_array = pipeline.predict_proba(input_df)
            probability = float(proba_array[0][1])

        return PredictionResponse(
            prediction=prediction, probability=probability, status="success"
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Erreur lors de la prédiction : {str(e)}",
        )