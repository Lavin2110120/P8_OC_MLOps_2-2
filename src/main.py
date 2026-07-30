import json
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

# Dictionnaire global pour stocker le pipeline
ml_models: Dict[str, Any] = {}

# --- CONFIGURATION DU LOGGING POUR EVIDENTLY / MONITORING ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_LOG_FILE = LOGS_DIR / "predictions.jsonl"


def log_prediction(payload: Dict[str, Any]):
    """Écrit un enregistrement au format JSON Lines (JSONL) pour le monitoring."""
    try:
        with open(PREDICTIONS_LOG_FILE, mode="a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"⚠️ Erreur lors de l'écriture du log : {e}")


# --- GESTION DU CYCLE DE VIE (LIFESPAN) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le pipeline ML au démarrage et libère la mémoire à l'arrêt."""
    candidate_paths = [
        PROJECT_ROOT / "models" / "best_pipeline_xgboost_epure.pkl",
        PROJECT_ROOT / "models" / "best_pipeline_production.joblib",
        PROJECT_ROOT / "artifacts" / "best_model_pipeline.joblib",
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

    ml_models.clear()
    print("🧹 Ressources du modèle libérées.")


# --- INITIALISATION DE L'APPLICATION FASTAPI ---
app = FastAPI(
    title="API de Scoring Client (Projet Morel)",
    description="API MLOps d'inférence basée sur les 20 features réelles du modèle.",
    version="2.0.0",
    lifespan=lifespan,
)

# --- CONFIGURATION CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time-Ms"],
)


# --- MIDDLEWARE : MESURE DE LATENCE ET EN-TÊTE HTTP ---
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
    response.headers["X-Process-Time-Ms"] = str(process_time_ms)
    return response


# --- SCHÉMAS PYDANTIC ---
class ClientData(BaseModel):
    customer_value_score: Optional[float] = Field(None, description="Score de valeur client", examples=[50.0])
    Panier_Moyen_N_signature_3: float = Field(..., description="Panier moyen signature 3", examples=[120.5])
    GrandCompte: bool = Field(..., description="Indicateur Grand Compte", examples=[False])
    clp_contrat_ap_stat: Optional[str] = Field(None, description="Statut contrat AP", examples=["STAT_01"])
    annees_depuis_dernier_achat: float = Field(..., ge=0.0, description="Années depuis dernier achat", examples=[1.5])
    Turnover_N_signature_1: float = Field(..., description="CA signature 1", examples=[3500.0])
    Panier_Moyen_N_signature_1: float = Field(..., description="Panier moyen signature 1", examples=[150.0])
    percent_EC: float = Field(..., alias="%EC", description="Pourcentage EC", examples=[12.5])
    Nb_lignes_N_signature_1: float = Field(..., description="Nb lignes signature 1", examples=[8.0])
    Turnover_N_signature_3: float = Field(..., description="CA signature 3", examples=[1500.0])
    Famille_2_N_signature_2: float = Field(..., description="Famille 2 signature 2", examples=[0.0])
    Panier_Moyen_N_signature_2: float = Field(..., description="Panier moyen signature 2", examples=[135.0])
    act_val_cust_3M: bool = Field(..., description="Valeur client active 3 mois", examples=[True])
    annees_depuis_1ere_facture: float = Field(..., ge=0.0, description="Années depuis 1ère facture", examples=[4.2])
    Famille_0_N_signature_1: float = Field(..., description="Famille 0 signature 1", examples=[0.0])
    Famille_2_N_signature_1: float = Field(..., description="Famille 2 signature 1", examples=[0.0])
    Famille_11_N_signature_1: float = Field(..., description="Famille 11 signature 1", examples=[0.0])
    Famille_14_N_signature_1: float = Field(..., description="Famille 14 signature 1", examples=[0.0])
    division: Optional[str] = Field(None, description="Division", examples=["DIV_A"])
    Famille_9_N_signature_3: float = Field(..., description="Famille 9 signature 3", examples=[0.0])

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
    prediction: int = Field(..., description="Classe prédite (0 ou 1)")
    probability: Optional[float] = Field(None, description="Probabilité classe 1")
    status: str = Field("success", description="Statut")


# --- ENDPOINTS ---
@app.get("/", tags=["Général"])
def read_root():
    return {"message": "Bienvenue sur l'API de Scoring Client. Rendez-vous sur /docs."}


@app.get("/health", tags=["Monitoring"])
def health_check():
    if "pipeline" not in ml_models:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Le modèle ML n'est pas encore chargé.",
        )
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict", response_model=PredictionResponse, tags=["Machine Learning"])
def predict(data: ClientData):
    start_time = time.perf_counter()
    timestamp = datetime.now(timezone.utc).isoformat()

    pipeline = ml_models.get("pipeline")
    if not pipeline:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Le modèle n'est pas initialisé.",
        )

    input_dict = data.model_dump(by_alias=True)

    try:
        input_df = pd.DataFrame([input_dict])

        cat_cols = ["clp_contrat_ap_stat", "division"]
        for col in cat_cols:
            if col in input_df.columns:
                input_df[col] = input_df[col].astype("category")

        prediction = int(pipeline.predict(input_df)[0])

        probability = None
        if hasattr(pipeline, "predict_proba"):
            proba_array = pipeline.predict_proba(input_df)
            probability = float(proba_array[0][1])

        execution_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

        log_entry = {
            "timestamp": timestamp,
            "inputs": input_dict,
            "prediction": prediction,
            "probability": probability,
            "latency_ms": execution_time_ms,
            "status": "success",
        }
        log_prediction(log_entry)

        return PredictionResponse(
            prediction=prediction, probability=probability, status="success"
        )

    except Exception as e:
        execution_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
        log_entry = {
            "timestamp": timestamp,
            "inputs": input_dict,
            "error": str(e),
            "latency_ms": execution_time_ms,
            "status": "error",
        }
        log_prediction(log_entry)

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Erreur lors de la prédiction : {str(e)}",
        )