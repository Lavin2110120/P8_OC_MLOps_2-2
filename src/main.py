import asyncio
import json
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import onnxruntime as ort
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from fastapi.responses import PlainTextResponse

from src.database import AsyncSessionLocal, Base, engine
from src.models import PredictionLog

import os
import threading

# Dictionnaire global pour stocker la session ONNX et la config de production
ml_models: Dict[str, Any] = {}

# --- CONFIGURATION DU LOGGING POUR MONITORING / EVIDENTLY ---
# Cohérence avec le notebook 4 : LOGS_FILE = PROJECT_ROOT / "logs" / "predictions.jsonl"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_LOG_FILE = Path(os.getenv("PREDICTIONS_LOG_FILE", LOGS_DIR / "predictions.jsonl"))

# Verrou pour garantir qu'une ligne JSONL n'est jamais écrite entrelacée
# quand plusieurs background_tasks écrivent en même temps (benchmark concurrent)
_log_lock = threading.Lock()

# Clés minimales attendues par le notebook 4 (sections 5 et 6)
#   - santé opérationnelle : "status", "latency_ms"
#   - drift Evidently     : "inputs", "prediction" (pour status == "success")
_REQUIRED_SUCCESS_KEYS = ("status", "latency_ms", "inputs", "prediction")


def log_prediction(payload: Dict[str, Any]) -> None:
    """Écrit UN enregistrement JSON compact par ligne (JSONL) pour le monitoring.

    Format contractuel avec le notebook 4 :
      - 1 ligne = 1 JSON complet (pas de pretty-print, pas de print())
      - status == "success" => clés "status", "latency_ms", "inputs", "prediction"
      - status == "error"   => clés "status", "latency_ms", "inputs", "error"
    """
    try:
        json_payload = payload.copy()
        if isinstance(json_payload.get("timestamp"), datetime):
            json_payload["timestamp"] = json_payload["timestamp"].isoformat()

        # Garde-fou : ne jamais écrire un record "success" incomplet,
        # sinon le drift Evidently du notebook 4 plantera silencieusement
        if json_payload.get("status") == "success":
            missing = [k for k in _REQUIRED_SUCCESS_KEYS if k not in json_payload]
            if missing:
                print(f"⚠️ Log 'success' incomplet (clés manquantes : {missing}) — non écrit")
                return

        # Valeurs numpy -> natifs Python (json.dumps plante sur np.int64/np.float32)
        def _to_native(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {str(k): _to_native(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_to_native(v) for v in obj]
            return obj

        json_payload = _to_native(json_payload)

        with _log_lock:
            with open(PREDICTIONS_LOG_FILE, mode="a", encoding="utf-8") as f:
                f.write(json.dumps(json_payload, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"⚠️ Erreur lors de l'écriture du log JSONL : {e}")

async def log_prediction_to_db(log_data: dict):
    try:
        async with AsyncSessionLocal() as session:
            log_entry = PredictionLog(**log_data)
            session.add(log_entry)
            await session.commit()
    except Exception as e:
        print(f"[Logging DB Warning] Impossible d'enregistrer le log : {e}")


# --- GESTION DU CYCLE DE VIE (LIFESPAN) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise les tables PostgreSQL et charge le modèle ONNX au démarrage."""
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        print("✅ Tables PostgreSQL vérifiées / créées avec succès.")
    except Exception as e:
        print(f"⚠️ Avertissement BDD : Impossible de créer les tables : {e}")

    candidate_paths = [
        PROJECT_ROOT / "models" / "best_pipeline_xgboost.onnx",
        PROJECT_ROOT / "models" / "model_xgboost.onnx",
        PROJECT_ROOT / "artifacts" / "model.onnx",
    ]

    model_path = None
    for p in candidate_paths:
        if p.exists():
            model_path = p
            break

    if not model_path:
        raise FileNotFoundError(
            f"Aucun artefact ONNX trouvé parmi : {[str(p) for p in candidate_paths]}"
        )

    print(f"🔄 Chargement de la session ONNX Runtime depuis : {model_path}")

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    ml_models["onnx_session"] = session
    ml_models["input_names"] = [inp.name for inp in session.get_inputs()]
    ml_models["output_names"] = [out.name for out in session.get_outputs()]

    print("✅ Modèle ONNX chargé avec succès et prêt pour l'inférence !")

    yield

    ml_models.clear()
    print("🧹 Ressources ONNX libérées.")


# --- INITIALISATION DE L'APPLICATION FASTAPI ---
app = FastAPI(
    title="API de Scoring Client (Projet Morel - ONNX Runtime)",
    description="API MLOps haute performance optimisée avec ONNX Runtime.",
    version="2.1.0",
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


# --- MIDDLEWARE : MESURE DE LATENCE ---
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
    clp_contrat_ap_stat: Optional[str] = Field(None, description="Statut contrat AP (catégoriel)", examples=["ACTIF"])
    GrandCompte: bool = Field(..., description="Indicateur Grand Compte", examples=[False])
    Turnover_N_signature_1: float = Field(..., description="CA signature 1", examples=[3500.0])
    annees_depuis_dernier_achat: float = Field(..., ge=0.0, description="Années depuis dernier achat", examples=[1.5])
    Panier_Moyen_N_signature_1: float = Field(..., description="Panier moyen signature 1", examples=[150.0])
    Turnover_N_signature_3: float = Field(..., description="CA signature 3", examples=[1500.0])
    percent_EC: float = Field(..., alias="%EC", description="Pourcentage EC", examples=[12.5])
    Famille_11_N_signature_1: float = Field(..., description="Famille 11 signature 1", examples=[0.0])
    Panier_Moyen_N_signature_2: float = Field(..., description="Panier moyen signature 2", examples=[135.0])
    annees_depuis_1ere_facture: float = Field(..., ge=0.0, description="Années depuis 1ère facture", examples=[4.2])
    Famille_2_N_signature_1: float = Field(..., description="Famille 2 signature 1", examples=[0.0])
    act_val_cust_3M: bool = Field(..., description="Valeur client active 3 mois", examples=[True])
    Famille_2_N_signature_2: float = Field(..., description="Famille 2 signature 2", examples=[0.0])
    Famille_0_N_signature_1: float = Field(..., description="Famille 0 signature 1", examples=[0.0])
    Nb_lignes_N_signature_1: float = Field(..., description="Nb lignes signature 1", examples=[8.0])
    Famille_1_N_signature_1: float = Field(..., description="Famille 1 signature 1", examples=[0.0])
    division: str = Field(..., description="Division (catégoriel)", examples=["DIV_A"])
    Famille_12_N_signature_2: float = Field(..., description="Famille 12 signature 2", examples=[0.0])

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "customer_value_score": 50.0,
                "Panier_Moyen_N_signature_3": 120.5,
                "clp_contrat_ap_stat": "ACTIF",
                "GrandCompte": False,
                "Turnover_N_signature_1": 3500.0,
                "annees_depuis_dernier_achat": 1.5,
                "Panier_Moyen_N_signature_1": 150.0,
                "Turnover_N_signature_3": 1500.0,
                "%EC": 12.5,
                "Famille_11_N_signature_1": 0.0,
                "Panier_Moyen_N_signature_2": 135.0,
                "annees_depuis_1ere_facture": 4.2,
                "Famille_2_N_signature_1": 0.0,
                "act_val_cust_3M": True,
                "Famille_2_N_signature_2": 0.0,
                "Famille_0_N_signature_1": 0.0,
                "Nb_lignes_N_signature_1": 8.0,
                "Famille_1_N_signature_1": 0.0,
                "division": "DIV_A",
                "Famille_12_N_signature_2": 0.0,
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
    return {"message": "Bienvenue sur l'API de Scoring Client (ONNX Runtime). Rendez-vous sur /docs."}

@app.get("/debug/schema", tags=["Général"])
def get_onnx_schema():
    """Permet d'inspecter dynamiquement les entrées/sorties du modèle ONNX chargé."""
    session: ort.InferenceSession = ml_models.get("onnx_session")
    if not session:
        raise HTTPException(status_code=500, detail="Modèle non chargé")
    
    inputs_info = [
        {"name": inp.name, "type": inp.type, "shape": inp.shape}
        for inp in session.get_inputs()
    ]
    outputs_info = [
        {"name": out.name, "type": out.type, "shape": out.shape}
        for out in session.get_outputs()
    ]
    
    return {
        "inputs_count": len(inputs_info),
        "inputs": inputs_info,
        "outputs": outputs_info
    }

@app.get("/health", tags=["Général"])
async def health_check():
    session = ml_models.get("onnx_session")
    is_loaded = session is not None

    return {
        "status": "healthy" if is_loaded else "unhealthy",
        "engine": "onnxruntime",
        "model_loaded": is_loaded,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Machine Learning"])
async def predict(data: ClientData, background_tasks: BackgroundTasks):
    start_time = time.perf_counter()
    timestamp = datetime.now(timezone.utc)

    session: ort.InferenceSession = ml_models.get("onnx_session")
    if not session:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="La session ONNX n'est pas initialisée.",
        )

    input_dict = data.model_dump(by_alias=True)

    try:
        input_df = pd.DataFrame([input_dict])


        sanitize_mapping = {"%EC": "_EC"  } # ONNX attend _EC donc on remplace
        for original_col, onnx_col in sanitize_mapping.items():
            if original_col in input_df.columns:
                input_df.rename(columns={original_col: onnx_col}, inplace=True)

        inputs_onnx = {}
        input_inputs = session.get_inputs()

        # Cas 1 : Modèle ONNX qui attend une unique matrice 2D
        if len(input_inputs) == 1 and input_inputs[0].type == "tensor(float)":
            numeric_df = input_df.copy()

            for col in numeric_df.columns:
                if numeric_df[col].dtype == "bool":
                    numeric_df[col] = numeric_df[col].astype(np.float32)
                elif numeric_df[col].dtype == "object":
                    converted = pd.to_numeric(numeric_df[col], errors="coerce")
                    if converted.isna().all() and numeric_df[col].notna().any():
                        numeric_df[col] = pd.factorize(numeric_df[col])[0].astype(np.float32)
                    else:
                        numeric_df[col] = converted.fillna(0.0).astype(np.float32)

            arr = numeric_df.to_numpy().astype(np.float32)

            expected_shape = input_inputs[0].shape
            if len(expected_shape) > 1 and isinstance(expected_shape[1], int):
                expected_dim = expected_shape[1]
                if arr.shape[1] < expected_dim:
                    padding = np.zeros((arr.shape[0], expected_dim - arr.shape[1]), dtype=np.float32)
                    arr = np.hstack([arr, padding])

            inputs_onnx[input_inputs[0].name] = arr

        # Cas 2 : Pipeline ONNX complet avec inputs nommés
        else:
            for inp in input_inputs:
                col_name = inp.name
                if col_name in input_df:
                    val = input_df[col_name].values
                    
                    if "float" in inp.type:
                        numeric_val = pd.to_numeric(val, errors="coerce")
                        if pd.isna(numeric_val).all() and pd.notna(val).any():
                            val = pd.factorize(val)[0].astype(np.float32)
                        else:
                            val = np.nan_to_num(numeric_val.astype(np.float32), nan=0.0)
                    elif "int" in inp.type:
                        val = pd.to_numeric(val, errors="coerce").fillna(0).astype(np.int64)
                    elif "string" in inp.type:
                        val = np.asarray(val, dtype=object).astype(str)
                    else:
                        val = np.asarray(val, dtype=object).astype(str)
                    
                    inputs_onnx[col_name] = val.reshape(-1, 1)
                else:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Colonne attendue par le modèle absente du payload : '{col_name}'",
                    )

        # Inférence ONNX Runtime sur Worker Thread
        outputs = await asyncio.to_thread(session.run, None, inputs_onnx)

        if len(outputs) >= 2:
            prediction = int(outputs[0][0])
            raw_proba = outputs[1]

            if isinstance(raw_proba, list) and isinstance(raw_proba[0], dict):
                probability = float(raw_proba[0].get(1, raw_proba[0].get("1", 0.0)))
            elif isinstance(raw_proba, np.ndarray):
                probability = float(raw_proba[0][1])
            else:
                probability = float(outputs[1][0])
        else:
            prediction = int(outputs[0][0])
            probability = float(outputs[0][0])

        execution_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

        log_entry = {
            "timestamp": timestamp,
            "inputs": input_dict,
            "prediction": prediction,
            "probability": probability,
            "latency_ms": execution_time_ms,
            "engine": "onnxruntime",
            "status": "success",
        }

        background_tasks.add_task(log_prediction, log_entry)
        background_tasks.add_task(log_prediction_to_db, log_entry)

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
            "engine": "onnxruntime",
            "status": "error",
        }

        background_tasks.add_task(log_prediction, log_entry)
        background_tasks.add_task(log_prediction_to_db, log_entry)

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Erreur lors de l'inférence ONNX : {str(e)}",
        )

@app.get("/logs/export", response_class=PlainTextResponse, tags=["Monitoring"])
def export_prediction_logs(
    status_filter: Optional[str] = None,
    limit: Optional[int] = None,
):
    """Exporte le journal de prédictions JSONL (pour le monitoring NB4).

    - status_filter : "success" ou "error" pour ne garder qu'un type de record
    - limit         : ne renvoyer que les N derniers enregistrements
    """
    if not PREDICTIONS_LOG_FILE.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Aucun log de prédiction disponible pour l'instant.",
        )

    with _log_lock:  # évite de lire pendant une écriture
        lines = PREDICTIONS_LOG_FILE.read_text(encoding="utf-8").splitlines()

    # On ne renvoie que les lignes JSON valides : le NB4 n'aura jamais
    # à gérer de JSONDecodeError sur l'export
    valid_lines = []
    for line in lines:
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if status_filter and rec.get("status") != status_filter:
            continue
        valid_lines.append(line)

    if limit is not None and limit > 0:
        valid_lines = valid_lines[-limit:]

    return PlainTextResponse("\n".join(valid_lines) + "\n", media_type="application/x-ndjson")

@app.get("/logs/stats", tags=["Monitoring"])
def prediction_logs_stats():
    """Résumé du journal : volume, taux d'erreur, latence moyenne."""
    if not PREDICTIONS_LOG_FILE.exists():
        return {"n_records": 0, "n_success": 0, "n_error": 0}

    with _log_lock:
        lines = PREDICTIONS_LOG_FILE.read_text(encoding="utf-8").splitlines()

    n_success, n_error, n_corrupt, latencies = 0, 0, 0, []
    for line in lines:
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            n_corrupt += 1
            continue
        if rec.get("status") == "success":
            n_success += 1
            if "latency_ms" in rec:
                latencies.append(rec["latency_ms"])
        else:
            n_error += 1

    return {
        "n_records": n_success + n_error,
        "n_success": n_success,
        "n_error": n_error,
        "n_corrupt_lines": n_corrupt,
        "error_rate": round(n_error / (n_success + n_error), 4) if (n_success + n_error) else None,
        "latency_ms_mean": round(float(np.mean(latencies)), 2) if latencies else None,
    }