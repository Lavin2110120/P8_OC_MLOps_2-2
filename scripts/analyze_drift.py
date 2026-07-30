import json
from pathlib import Path

import joblib
import pandas as pd

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.pipeline.column_mapping import ColumnMapping

# --- 1. CONFIGURATION DES CHEMINS ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_FILE = PROJECT_ROOT / "logs" / "predictions.jsonl"
REFERENCE_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "X_train.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_HTML_PATH = REPORTS_DIR / "data_drift_report.html"

# Mêmes chemins candidats que dans src/main.py, pour rester cohérent avec l'API
MODEL_CANDIDATE_PATHS = [
    PROJECT_ROOT / "models" / "best_pipeline_xgboost_epure.pkl",
    PROJECT_ROOT / "models" / "best_pipeline_production.joblib",
    PROJECT_ROOT / "artifacts" / "best_model_pipeline.joblib",
]

NUMERICAL_FEATURES = [
    "customer_value_score",
    "Panier_Moyen_N_signature_3",
    "annees_depuis_dernier_achat",
    "Turnover_N_signature_1",
    "Panier_Moyen_N_signature_1",
    "%EC",
    "Nb_lignes_N_signature_1",
    "Turnover_N_signature_3",
    "Famille_2_N_signature_2",
    "Panier_Moyen_N_signature_2",
    "annees_depuis_1ere_facture",
    "Famille_0_N_signature_1",
    "Famille_2_N_signature_1",
    "Famille_11_N_signature_1",
    "Famille_14_N_signature_1",
    "Famille_9_N_signature_3",
]

# Booléens et catégorielles au sens strict (traités comme catégoriels par Evidently)
CATEGORICAL_FEATURES = [
    "GrandCompte",
    "act_val_cust_3M",
    "clp_contrat_ap_stat",
    "division",
]

FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def load_production_logs(log_path: Path) -> pd.DataFrame:
    """
    Lit le fichier predictions.jsonl et extrait les 'inputs' et la 'prediction'.
    """
    if not log_path.exists():
        raise FileNotFoundError(f"Le fichier de log {log_path} n'existe pas.")

    records = []
    with open(log_path, mode="r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                if data.get("status") == "success":
                    entry = data["inputs"].copy()
                    entry["prediction"] = data.get("prediction")
                    records.append(entry)

    return pd.DataFrame(records)


def load_model():
    """
    Charge le pipeline entraîné, en suivant le même ordre de recherche que l'API.
    Retourne None si aucun artefact n'est trouvé (le script reste utilisable
    en mode dégradé, sans prediction drift sur la référence).
    """
    for model_path in MODEL_CANDIDATE_PATHS:
        if model_path.exists():
            print(f"🔄 Chargement du pipeline depuis : {model_path}")
            return joblib.load(model_path)
    print("⚠️ Aucun artefact de modèle trouvé, le prediction drift sur la référence sera ignoré.")
    return None


def ensure_reference_predictions(reference_df: pd.DataFrame) -> pd.DataFrame:
    """
    Le CSV de référence (X_train.csv) ne contient que les features d'entraînement,
    pas de colonne 'prediction'. Sans cette étape, TargetDriftPreset ne se déclenche
    jamais avec la vraie référence (seulement dans le fallback de split simulé),
    ce qui prive le rapport du suivi de la distribution des scores prédits demandé
    par Chloé. On score donc la référence avec le pipeline en production.
    """
    if "prediction" in reference_df.columns:
        return reference_df

    pipeline = load_model()
    if pipeline is None:
        return reference_df

    df = reference_df.copy()
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # IMPORTANT : on conserve l'ordre des colonnes tel qu'il existe déjà dans df
    # (= l'ordre d'entraînement, features_model_epure trié par importance),
    # et non l'ordre de la liste FEATURES qui, elle, ne sert qu'à regrouper
    # numériques/catégorielles pour le ColumnMapping Evidently. Le pipeline
    # (via SimpleImputer.feature_names_in_) valide un ordre strict des colonnes.
    cols_for_model = [c for c in df.columns if c in FEATURES]

    # Filet de sécurité : si le pipeline expose l'ordre exact appris au fit
    # (SimpleImputer le fait toujours), on s'aligne dessus explicitement.
    expected_order = getattr(
        getattr(pipeline, "named_steps", {}).get("imputer"), "feature_names_in_", None
    )
    if expected_order is not None:
        cols_for_model = list(expected_order)

    df["prediction"] = pipeline.predict(df[cols_for_model])
    return df

def analyze_operational_metrics(log_path: Path):
    df_raw = []
    with open(log_path, mode="r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                df_raw.append(json.loads(line))
    
    df_logs = pd.DataFrame(df_raw)
    
    total_reqs = len(df_logs)
    success_rate = (df_logs["status"] == "success").mean() * 100
    
    print("\n========================================")
    print("📈 MÉTRIQUES OPÉRATIONNELLES")
    print(f" • Volume total de requêtes : {total_reqs}")
    print(f" • Taux de succès : {success_rate:.2f}%")
    
    if "latency_ms" in df_logs.columns:
        mean_lat = df_logs["latency_ms"].mean()
        p95_lat = df_logs["latency_ms"].quantile(0.95)
        print(f" • Latence moyenne : {mean_lat:.2f} ms")
        print(f" • Latence p95     : {p95_lat:.2f} ms")
    print("========================================\n")

def generate_drift_report():
    print("🔄 Chargement des données de référence et de production...")

    current_df = load_production_logs(LOGS_FILE)
    print(f"📊 Logs de production chargés : {len(current_df)} enregistrements.")

    if REFERENCE_DATA_PATH.exists():
        reference_df = pd.read_csv(REFERENCE_DATA_PATH)
        reference_df = ensure_reference_predictions(reference_df)
    else:
        print(f"⚠️ Fichier de référence non trouvé à {REFERENCE_DATA_PATH}. Découpage simulé des logs.")
        split_idx = int(len(current_df) * 0.5)
        reference_df = current_df.iloc[:split_idx].copy()
        current_df = current_df.iloc[split_idx:].copy()

    cols_to_compare = [c for c in FEATURES if c in current_df.columns and c in reference_df.columns]

    # Configuration du ColumnMapping pour Evidently 0.6.x
    # (numerical_features / categorical_features explicites : plus fiable que
    # l'inférence automatique, en particulier pour les colonnes booléennes)
    column_mapping = ColumnMapping()
    column_mapping.numerical_features = [c for c in NUMERICAL_FEATURES if c in cols_to_compare]
    column_mapping.categorical_features = [c for c in CATEGORICAL_FEATURES if c in cols_to_compare]

    has_prediction = "prediction" in current_df.columns and "prediction" in reference_df.columns
    if has_prediction:
        cols_to_compare.append("prediction")
        column_mapping.prediction = "prediction"

    ref_subset = reference_df[cols_to_compare]
    curr_subset = current_df[cols_to_compare]

    print("📊 Génération du rapport Evidently AI...")

    metrics = [DataDriftPreset()]
    if has_prediction:
        metrics.append(TargetDriftPreset())
    else:
        print("⚠️ Pas de colonne 'prediction' disponible des deux côtés : TargetDriftPreset ignoré.")

    drift_report = Report(metrics=metrics)
    drift_report.run(reference_data=ref_subset, current_data=curr_subset, column_mapping=column_mapping)

    drift_report.save_html(str(REPORT_HTML_PATH))
    print(f"✅ Rapport Data Drift généré avec succès dans : {REPORT_HTML_PATH}")


if __name__ == "__main__":
    generate_drift_report()