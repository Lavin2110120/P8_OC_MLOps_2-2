import os
import pandas as pd
from datetime import datetime, timedelta, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Modules Evidently (version >= 0.4.0)
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset


# 1. Connexion synchronisée à PostgreSQL (via psycopg2)
DATABASE_URL = os.getenv(
    "DATABASE_URL_SYNC",
    "postgresql://postgres:postgrespassword@localhost:5432/scoring_db"
)

# Remplacement du driver asyncpg en psycopg2 pour Pandas si nécessaire
if "postgresql+asyncpg://" in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")

engine = create_engine(DATABASE_URL)


def load_data_from_db(days_offset_start: int, days_offset_end: int) -> pd.DataFrame:
    """
    Extrait les prédictions entre deux intervalles de jours.
    Exemple: (30, 1) = du jour -30 au jour -1.
    """
    now = datetime.now(timezone.utc)
    start_date = now - timedelta(days=days_offset_start)
    end_date = now - timedelta(days=days_offset_end)

    query = """
        SELECT timestamp, inputs, prediction, probability
        FROM prediction_logs
        WHERE timestamp >= %(start)s AND timestamp < %(end)s
          AND status = 'success'
    """
    
    # Lecture depuis PostgreSQL via Pandas
    df_raw = pd.read_sql(query, con=engine, params={"start": start_date, "end": end_date})

    if df_raw.empty:
        raise ValueError(f"Aucune donnée trouvée entre {start_date} et {end_date}")

    # Unpack de la colonne JSON 'inputs' en colonnes individuelles
    inputs_df = pd.json_normalize(df_raw["inputs"])
    
    # Combinaison des métadonnées (prediction, probability) avec les features
    df_final = pd.concat([inputs_df, df_raw[["prediction", "probability"]]], axis=1)
    
    return df_final


def generate_drift_report():
    print("🔄 Extraction des jeux de données depuis PostgreSQL...")
    
    # Dataset de Référence : Les données passées (ex: 30 jours à 1 jour)
    reference_data = load_data_from_db(days_offset_start=30, days_offset_end=1)
    
    # Dataset Actuel / Production : Les données récentes (ex: dernières 24h)
    current_data = load_data_from_db(days_offset_start=1, days_offset_end=0)

    print(f"📊 Données de référence : {len(reference_data)} lignes")
    print(f"📊 Données de production actuelles : {len(current_data)} lignes")

    # Configuration et génération du rapport Evidently
    print("🧪 Calcul des métriques de Data Drift...")
    report = Report(metrics=[
        DataDriftPreset(),      # Analyse la dérive de chaque variable/feature
        TargetDriftPreset(),    # Analyse la dérive des prédictions (target)
    ])

    report.run(reference_data=reference_data, current_data=current_data)

    # Exportation du rapport HTML
    output_path = "logs/data_drift_report.html"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    report.save_html(output_path)
    
    print(f"✅ Rapport de Data Drift généré avec succès : {output_path}")


if __name__ == "__main__":
    generate_drift_report()