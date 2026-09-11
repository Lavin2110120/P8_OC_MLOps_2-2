import json
import os
import time
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

# --- Définition des chemins dynamiques ---
# BASE_DIR correspond au dossier parent de 'dashboard', donc la racine du projet
BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"
CONFIG_PATH = BASE_DIR / "models" /"config_production.json"

st.set_page_config(page_title="Dashboard MLOps", layout="wide")
st.title("📊 Dashboard MLOps - Scoring Client & Monitoring")


# --- CHARGEMENT DE LA CONFIG DE PRODUCTION (SOURCE DE VÉRITÉ) ---
@st.cache_data(show_spinner=False)
def load_production_config(path: Path) -> dict:
    """Charge config_production.json (source de vérité alignée sur main.py)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier de configuration introuvable : {path}. "
            "Assurez-vous que config_production.json est à la racine du projet."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


try:
    config = load_production_config(CONFIG_PATH)
    FEATURES_MODEL = list(config.get("features_model", []))
    COLONNES_CAT = set(config.get("colonnes_cat", []))
    SEUIL_PRODUCTION = float(config.get("seuil_production", 0.05))
    config_loaded_ok = True
    config_error = None
except Exception as e:
    config_loaded_ok = False
    config_error = str(e)
    FEATURES_MODEL, COLONNES_CAT, SEUIL_PRODUCTION = [], set(), 0.05


# --- CONSTRUCTION D'UN PAYLOAD CONFORME À LA CONFIG ---
# Valeurs par défaut cohérentes avec les types définis dans main.py (ClientData) :
#   - colonnes_cat       -> str
#   - GrandCompte, act_val_cust_3M -> bool
#   - toutes les autres  -> float
_DEFAULT_VALUES = {
    # Numériques "métier"
    "customer_value_score": 50.0,
    "Panier_Moyen_N_signature_3": 120.5,
    "Panier_Moyen_N_signature_1": 150.0,
    "Panier_Moyen_N_signature_2": 135.0,
    "EC": 12.5,
    "annees_depuis_dernier_achat": 1.5,
    "annees_depuis_1ere_facture": 4.2,
    "Turnover_N_signature_1": 3500.0,
    "Turnover_N_signature_3": 1500.0,
    "Nb_lignes_N_signature_1": 8.0,
    # Compteurs Famille
    "Famille_0_N_signature_1": 0.0,
    "Famille_1_N_signature_1": 0.0,
    "Famille_2_N_signature_1": 0.0,
    "Famille_2_N_signature_2": 0.0,
    "Famille_6_N_signature_3": 0.0,
    "Famille_11_N_signature_1": 0.0,
    # Booléens (castés en numérique côté inférence ONNX dans main.py)
    "GrandCompte": False,
    "act_val_cust_3M": True,
    # Catégorielles
    "clp_contrat_ap_stat": "BK",
    "division": "DIV1",
}


def build_sample_payload() -> dict:
    """Génère un payload aligné sur config_production.json (features_model + colonnes_cat)."""
    payload: dict = {}
    for feat in FEATURES_MODEL:
        if feat in COLONNES_CAT:
            payload[feat] = _DEFAULT_VALUES.get(feat, "UNKNOWN")
        else:
            payload[feat] = _DEFAULT_VALUES.get(feat, 0.0)
    return payload


# --- VALIDATION DU PAYLOAD CONTRE LA CONFIG ---
def validate_payload(payload: dict) -> list[str]:
    """Retourne la liste des erreurs (vide = OK). Aligne sur features_model."""
    errors: list[str] = []
    expected = set(FEATURES_MODEL)
    received = set(payload.keys())

    missing = sorted(expected - received)
    extra = sorted(received - expected)

    if missing:
        errors.append(f"Variables manquantes ({len(missing)}) : {missing}")
    if extra:
        errors.append(f"Variables non attendues par le modèle ({len(extra)}) : {extra}")

    for feat in FEATURES_MODEL:
        if feat not in payload:
            continue
        val = payload[feat]
        if feat in COLONNES_CAT:
            if not isinstance(val, str):
                errors.append(f"'{feat}' doit être une chaîne (catégoriel), reçu : {type(val).__name__}")
        else:
            # On accepte bool/int/float côté client, le backend cast proprement
            if not isinstance(val, (bool, int, float)) or isinstance(val, bool) and feat not in {
                "GrandCompte",
                "act_val_cust_3M",
            }:
                # Pas une erreur dure, on ne bloque pas : main.py gère la conversion bool->float
                pass
    return errors


# Définir 3 onglets pour bien séparer les concepts
tab1, tab2, tab3 = st.tabs(
    ["🎯 Scoring Client", "📉 Data Drift (Evidently)", "🚀 Performances & MLflow"]
)


# ONGLET 1 : SCORING CLIENT (Appel API)
with tab1:
    st.header("Tester l'API de Scoring")

    if not config_loaded_ok:
        st.error(
            f"❌ Impossible de charger la configuration de production : {config_error}\n\n"
            "L'onglet de scoring est désactivé tant que `config_production.json` n'est pas accessible."
        )
    else:
        # Bandeau d'alignement config / API
        with st.expander("ℹ️ Configuration de production chargée", expanded=False):
            st.markdown(
                f"""
                - **Features attendues par le modèle** : `{len(FEATURES_MODEL)}`
                - **Colonnes catégorielles** : `{sorted(COLONNES_CAT)}`
                - **Seuil de décision** : `{SEUIL_PRODUCTION}`
                - **Source** : `{CONFIG_PATH}`
                """
            )

        RENDER_URL = "https://p8-oc-mlops-2-2.onrender.com/predict"
        LOCAL_URL = "http://localhost:8000/predict"

        st.write("Exemple de test avec un profil client complet :")

        # 1. Payload généré dynamiquement depuis config_production.json
        sample_payload = build_sample_payload()

        # Validation côté UI avant envoi
        validation_errors = validate_payload(sample_payload)
        if validation_errors:
            st.error("❌ Payload示例 non conforme à `config_production.json` :")
            for err in validation_errors:
                st.write(f"- {err}")
        else:
            st.success(
                f"✅ Payload aligné sur les {len(FEATURES_MODEL)} features "
                f"de production (catégorielles : {sorted(COLONNES_CAT)})."
            )

        st.json(sample_payload)

        if st.button("Prédire le score"):
            if validation_errors:
                st.error("Envoi annulé : payload non conforme à la configuration.")
            else:
                with st.spinner("Appel de l'API de production..."):
                    t0 = time.perf_counter()
                    try:
                        # 2. On envoie le payload conforme à la config
                        response = requests.post(RENDER_URL, json=sample_payload, verify=False)
                        response.raise_for_status()
                        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
                        st.success(
                            f"✅ Prédiction réussie via le Cloud (Render) en {latency_ms} ms."
                        )
                        result = response.json()
                        st.json(result)

                        # Mise en évidence de la décision vs seuil de production
                        proba = result.get("probability")
                        if proba is not None:
                            decision = "🎯 CLIENT À CIBLER" if proba >= SEUIL_PRODUCTION else "CLIENT IGNORE"
                            st.metric(
                                label=f"Probabilité (seuil = {SEUIL_PRODUCTION})",
                                value=f"{proba:.4f}" if isinstance(proba, (int, float)) else proba,
                                delta=decision,
                            )

                    except requests.exceptions.RequestException as e:
                        st.warning(f"⚠️ API Cloud indisponible : {e}. Bascule sur l'API locale...")
                        try:
                            with st.spinner("Appel de l'API locale..."):
                                t0_loc = time.perf_counter()
                                response_local = requests.post(
                                    LOCAL_URL, json=sample_payload, timeout=5
                                )
                                latency_ms = round((time.perf_counter() - t0_loc) * 1000, 2)
                                if response_local.status_code == 200:
                                    st.success(
                                        f"✅ Prédiction réussie via l'API Locale (localhost) en {latency_ms} ms."
                                    )
                                    st.json(response_local.json())
                                else:
                                    st.error(
                                        f"❌ Erreur API locale (Status {response_local.status_code}) : "
                                        f"{response_local.text}"
                                    )

                        except requests.exceptions.RequestException:
                            st.error(
                                "❌ Échec total : l'API Render ET l'API locale sont inaccessibles."
                            )


# ONGLET 2 : DATA DRIFT (Lecture du rapport Evidently)
with tab2:
    st.header("Analyse de la Dérive des Données")
    st.write("Ce rapport est généré automatiquement lors de l'analyse de production.")

    if REPORTS_DIR.exists():
        html_files = list(REPORTS_DIR.glob("data_drift_report_*.html"))
        if html_files:
            # Prendre le fichier le plus récent
            latest_report = max(html_files, key=os.path.getctime)
            st.success(f"Dernier rapport chargé : {latest_report.name}")

            # Afficher le HTML directement dans Streamlit
            with open(latest_report, "r", encoding="utf-8") as f:
                html_content = f.read()
            components.html(html_content, height=1000, scrolling=True)
        else:
            st.warning("Aucun rapport Evidently trouvé dans le dossier 'reports'.")
    else:
        st.error(f"Le dossier {REPORTS_DIR} n'existe pas.")


# ONGLET 3 : PERFORMANCES & MLFLOW (Lecture du CSV)
with tab3:
    st.header("Historique des Optimisations & Santé de l'API")

    csv_path = REPORTS_DIR / "historique_optimisations.csv"
    if csv_path.exists():
        df_history = pd.read_csv(csv_path)
        st.dataframe(df_history, use_container_width=True)

        # Graphique direct sur l'évolution de la latence
        if "metrics.health_latency_mean_ms" in df_history.columns:
            st.subheader("Évolution de la Latence Moyenne (ms)")
            if "tags.optimization_phase" in df_history.columns:
                st.line_chart(
                    df_history.set_index("tags.optimization_phase")[
                        "metrics.health_latency_mean_ms"
                    ]
                )
            else:
                st.line_chart(df_history["metrics.health_latency_mean_ms"])
    else:
        st.info(
            "Le fichier d'historique MLflow n'a pas encore été généré dans le dossier 'reports'."
        )
