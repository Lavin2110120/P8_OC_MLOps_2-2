import streamlit as st
import pandas as pd
import requests
import streamlit.components.v1 as components
from pathlib import Path
import os

# --- Définition des chemins dynamiques ---
# BASE_DIR correspond au dossier parent de 'dashboard', donc la racine du projet
BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"

st.set_page_config(page_title="Dashboard MLOps", layout="wide")
st.title("📊 Dashboard MLOps - Scoring Client & Monitoring")

# Définir 3 onglets pour bien séparer les concepts
tab1, tab2, tab3 = st.tabs(["🎯 Scoring Client", "📉 Data Drift (Evidently)", "🚀 Performances & MLflow"])

# ONGLET 1 : SCORING CLIENT (Appel API)
with tab1:
    st.header("Tester l'API de Scoring")
    
    RENDER_URL = "https://p8-oc-mlops-2-2.onrender.com/predict" 
    LOCAL_URL = "http://localhost:8000/predict"
    
    st.write("Exemple de test avec un profil client complet :")
    
    # 1. On prépare un dictionnaire avec TOUTES les variables attendues par l'API
    sample_payload = {
        "customer_value_score": 50.0,
        "clp_contrat_ap_stat": "BK",
        "Panier_Moyen_N_signature_3": 120.5,
        "GrandCompte": False,
        "act_val_cust_3M": True,
        "Nb_lignes_N_signature_4": 5.0,
        "EC": 12.5,
        "annees_depuis_dernier_achat": 1.5,
        "Panier_Moyen_N_signature_1": 150.0,
        "Nb_lignes_N_signature_1": 8.0,
        "Panier_Moyen_N_signature_2": 135.0,
        "Turnover_N_signature_3": 1500.0,
        "Turnover_N_signature_1": 3500.0,
        "Famille_5_N_signature_1": 0.0,
        "Famille_2_N_signature_2": 0.0,
        "annees_depuis_1ere_facture": 4.2,
        "Famille_11_N_signature_1": 0.0,
        "Famille_14_N_signature_2": 0.0,
        "Famille_1_N_signature_1": 0.0,
        "Famille_2_N_signature_1": 0.0,
    }
    
    # On l'affiche dans Streamlit pour que l'utilisateur voie ce qui est envoyé
    st.json(sample_payload)
    
    if st.button("Prédire le score"):
        with st.spinner("Appel de l'API de production..."):
            try:
                # 2. On envoie json=sample_payload au lieu de json={"client_id": ...}
                response = requests.post(RENDER_URL, json=sample_payload, timeout=15)
                response.raise_for_status() 
                st.success("✅ Prédiction réussie via le Cloud (Render) !")
                st.json(response.json())
                    
            except requests.exceptions.RequestException as e:
                st.warning(f"⚠️ API Cloud indisponible. Bascule sur l'API locale...")
                try:
                    with st.spinner("Appel de l'API locale..."):
                        # 3. Pareil pour l'API locale
                        response_local = requests.post(LOCAL_URL, json=sample_payload, timeout=5)
                        if response_local.status_code == 200:
                            st.success("✅ Prédiction réussie via l'API Locale (localhost) !")
                            st.json(response_local.json())
                        else:
                            st.error(f"❌ Erreur API locale (Status {response_local.status_code}) : {response_local.text}")
                            
                except requests.exceptions.RequestException:
                    st.error("❌ Échec total : L'API Render ET l'API locale sont inaccessibles.")

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
            st.line_chart(df_history.set_index("tags.optimization_phase")["metrics.health_latency_mean_ms"])
    else:
        st.info("Le fichier d'historique MLflow n'a pas encore été généré dans le dossier 'reports'.")