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
    RENDER_URL= "postgresql+asyncpg://p8_oc_mlops_part2_db_user:llJ81uZProdA5WKLyn6xs3PEldHsa0mo@dpg-daaive942hec73akonog-a.frankfurt-postgres.render.com/p8_oc_mlops_part2_db?ssl=require"
    LOCAL_URL = "http://localhost:8000/predict"
    client_id = st.number_input("Entrez l'ID du client (ou utilisez un ID de test)", min_value=100000, step=1, value=100001)
    
    if st.button("Prédire le score"):
        # On ajoute un "spinner" visuel pour que l'utilisateur patiente
        with st.spinner("Appel de l'API de production (Render)..."):
            try:
                # 1ère tentative : API Render (timeout de 15s au cas où l'API est en veille)
                response = requests.post(RENDER_URL, json={"client_id": client_id}, timeout=15)
                
                if response.status_code == 200:
                    st.success("✅ Prédiction réussie via le Cloud (Render) !")
                    st.json(response.json())
                else:
                    # Si Render répond mais avec une erreur (ex: 500), on force le passage au bloc except
                    response.raise_for_status() 
                    
            except requests.exceptions.RequestException as e:
                st.warning(f"⚠️ API Cloud indisponible (ou en veille). Bascule sur l'API locale...")
                
                # 2ème tentative : API Locale
                try:
                    with st.spinner("Appel de l'API locale..."):
                        response_local = requests.post(LOCAL_URL, json={"client_id": client_id}, timeout=5)
                        
                        if response_local.status_code == 200:
                            st.success("✅ Prédiction réussie via l'API Locale (localhost) !")
                            st.json(response_local.json())
                        else:
                            st.error(f"❌ Erreur API locale (Status {response_local.status_code})")
                            
                except requests.exceptions.RequestException:
                    st.error("❌ Échec total : L'API Render ET l'API locale sont inaccessibles. Pense à lancer `uvicorn` pour ton API locale !")

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