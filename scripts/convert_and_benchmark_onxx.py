import os
import time
import numpy as np
import pandas as pd

# Machine Learning & ONNX Conversion
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType
import onnxruntime as ort


# -------------------------------------------------------------------
# 1. Génération de données synthétiques & Entraînement du Modèle
# -------------------------------------------------------------------
def build_and_train_pipeline():
    print("🔄 [1/4] Entraînement d'un pipeline XGBoost d'exemple...")
    
    X, y = make_classification(
        n_samples=5000, 
        n_features=50, 
        n_informative=30, 
        random_state=42
    )
    # 💡 Remplacer "feature_i" par "fi"
    feature_names = [f"f{i}" for i in range(50)]
    X_df = pd.DataFrame(X, columns=feature_names)

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss"
    )
    
    model.fit(X_df, y)
    print("✅ Modèle XGBoost entraîné avec succès.")
    return model, X_df, feature_names


# -------------------------------------------------------------------
# 2. Conversion XGBoost -> ONNX
# -------------------------------------------------------------------
def convert_to_onnx(model, num_features, output_path="model_xgboost.onnx"):
    print(f"\n🔄 [2/4] Conversion du modèle vers {output_path}...")
    
    # Définition du shape d'entrée : [Batch Size (dynamique), Nombre de features]
    initial_types = [('input', FloatTensorType([None, num_features]))]
    
    # Conversion du modèle XGBoost via onnxmltools
    onnx_model = onnxmltools.convert_xgboost(
        model, 
        initial_types=initial_types,
        target_opset=15
    )
    
    # Sauvegarde sur disque
    onnxmltools.utils.save_model(onnx_model, output_path)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ Conversion réussie ! Taille du fichier ONNX : {file_size_mb:.2f} MB")
    return output_path


# -------------------------------------------------------------------
# 3. Validation de la Cohérence des Prédictions
# -------------------------------------------------------------------
def validate_predictions(xgb_model, ort_session, sample_input):
    print("\n🔍 [3/4] Vérification de la cohérence des prédictions...")
    
    # Prédiction XGBoost natif (Probabilités de la classe 1)
    xgb_probs = xgb_model.predict_proba(sample_input)[:, 1]
    
    # Prédiction ONNX Runtime
    input_name = ort_session.get_inputs()[0].name
    # ONNX attend un array float32 numpy
    onnx_input = {input_name: sample_input.to_numpy().astype(np.float32)}
    
    # Les sorties ONNX pour un classifieur sont : [labels, probabilities_map]
    onnx_outputs = ort_session.run(None, onnx_input)
    # Extraction de la probabilité de la classe 1
    onnx_probs = np.array([res[1] for res in onnx_outputs[1]])

    # Vérification du de la différence maximale
    max_diff = np.max(np.abs(xgb_probs - onnx_probs))
    print(f"Différence maximale de probabilité : {max_diff:.8f}")
    
    assert max_diff < 1e-4, "⚠️ Écart trop élevé entre XGBoost et ONNX !"
    print("✅ Validation OK : Les prédictions sont identiques.")


# -------------------------------------------------------------------
# 4. Benchmark de Latence (XGBoost vs ONNX Runtime)
# -------------------------------------------------------------------
def benchmark_latency(xgb_model, ort_session, X_sample, iterations=1000):
    print(f"\n📊 [4/4] Lancement du Benchmark ({iterations} itérations)...")
    
    # Préparation des inputs
    single_df = X_sample.iloc[[0]] # 1 seul individu (Cas usuel d'une API de scoring)
    single_numpy = single_df.to_numpy().astype(np.float32)
    input_name = ort_session.get_inputs()[0].name

    # --- Warmup (Chauffage pour éviter la sur-estimation de la 1ère exécution) ---
    for _ in range(50):
        _ = xgb_model.predict_proba(single_df)
        _ = ort_session.run(None, {input_name: single_numpy})

    # --- Benchmark XGBoost Natif ---
    start_time = time.perf_counter()
    for _ in range(iterations):
        _ = xgb_model.predict_proba(single_df)
    xgb_total_time = time.perf_counter() - start_time
    xgb_avg_ms = (xgb_total_time / iterations) * 1000

    # --- Benchmark ONNX Runtime ---
    start_time = time.perf_counter()
    for _ in range(iterations):
        _ = ort_session.run(None, {input_name: single_numpy})
    onnx_total_time = time.perf_counter() - start_time
    onnx_avg_ms = (onnx_total_time / iterations) * 1000

    # --- Résultats ---
    speedup = xgb_avg_ms / onnx_avg_ms if onnx_avg_ms > 0 else 0
    
    print("\n" + "=" * 55)
    print("📈 RÉSULTATS DU BENCHMARK DE LATENCE (Inférence Unitaire)")
    print("=" * 55)
    print(f"• XGBoost Natif : {xgb_avg_ms:.3f} ms / requête")
    print(f"• ONNX Runtime  : {onnx_avg_ms:.3f} ms / requête")
    print(f"🚀 Gain de vitesse (Speedup) : x{speedup:.2f}")
    print("=" * 55)


# -------------------------------------------------------------------
# Main Execution Flow
# -------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Entraînement
    model, X_data, feature_names = build_and_train_pipeline()
    
    # 2. Conversion
    onnx_path = convert_to_onnx(model, num_features=len(feature_names))
    
    # 3. Initialisation de la session ONNX Runtime
    # Utilise 'CPUExecutionProvider' par défaut
    ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    # 4. Validation
    validate_predictions(model, ort_session, X_data.head(10))
    
    # 5. Benchmark
    benchmark_latency(model, ort_session, X_data, iterations=2000)