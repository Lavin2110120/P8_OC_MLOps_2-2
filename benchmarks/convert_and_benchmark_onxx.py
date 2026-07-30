import os
import time
import numpy as np
import pandas as pd
import json
import datetime

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
    
    initial_types = [('input', FloatTensorType([None, num_features]))]
    
    onnx_model = onnxmltools.convert_xgboost(
        model, 
        initial_types=initial_types,
        target_opset=15
    )
    
    onnxmltools.utils.save_model(onnx_model, output_path)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ Conversion réussie ! Taille du fichier ONNX : {file_size_mb:.2f} MB")
    return output_path


# -------------------------------------------------------------------
# 3. Validation de la Cohérence des Prédictions
# -------------------------------------------------------------------
def validate_predictions(xgb_model, ort_session, sample_input):
    print("\n🔍 [3/4] Vérification de la cohérence des prédictions...")
    
    xgb_probs = xgb_model.predict_proba(sample_input)[:, 1]
    
    input_name = ort_session.get_inputs()[0].name
    onnx_input = {input_name: sample_input.to_numpy().astype(np.float32)}
    
    onnx_outputs = ort_session.run(None, onnx_input)
    onnx_probs = np.array([res[1] for res in onnx_outputs[1]])

    max_diff = np.max(np.abs(xgb_probs - onnx_probs))
    print(f"Différence maximale de probabilité : {max_diff:.8f}")
    
    assert max_diff < 1e-4, "⚠️ Écart trop élevé entre XGBoost et ONNX !"
    print("✅ Validation OK : Les prédictions sont identiques.")


# -------------------------------------------------------------------
# 4. Benchmark de Latence (XGBoost vs ONNX Runtime)
# -------------------------------------------------------------------
def benchmark_latency(xgb_model, ort_session, X_sample, iterations=1000):
    print(f"\n📊 [4/4] Lancement du Benchmark ({iterations} itérations)...")
    
    single_df = X_sample.iloc[[0]]
    single_numpy = single_df.to_numpy().astype(np.float32)
    input_name = ort_session.get_inputs()[0].name

    # --- Warmup ---
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

    # --- Calculs & Résultats ---
    speedup = xgb_avg_ms / onnx_avg_ms if onnx_avg_ms > 0 else 0
    
    print("\n" + "=" * 55)
    print("📈 RÉSULTATS DU BENCHMARK DE LATENCE (Inférence Unitaire)")
    print("=" * 55)
    print(f"• XGBoost Natif : {xgb_avg_ms:.3f} ms / requête")
    print(f"• ONNX Runtime  : {onnx_avg_ms:.3f} ms / requête")
    print(f"🚀 Gain de vitesse (Speedup) : x{speedup:.2f}")
    print("=" * 55)

    # --- Sauvegarde des résultats en JSON ---
    os.makedirs("benchmarks", exist_ok=True)
    results = {
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "iterations": iterations,
        "xgb_avg_ms": round(xgb_avg_ms, 4),
        "onnx_avg_ms": round(onnx_avg_ms, 4),
        "speedup": round(speedup, 2),
    }
    json_path = f"benchmarks/onnx_bench_{datetime.date.today()}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"💾 Résultats enregistrés dans : {json_path}\n")


# -------------------------------------------------------------------
# Main Execution Flow
# -------------------------------------------------------------------
if __name__ == "__main__":
    model, X_data, feature_names = build_and_train_pipeline()
    onnx_path = convert_to_onnx(model, num_features=len(feature_names))
    ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    validate_predictions(model, ort_session, X_data.head(10))
    benchmark_latency(model, ort_session, X_data, iterations=2000)