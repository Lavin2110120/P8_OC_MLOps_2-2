import cProfile
import pstats
import io
import time
import numpy as np
from pstats import SortKey

# -------------------------------------------------------------------
# 1. OPTION A : Profilage direct d'une fonction d'inférence
# -------------------------------------------------------------------
def profile_function(predict_fn, sample_data, num_iterations=100, top_n=20, output_file="inference_stats.prof"):
    print(f"\n" + "=" * 60)
    print(f"📊 PROFILAGE DIRECT : {predict_fn.__name__} ({num_iterations} itérations)")
    print(f"=" * 60)

    profiler = cProfile.Profile()
    
    # Warmup
    _ = predict_fn(sample_data)

    # Lancement du profilage
    profiler.enable()
    start_time = time.perf_counter()
    
    for _ in range(num_iterations):
        _ = predict_fn(sample_data)
        
    elapsed_time = time.perf_counter() - start_time
    profiler.disable()

    # 💾 SAUVEGARDE DU FICHIER POUR SNAKEVIZ
    profiler.dump_stats(output_file)
    print(f"✅ Statistiques enregistrées dans : {output_file}")

    # Affichage classique dans la console
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(top_n)
    
    print(f"⏱️ Temps total : {elapsed_time:.4f}s | Moyen : {(elapsed_time / num_iterations) * 1000:.2f} ms/it")
    print(stream.getvalue())


# -------------------------------------------------------------------
# 2. OPTION B : Middleware FastAPI pour profiler les endpoints HTTP
# -------------------------------------------------------------------
def add_cprofile_middleware(app, output_file="api_profile.prof"):
    """
    Middleware FastAPI qui enregistre le profilage de chaque requête dans un fichier .prof
    particulièrement utile avec SnakeViz ou pyprof2calltree.
    """
    from fastapi import Request

    @app.middleware("http")
    async def profile_request(request: Request, call_next):
        profiler = cProfile.Profile()
        profiler.enable()

        response = await call_next(request)

        profiler.disable()
        # Exporte les statistiques au format cProfile
        profiler.dump_stats(output_file)
        profiler.dump_stats("inference_stats.prof")
        return response

    print(f"✅ Middleware cProfile activé. Sauvegarde dans : {output_file}")


# -------------------------------------------------------------------
# 3. EXEMPLE D'UTILISATION (Dummy Model / ONNX)
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Simulation d'une fonction de prédiction (remplace par ton modèle chargé)
    def dummy_predict_pipeline(data):
        # Ex: Prétraitement
        scaled_data = data * 1.05 + 0.2
        # Ex: Inférence
        predictions = np.dot(scaled_data, np.ones((scaled_data.shape[1], 1)))
        # Ex: Post-traitement / Formatting
        return predictions.tolist()

    # Génération de données synthétiques (ex: 50 features)
    dummy_input = np.random.randn(1, 50)

    # Exécution du profilage direct
    profile_function(
        predict_fn=dummy_predict_pipeline,
        sample_data=dummy_input,
        num_iterations=500,
        top_n=15
    )