import os
import random
import time
from typing import Dict, Any
import requests

# 🌐 URL de ton API déployée sur Render (ou http://127.0.0.1:8000 pour des tests locaux)
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")

# 📊 Payload de base conforme au schéma ClientData
BASE_PAYLOAD: Dict[str, Any] = {
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


def generate_varied_payload(drift: bool = False) -> Dict[str, Any]:
    """
    Génère un payload avec des variations aléatoires autour des valeurs de base.
    Si drift=True, injecte une dérive statistique (valeurs plus élevées).
    """
    payload = BASE_PAYLOAD.copy()

    # Facteur de dérive
    drift_factor = random.uniform(1.5, 3.0) if drift else 1.0

    # Perturbations réalistes
    payload["Panier_Moyen_N_signature_1"] = round(
        random.uniform(50.0, 300.0) * drift_factor, 2
    )
    payload["Panier_Moyen_N_signature_2"] = round(
        random.uniform(40.0, 250.0) * drift_factor, 2
    )
    payload["Panier_Moyen_N_signature_3"] = round(
        random.uniform(30.0, 200.0) * drift_factor, 2
    )
    payload["Turnover_N_signature_1"] = round(
        random.uniform(1000.0, 8000.0) * drift_factor, 2
    )
    payload["annees_depuis_dernier_achat"] = round(
        random.uniform(0.1, 5.0) * (1.8 if drift else 1.0), 2
    )
    payload["%EC"] = round(random.uniform(0.0, 40.0), 2)
    payload["GrandCompte"] = random.choice([True, False])
    payload["act_val_cust_3M"] = random.choice([True, False])

    # Gestion optionnelle des champs nullables
    if random.random() < 0.1:  # 10% de chance d'avoir None sur customer_value_score
        payload["customer_value_score"] = None
    else:
        payload["customer_value_score"] = round(
            random.uniform(10.0, 90.0) * drift_factor, 2
        )

    return payload


def run_simulation(num_requests: int = 50, delay_seconds: float = 0.5):
    """
    Envoie 'num_requests' requêtes à l'API Render avec un mix de données normales et dérivées.
    """
    print(f"🚀 Démarrage de la simulation de trafic vers : {API_URL}")
    print(f"📦 Nombre de requêtes prévues : {num_requests}\n")

    success_count = 0
    error_count = 0

    for i in range(1, num_requests + 1):
        # On injecte du drift artificiel sur les 30% dernières requêtes
        is_drifted = i > int(num_requests * 0.7)
        payload = generate_varied_payload(drift=is_drifted)

        try:
            response = requests.post(API_URL, json=payload, timeout=10)
            if response.status_code == 200:
                data = response.json()
                success_count += 1
                pred = data.get("prediction")
                proba = round(data.get("probability", 0.0), 4)
                drift_tag = "⚠️ [DRIFT]" if is_drifted else "🟢 [NORMAL]"
                print(
                    f"Req #{i:02d} | Status: 200 OK | Pred: {pred} | Proba: {proba} | {drift_tag}"
                )
            else:
                error_count += 1
                print(f"Req #{i:02d} | Status: {response.status_code} | Error: {response.text}")

        except Exception as e:
            error_count += 1
            print(f"Req #{i:02d} | Échec de connexion : {e}")

        time.sleep(delay_seconds)

    print("\n" + "=" * 40)
    print("📊 Bilan de la simulation :")
    print(f" - Requêtes réussies : {success_count}/{num_requests}")
    print(f" - Échecs/Erreurs    : {error_count}/{num_requests}")
    print("=" * 40)


if __name__ == "__main__":
    # Ajuste le nombre de requêtes souhaité
    run_simulation(num_requests=200, delay_seconds=0.3)