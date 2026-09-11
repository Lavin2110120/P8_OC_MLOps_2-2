import http from 'k6/http';
import { check } from 'k6';

export const options = {
  stages: [
    { duration: '10s', target: 50 },   // Montée rapide
    { duration: '20s', target: 200 },  // Charge lourde
    { duration: '30s', target: 500 },  // Seuil de rupture potentiel
    { duration: '10s', target: 0 },
  ],
  thresholds: {
    // Alerte si le taux d'erreur dépasse 5% ou si p95 > 2000ms
    http_req_failed: ['rate<0.05'],
    http_req_duration: ['p(95)<2000'],
  },
};

export default function () {
  const url = 'http://127.0.0.1:8000/predict';
  
  const payload = JSON.stringify({
    customer_value_score: 50.0,
    Panier_Moyen_N_signature_3: 120.5,
    clp_contrat_ap_stat: "BK",
    GrandCompte: false,
    Turnover_N_signature_1: 3500.0,
    annees_depuis_dernier_achat: 1.5,
    Panier_Moyen_N_signature_1: 150.0,
    Turnover_N_signature_3: 1500.0,
    EC: 12.5,
    Panier_Moyen_N_signature_2: 135.0,
    Famille_11_N_signature_1: 0.0,
    annees_depuis_1ere_facture: 4.2,
    Famille_0_N_signature_1: 0.0,
    Famille_2_N_signature_2: 0.0,
    Nb_lignes_N_signature_1: 8.0,
    Famille_2_N_signature_1: 0.0,
    act_val_cust_3M: true,
    Famille_1_N_signature_1: 0.0,
    division: "DIV1",
    Famille_6_N_signature_3: 0.0
  });

  const params = {
    headers: { 'Content-Type': 'application/json' },
  };

  const res = http.post(url, payload, params);

  check(res, {
    'status est 200': (r) => r.status === 200,
  });

}