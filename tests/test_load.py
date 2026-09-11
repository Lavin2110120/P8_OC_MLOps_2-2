import asyncio
import time

import pytest


class TestLoadScaling:
    """Vérifie la stabilité de /predict à différentes échelles de charge."""

    @pytest.mark.parametrize("n_requests", [1, 10])
    @pytest.mark.asyncio
    async def test_predict_sequential_scaling(self, async_client, valid_payload, n_requests):
        """Requêtes séquentielles : l'API doit répondre 200 à chaque appel,
        quel que soit le volume, sans dégradation de latence anormale."""
        latencies = []
        errors = []

        for i in range(n_requests):
            start = time.perf_counter()
            response = await async_client.post("/predict", json=valid_payload)
            elapsed_ms = (time.perf_counter() - start) * 1000

            if response.status_code != 200:
                errors.append((i, response.status_code, response.text[:200]))
            else:
                latencies.append(elapsed_ms)

        assert not errors, f"{len(errors)}/{n_requests} échecs : {errors[:5]}"
        assert len(latencies) == n_requests

        mean_latency = sum(latencies) / len(latencies)
        print(f"\n[n={n_requests}] latence moyenne={mean_latency:.2f}ms")

        assert mean_latency < 200, f"Latence moyenne trop élevée à n={n_requests} : {mean_latency:.2f}ms"

    @pytest.mark.slow
    @pytest.mark.parametrize("n_requests", [1000])
    @pytest.mark.asyncio
    async def test_predict_sequential_scaling_large(self, async_client, valid_payload, n_requests):
        """Version lourde (1000 requêtes), exclue par défaut de la CI rapide."""
        errors = []
        for i in range(n_requests):
            response = await async_client.post("/predict", json=valid_payload)
            if response.status_code != 200:
                errors.append((i, response.status_code, response.text[:200]))
        assert not errors, f"{len(errors)}/{n_requests} échecs : {errors[:5]}"

    @pytest.mark.parametrize("n_requests", [1, 10])
    @pytest.mark.asyncio
    async def test_predict_concurrent_scaling(self, async_client, valid_payload, n_requests):
        """Requêtes concurrentes (asyncio.gather) : vérifie l'absence de
        race condition sur la session ONNX partagée et la stabilité globale."""
        async def _one_call():
            return await async_client.post("/predict", json=valid_payload)

        start = time.perf_counter()
        responses = await asyncio.gather(*[_one_call() for _ in range(n_requests)])
        elapsed_s = time.perf_counter() - start

        status_codes = [r.status_code for r in responses]
        n_success = sum(1 for s in status_codes if s == 200)
        n_error = n_requests - n_success

        throughput = n_requests / elapsed_s if elapsed_s > 0 else 0
        print(f"\n[concurrent n={n_requests}] succès={n_success}/{n_requests} | "
              f"throughput={throughput:.1f} req/s | durée={elapsed_s:.2f}s")

        assert n_error == 0, (
            f"{n_error}/{n_requests} requêtes concurrentes en échec. "
            f"Codes observés : {set(status_codes)}"
        )

        predictions = [r.json()["prediction"] for r in responses if r.status_code == 200]
        assert len(set(predictions)) == 1, (
            "Incohérence : des requêtes identiques donnent des prédictions différentes "
            "→ possible race condition sur la session ONNX partagée."
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("n_requests", [1000])
    @pytest.mark.asyncio
    async def test_predict_concurrent_scaling_large(self, async_client, valid_payload, n_requests):
        """Version lourde concurrente (1000 requêtes), exclue par défaut de la CI rapide."""
        async def _one_call():
            return await async_client.post("/predict", json=valid_payload)

        responses = await asyncio.gather(*[_one_call() for _ in range(n_requests)])
        status_codes = [r.status_code for r in responses]
        n_error = sum(1 for s in status_codes if s != 200)

        assert n_error == 0, f"{n_error}/{n_requests} échecs concurrents"

    def test_health_survives_1000_requests(self, client_sync):
        """Sanity check rapide sur un endpoint léger à très haute fréquence."""
        for _ in range(1000):
            response = client_sync.get("/health")
            assert response.status_code == 200