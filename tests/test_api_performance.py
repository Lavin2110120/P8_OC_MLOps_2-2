import pytest
from httpx import AsyncClient, ASGITransport
from src.main import app


@pytest.mark.asyncio
async def test_async_predict_endpoint(valid_payload):  # Passer le fixture valide
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/predict", json=valid_payload)
        assert response.status_code == 200