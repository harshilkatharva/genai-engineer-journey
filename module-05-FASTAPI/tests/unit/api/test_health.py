def test_health(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200

    response = response.json()
    assert isinstance(response, dict)
