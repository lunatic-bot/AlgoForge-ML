# Ensure endpoints return correct status codes
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


class TestRootEndpoint:
    def test_root(self):
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()

    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


class TestModelsEndpoint:
    def test_list_models(self):
        response = client.get("/api/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert isinstance(data["models"], list)
        assert "random_forest" in data["models"]


class TestDatasetsEndpoint:
    def test_list_datasets(self):
        response = client.get("/api/datasets")
        assert response.status_code == 200
        data = response.json()
        assert "datasets" in data
        assert isinstance(data["datasets"], list)


class TestTrainEndpoint:
    def test_train_missing_data(self):
        response = client.post(
            "/api/train",
            json={
                "model_type": "random_forest",
                "data_path": "nonexistent.csv",
                "target_column": "target"
            }
        )
        # Should fail because file doesn't exist
        assert response.status_code == 500

    def test_train_invalid_model(self):
        response = client.post(
            "/api/train",
            json={
                "model_type": "invalid_model",
                "data_path": "data/raw/titanic.csv",
                "target_column": "survived"
            }
        )
        assert response.status_code == 400


class TestPredictEndpoint:
    def test_predict_model_not_found(self):
        response = client.post(
            "/api/predict",
            json={
                "model_id": "nonexistent_model",
                "features": [1, 2, 3, 4, 5]
            }
        )
        assert response.status_code == 404

    def test_predict_invalid_features(self):
        response = client.post(
            "/api/predict",
            json={
                "model_id": "test_model",
                "features": []
            }
        )
        # Should fail due to empty features
        assert response.status_code in [404, 500]