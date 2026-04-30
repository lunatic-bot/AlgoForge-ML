# # Ensure your ML classes work isolated from the API
# import pytest
# import numpy as np
# from core.models.tree_models import RandomForestRunner, DecisionTreeRunner
# from core.models.linear_models import LogisticRegressionRunner, LinearRegressionRunner
# from core.models.distance import KNNRunner, SVMRunner
# from core.data_loader import DataLoader


# # Sample data for testing
# @pytest.fixture
# def sample_data():
#     X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
#     y = np.array([0, 0, 1, 1, 1])
#     return X, y


# class TestRandomForestRunner:
#     def test_fit(self, sample_data):
#         X, y = sample_data
#         model = RandomForestRunner()
#         model.fit(X, y)
#         assert model.model is not None

#     def test_predict(self, sample_data):
#         X, y = sample_data
#         model = RandomForestRunner()
#         model.fit(X, y)
#         predictions = model.predict(X)
#         assert len(predictions) == len(y)

#     def test_score(self, sample_data):
#         X, y = sample_data
#         model = RandomForestRunner()
#         model.fit(X, y)
#         score = model.score(X, y)
#         assert 0 <= score <= 1


# class TestDecisionTreeRunner:
#     def test_fit(self, sample_data):
#         X, y = sample_data
#         model = DecisionTreeRunner()
#         model.fit(X, y)
#         assert model.model is not None

#     def test_predict(self, sample_data):
#         X, y = sample_data
#         model = DecisionTreeRunner()
#         model.fit(X, y)
#         predictions = model.predict(X)
#         assert len(predictions) == len(y)


# class TestLogisticRegressionRunner:
#     def test_fit(self, sample_data):
#         X, y = sample_data
#         model = LogisticRegressionRunner()
#         model.fit(X, y)
#         assert model.model is not None

#     def test_predict(self, sample_data):
#         X, y = sample_data
#         model = LogisticRegressionRunner()
#         model.fit(X, y)
#         predictions = model.predict(X)
#         assert len(predictions) == len(y)


# class TestLinearRegressionRunner:
#     def test_fit(self):
#         X = np.array([[1], [2], [3], [4], [5]])
#         y = np.array([2, 4, 6, 8, 10])
#         model = LinearRegressionRunner()
#         model.fit(X, y)
#         assert model.model is not None

#     def test_predict(self):
#         X = np.array([[1], [2], [3], [4], [5]])
#         y = np.array([2, 4, 6, 8, 10])
#         model = LinearRegressionRunner()
#         model.fit(X, y)
#         predictions = model.predict(X)
#         assert len(predictions) == len(y)


# class TestKNNRunner:
#     def test_fit(self, sample_data):
#         X, y = sample_data
#         model = KNNRunner()
#         model.fit(X, y)
#         assert model.model is not None

#     def test_predict(self, sample_data):
#         X, y = sample_data
#         model = KNNRunner()
#         model.fit(X, y)
#         predictions = model.predict(X)
#         assert len(predictions) == len(y)


# class TestSVMRunner:
#     def test_fit(self, sample_data):
#         X, y = sample_data
#         model = SVMRunner()
#         model.fit(X, y)
#         assert model.model is not None

#     def test_predict(self, sample_data):
#         X, y = sample_data
#         model = SVMRunner()
#         model.fit(X, y)
#         predictions = model.predict(X)
#         assert len(predictions) == len(y)


# class TestDataLoader:
#     def test_split(self):
#         X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
#         y = np.array([0, 1, 0, 1])
#         loader = DataLoader()
#         X_train, X_test, y_train, y_test = loader.split(X, y)
#         assert len(X_train) + len(X_test) == len(X)

#     def test_scale_features(self):
#         X_train = np.array([[1, 2], [3, 4], [5, 6]])
#         X_test = np.array([[7, 8], [9, 10]])
#         loader = DataLoader()
#         X_train_scaled, X_test_scaled = loader.scale_features(X_train, X_test)
#         assert X_train_scaled.shape == X_train.shape
#         assert X_test_scaled.shape == X_test.shape