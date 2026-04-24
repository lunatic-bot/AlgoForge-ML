# Distance-based Models - KNNRunner, SVMRunner
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from core.base_model import BaseMLModel


class KNNRunner(BaseMLModel):
    """K-Nearest Neighbors model runner for classification and regression."""

    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.model = None

    def fit(self, X, y):
        # Determine if classification or regression based on y values
        if len(set(y)) <= 2:
            self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        else:
            self.model = KNeighborsRegressor(n_neighbors=self.n_neighbors)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)


class SVMRunner(BaseMLModel):
    """Support Vector Machine model runner for classification and regression."""

    def __init__(self, kernel='rbf', random_state=42):
        self.kernel = kernel
        self.random_state = random_state
        self.model = None

    def fit(self, X, y):
        # Determine if classification or regression based on y values
        if len(set(y)) <= 2:
            self.model = SVC(kernel=self.kernel, random_state=self.random_state)
        else:
            self.model = SVR(kernel=self.kernel)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)