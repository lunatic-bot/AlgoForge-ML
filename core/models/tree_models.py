# Tree Models - RandomForestRunner, DecisionTreeRunner
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from core.base_model import BaseMLModel


class RandomForestRunner(BaseMLModel):
    """Random Forest model runner for classification and regression."""

    def __init__(self, n_estimators=100, random_state=42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = None

    def fit(self, X, y):
        if len(set(y)) <= 2:
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state
            )
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)


class DecisionTreeRunner(BaseMLModel):
    """Decision Tree model runner for classification and regression."""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None

    def fit(self, X, y):
        if len(set(y)) <= 2:
            self.model = DecisionTreeClassifier(random_state=self.random_state)
        else:
            self.model = DecisionTreeRegressor(random_state=self.random_state)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)