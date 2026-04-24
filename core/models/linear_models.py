# Linear Models - LogisticRegressionRunner, etc.
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from core.base_model import BaseMLModel


class LogisticRegressionRunner(BaseMLModel):
    """Logistic Regression model runner for classification."""

    def __init__(self, random_state=42, max_iter=1000):
        self.random_state = random_state
        self.max_iter = max_iter
        self.model = None

    def fit(self, X, y):
        self.model = LogisticRegression(
            random_state=self.random_state,
            max_iter=self.max_iter
        )
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)


class LinearRegressionRunner(BaseMLModel):
    """Linear Regression model runner for regression tasks."""

    def __init__(self):
        self.model = None

    def fit(self, X, y):
        self.model = LinearRegression()
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)


class RidgeRunner(BaseMLModel):
    """Ridge Regression model runner."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.model = None

    def fit(self, X, y):
        self.model = Ridge(alpha=self.alpha)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)


class LassoRunner(BaseMLModel):
    """Lasso Regression model runner."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.model = None

    def fit(self, X, y):
        self.model = Lasso(alpha=self.alpha)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)