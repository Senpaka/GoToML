from numpy.random import weibull
from sklearn.datasets import load_diabetes
import pandas as pd
import numpy as np

data = load_diabetes(as_frame=True)
X, y = data['data'], data['target']


class MyLineReg():
    def __init__(self, n_iter=100, learning_rate=0.1, metric=None):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weight = 0
        self.metric = metric

    def __str__(self):
        return format(f"MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}")

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose=False):
        self.X_train = X.copy()
        self.y_train = y.copy()

        X_copy = X.copy()
        X_copy.insert(0, "once", 1)
        self.weight = np.ones(X_copy.shape[1])

        X_matrix = X_copy.values
        Y_vector = y.values
        n_samples = X_matrix.shape[0]

        for i in range(self.n_iter):
            Y = X_matrix @ self.weight
            loss = Y - Y_vector
            grad = 2 / n_samples * (X_matrix.T @ loss)
            self.weight = self.weight - self.learning_rate * grad

            if verbose and i % verbose == 0:
                if self.metric is None:
                    self.log(i, (loss**2).mean())
                else:
                    metric_value = self._calc_metric(X, y)
                    self.log(i, (loss**2).mean(), metric_value)

    def predict(self, X: pd.DataFrame):

        X = X.copy()
        X.insert(0, "once", 1)
        pred = X @ self.weight

        return pred # Для задания здесь была сумма

    def log(self, n: int, loss: float, metric_value=0):
        if self.metric is None:
            print(f"{n} | loss: {loss}")
        else:
            print(f"{n} | loss: {loss} | {self.metric}: {metric_value}")

    def get_coef(self):
        return self.weight[1:]

    # Получение вектора остатков
    def get_residuals(self, X: pd.DataFrame, y: pd.Series):
        X = X.copy()
        X.insert(0, "once", 1)

        X_matrix = X.values
        y_vector = y.values

        Y_pred = X_matrix @ self.weight
        return Y_pred - y_vector

    # Расчет метрик
    def mse(self, X: pd.DataFrame, y: pd.Series):
        loss = self.get_residuals(X, y)

        return np.mean(loss**2)

    def mae(self, X: pd.DataFrame, y: pd.Series):
        loss = self.get_residuals(X, y)

        return np.mean(np.abs(loss))

    def rmse(self, X: pd.DataFrame, y: pd.Series):
        loss = self.get_residuals(X, y)

        return np.sqrt(np.mean(loss**2))

    def r_2(self, X: pd.DataFrame, y: pd.Series):
        y_vector = y.values

        ss_res = np.sum(self.get_residuals(X, y) ** 2)
        ss_tot = np.sum((y_vector - np.mean(y_vector))**2)

        coef_det = 1 - ss_res / ss_tot

        return coef_det

    def mape(self, X, y):
        loss = self.get_residuals(X, y)
        mape_loss = 100 * np.mean(np.abs(loss / np.where(y.values == 0, 1e-10, y.values)))

        return mape_loss

    # Внутренний метод для выбора метрики
    def _calc_metric(self, X, y):
        if self.metric == "mse":
            return self.mse(X,y)
        elif self.metric == "mae":
            return self.mae(X, y)
        elif self.metric == "rmse":
            return self.rmse(X, y)
        elif self.metric == "r2":
            return self.r_2(X, y)
        elif self.metric == "mape":
            return self.mape(X, y)
        else:
            return self.mse(X, y)

    def get_best_score(self):
        return self._calc_metric(self.X_train, self.y_train)

linReg = MyLineReg(1000, 0.1, "mape")
linReg.fit(X, y, 100)
print(linReg.get_coef())
print(linReg.predict(X))
print(y.values)
print(linReg.mse(X, y))
print(linReg.mae(X, y))
print(linReg.rmse(X, y))
print(linReg.r_2(X, y))
print(linReg.mape(X, y))
