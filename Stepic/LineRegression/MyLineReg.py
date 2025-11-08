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
        self.last_metric_value = None

    def __str__(self):
        return format(f"MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}")

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose=False):
        X_copy = X.copy()
        X_copy.insert(0, "once", 1)
        self.weight = np.ones(X_copy.shape[1])

        X_matrix = X_copy.values
        Y_vector = y.values
        n_samples = X_matrix.shape[0]

        for i in range(self.n_iter):
            Y = X_matrix @ self.weight
            loss = Y - Y_vector
            mse = (loss**2).mean()
            grad = 2 / n_samples * (loss @ X_copy)

            self.weight = self.weight - self.learning_rate * grad

            if self.metric is None:
                self.last_metric_value = getattr(self, "_mse")(y, X_matrix @ self.weight)
            else:
                self.last_metric_value = getattr(self, "_" + self.metric)(y, X_matrix @ self.weight)  # self._calc_metric(X, y)

            if verbose and i % verbose == 0:
                if self.metric is None:
                    self.log(i, mse)
                else:
                    self.log(i, mse, self.last_metric_value)

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

    # Расчет метрик
    @staticmethod
    def _mse(y_true: np.array, y_pred: np.array):
        loss = y_pred - y_true

        return np.mean(loss**2)

    @staticmethod
    def _mae(y_true: np.array, y_pred: np.array):
        loss = y_pred - y_true

        return np.mean(np.abs(loss))

    @staticmethod
    def _rmse(y_true: np.array, y_pred: np.array):
        loss = y_pred - y_true

        return np.sqrt(np.mean(loss**2))

    @staticmethod
    def _r2(y_true: np.array, y_pred: np.array):
        loss = y_pred - y_true
        ss_res = np.sum(loss ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)

        coef_det = 1 - ss_res / ss_tot

        return coef_det

    @staticmethod
    def _mape(y_true: np.array, y_pred: np.array):
        loss = y_pred - y_true
        mape_loss = 100 * np.mean(np.abs(loss / np.where(y_true == 0, 1e-10, y_true)))

        return mape_loss

    def get_best_score(self):
        return self.last_metric_value

linReg = MyLineReg(1000, 0.1)
linReg.fit(X, y, 100)
print(linReg.get_coef())
print(linReg.predict(X))
print(y.values)
print(linReg.get_best_score())
