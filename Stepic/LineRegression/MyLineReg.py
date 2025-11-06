from numpy.random import weibull
from sklearn.datasets import load_diabetes
import pandas as pd
import numpy as np

data = load_diabetes(as_frame=True)
X, y = data['data'], data['target']


class MyLineReg():
    def __init__(self, n_iter=100, learning_rate=0.1):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weight = 0

    def __str__(self):
        return format(f"MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}")

    def log(self, n: int, loss: int):
        print(f"{n} | loss: {loss}")

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose=False):
        X = X.copy()
        X.insert(0, "once", 1)
        self.weight = np.ones(X.shape[1])

        X_matrix = X.values
        Y_vector = y.values

        n_samples = X_matrix.shape[0]

        for i in range(self.n_iter):
            Y = X_matrix @ self.weight
            loss = Y - Y_vector
            grad = 2 / n_samples * (X_matrix.T @ loss)
            self.weight = self.weight - self.learning_rate * grad

            if verbose and i % verbose == 0:
                self.log(i, (loss**2).mean())

    def get_coef(self):
        return self.weight[1:]

    def predict(self, X: pd.DataFrame):

        X = X.copy()
        X.insert(0, "ones", 1)
        pred = X @ self.weight

        return pred.sum()

linReg = MyLineReg(50, 0.1)
linReg.fit(X, y, 10)
print(linReg.get_coef())
print(linReg.predict(X))
