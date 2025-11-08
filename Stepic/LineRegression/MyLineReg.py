from sklearn.datasets import load_diabetes
import pandas as pd
import numpy as np

data = load_diabetes(as_frame=True)
X, y = data['data'], data['target']


class MyLineReg:
    def __init__(self, n_iter=100, learning_rate=0.1, metric=None, reg=None, l1_coef=0., l2_coef=0.):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weight = None
        self.metric = metric
        self.last_metric_value = None
        self.last_loss = None
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef

    def __str__(self):
        return format(f"MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}")

    # Расчет метрик
    @staticmethod
    def _mse(y_true: np.array, y_pred: np.array):
        diff = y_pred - y_true

        return np.mean(diff ** 2)

    @staticmethod
    def _mae(y_true: np.array, y_pred: np.array):
        diff = y_pred - y_true

        return np.mean(np.abs(diff))

    @staticmethod
    def _rmse(y_true: np.array, y_pred: np.array):
        diff = y_pred - y_true

        return np.sqrt(np.mean(diff ** 2))

    @staticmethod
    def _r2(y_true: np.array, y_pred: np.array):
        diff = y_pred - y_true
        ss_res = np.sum(diff ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

        coef_det = 1 - ss_res / ss_tot

        return coef_det

    @staticmethod
    def _mape(y_true: np.array, y_pred: np.array):
        diff = y_pred - y_true
        mape_loss = 100 * np.mean(np.abs(diff / np.where(y_true == 0, 1e-10, y_true)))

        return mape_loss

    # Методы регуляризации

    @staticmethod
    def _l1(weight, loss_grad, loss, l1_coef):
        grad = loss_grad + l1_coef * np.sign(weight)
        loss = loss + l1_coef * np.sum(np.abs(weight))
        return grad, loss

    @staticmethod
    def _l2(weight, loss_grad, loss, l2_coef):
        grad = loss_grad + l2_coef * 2 * weight
        loss = loss + l2_coef * np.sum(weight ** 2)
        return grad, loss

    @staticmethod
    def _elasticnet(weight, loss_grad, loss, l1_coef, l2_coef):
        grad = loss_grad + l1_coef * np.sign(weight) + l2_coef * 2 * weight
        loss = loss + l1_coef * np.sum(np.abs(weight)) + l2_coef * np.sum(weight ** 2)
        return grad, loss

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose=0):
        X_copy = X.copy()
        X_copy.insert(0, "once", 1)
        self.weight = np.ones(X_copy.shape[1])

        X_matrix = X_copy.values
        Y_vector = y.values
        n_samples = X_matrix.shape[0]

        for i in range(self.n_iter):
            Y = X_matrix @ self.weight
            diff = Y - Y_vector
            loss = (diff ** 2).mean()
            self.last_loss = loss

            loss_grad = 2 / n_samples * (X_matrix.T @ diff)
            grad, loss = self.apply_reg(loss_grad, loss)
            #grad = getattr(self, "_" + self.reg)(self.weight, loss_grad, self.l1_coef, self.l2_coef)

            self.weight = self.weight - self.learning_rate * grad

            if self.metric is None:
                self.last_metric_value = getattr(self, "_mse")(y, X_matrix @ self.weight)
            else:
                self.last_metric_value = getattr(self, "_" + self.metric)(y,
                                                                          X_matrix @ self.weight)

            if verbose and i % verbose == 0:
                if self.metric is None:
                    self.log(i, loss)
                else:
                    self.log(i, loss, self.last_metric_value)

    def predict(self, X: pd.DataFrame):
        X = X.copy()
        X.insert(0, "once", 1)

        return X @ self.weight  # Для задания здесь была сумма

    def get_coef(self):
        return self.weight[1:]

    def get_last_metric(self):
        return self.last_metric_value

    def get_last_loss(self):
        return self.last_loss

    def apply_reg(self, loss_grad, loss):
        if self.reg in ["l1", "l2", "elasticnet"]:
            func = getattr(self, f"_{self.reg}")
            if self.reg == "l1":
                return func(self.weight, loss_grad, loss, self.l1_coef)
            elif self.reg == "l2":
                return func(self.weight, loss_grad, loss, self.l2_coef)
            else:
                return func(self.weight, loss_grad, loss, self.l1_coef, self.l2_coef)
        return loss_grad, loss

    def log(self, n: int, loss: float, metric_value=0):
        if self.metric is None:
            print(f"{n} | loss: {loss}")
        else:
            print(f"{n} | loss: {loss} | {self.metric}: {metric_value}")


linReg = MyLineReg(10000, 0.1, reg="elasticnet", l1_coef=0.5, l2_coef=0.1)
linReg.fit(X, y)
print(linReg.get_coef())
print(linReg.predict(X))
print(y.values)
print(linReg.get_last_metric())
print(linReg.get_last_loss())
