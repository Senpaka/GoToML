import pandas as pd
import numpy as np
import random

class MyLineReg:
    """
    Линейная регрессия

    Параметры:
    n_iter : int
        Кол-во итерация для градиентного спуска
    learning_rate : float
        Скорость обучения
    metric : str
        Метрика для оценки качества модели ["mse", "mae", "rmse", "r2", "mape"]
    reg : str
        Тип регуляризации ["l1", "l2", "elasticnet"]
    l1_coef : float
        Коэффициент L1-регуляризации
    l2_coef : float
        Коэффициент L2-регуляризации
    sgd_sample : float, int
        Rол-во образцов, которое будет использоваться на каждой итерации обучения. Может принимать либо целые числа, либо дробные от 0.0 до 1.0.
    random_state : int
        Сид для многократноой выборки одинаковых наборов батчей

    Атрибуты:
    weight : np.array
        Вектор весов модели, включая bias
    last_metric : float
        Последнее значение выбранной метрики (без учета регуляризации)
    last_loss : float
        Последнее значение loss (без учетом регуляризации)
    """

    def __init__(self, n_iter=100, learning_rate=0.1, metric=None, reg=None, l1_coef=0., l2_coef=0., sgd_sample=None,
                 random_state=42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weight = None
        self.metric = metric
        self.last_metric = None
        self.last_loss = None
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state

    def __str__(self):
        return format(f"MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}")

    # ------------------------Метрики------------------------#
    @staticmethod
    def _mse(y_true: np.ndarray, y_pred: np.ndarray):
        diff = y_pred - y_true

        return np.mean(diff ** 2)

    @staticmethod
    def _mae(y_true: np.ndarray, y_pred: np.ndarray):
        return np.mean(np.abs(y_pred - y_true))

    @staticmethod
    def _rmse(y_true: np.ndarray, y_pred: np.ndarray):
        return np.sqrt(np.mean((y_pred - y_true) ** 2))

    @staticmethod
    def _r2(y_true: np.ndarray, y_pred: np.ndarray):
        ss_res = np.sum((y_pred - y_true) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

        return 1 - ss_res / ss_tot

    @staticmethod
    def _mape(y_true: np.ndarray, y_pred: np.ndarray):
        diff = y_pred - y_true
        mape_loss = 100 * np.mean(np.abs(diff / np.where(y_true == 0, 1e-10, y_true)))

        return mape_loss

    # ------------------------Регуляризация------------------------#
    @staticmethod
    def _l1(weight, loss_grad, loss, l1_coef):
        grad = loss_grad.copy()
        grad[1:] += l1_coef * np.sign(weight[1:])
        loss += l1_coef * np.sum(np.abs(weight[1:]))
        return grad, loss

    @staticmethod
    def _l2(weight, loss_grad, loss, l2_coef):
        grad = loss_grad.copy()
        grad[1:] += l2_coef * 2 * weight[1:]
        loss += l2_coef * np.sum(weight[1:] ** 2)
        return grad, loss

    @staticmethod
    def _elasticnet(weight, loss_grad, loss, l1_coef, l2_coef):
        grad = loss_grad.copy()
        grad[1:] += l1_coef * np.sign(weight[1:]) + l2_coef * 2 * weight[1:]
        loss += l1_coef * np.sum(np.abs(weight[1:])) + l2_coef * np.sum(weight[1:] ** 2)
        return grad, loss

    # ------------------------Батчи------------------------#
    @staticmethod
    def _get_batch(X: pd.DataFrame, y: pd.Series, sgd_sample) -> tuple[np.ndarray, np.ndarray]:
        if sgd_sample is None:
            return X.values, y.values
        else:
            if type(sgd_sample) is float:
                sgd_sample = int(sgd_sample * X.shape[0])

            sample_row_index = random.sample(range(X.shape[0]), sgd_sample)
            X_matrix = X.iloc[sample_row_index].values
            Y_vector = y.iloc[sample_row_index].values
            return X_matrix, Y_vector

    # ------------------------Обновление весов------------------------#
    def _update_weight(self, X_batch: np.ndarray, y_batch: np.ndarray, batch_size: int, i: int) -> float:
        Y_pred_batch = X_batch @ self.weight
        diff = Y_pred_batch - y_batch
        loss = (diff ** 2).mean()
        loss_grad = 2 / batch_size * (X_batch.T @ diff)
        grad, loss = self.apply_reg(loss_grad, loss)

        lr = self.learning_rate(i) if callable(self.learning_rate) else self.learning_rate
        self.weight -= lr * grad

        return loss

    # ------------------------Обучение------------------------#
    def fit(self, X: pd.DataFrame, y: pd.Series, verbose=0):
        """
        Обучение модели с помощью градиентного спуска


        Параметры:
        X : pd.DataFrame
            Признаки (фичи, features)
        y : pd.Series
            Целевая переменная
        verbose : int
            Каждое verbose-е значение показывает лог

        Поведение:
        Обучение будет происходить на случайных мини батчах, если указан sgd_sample
        Вычисляет loss, если указана то с регуляризацией,
        Считает градиент и обновляет веса
        Сохраняет last_metric и last_loss (На полном наборе данных)
        """

        random.seed(self.random_state)

        X_copy = X.copy()
        X_copy.insert(0, "once", 1)
        self.weight = np.ones(X_copy.shape[1])

        X_matrix = X_copy.values
        Y_vector = y.values

        for i in range(1, self.n_iter + 1):

            X_batch, y_batch = self._get_batch(X_copy, y, self.sgd_sample)
            batch_size = X_batch.shape[0]

            loss = self._update_weight(X_batch, y_batch, batch_size, i)
            self.last_loss = (((X_matrix @ self.weight) - Y_vector) ** 2).mean()

            if self.metric is None:
                self.last_metric = getattr(self, "_mse")(y, X_matrix @ self.weight)
            else:
                self.last_metric = getattr(self, "_" + self.metric)(y,
                                                                    X_matrix @ self.weight)

            if verbose and i % verbose == 0:
                if self.metric is None:
                    self.log(i, loss)
                else:
                    self.log(i, loss, self.last_metric)

    # ------------------------Предсказание------------------------#
    def predict(self, X: pd.DataFrame):
        X = X.copy()
        X.insert(0, "once", 1)

        return X @ self.weight

    # ------------------------Методы доступа------------------------#
    def get_coef(self):
        return self.weight[1:]

    def get_last_metric(self):
        return self.last_metric

    def get_last_loss(self):
        return self.last_loss

    # ------------------------Применение регуляризации------------------------#
    def apply_reg(self, loss_grad, loss) -> tuple[np.ndarray, float]:
        """
        Применяе регуляризацию к градиенту и loss,
        если регуляризация не указана, то возвращает исходные градиент и loss

        Параметры:
        loss_grad : np.array
            Вектор градиент без регуляризации
        loss : float
            Текущий loss без регуляризации

        Возврящает:
        grad : np.array
            Вектор градиент с регуляризацией
        loss : float
            Текущий loss с регуляризацией
        """

        if self.reg in ["l1", "l2", "elasticnet"]:
            func = getattr(self, f"_{self.reg}")
            if self.reg == "l1":

                return func(self.weight, loss_grad, loss, self.l1_coef)
            elif self.reg == "l2":
                return func(self.weight, loss_grad, loss, self.l2_coef)
            else:
                return func(self.weight, loss_grad, loss, self.l1_coef, self.l2_coef)
        return loss_grad, loss

    # ------------------------Логи------------------------#
    def log(self, n: int, loss: float, metric_value=0):
        if self.metric is None:
            print(f"{n} | loss: {loss}")
        else:
            print(f"{n} | loss: {loss} | {self.metric}: {metric_value}")
