import pandas as pd
import numpy as np

class MyLogReg:
    """
    Логистическая регрессия

    Параметры:
    n_iter: int
        Кол-во итераций для обучения модели
    learning_rate: float
        Скорость обучения модели

    Атрибуты:
    weights : np.array
        Вектор весов модели, включая bias

    """

    def __init__(self, n_iter=10, learning_rate=0.1):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None

    def __str__(self) -> str:
        return f"MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"


    def __update_weights(self, X_matrix: np.ndarray, y_vector: np.ndarray):
        """
        Метод обновления весов

        Параметры:
        X_matrix: np.array
            Матрица признаков X
        y_vector: np.array
            Вектор целевых переменных

        Считает предсказанные переменные (y_pred)
        Находит логистическую функцию потерь с учетом небольшого прибавления exp
        Ищет градиент и находит веса

        """
        exp = 1e-15
        y_pred = 1 / (1 + np.exp(-np.dot(X_matrix, self.weights)))
        log_loss = -np.mean(y_vector * np.log(y_pred + exp) + (1 - y_vector) * np.log(1 - y_pred + exp))
        grad = 1 / y_vector.shape[0] * np.dot((y_pred - y_vector), X_matrix)
        self.weights -= self.learning_rate * grad

        return log_loss

    def __calculate_predict(self, X: pd.DataFrame, weights: np.ndarray):
        """
        Подсчет предсказаний

        Параметры :
        X : pd.DataFrame
            Массив признаков
        weights : np.array
            Вектор весов

        Возвращает:
        y_predict : np.array
            Вектор предсказаний

        Считает предсказания (вероятности)
        Использует функцию сигмоиды
        """
        X_copy = X.copy()
        X_copy.insert(0, "bias", 1)

        X_matrix = X_copy.values
        y_predict = 1 / (1 + np.exp(-np.dot(X_matrix, weights)))

        return y_predict

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose=False):
        """
        Метод обучения логистичестой регрессии

        Параметры:
        X : pd.DataFrame
            Признаки
        y : pd.Series
            Целевые переменные

        Добавляет к признакам bias
        И находит веса путем логистической регрессии

        """
        X_copy = X.copy()
        X_copy.insert(0, "bias", 1)

        self.weights = np.ones(X_copy.shape[1])

        X_matrix = X_copy.values
        y_vector = y.values

        for i in range(1, self.n_iter + 1):

            log_loss = self.__update_weights(X_matrix, y_vector)

            if verbose and i % verbose == 0:
                print(f"inter: {i}| loss: {log_loss}")

    def predict(self, X: pd.DataFrame):
        """
                Возвращение предсказаний в виде классов

                Параметры:
                X : pd.DataFrame
                    Матрица признаков

                Возвращает:
                y_predict : np.array
                    предсказания в виде класов

                Преобразует вероятности больше 0.5 в 1
                А меньше 0.5 в 0
                """
        y_predict = self.__calculate_predict(X, self.weights)
        y_predict = np.where(y_predict > 0.5, 1, 0)

        return y_predict

    def predict_proba(self, X: pd.DataFrame):
        """
        Возвращение предсказаний в виде вероятностей

        Параметры:
        X : pd.DataFrame
            Матрица признаков

        Возвращает:
        y_predict : np.array
            предсказания в виде вероятностей
        """
        y_predict_proba = self.__calculate_predict(X, self.weights)

        return y_predict_proba


    def get_coef(self):
        return self.weights[1:]




log_reg = MyLogReg(n_iter=10, learning_rate=0.1)
log_reg
print(repr(log_reg))