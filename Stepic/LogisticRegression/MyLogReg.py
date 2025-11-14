import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score


class MyLogReg:
    """
    Логистическая регрессия

    Параметры:
    n_iter: int
        Кол-во итераций для обучения модели
    learning_rate: float
        Скорость обучения модели
    metric : str
        Выбранная метрика оценки (accuracy, precision, recall, f1, roc_auc)

    Атрибуты:
    weights : np.array
        Вектор весов модели, включая bias
    last_metric : float
        Последний результат подсчета метрики
    """

    def __init__(self, n_iter=10, learning_rate=0.1, metric=None):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None
        self.metric = metric
        self.last_metric = None


    def __str__(self) -> str:
        return f"MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"


    #--------------------Обновление весов--------------------#
    def __update_weights(self, X_matrix: np.ndarray, y_vector: np.ndarray, y_pred: np.ndarray):
        """
        Метод обновления весов

        Параметры:
        X_matrix: np.array
            Матрица признаков X
        y_vector: np.array
            Вектор целевых переменных

        Считает предсказанные переменные (y_pred)
        Находит логистическую функцию потерь с учетом небольшого прибавления exp (для исключения случаев log(0))
        Ищет градиент и находит веса
        """

        exp = 1e-15
        log_loss = -np.mean(y_vector * np.log(y_pred + exp) + (1 - y_vector) * np.log(1 - y_pred + exp))
        grad = 1 / y_vector.shape[0] * np.dot((y_pred - y_vector), X_matrix)
        self.weights -= self.learning_rate * grad

        return log_loss

    #--------------------Подсчет предсказаний--------------------#
    @staticmethod
    def _calculate_predict(X: pd.DataFrame, weights: np.ndarray):
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

    #--------------------Метрики--------------------
    @staticmethod
    def _accuracy(y_true: np.ndarray, y_pred: np.ndarray):
        """
        Метрика accuracy

        Параметры:
        y_true: np.array
            Вектор искомых значений
        y_pred: np.array
            Вектор предсказаний

        Возвращает:
        accuracy: float
            Отношение всех правильных ответов модели ко всем ответам

        Показывает общую эффективность модели (неэффективная оценка)
        """
        accuracy = np.mean(y_true == y_pred)

        return accuracy

    @staticmethod
    def _precision(y_true: np.ndarray, y_pred: np.ndarray):
        """
        Метрика precision (точность)

        Параметры
        y_true: np.array
            Вектор искомых значений
        y_pred: np.array
            Вектор предсказаний

        Возвращает:
        precision: float
            Вероятность что положительный результат является положительным

        Доля объектов, которые были выделены как положительные,
        действительно являются положительными
        """
        TP = np.sum((y_pred == 1) & (y_true == 1))
        FP = np.sum((y_pred == 1) & (y_true == 0))

        return TP / (TP + FP)

    @staticmethod
    def _recall(y_true: np.ndarray, y_pred: np.ndarray):
        """
        Метрика recall (полнота)

        Параметры
        y_true: np.array
            Вектор искомых значений
        y_pred: np.array
            Вектор предсказаний

        Возвращает:
        recall: float
            Показывает какую часть положительных ответов правильно классифицировала модель

        Дает понять может ли модель отличать выбранный класс от другого
        """
        TP = np.sum((y_pred == 1) & (y_true == 1))
        FN = np.sum((y_pred == 0) & (y_true == 1))

        return TP / (TP + FN)

    @staticmethod
    def _f1(y_true: np.ndarray, y_pred: np.ndarray):
        """
        Метрика f1-мера

        Параметры
        y_true: np.array
            Вектор искомых значений
        y_pred: np.array
            Вектор предсказаний

        Возвращает:
        f1: float
            Объединение recall и precision

        Когда recall и precision равны 1 метрика будет максимальной,
        Когда один из двух (recall, precision) близок к 0,
        Метрика близка 0
        (Не использует True Negative значение, из-за этого может быть неинформативной
        В случаях когда нет или мало True Positive)
        """

        TP = np.sum((y_true == 1) & (y_pred == 1))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_pred == 0) & (y_true == 1))

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * ((precision * recall) / (precision + recall))

        return f1

    @staticmethod
    def _roc_auc(y_true: np.ndarray, y_pred_proba: np.ndarray):
        """
        Метрика roc_auc

        Параметры:
        y_true: np.array
            Вектор искомых значений
        y_pred_proba: np.array
            Вектор предсказанных значений

        Возвращает:
        roc_auc: float
            Вероятность того что случайно выбранный положительный объект окажется выше
            Случайно выбранного отрицательного объекта в отранжированном списке

        Считает площать под крвой ошибок

        Ищет индексы позитивных и отрицательных событий
        Создаются на их основе списки
        По длине neg(pos)_ind находятся N и P (кол-во отрицательных и положительных событий)
        считаются кол-во отрицательных классов,
        для которых считаются для каждого негативного класса все положительные классы,
        которые больше него по вероятности(y_pred_proba)
        Таким же способом считаются кол-во классов у которых совпадают вероятности(y_pred_proba)
        Далее высчитывается rog_auc (с округлением до 10 знаков)
        """

        pos_ind = np.where(y_true == 1)[0]
        neg_ind = np.where(y_true == 0)[0]

        neg_proba = y_pred_proba[neg_ind]
        pos_proba = y_pred_proba[pos_ind]

        N = len(neg_ind)
        P = len(pos_ind)

        compr = neg_proba[:, None] < pos_proba[None, :]
        piece = neg_proba[:, None] == pos_proba[None, :]
        roc_auc = np.sum(compr + 0.5 * piece) / (N * P)

        return round(roc_auc, 10)

    #--------------------Обучение модели--------------------#
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
        По обновленным весам считается новый y_pred_proba_updated
        На основе y_pred_proba_updated считается метрика (если была указана)
        """
        X_copy = X.copy()
        X_copy.insert(0, "bias", 1)

        self.weights = np.ones(X_copy.shape[1])

        X_matrix = X_copy.values
        y_vector = y.values
        metric = None

        for i in range(1, self.n_iter + 1):

            y_pred_proba = 1 / (1 + np.exp(-np.dot(X_matrix, self.weights)))

            log_loss = self.__update_weights(X_matrix, y_vector, y_pred_proba)

            y_pred_proba_updated = 1 / (1 + np.exp(-np.dot(X_matrix, self.weights)))

            if self.metric is not None:
                if self.metric == "roc_auc":
                    metric = self._roc_auc(y_vector, y_pred_proba_updated)
                else:
                    y_predict = np.where(y_pred_proba_updated > 0.5, 1, 0)
                    metric = getattr(self, "_" + self.metric)(y_vector, y_predict)

                self.last_metric = metric

            if verbose and i % verbose == 0:
                if self.metric is not None:
                    print(f"inter: {i}| loss: {log_loss} | metric: {metric}")
                else:
                    print(f"inter: {i}| loss: {log_loss}")


    #--------------------Предсказания--------------------#
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
        y_predict = self._calculate_predict(X, self.weights)
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
        y_predict_proba = self._calculate_predict(X, self.weights)

        return y_predict_proba

    #--------------------Методы доступа--------------------#
    def get_coef(self):
        return self.weights[1:]

    def get_best_score(self):
        return self.last_metric




log_reg = MyLogReg(n_iter=10, learning_rate=0.1)
log_reg
print(repr(log_reg))