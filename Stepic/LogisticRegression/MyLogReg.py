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
    metric : str
        Выбранная метрика оценки (accuracy, precision, recall, f1, roc_auc)

    Атрибуты:
    weights : np.array
        Вектор весов модели, включая bias
    last_metric : float
        Последний результат подсчета метрики
    """

    def __init__(self, n_iter=10, learning_rate=0.1, metric=None, reg=None, l1_coef=None, l2_coef=None):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None
        self.metric = metric
        self.last_metric = None
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef

    def __str__(self) -> str:
        return f"MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"

    # --------------------Обновление весов--------------------#
    def __update_weights(self, X_matrix: np.ndarray, y_vector: np.ndarray, y_prediction: np.ndarray) -> None:
        """
        Метод обновления весов

        Параметры:
        X_matrix: np.array
            Матрица признаков X
        y_vector: np.array
            Вектор целевых переменных

        Считает предсказанные переменные (y_prediction)
        Находит логистическую функцию потерь
        С учетом небольшого прибавления exp (для исключения случаев log(0))
        Ищет градиент и учитывает регуляризацию
        (При указании (reg, l1_coef, l2_coef))
        Находит веса
        """

        grad = 1 / y_vector.shape[0] * np.dot((y_prediction - y_vector), X_matrix)
        grad_reg = self.__calculate_gradient_regularization()
        self.weights -= self.learning_rate * (grad + grad_reg)

    # --------------------Подсчет предсказаний--------------------#
    @staticmethod
    def _calculate_predict(X: pd.DataFrame, weights: np.ndarray) -> np.ndarray:
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

    # --------------------Применение регуляризации--------------------#
    def __calculate_gradient_regularization(self) -> np.ndarray:
        """
        Применение регуляризации

        Возвращает:
        np.ndarray
            Градиент регуляризации

        Проверяет передан ли параметр регуляризации
        Если передан то применяет указанную
        """
        if self.reg == "l1" and self.l1_coef is not None:
            return self.__l1_regularization(self.weights, self.l1_coef)
        elif self.reg == "l2" and self.l2_coef is not None:
            return self.__l2_regularization(self.weights, self.l2_coef)
        elif (self.reg == "elasticnet"
              and self.l1_coef is not None
              and self.l2_coef is not None):
            return self.__elasticnet_regularization(self.weights, self.l1_coef, self.l2_coef)
        else:
            return np.zeros(1)

    # --------------------Подсчет регуляризаций--------------------#
    @staticmethod
    def __l1_regularization(weights: np.ndarray, l1_coef: float) -> np.ndarray:
        """
        Подсчитывает l1 регуляризацию

        Параметры:
        weights: np.array
            Вектор весов
        l1_coef: float
            Коэффициент регуляризации

        Возвращает:
        np.ndarray
            Градиент l1 регуляризации
        """
        return l1_coef * np.sign(weights)

    @staticmethod
    def __l2_regularization(weights: np.ndarray, l2_coef: float) -> np.ndarray:
        """
        Подсчитывает l1 регуляризацию

        Параметры:
        weights: np.array
            Вектор весов
        l1_coef: float
            Коэффициент регуляризации

        Возвращает:
        np.ndarray
            Градиент l1 регуляризации
        """
        return l2_coef * 2 * weights

    @staticmethod
    def __elasticnet_regularization(weights: np.ndarray, l1_coef: float, l2_coef: float) -> np.ndarray:
        """
        Подсчитывает elasticnet регуляризацию

        Параметры:
        weights: np.array
            Вектор весов
        l1_coef: float
            Коэффициент l1 регуляризации
        l2_coef: float
            Коэффициент l2 регуляризации

        Возвращает:
        np.ndarray
            Градиент elasticnet регуляризации
        """
        l1_reg =  l1_coef * np.sign(weights)
        l2_reg = l2_coef * 2 * weights
        return l1_reg + l2_reg

    # --------------------Метрики--------------------
    @staticmethod
    def _accuracy(y_true: np.ndarray, y_prediction: np.ndarray) -> float:
        """
        Метрика accuracy

        Параметры:
        y_true: np.array
            Вектор искомых значений
        y_prediction: np.array
            Вектор предсказаний

        Возвращает:
        accuracy: float
            Отношение всех правильных ответов модели ко всем ответам

        Показывает общую эффективность модели (неэффективная оценка)
        """
        accuracy = np.mean(y_true == y_prediction)

        return accuracy

    @staticmethod
    def _precision(y_true: np.ndarray, y_prediction: np.ndarray) -> float:
        """
        Метрика precision (точность)

        Параметры
        y_true: np.array
            Вектор искомых значений
        y_prediction: np.array
            Вектор предсказаний

        Возвращает:
        precision: float
            Вероятность что положительный результат является положительным

        Доля объектов, которые были выделены как положительные,
        действительно являются положительными
        """
        TP = np.sum((y_prediction == 1) & (y_true == 1))
        FP = np.sum((y_prediction == 1) & (y_true == 0))

        return TP / (TP + FP)

    @staticmethod
    def _recall(y_true: np.ndarray, y_prediction: np.ndarray) -> float:
        """
        Метрика recall (полнота)

        Параметры
        y_true: np.array
            Вектор искомых значений
        y_prediction: np.array
            Вектор предсказаний

        Возвращает:
        recall: float
            Показывает какую часть положительных ответов правильно классифицировала модель

        Дает понять может ли модель отличать выбранный класс от другого
        """
        TP = np.sum((y_prediction == 1) & (y_true == 1))
        FN = np.sum((y_prediction == 0) & (y_true == 1))

        return TP / (TP + FN)

    @staticmethod
    def _f1(y_true: np.ndarray, y_prediction: np.ndarray) -> float:
        """
        Метрика f1-мера

        Параметры
        y_true: np.array
            Вектор искомых значений
        y_prediction: np.array
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

        TP = np.sum((y_true == 1) & (y_prediction == 1))
        FP = np.sum((y_true == 0) & (y_prediction == 1))
        FN = np.sum((y_prediction == 0) & (y_true == 1))

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * ((precision * recall) / (precision + recall))

        return f1

    @staticmethod
    def _roc_auc(y_true: np.ndarray, y_prediction_proba: np.ndarray) -> float:
        """
        Метрика roc_auc

        Параметры:
        y_true: np.array
            Вектор искомых значений
        y_prediction_proba: np.array
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
        которые больше него по вероятности(y_prediction_proba)
        Таким же способом считаются кол-во классов у которых совпадают вероятности(y_prediction_proba)
        Далее высчитывается rog_auc (с округлением до 10 знаков)
        """

        pos_ind = np.where(y_true == 1)[0]
        neg_ind = np.where(y_true == 0)[0]

        neg_proba = y_prediction_proba[neg_ind]
        pos_proba = y_prediction_proba[pos_ind]

        N = len(neg_ind)
        P = len(pos_ind)

        compr = neg_proba[:, None] < pos_proba[None, :]
        piece = neg_proba[:, None] == pos_proba[None, :]
        roc_auc = np.sum(compr + 0.5 * piece) / (N * P)

        return round(roc_auc, 10)

    # --------------------Обучение модели--------------------#
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
        По обновленным весам считается новый y_prediction_proba_updated
        На основе y_prediction_proba_updated считается метрика (если была указана)
        """
        X_copy = X.copy()
        X_copy.insert(0, "bias", 1)

        self.weights = np.ones(X_copy.shape[1])

        X_matrix = X_copy.values
        y_vector = y.values
        metric = None
        exp = 1e-15

        for i in range(1, self.n_iter + 1):

            y_prediction_proba = 1 / (1 + np.exp(-np.dot(X_matrix, self.weights)))

            log_loss = -np.mean(y_vector * np.log(y_prediction_proba + exp) + (1 - y_vector) * np.log(1 - y_prediction_proba + exp))

            self.__update_weights(X_matrix, y_vector, y_prediction_proba)
            y_prediction_proba_updated = 1 / (1 + np.exp(-np.dot(X_matrix, self.weights)))

            if self.metric is not None:
                if self.metric == "roc_auc":
                    metric = self._roc_auc(y_vector, y_prediction_proba_updated)
                else:
                    y_predict = np.where(y_prediction_proba_updated > 0.5, 1, 0)
                    metric = getattr(self, "_" + self.metric)(y_vector, y_predict)

                self.last_metric = metric

            if verbose and i % verbose == 0:
                if self.metric is not None:
                    print(f"inter: {i}| loss: {log_loss} | metric: {metric}")
                else:
                    print(f"inter: {i}| loss: {log_loss}")

    # --------------------Предсказания--------------------#
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
                Возвращение предсказаний в виде классов

                Параметры:
                X : pd.DataFrame
                    Матрица признаков

                Возвращает:
                y_prediction : np.array
                    предсказания в виде класов

                Преобразует вероятности больше 0.5 в 1
                А меньше 0.5 в 0
                """
        y_prediction = self._calculate_predict(X, self.weights)
        y_prediction = np.where(y_prediction > 0.5, 1, 0)

        return y_prediction

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
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

    # --------------------Методы доступа--------------------#
    def get_coef(self) -> np.ndarray:
        return self.weights[1:]

    def get_best_score(self) -> float:
        return self.last_metric


log_reg = MyLogReg(n_iter=10, learning_rate=0.1)
print(repr(log_reg))
