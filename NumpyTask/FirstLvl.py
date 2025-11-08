import numpy as np
import matplotlib.pyplot as plt


arr = np.array([1,2,3,4,5])
arrr = np.array([[1,2,3], [1,2,3], [1,2,3]])
arr0 = np.zeros((5,5))
arr1 = np.ones((4,4))
arr7 = np.full((3,3), 7)
arr9 = np.arange(10)
arr20 = np.linspace(0, 1, 20)

print(arr, arr.shape, arr.size, arr.dtype, sep="\n")
print("====================================")
print(arrr, arrr.shape, arrr.size, arrr.dtype, sep="\n")
print("====================================")
print(arr0, arr0.shape, arr0.size, arr0.dtype, sep="\n")
print("====================================")
print(arr1, arr1.shape, arr1.size, arr1.dtype, sep="\n")
print("====================================")
print(arr7, arr7.shape, arr7.size, arr7.dtype, sep="\n")
print("====================================")
print(arr9, arr9.shape, arr9.size, arr9.dtype, sep="\n")
print("====================================")
print(arr20, arr20.shape, arr20.size, arr20.dtype, sep="\n")

arr = np.array([ [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12] ])
print(arr[1][2])
print(arr[1])
print(arr[:, -1])
print(arr[0:2, -2:])

print("====================================")
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(a + b)
print(a - b)
print(a * b)
print(a / b)
print(a**2)

#================2LVL========================
arr = np.linspace(0, 2 * np.pi, 100)
print(arr)
arr_sin = np.sin(arr)
print(arr_sin)
arr_cos = np.cos(arr)
print(arr_cos)
arr_2 = np.power(arr, 2)
print(arr_2)
arr_exp = np.exp(arr)
print(arr_exp)
print("=========================")
arr = np.random.rand(100)
print(arr)
print(arr.sum())
print(arr.mean())
print(arr.min())
print(arr.max())
print(arr.std())
print(np.median(arr))
print("=========================")

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
mask = arr % 2 == 0
print(arr[mask])
mask = arr > 5
print(arr[mask])
print("=========================")
#===================3LVL=================
arr = np.arange(1, 13)
print(arr)
arr = np.reshape(arr, (3, 4))
print(arr)
arr = np.reshape(arr, (4,3))
print(arr)
print("=========================")
arr = np.reshape(arr, (2,3,2))
print(arr)
flat_arr = arr.flatten()
flat_arr[0] = 123
print(flat_arr)
print(arr)
print("=========================")
flat_arr = arr.ravel()
flat_arr[0] = 123
print(flat_arr)
print(arr)
print("=========================")
arr = np.arange(9).reshape(3,3)
arr_T = arr.T
print(arr)
print("=========================")
print(arr_T)
#=======================LVL4===========================
print("=========================")
scores = np.array([95, 87, 92, 78, 100, 85, 91, 70])
min_ = scores.min()
max_ = scores.max()
normal_scores = (scores - min_) / (max_ - min_)

print(normal_scores)

points1 = np.random.rand(5,2)
points2 = np.random.rand(5,2)
new_points1 = points1[:, np.newaxis,:]
diss = new_points1 - points2[np.newaxis, :, :]
print(np.linalg.norm(diss, axis=2))
print("=========================")

vectors1 = np.random.rand(4, 10)  # 4 вектора по 10 элементов
vectors2 = np.random.rand(6, 10)  # 6 векторов по 10 элементов
print(vectors1)
print("=========================")
print(vectors2)
scal_proizv = np.dot(vectors1, vectors2.T)
mod_vect1 = np.linalg.norm(vectors1, axis=1)
mod_vect2 = np.linalg.norm(vectors2, axis=1)
print("=========================")
print(mod_vect1)
print("=========================")
print(mod_vect2)
print("=========================")

znam = mod_vect2[np.newaxis,:] * mod_vect1[:,np.newaxis]
print(znam)
print("=========================")
print(scal_proizv / znam)

# Даны две матрицы
A = np.random.rand(3, 5)  # 3 вектора по 5 элементов
B = np.random.rand(4, 5)  # 4 вектора по 5 элементов

# Вычислите матрицу 3x4, где элемент [i,j] = скалярное произведение A[i] и B[j]

print(np.dot(A, B.T))
print("=========================")

#==================================================
# arr = np.random.rand(3,3)
# print(arr.sum(axis=0))
#
# vectors1 = np.random.rand(3,5)
# vectors2 = np.random.rand(3,5)
# print(np.dot(vectors2, vectors1.T))
# hundr = np.random.rand(100)
# print(hundr.mean(), hundr.std(), np.cov(hundr))
# age = np.random.randint(1,25, 100)
# plt.hist(age, bins=25)
# plt.xlabel("Возраст")
# plt.ylabel("Кол-во")
# plt.title("Гистограма распределения возраста")
# plt.show()

# vectors1 = np.random.randint(0, 10, (4, 5))
# vectors2 = np.random.randint(0, 10, (4, 5))
# print(np.linalg.norm(vectors1, axis=1))
# print(np.dot(vectors1, vectors2.T))
# matrix33 = np.random.rand(3,3)
# matrix44 = np.random.rand(4,4)
# print(np.linalg.inv(matrix33))
# print(np.linalg.det(matrix44))
# sys = np.array([[1, 2], [3,5]])
# eq = np.array([1,2])
# X = np.linalg.solve(sys, eq)
# print(X)
# print(np.allclose(np.dot(sys, X), eq))

X = np.random.rand(100, 3)
w = np.array([0.2, 0.5, 0.3])
print((X * w).sum(axis=1))

X = np.random.rand(100,4)
mean = X.mean(axis=0)[np.newaxis,:]
std = X.std(axis=0)[np.newaxis,:]
normalizeX = (X - mean) / std
print(normalizeX)

X = np.random.rand(10,3,32,32)
X = np.reshape(X, (10, -1))
print(X, X.shape)

X = np.random.rand(100, 3)
noise = np.random.rand(100)
w_true = [1.5, -2, 0.5]
Y = (X @ w_true) + noise
w_est = np.linalg.inv((X.T @ X)) @ (X.T @ Y)
print(w_est)