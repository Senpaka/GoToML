import numpy as np
import matplotlib.pyplot as plt

# Точки сегментов
x1, x2, x3, x4 = 0, 2, 5, 8

a0, a1, a2, a3 = 0, 3.1333333333, 0, -0.1583333333
b0, b1, b2, b3 = 5, 1.2333333333, -0.95, 0.1055555556

aa0, aa1, aa2, aa3 = 5, -1.25, 0, 0.0648148148
bb0, bb1, bb2, bb3 = 3, 0.5, 0.5833333333, -0.0648148148

def S1(x):
    dx = x - x1
    return a0 + a1*dx + a2*dx**2 + a3*dx**3

def S2(x):
    dx = x - x2
    return b0 + b1*dx + b2*dx**2 + b3*dx**3

def S3(x):
    dx = x - x2
    return aa0 + aa1*dx + aa2*dx**2 + aa3*dx**3
def S4(x):
    dx = x - x3
    return bb0 + bb1*dx + bb2*dx**2 + bb3*dx**3

# Значения для графика
x_vals1 = np.linspace(x1, x2, 100)
x_vals2 = np.linspace(x2, x3, 100)
x_vals3 = np.linspace(x2, x3, 100)
x_vals4 = np.linspace(x3, x4, 100)


y_vals1 = S1(x_vals1)
y_vals2 = S2(x_vals2)
y_vals3 = S3(x_vals3)
y_vals4 = S4(x_vals4)

# Рисуем график
plt.figure(figsize=(8,5))
plt.plot(x_vals1, y_vals1, label="Сегмент 1")
plt.plot(x_vals2, y_vals2, label="Сегмент 2")
plt.plot(x_vals3, y_vals3, label="Сегмент 2new")

plt.plot(x_vals4, y_vals4, label="Сегмент 3")

plt.scatter([x1, x2, x3, x4], [0, 5, 3, -4], color='red', label="Точки")
plt.title("Кубический сплайн с заданными коэффициентами")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()
plt.show()
