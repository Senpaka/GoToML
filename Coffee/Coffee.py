import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox

def simulate_coffee_cooling(T0, T_env, total_time, T_end, k, dt):
    T = T0
    time = 0
    timeList = []
    tempList = []
    print("Моделирование охлаждения кофе")
    print("=" * 40)
    print(f"Начальная температура: {T0}°C")
    print(f"Температура в комнате: {T_env}°C")
    print(f"Коэффициент охлаждения k: {k}")
    print("=" * 40)

    while time < total_time and T > T_end:
        dT_dt = -k * (T - T_env)
        T = T + dT_dt * dt
        time += dt
        timeList.append(time)
        tempList.append(T)

    print(f"Коффе остыл за {time} до {T} градусов")

    return timeList, tempList



fig, ax = plt.subplots(figsize=(10,10))
plt.subplots_adjust(bottom=0.25)

T0 = 100
T_env = 30
total_time = 50
T_end = 60
k = 0.1
dt = 0.1

timeList, tempList = simulate_coffee_cooling(T0, T_env, total_time, T_end, k, dt)


line, = ax.plot(timeList, tempList, lw=2)
plt.xlabel("Время")
plt.ylabel("Температура")
plt.title("График по координатам")
plt.grid(True)

ax_T_env = plt.axes([0.25, 0.05, 0.2, 0.03])
ax_T0 = plt.axes([0.25, 0.1, 0.2, 0.03])
ax_T_end = plt.axes([0.25, 0.15, 0.2, 0.03])
ax_total_time = plt.axes([0.75, 0.05, 0.2, 0.03])
ax_k = plt.axes([0.75, 0.1, 0.2, 0.03])
ax_dt = plt.axes([0.75, 0.15, 0.2, 0.03])

slider_T0 = Slider(ax_T0, "Начальная температура коффе", 70, 120, valinit=T0)
slider_T_env = Slider(ax_T_env, "Начальная температура комнаты", -30, 30, valinit=T_env)
slider_T_end = Slider(ax_T_end, "Нужная температура", -10, 80, valinit=T_end)
slider_total_time = Slider(ax_total_time, "Время в минутах", 1, 60, valinit=total_time)
slider_k = Slider(ax_k, "Коэффициэнт охлаждения", 0.05, 0.3, valinit=k)
slider_dt = Slider(ax_dt, "Точность", 0.05, 0.2, valinit=dt)

def update(val):
    T0 = slider_T0.val
    T_env = slider_T_env.val
    T_end = slider_T_end.val
    total_time = slider_total_time.val
    k = slider_k.val
    dt = slider_dt.val

    timeList, tempList = simulate_coffee_cooling(T0, T_env, total_time, T_end, k, dt)
    line.set_data(timeList, tempList)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw_idle()

slider_T0.on_changed(update)
slider_T_env.on_changed(update)
slider_T_end.on_changed(update)
slider_total_time.on_changed(update)
slider_k.on_changed(update)
slider_dt.on_changed(update)

plt.show()