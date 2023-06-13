import numpy as np
import matplotlib.pyplot as plt

# Ecuación diferencial a reolver mediante Runge Kutta para F z F' dada F''

# Condición constante de la ecuación
# k = 10
# r = 10
k = 1480000
r = 10
# k = 1470000
# r = 10


def f(x, y, y1):
    # x, y1 no se utilizan pero así es el método completo
    return k*y/(r**2+y**2)**(5/2)-9.81


# Conficiones iniciales
t0 = 0      # tiempo
z0 = 40     # altura inicial
z1_0 = 0    # velocidad inicial
dt = 0.01    # delta/diferencial tiempo
n = 500    # cantidad de pasos

# Crear matriz para guaradar los datos
t = np.zeros(n + 1)
z = np.zeros(n + 1)
z1 = np.zeros(n + 1)
z2 = np.zeros(n + 1)
t[0] = t0   # inicializar tiempo
z[0] = z0   # inicializar velocidad
z1[0] = z1_0    # incializar velocidad

for i in range(n):
    # k para Posición vs Tiempo z(t)
    # l para Velcoidad vs Tiempo z'(t)

    # Primer pendienete
    k1 = dt * z1[i]
    l1 = dt * f(t[i], z[i], z1[i])

    # Segunda pendiente
    k2 = dt * (z1[i] + 0.5 * l1)
    l2 = dt * f(t[i] + 0.5 * dt, z[i] + 0.5 * k1, z1[i] + 0.5 * l1)

    # Tercer pendiente
    k3 = dt * (z1[i] + 0.5 * l2)
    l3 = dt * f(t[i] + 0.5 * dt, z[i] + 0.5 * k2, z1[i] + 0.5 * l2)

    # Cuarta Pendiente
    k4 = dt * (z1[i] + l3)
    l4 = dt * f(t[i] + dt, z[i] + k3, z1[i] + l3)

    # Cálculo valores
    t[i + 1] = t[i] + dt
    z[i + 1] = z[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    z1[i+1] = z1[i] + (l1 + 2*l2 + 2*l3 + l4) / 6
    z2[i+1] = f(t[i + 1], z[i + 1], z1[i + 1])


# Graficas
plt.figure()

plt.plot(t, z, label='z(t)')
plt.plot(t, z1, label='v(t)')
plt.plot(t, z2, label='a(t)')

plt.xlabel("tiempo (s)")
plt.grid(color='grey', linestyle='-', linewidth=0.3)
plt.legend()
plt.show()
