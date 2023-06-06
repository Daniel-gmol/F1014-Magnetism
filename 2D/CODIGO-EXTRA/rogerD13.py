import numpy as np
from matplotlib import pyplot as plt


# Parametrización

# Circunferencia del anillo
anguloInicial = 0  # Radianes
anguloFinal = 2 * np.pi
pasos = 500  # Cantidad partículas del anillo
radio = 2

# Vector [Θ1, Θ2, ...] Cambio del ángulo
theta, deltaTheta = np.linspace(anguloInicial, anguloFinal, pasos, retstep=True)

xCirculo = radio * np.cos(theta)
yCirculo = radio * np.sin(theta)
# Fin de parametrizacion


# Cálculo de campo magnético para puntos en la red

# B = B = g * SUMATORIA[k * c]

# Constantes
mu_0 = 4 * np.pi * 10 ** -7     # Tesla * metros / Ampers
corriente = -10                   # Ampers
g = (mu_0 * radio * corriente) / (4 * np.pi)    # Tesla * metros
# g ✓


Bsubx = (np.cos(theta))/(radio**2 + (-4)**2 + (-4)**2 - 2*radio*(-4)*np.sin(theta))**(3/2)
Bsubx2 = Bsubx * deltaTheta
Bsubx3 = np.sum(Bsubx2)
Bsubx4 = Bsubx3 * g
Bsubx5 = Bsubx4 * (-4)

Bsuby = np.sin(theta)/(radio**2 + (-4)**2 + (-4)**2 - 2*radio*(-4)*np.sin(theta))**(3/2)
Bsuby2 = Bsuby * deltaTheta
Bsuby3 = np.sum(Bsuby2)
Bsuby4 = Bsuby3 * g
Bsuby5 = Bsuby4 * (-4)

Bsubz = (radio - (-4)*np.sin(theta))/(radio**2 + (-4)**2 + (-4)**2 - 2*radio*(-4)*np.sin(theta))**(3/2)
Bsubz2 = Bsubz * deltaTheta
Bsubz3 = np.sum(Bsubz2)
Bsubz4 = Bsubz3 * g
Bsubz5 = Bsubz4 * (-4)


losRealesy = (3/2) * mu_0 * (radio**2) * corriente * (-4*-4)/(((-4)**2 + (-4)**2)**(5/2))
losRealesz = 1/2 * mu_0 * (radio**2) * corriente * (1/(((-4)**2+(-4)**2)**(3/2))) * (1 - 3 * ((-4)**2) / (
            (-4) ** 2 + (-4) ** 2))
print("Y exacta: ", losRealesy)
print("Z exacta: ", losRealesz)

for i in range(5):
    detenerse = 1

print("X aprox: ", Bsubx5)
print("Y aprox: ", Bsuby5)
print("Z aprox: ", Bsubz5)
# Gráficación
# Vectores
figure = plt.figure(1)
plt.quiver(-4, -4, Bsuby5, Bsubz5)
plt.show()
