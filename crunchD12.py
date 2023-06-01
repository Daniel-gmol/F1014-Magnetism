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


with np.errstate(divide='ignore', invalid='ignore'):
    dividend = ((-radio*np.cos(theta))**2 + ((-4)-radio*np.sin(theta))**2 + (-4)**2) ** (3 / 2)
    k = np.where(dividend != 0, deltaTheta / dividend, 0)
    where_nans = np.isnan(k)
    k[where_nans] = 0
    # k ✓

# c = [c1] x [c2] Producto cruz de dos matrices
# create c1
c1 = np.zeros((pasos, 2))
c1[:, 0] = np.sin(theta)
c1[:, 1] = -np.cos(theta)
c1 = np.hstack((c1, np.zeros((pasos, 1))))

c2 = np.zeros((pasos, 3))
c2[:, 0] = -radio * np.cos(theta)
c2[:, 1] = -4 - radio * np.sin(theta)
c2[:, 2] = -4

c = np.cross(c1, c2)
# c ✓

# Cálculo c * k
multiplicacion = c * k[:, np.newaxis]
B = g * np.sum(multiplicacion, axis=0)

print(B[0])
print(B[1])
print(B[2])
for i in range(5):
    detenerse = 1

# Gráficación

# Vectores
figure = plt.figure(1)
plt.quiver(-4, -4, B[1], B[2])
plt.show()