import numpy as np
from matplotlib import pyplot as plt


# Parametrización

# Circunferencia del anillo
anguloInicial = 0  # Radianes
anguloFinal = 2 * np.pi
pasos = 50  # Cantidad partículas del anillo
radio = 2

# Vector [Θ1, Θ2, ...] Cambio del ángulo
theta, deltaTheta = np.linspace(anguloInicial, anguloFinal, pasos, retstep=True)

xCirculo = radio * np.cos(theta)
yCirculo = radio * np.sin(theta)
# Fin de parametrizacion


# Reja de puntos a evaluar el campo magnetico

cantidadPuntos = 25  # Por lado XYZ
distancia = 2       # Por defecto: radio
offset = 2          # Desplazamiento fuera del radio de puntos

# Vectores de coordenandas de puntos
rX = np.linspace(-distancia - offset, distancia + offset, cantidadPuntos)     # Coordenadas X
rY = np.linspace(-distancia - offset, distancia + offset, cantidadPuntos)    # Coordenadas Y
rZ = np.linspace(-distancia - offset, distancia + offset, cantidadPuntos)    # Coordenadas Z
# Fin red de puntos

# Cálculo de campo magnético para puntos en la red

# Ecuación de campo magnético
"""
(...) Parentesis ---> nombre de las variables
↓↑    Flechas    ---> lugar en donde se nombra la variable

Campo magnético:

B = g * SUMATORIA[k * c]

g  = mu_O * RX * I / 4 * pi  # Constante fuera de la Sumatoria/~Integral~

en donde 
        (mu_0);   permeabilidad del vacío (↓)
        RX;        (radio ↑)
        I;        corriente eléctrica (corriente ↓)


Sumatoria/~Integral~ de k * c
    de cada patícula en el anillo (pasos ↑)
    sobre cada posición en la red (rX, rY, rZ ↑) para cierta (cantidadPuntos ↑)^3*dimensiones*:

        k =                     ∆Θ
            ---------------------------------------------
            [(-Rcos(Θn))^2 + (y - Rsin(Θn))^2 + z^2]^3/2 


        c = [+-sin(Θn) î  +-cos(Θn) ĵ] xCirculo [-Rcos(Θn) î  +[y - Rsin(Θn)] ĵ  +z k̂]  # Producto Cruz

en donde 
        ∆Θ;     cambio del ángulo (deltaTheta ↑)
        RX;      (radio ↑)
        Θn;     vector [Θ1, Θ2, ...] Cambio del ángulo (theta ↑)
        y,z;    posiciones en dónde evaluar campo magnético (rX, rY, rZ ↑)



RESUMEN:

Calcular: B = g * SUMATORIA[k * c]        

"""

# B = B = g * SUMATORIA[k * c]

# Constantes
mu_0 = 4 * np.pi * 10 ** -7     # Tesla * metros / Ampers
corriente = -10                   # Ampers

g = (mu_0 * radio * corriente) / (4 * np.pi)    # Tesla * metros
# g ✓


b_kc = np.zeros([pasos, 3])
B = np.zeros([cantidadPuntos * cantidadPuntos, 3])
contador = 0
for w in range(cantidadPuntos):
    for vv in range(cantidadPuntos):
        for j in range(pasos):
            # Warning con ello eliminado de la consola debido a --> Posible división por 0
            with np.errstate(divide='ignore', invalid='ignore'):
                dividend = ((- radio * np.cos(theta[j])) ** 2 + (rY[vv] - radio * np.sin(theta[j])) ** 2
                            + rZ[w] ** 2) ** (3 / 2)
                k = np.where(dividend != 0, deltaTheta / dividend, 0)
            # k ✓

            # c = [c1] x [c2] Producto cruz de dos matrices
            c1 = np.array([np.sin(theta[j]), -np.cos(theta[j])])
            c2 = np.array([-radio * np.cos(theta[j]), rY[vv] - radio * np.sin(theta[j]), rZ[w]])

            c = np.cross(c1, c2)
            # c ✓

            # Cálculo c * k
            multiplicacion = c * k
            b_kc[j] = multiplicacion
        B[contador] = g * np.sum(b_kc, axis=0)
        contador += 1



print(B[0])

# Gráficación

# Circunferencia - No es necesaria para ejes ZY (el círuclo se vería como una línea)
# plt.plot(xCirculo, yCirculo)


# Red de puntos a evaluar el campo magnetico
xx, yy = np.meshgrid(rY, rZ)
plt.scatter(xx, yy, marker='.', s=0.5)


# Calcular el color de los vectores según su cambio de dirección. CRÉDITOS: GPT DaVinci
angles = np.arctan2(B[:, 1], B[:, 2])
# Create anguloInicial color map based on the angles
cmap = plt.colormaps['hsv']
colors = cmap(angles / (2*np.pi))  # Normalize angles to [0, 1]

# Vectores
figure = plt.figure(1)
plt.quiver(xx, yy, B[:, 1], B[:, 2], color=colors)
figure.show()

"""
# Vectores normalizados, aka solo dirección
plt.scatter(xx, yy, marker='.', s=0.5)

BnormY = B[:, 1] / (np.sqrt(B[:, 1]**2 + B[:, 2]**2))
BnormZ = B[:, 2] / (np.sqrt(B[:, 2]**2 + B[:, 2]**2))

figureNorm = plt.figure(2)
plt.quiver(xx, yy, BnormY, BnormZ, color=colors)
figureNorm.show()
"""
plt.show()
