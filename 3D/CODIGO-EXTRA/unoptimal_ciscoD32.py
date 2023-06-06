import numpy as np
from matplotlib import pyplot as plt
import time
start = time.process_time()  # Para calcular tiempo de ejecución


# Parametrización

# Circunferencia del anillo
anguloInicial = 0  # Radianes
anguloFinal = 2 * np.pi
pasos = 10  # Cantidad partículas del anillo
radio = 2

# Vector [Θ1, Θ2, ...] Cambio del ángulo
theta, deltaTheta = np.linspace(anguloInicial, anguloFinal, pasos, retstep=True)

xCirculo = radio * np.cos(theta)
yCirculo = radio * np.sin(theta)
# Fin de parametrización


# Red de puntos a evaluar el campo magnetico

cantidadPuntos = 10     # Por lado XYZ
distancia = 5      # Por defecto: radio
offset = 0  # Desplazamiento fuera del radio de puntos

# Vectores de coordenandas de puntos
rX = np.linspace(-distancia - offset, distancia + offset, cantidadPuntos)   # Coordenadas X
rY = np.linspace(-0.5, 0.5, cantidadPuntos)                     # Coordenadas Y
rZ = np.linspace(-distancia - offset, distancia + offset, cantidadPuntos)   # Coordenadas Z
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
mu_0 = 4 * np.pi * 10 ** -7      # Tesla * metros / Ampers
corriente = -1                    # Ampers

g = (mu_0 * radio * corriente) / (4 * np.pi)  # Tesla * metros
# g ✓


b_kc = np.zeros([pasos, 3])
B = np.zeros([cantidadPuntos * cantidadPuntos * cantidadPuntos, 3])

contador = 0
for w in range(cantidadPuntos):
    for vv in range(cantidadPuntos):
        for nn in range(cantidadPuntos):
            for j in range(pasos):
                # Warning con ello eliminado de la consola debido a --> Posible división por 0
                with np.errstate(divide='ignore', invalid='ignore'):
                    dividend = ((rX[vv] - radio * np.cos(theta[j])) ** 2 + (rY[w] - radio * np.sin(theta[j])) ** 2
                                + rZ[nn] ** 2) ** (3 / 2)
                    k = np.where(dividend != 0, deltaTheta / dividend, 0)
                # k ✓

                # c = [c1] x [c2] Producto cruz de dos matrices
                c1 = np.array([ np.sin(theta[j]), -np.cos(theta[j])])
                c2 = np.array([rX[vv] - radio * np.cos(theta[j]), rY[w] - radio * np.sin(theta[j]), rZ[nn]])
                c = np.cross(c1, c2)

                multiplicacion = c * k
                b_kc[j] = multiplicacion
            B[contador] = g * np.sum(b_kc, axis=0)
            contador += 1


# Gráficación

# Circunferencia
ax = plt.figure().add_subplot(projection='3d')
ax.plot(xCirculo, yCirculo, zs=0, zdir='z', label='Circunferencia campo magnetico')


# Red de puntos a evaluar el campo magnetico
xx, yy, zz = np.meshgrid(rX, rY, rZ)
ax.scatter(xx, yy, zz, marker='.', s=0.2)


# Calculate arrow colors based on direction.  CRÉDITOS: GPT DaVinci
color = np.arctan2(B[:, 2], np.sqrt(B[:, 0] ** 2 + B[:, 1] ** 2))
cmap = plt.colormaps['PiYG']  # Choose a colormap
norm = plt.Normalize(color.min(), color.max())  # Normalize color values
colors = cmap(norm(color))  # Convert color values to RGBA


ax.quiver(xx.ravel(), yy.ravel(), zz.ravel(), B[:, 0], B[:, 1], B[:, 2], color=colors, length=0.5, normalize=True)


# Set plot limits
ax.set_xlim([-4, 4])
ax.set_ylim([-4, 4])
ax.set_zlim([-7, 7])


end = time.process_time()
print(end - start)

plt.show()
