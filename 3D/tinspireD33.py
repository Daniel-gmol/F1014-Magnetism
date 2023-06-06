import numpy as np
from matplotlib import pyplot as plt
import time
start = time.process_time()  # Para calcular tiempo de ejecución


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
# Fin de parametrización


# Red de puntos a evaluar el campo magnetico

cantidadPuntos = 15     # Por lado XYZ
distancia = 2      # Por defecto: radio
offset = 2  # Desplazamiento fuera del radio de puntos

# Vectores de coordenandas de puntos
rX = np.linspace(-distancia - offset, distancia + offset, cantidadPuntos)   # Coordenadas X
rY = np.linspace(-0.5, 0.5, cantidadPuntos)               # Coordenadas Y
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
            
            
        c = [+-sin(Θn) î  +-cos(Θn) ĵ] x [-Rcos(Θn) î  +[y - Rsin(Θn)] ĵ  +z k̂]  # Producto Cruz

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
mu_0 = 4 * np.pi * 10 ** -7                     # Tesla * metros / Ampers
corriente = -10                                   # Ampers

g = (mu_0 * radio * corriente) / (4 * np.pi)    # Tesla * metros
# g ✓


# Meshgrid de coorednadas X, Y, Z y Theta para operaciones vectoriales
RX, RY, RZ, T = np.meshgrid(rX, rY, rZ, theta, indexing='ij')

# Warning con ello eliminado de la consola debido a --> Posible división por 0
with np.errstate(divide='ignore', invalid='ignore'):
    dividend = ((RX - radio * np.cos(T)) ** 2 + (RY - radio * np.sin(T)) ** 2 + RZ ** 2) ** (3 / 2)
    k = np.where(dividend != 0, deltaTheta / dividend, 0)
# Código de antes por si acaso
# k = deltaTheta / (((RX - radio * np.cos(T)) ** 2 + (RY - radio * np.sin(T)) ** 2 + RZ ** 2) ** (3 / 2))
# k ✓


# c = [c1] x [c2] Producto cruz de dos matrices
c1 = np.array([np.sin(T), -np.cos(T)])
c2 = np.array([RX - radio * np.cos(T), RY - radio * np.sin(T), RZ])

c = np.cross(c1.transpose(), c2.transpose())
# c ✓


# Cambio de orden de dimensiones de matrices
"""
Permutacion para arreglar anguloInicial matriz de c

Actualmente funcionando:
index = 56 ---> (2, 1, 3, 0, 4)
print(combinacionDimensiones[indez])

Encontrado mediante:

import itertools
combinacionDimensiones = list(itertools.permutations([0, 1, 2, 3, 4]))
nosquedamos = 57
for z in range(nosquedamos, len(combinacionDimensiones)):
    cr = c.transpose(combinacionDimensiones[z])
    cz = cr.reshape((cantidadPuntos**4, 3))
"""

# Arreglo de dimensiones y forma de matrices de c y k
"""
Para poder calcular c * k
y
SUMATORIA[c * k]
hay que modificar la forma de las matrices en donde se almacena
el resultado de c y de k debdio anguloInicial la forma en que se obtuviron sus valores, esto es
anguloInicial través de operaciones vectoriales con el
meshgrid RX, RY, RZ, T = np.meshgrid(rX, rY, rZ, theta, indexing='ij') 
"""
configNorm = cantidadPuntos ** 3 * pasos        # Toda la cantidad de valores calculados


c = c.transpose((2, 3, 1, 0, 4))            # Cambio de orden de dimenensiones // 2,1,3,0,4
c = c.reshape((configNorm, 3))     # Reconfigruación eliminando dimensiones extra



k = k.transpose((1, 0, 2, 3)) # 1, 0, 2, 3
k = k.reshape((configNorm, 1))


# Cálculo c * k
multiplicacion = np.multiply(k, c)


# Arreglo de Matriz multiplicacion para poder hacer SUMATORIA[c * k]
multiplicacion = multiplicacion.reshape((configNorm//pasos, pasos, 3))

sumatoria = np.sum(multiplicacion, axis=1)
"""
 Matriz final de sumatoria de forma (cantidadPuntos^3*dimensiones*, 3)
 cada renglon es un punto de la red/reja/malla
 cada columna es una coordenada xCirculo/y/z o i/j/k
"""


# Matriz Campo Magnético
B = sumatoria * g


# Gráficación

# Circunferencia
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(xCirculo, yCirculo, zs=0, zdir='z', label='Circunferencia campo magnetico')


# Red de puntos a evaluar el campo magnetico
xx, yy, zz = np.meshgrid(rX, rY, rZ)
ax.scatter(xx, yy, zz, marker='.', s=0.2)


# Calcular el color de los vectores según su cambio de dirección. CRÉDITOS: GPT DaVinci
color = np.arctan2(B[:, 2], np.sqrt(B[:, 0] ** 2 + B[:, 1] ** 2))
cmap = plt.colormaps['PRGn']  # Elegir mapa de colores
norm = plt.Normalize(color.min(), color.max())  # Normalizar valores de colores
colors = cmap(norm(color))  # Convertir valores de colores en RGBA


# Vectores Campo magnético

ax.quiver(xx.ravel(), yy.ravel(), zz.ravel(), B[:, 0], B[:, 1], B[:, 2], color=colors, length=1, normalize=True)


# Ajustes misceláneos de visaulización

# Limítes de ejes
ax.set_xlim([-5, 5])
ax.set_ylim([-5, 5])
ax.set_zlim([-5, 5])


""" 
Por si se ofrece, ratio de ejes
# print(np.ptp(xx))
# print(np.ptp(yy))
# print(np.ptp(zz))
# ax.set_box_aspect((np.ptp(xx), 26.0, np.ptp(zz)))
"""


end = time.process_time()
print(end - start)


# Mostrar la gráfica
plt.show()
