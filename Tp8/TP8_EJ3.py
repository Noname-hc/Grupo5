""" 
Ejercicio 3 - K-means + KNN

Objetivo:
- Generar puntos aleatorios (conjunto de entrenamiento y prueba).
- Ejecutar un K-means simple (k=2) implementado manualmente sobre los puntos de entrenamiento.
- Permitir al usuario elegir un valor de K para un clasificador KNN (1,3,5).
- Entrenar KNN con las etiquetas resultantes de K-means y predecir las etiquetas de los puntos de prueba.
- Mostrar resultados numéricos y tres gráficos explicativos.

Dependencias:
- numpy
- matplotlib
- scikit-learn (KNeighborsClassifier)

Cómo usar:
- Ejecutar: python TP8_EJ3.py
- El programa pedirá por consola la opción de K (1/3/5) para KNN.

Notas:
- El K-means está implementado aquí para fines didácticos; en la práctica se recomienda usar sklearn.cluster.KMeans
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# --------------------------
# 1. Generar puntos aleatorios
# --------------------------
# np.random.seed(None) deja la semilla aleatoria basada en el reloj del sistema,
# por lo que cada ejecución producirá datos distintos. Si quiere reproducibilidad,
# ponga un entero fijo (por ejemplo: np.random.seed(42)).
np.random.seed(None)  # diferente cada ejecución

# Generamos 23 puntos 2D uniformes en el cuadrado [0,5] x [0,5]
# La forma (23, 2) indica 23 filas (puntos) y 2 columnas (coordenadas x,y).
points = np.random.uniform(0, 5, (23, 2))

# Usamos los primeros 20 puntos como conjunto de entrenamiento y los 3 últimos como test
train_points = points[:20]
test_points = points[20:]

# --------------------------
# 2. Inicializar centroides aleatorios
# --------------------------
# Centroides iniciales para K-means (k=2). Se generan aleatoriamente en el mismo rango [0,5].
centroids_init = np.random.uniform(0, 5, (2, 2))  # 2 centroides, cada uno con 2 coordenadas


def kmeans(train_points, centroids, tol=1e-3, max_iter=100):
    """
    Implementación simple de K-means para 2 clusters (pero el código está escrito
    de forma que adaptarlo a k>2 es directo cambiando la forma de `centroids`).

    Parámetros:
    - train_points: array (n_samples, 2) con los puntos de entrenamiento.
    - centroids: array (k, 2) con las posiciones iniciales de los centroides.
    - tol: tolerancia para la convergencia (norma del desplazamiento de centroides).
    - max_iter: número máximo de iteraciones.

    Retorna:
    - labels: array (n_samples,) con la etiqueta (0..k-1) asignada a cada punto.
    - centroids: array (k, 2) con la posición final de los centroides.

    Algoritmo (iterativo):
    1) Calcular la distancia euclidiana de cada punto a cada centroide.
    2) Asignar cada punto al centroide más cercano (etiquetas).
    3) Recalcular cada centroide como la media de los puntos asignados.
    4) Si algún centroide queda sin puntos asignados, se mantiene su posición anterior.
    5) Calcular el desplazamiento (norma) de los centroides y comparar con `tol`.
       Si es menor, considerar convergencia y parar.
    6) Repetir hasta `max_iter`.
    """

    # Bucle de iteración del K-means
    for _ in range(max_iter):
        # distances: matriz (n_samples, k) con la distancia de cada punto a cada centroide.
        # train_points[:, np.newaxis] expande la dimensión para poder restar centroids
        # de forma vectorizada (broadcasting).
        distances = np.linalg.norm(train_points[:, np.newaxis] - centroids, axis=2)

        # labels: para cada punto, el índice del centroide más cercano (0..k-1)
        labels = np.argmin(distances, axis=1)

        # Recalcular centroides: para cada cluster i, sacar la media de los puntos asignados
        # Si no hay puntos asignados a un centroide, mantenemos el centroide anterior
        new_centroids = np.array([
            train_points[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
            for i in range(centroids.shape[0])
        ])

        # shift es la norma del vector diferencia entre los centroides nuevos y viejos.
        # Si el cambio total es menor que `tol`, consideramos que convergió.
        shift = np.linalg.norm(new_centroids - centroids)
        if shift < tol:
            # Convergencia alcanzada: salimos del bucle
            break

        # Actualizamos los centroides y seguimos iterando
        centroids = new_centroids

    return labels, centroids


# Ejecutamos K-means sobre los puntos de entrenamiento:
labels, centroids_final = kmeans(train_points, centroids_init)

# --------------------------
# 3. Selección de K por consola (KNN)
# --------------------------
# Mostramos opciones simples para que el usuario elija K (1,3 o 5).
print("Seleccione un valor de K para KNN:")
print("1. K = 1")
print("2. K = 3")
print("3. K = 5")
opcion = int(input("Ingrese opción (1-3): "))

# Mapeo de la opción ingresada al valor real de k. Si la opción no es válida,
# se avisa y se fija K=3 por defecto.
if opcion == 1:
    k = 1
elif opcion == 2:
    k = 3
elif opcion == 3:
    k = 5
else:
    print("Opción inválida, se usará K=3 por defecto.")
    k = 3

# --------------------------
# 4. KNN con valor elegido
# --------------------------
# Construimos el clasificador KNN (scikit-learn). `labels` son las etiquetas
# obtenidas por K-means (valores 0 o 1). Entrenamos (fit) el KNN con los
# puntos de entrenamiento y sus etiquetas, y luego usamos `predict` sobre test_points.
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(train_points, labels)
predictions = knn.predict(test_points)

# --------------------------
# 5. Mostrar resultados numéricos ANTES de mostrar figuras
# --------------------------
# Imprimimos en consola la clasificación de cada punto de test. Sumamos +1 para
# presentar los grupos de forma humana (1 y 2) en lugar de (0 y 1).
print(f"\nResultados con K = {k}:")
for i, p in enumerate(test_points):
    print(f"Punto {i+1} {p} clasificado en grupo {predictions[i]+1}")

# --------------------------
# 6. Graficar resultados (3 gráficos), con leyendas explícitas por grupo
# --------------------------
# Creamos una figura con 1 fila y 3 columnas de subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# (a) Puntos iniciales + centroides iniciales
# Mostramos todos los puntos de entrenamiento en negro y los centroides iniciales en rojo.
axs[0].scatter(train_points[:, 0], train_points[:, 1], c="black", label="Train points")
axs[0].scatter(centroids_init[:, 0], centroids_init[:, 1],
               c="red", marker="X", s=200, label="Centroides iniciales")
axs[0].set_title("Puntos iniciales + centroides iniciales")
axs[0].set_xlim(0, 5)
axs[0].set_ylim(0, 5)
axs[0].grid(True)
axs[0].legend()

# (b) Resultado final de K-means
# Para que la leyenda muestre claramente ambos grupos, separamos los puntos por su etiqueta.
mask_g1 = labels == 0  # grupo 1 (vamos a mostrarlo en rojo)
mask_g2 = labels == 1  # grupo 2 (azul)

# Solo dibujamos cada grupo si tiene al menos un punto (evitamos errores si un grupo queda vacío)
if np.any(mask_g1):
    axs[1].scatter(train_points[mask_g1, 0], train_points[mask_g1, 1],
                   c='red', label='puntos del grupo 1')
if np.any(mask_g2):
    axs[1].scatter(train_points[mask_g2, 0], train_points[mask_g2, 1],
                   c='blue', label='puntos del grupo 2')

# Centroides finales (cruces amarillas)
axs[1].scatter(centroids_final[:, 0], centroids_final[:, 1],
               c='yellow', marker='X', s=200, label='centroides finales')
axs[1].set_title("Clusters finales con K-means")
axs[1].set_xlim(0, 5)
axs[1].set_ylim(0, 5)
axs[1].grid(True)
axs[1].legend()

# (c) Clasificación con KNN (mostrar train por grupos y test por grupo predicho)
# Primero dibujamos los puntos de entrenamiento coloreados por su etiqueta original
if np.any(mask_g1):
    axs[2].scatter(train_points[mask_g1, 0], train_points[mask_g1, 1],
                   c='red', label='puntos del grupo 1', alpha=0.8)
if np.any(mask_g2):
    axs[2].scatter(train_points[mask_g2, 0], train_points[mask_g2, 1],
                   c='blue', label='puntos del grupo 2', alpha=0.8)

# Ahora dibujamos los puntos test según la predicción del KNN, de forma que
# cada subconjunto de puntos test tenga su propia entrada en la leyenda.
mask_t_g1 = predictions == 0
mask_t_g2 = predictions == 1

# Usamos marker 'D' (diamond) y borde (edgecolors='black') para resaltar los puntos test.
# Nota: matplotlib puede emitir una advertencia si se usa edgecolors en marcadores rellenos.
if np.any(mask_t_g1):
    axs[2].scatter(test_points[mask_t_g1, 0], test_points[mask_t_g1, 1],
                   c='red', edgecolors='black', marker='D', s=120, label='puntos test (predicción grupo 1)')
if np.any(mask_t_g2):
    axs[2].scatter(test_points[mask_t_g2, 0], test_points[mask_t_g2, 1],
                   c='blue', edgecolors='black', marker='D', s=120, label='puntos test (predicción grupo 2)')

# Centroides finales en el tercer gráfico también, para referencia visual
axs[2].scatter(centroids_final[:, 0], centroids_final[:, 1],
               c='yellow', marker='X', s=200, label='centroides finales')

axs[2].set_title(f"Clasificación de {len(test_points)} puntos con KNN (K={k})")
axs[2].set_xlim(0, 5)
axs[2].set_ylim(0, 5)
axs[2].grid(True)
axs[2].legend()

plt.tight_layout()
plt.show()

# --------------------------
# FIN