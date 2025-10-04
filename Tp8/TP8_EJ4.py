import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# =============================================================================
# CONFIGURACIÓN DE NUMPY PARA IMPRESIÓN DE MATRICES
# =============================================================================

# Configuramos cómo numpy imprime los arrays:
# - precision=2: muestra solo 2 decimales
# - suppress=True: suprime la notación científica (no muestra 1e-10, etc.)
# - linewidth=100: permite líneas más largas antes de dividir
np.set_printoptions(precision=2, suppress=True, linewidth=100)

# =============================================================================
# PASO 1: DEFINIR LA ESTRUCTURA DEL PROBLEMA
# =============================================================================

# Número de estados (salas)
N = 6  # Tenemos 6 salas numeradas del 0 al 5

# Estado objetivo (sala del tesoro)
TESORO = 2  # La sala 2 contiene el tesoro

# Factor de descuento gamma (γ = 0.9)
# Este valor indica cuánto valoramos las recompensas futuras
# γ cercano a 1 = valoramos mucho el futuro
# γ cercano a 0 = solo nos importan las recompensas inmediatas
gamma = 0.9

# =============================================================================
# PASO 2: DEFINIR LAS CONEXIONES ENTRE SALAS (GRAFO)
# =============================================================================

# Analizando la imagen, las conexiones bidireccionales son:
# 0 ↔ 1: sala 0 conecta con sala 1
# 0 ↔ 3: sala 0 conecta con sala 3
# 1 ↔ 2: sala 1 conecta con sala 2 (tesoro)
# 1 ↔ 4: sala 1 conecta con sala 4
# 2 ↔ 5: sala 2 conecta con sala 5
# 3 ↔ 4: sala 3 conecta con sala 4
# 4 ↔ 5: sala 4 conecta con sala 5

# Creamos un diccionario con las adyacencias (conexiones válidas)
adyacencias = {
    0: [1, 3],           # Desde sala 0 puedo ir a salas 1 o 3
    1: [0, 2, 4],        # Desde sala 1 puedo ir a salas 0, 2 o 4
    2: [1, 2, 5],        # Desde sala 2 puedo ir a salas 1, 5 o quedarme en 2
    3: [0, 4],           # Desde sala 3 puedo ir a salas 0 o 4
    4: [1, 3, 5],        # Desde sala 4 puedo ir a salas 1, 3 o 5
    5: [2, 4]            # Desde sala 5 puedo ir a salas 2 o 4
}
# Nota: Incluimos 2→2 (autotransición) para tratar el tesoro como estado terminal

# =============================================================================
# PASO 3: CONSTRUIR LA MATRIZ DE RECOMPENSAS R
# =============================================================================

# Inicializamos la matriz R con -1 (indica movimientos NO permitidos)
# Dimensiones: 6x6 (filas=estado origen, columnas=estado destino/acción)
R = np.full((N, N), -1, dtype=int)  # Usamos dtype=int porque solo tenemos enteros

# Asignamos recompensas según las reglas:
# - Camino accesible que NO lleva al tesoro → recompensa = 0
# - Camino que lleva directamente al tesoro (estado 2) → recompensa = 100
# - Movimiento no permitido → recompensa = -1 (ya inicializado)

for estado_origen in range(N):
    # Para cada sala, revisamos sus conexiones válidas
    for estado_destino in adyacencias[estado_origen]:
        # Si el movimiento lleva directamente al tesoro
        if estado_destino == TESORO:
            R[estado_origen, estado_destino] = 100  # Recompensa alta
        else:
            R[estado_origen, estado_destino] = 0    # Recompensa neutra

# Imprimimos la matriz de recompensas para verificar
print("=" * 70)
print("MATRIZ DE RECOMPENSAS R")
print("=" * 70)
print("Filas = Estado origen (desde dónde estoy)")
print("Columnas = Estado destino (hacia dónde voy)")
print("Valores: -1 = movimiento NO permitido, 0 = camino válido, 100 = llega al tesoro")
print()
# Convertimos temporalmente a int para imprimir sin decimales
print(R.astype(int))
print()

# =============================================================================
# PASO 4: INICIALIZAR LA MATRIZ Q
# =============================================================================

# La matriz Q almacena los valores Q(s,a) para cada par estado-acción
# Inicialmente todos los valores son 0 (no sabemos nada)
Q = np.zeros((N, N), dtype=float)

print("=" * 70)
print("MATRIZ Q INICIAL (todos los valores en 0)")
print("=" * 70)
print(Q.astype(int)) #Esto es para que imprima solo los ceros, sin el punto del float
print()

# =============================================================================
# PASO 5: ALGORITMO ITERATIVO PARA CALCULAR Q ÓPTIMA
# =============================================================================

# Tolerancia para detectar convergencia
# Cuando los cambios en Q sean menores a este valor, paramos
tol = 1e-6

# Contador de iteraciones para seguimiento
iteracion = 0

print("=" * 70)
print("INICIANDO ITERACIONES DE Q-LEARNING")
print("=" * 70)

# Bucle principal: repetimos hasta que Q converja
while True:
    # Variable para rastrear el cambio máximo en esta iteración
    delta = 0.0
    
    # Creamos una copia de Q para hacer las actualizaciones
    # (actualizamos todos los valores al mismo tiempo - barrido síncrono)
    Q_new = Q.copy()
    
    # Recorremos todos los estados (salas)
    for s in range(N):
        # Para cada estado, recorremos todas las posibles acciones (destinos)
        for a in range(N):
            # Solo procesamos acciones válidas (las que tienen R >= 0)
            if R[s, a] >= 0:
                # El estado siguiente es el destino de la acción
                s_next = a
                
                # FÓRMULA DE BELLMAN PARA Q-LEARNING:
                # Q(s,a) = R(s,a) + γ * max(Q(s', A))
                #
                # Donde:
                # - R(s,a) = recompensa inmediata por hacer la acción a en estado s
                # - γ = factor de descuento (0.9)
                # - max(Q(s', A)) = mejor valor Q posible desde el siguiente estado
                # - s' = estado siguiente (s_next)
                # - A = todas las acciones posibles desde s'
                
                # Calculamos el máximo Q del estado siguiente
                max_q_siguiente = np.max(Q[s_next, :])
                
                # Aplicamos la fórmula de actualización
                Q_new[s, a] = R[s, a] + gamma * max_q_siguiente
                
                # Calculamos cuánto cambió este valor Q
                cambio = abs(Q_new[s, a] - Q[s, a])
                
                # Actualizamos el delta si este cambio es el mayor hasta ahora
                delta = max(delta, cambio)
    
    # Reemplazamos Q con los nuevos valores calculados
    Q = Q_new
    
    # Incrementamos el contador de iteraciones
    iteracion += 1
    
    # Mostramos el progreso cada 10 iteraciones
    if iteracion % 10 == 0:
        print(f"Iteración {iteracion}: delta máximo = {delta:.8f}")
    
    # Verificamos convergencia: si el cambio máximo es menor que la tolerancia, terminamos
    if delta < tol:
        print(f"\n¡Convergencia alcanzada en la iteración {iteracion}!")
        print(f"Delta final: {delta:.10f}")
        break

print()

# =============================================================================
# PASO 6: MOSTRAR LA MATRIZ Q ÓPTIMA
# =============================================================================

print("=" * 70)
print("MATRIZ Q ÓPTIMA")
print("=" * 70)
print("Estos valores representan la utilidad esperada de cada acción")
print()
print(Q)
print()

# Encontramos el valor máximo de Q
Q_max = np.max(Q)
print(f"Valor máximo en Q: {Q_max:.2f}")
print()

# =============================================================================
# PASO 7: NORMALIZAR LA MATRIZ Q
# =============================================================================

# Normalizamos dividiendo por el valor máximo
# Esto lleva todos los valores al rango [0, 1]
if Q_max > 0:
    Q_normalizada = Q / Q_max
else:
    Q_normalizada = Q  # Si Q_max es 0, no dividimos

print("=" * 70)
print("MATRIZ Q NORMALIZADA (valores entre 0 y 1)")
print("=" * 70)
print(Q_normalizada)
print()

# =============================================================================
# PASO 8: EXTRAER LA POLÍTICA ÓPTIMA
# =============================================================================

# La política nos dice qué acción tomar en cada estado
# Elegimos la acción con mayor valor Q (estrategia greedy)

politica = {}

for s in range(N):
    # Obtenemos las acciones válidas desde el estado s
    acciones_validas = [a for a in range(N) if R[s, a] >= 0]
    
    # Entre las acciones válidas, elegimos la que tiene mayor Q(s,a)
    mejor_accion = max(acciones_validas, key=lambda a: Q[s, a])
    
    # Guardamos la mejor acción para este estado
    politica[s] = mejor_accion

print("=" * 70)
print("POLÍTICA ÓPTIMA")
print("=" * 70)
print("Para cada sala, indica cuál es la mejor sala siguiente:")
print()
for estado, accion in politica.items():
    print(f"Desde sala {estado} → ir a sala {accion}")
print()

# =============================================================================
# PASO 9: VISUALIZACIÓN DEL GRAFO CON LA POLÍTICA
# =============================================================================

# Creamos un grafo dirigido para visualizar la red de salas
G = nx.DiGraph()

# Agregamos todos los nodos (salas)
G.add_nodes_from(range(N))

# Agregamos las aristas según las adyacencias
for origen, destinos in adyacencias.items():
    for destino in destinos:
        # Evitamos agregar autotransiciones para mejor visualización
        if origen != destino:
            G.add_edge(origen, destino)

# Creamos la figura para el diagrama
plt.figure(figsize=(10, 6))

# --- SUBPLOT 1: Red completa con todas las conexiones ---
plt.subplot(1, 2, 1)
# Definimos la posición de los nodos manualmente para que se vea similar a la imagen
pos = {
    0: (0, 2),    # Sala 0 arriba izquierda
    1: (1, 2),    # Sala 1 arriba centro
    2: (2, 2),    # Sala 2 arriba derecha (TESORO)
    3: (0, 1),    # Sala 3 abajo izquierda
    4: (1, 1),    # Sala 4 abajo centro
    5: (2, 1)     # Sala 5 abajo derecha
}

# Definimos colores: rojo para el tesoro, azul para las demás salas
colores_nodos = ['red' if nodo == TESORO else 'lightblue' for nodo in G.nodes()]

# Dibujamos el grafo completo
nx.draw(G, pos, with_labels=True, node_color=colores_nodos, 
        node_size=1500, font_size=16, font_weight='bold',
        arrows=True, arrowsize=20, edge_color='gray', width=2)

plt.title("Red de Salas Completa\n(Rojo = Tesoro)", fontsize=14, fontweight='bold')

# --- SUBPLOT 2: Política óptima ---
plt.subplot(1, 2, 2)

# Creamos un nuevo grafo solo con las aristas de la política óptima
G_politica = nx.DiGraph()
G_politica.add_nodes_from(range(N))

# Agregamos solo las aristas que corresponden a la política óptima
for origen, destino in politica.items():
    # Solo agregamos si no es autotransición (para mejor visualización)
    if origen != destino:
        G_politica.add_edge(origen, destino)

# Dibujamos el grafo con la política
nx.draw(G_politica, pos, with_labels=True, node_color=colores_nodos,
        node_size=1500, font_size=16, font_weight='bold',
        arrows=True, arrowsize=20, edge_color='green', width=3)

plt.title("Política Óptima\n(Flechas verdes = mejor camino)", fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

# =============================================================================
# PASO 10: ANÁLISIS DE LA POLÍTICA
# =============================================================================

print("=" * 70)
print("ANÁLISIS DE LA POLÍTICA")
print("=" * 70)
print()
print("Caminos óptimos desde cada sala hasta el tesoro:")
print()

for inicio in range(N):
    if inicio == TESORO:
        print(f"Sala {inicio}: ¡Ya estás en el tesoro!")
    else:
        # Seguimos la política para ver el camino
        camino = [inicio]
        estado_actual = inicio
        pasos = 0
        max_pasos = 10  # Límite para evitar bucles infinitos
        
        while estado_actual != TESORO and pasos < max_pasos:
            estado_actual = politica[estado_actual]
            camino.append(estado_actual)
            pasos += 1
        
        # Mostramos el camino
        camino_str = " → ".join(map(str, camino))
        print(f"Sala {inicio}: {camino_str} ({len(camino)-1} pasos)")

print()
print("=" * 70)
print("¡PROCESO COMPLETADO!")
print("=" * 70)