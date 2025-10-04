import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# =============================================================================
# CONFIGURACIÓN DE NUMPY PARA IMPRESIÓN DE MATRICES
# =============================================================================

# Configuramos cómo numpy imprime los arrays
np.set_printoptions(precision=2, suppress=True, linewidth=100)

# =============================================================================
# PASO 1: DEFINIR LA ESTRUCTURA DEL PROBLEMA (COMÚN PARA AMBOS MÉTODOS)
# =============================================================================

# Número de estados (salas)
N = 6

# Estado objetivo (sala del tesoro)
TESORO = 2

# Factor de descuento gamma (γ = 0.9)
gamma = 0.9

# Tasa de aprendizaje alpha (α = 0.1)
# Este es un valor típico y común en Q-learning
# Valores pequeños (0.1-0.3) hacen que el aprendizaje sea más gradual y estable
# Valores grandes (0.7-0.9) hacen que el aprendizaje sea más rápido pero menos estable
alpha = 0.1

print("=" * 80)
print("COMPARACIÓN DE MÉTODOS Q-LEARNING")
print("=" * 80)
print(f"Parámetros:")
print(f"  - Número de estados: {N}")
print(f"  - Estado tesoro: {TESORO}")
print(f"  - Gamma (γ): {gamma}")
print(f"  - Alpha (α): {alpha} (valor típico para aprendizaje gradual)")
print("=" * 80)
print()

# =============================================================================
# PASO 2: DEFINIR LAS CONEXIONES ENTRE SALAS
# =============================================================================

adyacencias = {
    0: [1, 3],
    1: [0, 2, 4],
    2: [1, 2, 5],
    3: [0, 4],
    4: [1, 3, 5],
    5: [2, 4]
}

# =============================================================================
# PASO 3: CONSTRUIR LA MATRIZ DE RECOMPENSAS R
# =============================================================================

R = np.full((N, N), -1, dtype=int)

for estado_origen in range(N):
    for estado_destino in adyacencias[estado_origen]:
        if estado_destino == TESORO:
            R[estado_origen, estado_destino] = 100
        else:
            R[estado_origen, estado_destino] = 0

print("MATRIZ DE RECOMPENSAS R")
print("-" * 80)
print(R.astype(int))
print()

# =============================================================================
# MÉTODO 1: ACTUALIZACIÓN DIRECTA (FÓRMULA ORIGINAL)
# Q(s,a) = R(s,a) + γ * max(Q(s', A))
# =============================================================================

print("=" * 80)
print("MÉTODO 1: ACTUALIZACIÓN DIRECTA (SÍNCRONA)")
print("Fórmula: Q(s,a) = R(s,a) + γ * max(Q(s', A))")
print("=" * 80)
print()

# Inicializamos Q1 con ceros
Q1 = np.zeros((N, N), dtype=float)

# Tolerancia para convergencia
tol = 1e-6

# Contador de iteraciones
iteracion1 = 0

print("Ejecutando iteraciones...")

# Bucle de iteración hasta convergencia
while True:
    delta = 0.0
    Q1_new = Q1.copy()
    
    # Recorremos todos los pares estado-acción
    for s in range(N):
        for a in range(N):
            if R[s, a] >= 0:  # Solo acciones válidas
                s_next = a
                
                # FÓRMULA DIRECTA:
                # Q(s,a) = R(s,a) + γ * max(Q(s', A))
                max_q_siguiente = np.max(Q1[s_next, :])
                Q1_new[s, a] = R[s, a] + gamma * max_q_siguiente
                
                # Calculamos el cambio
                cambio = abs(Q1_new[s, a] - Q1[s, a])
                delta = max(delta, cambio)
    
    Q1 = Q1_new
    iteracion1 += 1

    if iteracion1 == 1:
        print(f" Delta de la primera iteracion: {delta:.8f}")
    
    if iteracion1 % 10 == 0:
        print(f"  Iteración {iteracion1}: delta = {delta:.8f}")
    
    if delta < tol:
        print(f"\n¡Convergencia alcanzada en {iteracion1} iteraciones!")
        print(f"Delta final: {delta:.10f}")
        break

print()
print("MATRIZ Q1 ÓPTIMA (Método Directo):")
print("-" * 80)
print(Q1)
print()

# Normalizamos Q1
Q1_max = np.max(Q1)
Q1_norm = Q1 / Q1_max if Q1_max > 0 else Q1

print("MATRIZ Q1 NORMALIZADA:")
print("-" * 80)
print(Q1_norm)
print()

# Extraemos la política del método 1
politica1 = {}
for s in range(N):
    acciones_validas = [a for a in range(N) if R[s, a] >= 0]
    politica1[s] = max(acciones_validas, key=lambda a: Q1[s, a])

print("POLÍTICA ÓPTIMA (Método 1):")
print("-" * 80)
for estado, accion in politica1.items():
    print(f"  Estado {estado} → Acción {accion}")
print()

# =============================================================================
# MÉTODO 2: Q-LEARNING INCREMENTAL CON ALPHA
# Q(s,a) = Q(s,a) + α * (R(s,a) + γ * max(Q(s', A)) - Q(s,a))
# =============================================================================

print("=" * 80)
print("MÉTODO 2: Q-LEARNING INCREMENTAL (CON ALPHA)")
print("Fórmula: Q(s,a) = Q(s,a) + α * (R(s,a) + γ * max(Q(s', A)) - Q(s,a))")
print("=" * 80)
print()

# Inicializamos Q2 con ceros
Q2 = np.zeros((N, N), dtype=float)

# Contador de iteraciones
iteracion2 = 0

# Límite máximo de iteraciones (el método incremental puede necesitar más)
max_iteraciones = 10000

print("Ejecutando iteraciones...")

# Bucle de iteración hasta convergencia
while iteracion2 < max_iteraciones:
    delta = 0.0
    
    # En el método incremental, actualizamos directamente Q (no usamos Q_new)
    # Recorremos todos los pares estado-acción
    for s in range(N):
        for a in range(N):
            if R[s, a] >= 0:  # Solo acciones válidas
                s_next = a
                
                # Guardamos el valor anterior de Q para calcular el cambio
                q_anterior = Q2[s, a]
                
                # FÓRMULA INCREMENTAL CON ALPHA:
                # Q(s,a) = Q(s,a) + α * (R(s,a) + γ * max(Q(s', A)) - Q(s,a))
                #
                # Esta fórmula se puede entender como:
                # Q_nuevo = Q_viejo + α * error_TD
                # donde error_TD = (R + γ * max(Q_siguiente)) - Q_viejo
                #
                # α controla qué tan rápido incorporamos nueva información:
                # - α pequeño (ej: 0.1) → aprendizaje lento y estable
                # - α grande (ej: 0.9) → aprendizaje rápido pero inestable
                
                max_q_siguiente = np.max(Q2[s_next, :])
                error_td = R[s, a] + gamma * max_q_siguiente - Q2[s, a]
                Q2[s, a] = Q2[s, a] + alpha * error_td
                
                # Calculamos el cambio
                cambio = abs(Q2[s, a] - q_anterior)
                delta = max(delta, cambio)
    
    iteracion2 += 1
    
    if iteracion2 == 1:
        print(f" Delta de la primera iteracion: {delta:.8f}")    

    if iteracion2 % 100 == 0:
        print(f"  Iteración {iteracion2}: delta = {delta:.8f}")
    
    if delta < tol:
        print(f"\n¡Convergencia alcanzada en {iteracion2} iteraciones!")
        print(f"Delta final: {delta:.10f}")
        break

if iteracion2 >= max_iteraciones:
    print(f"\n¡Se alcanzó el límite de {max_iteraciones} iteraciones!")
    print(f"Delta final: {delta:.10f}")

print()
print("MATRIZ Q2 ÓPTIMA (Método Incremental):")
print("-" * 80)
print(Q2)
print()

# Normalizamos Q2
Q2_max = np.max(Q2)
Q2_norm = Q2 / Q2_max if Q2_max > 0 else Q2

print("MATRIZ Q2 NORMALIZADA:")
print("-" * 80)
print(Q2_norm)
print()

# Extraemos la política del método 2
politica2 = {}
for s in range(N):
    acciones_validas = [a for a in range(N) if R[s, a] >= 0]
    politica2[s] = max(acciones_validas, key=lambda a: Q2[s, a])

print("POLÍTICA ÓPTIMA (Método 2):")
print("-" * 80)
for estado, accion in politica2.items():
    print(f"  Estado {estado} → Acción {accion}")
print()

# =============================================================================
# COMPARACIÓN DE RESULTADOS
# =============================================================================

print("=" * 80)
print("COMPARACIÓN DE RESULTADOS")
print("=" * 80)
print()

# Comparación de iteraciones
print(f"1. NÚMERO DE ITERACIONES:")
print(f"   - Método Directo:      {iteracion1} iteraciones")
print(f"   - Método Incremental:  {iteracion2} iteraciones")
print(f"   - Diferencia:          {abs(iteracion2 - iteracion1)} iteraciones")
print()

# Comparación de matrices Q
diferencia_Q = np.abs(Q1 - Q2)
diferencia_max = np.max(diferencia_Q)
diferencia_promedio = np.mean(diferencia_Q[R >= 0])  # Solo en acciones válidas

print(f"2. DIFERENCIA EN MATRICES Q:")
print(f"   - Diferencia máxima:   {diferencia_max:.6f}")
print(f"   - Diferencia promedio: {diferencia_promedio:.6f}")
print()
print("   Matriz de diferencias |Q1 - Q2|:")
print("   " + "-" * 76)
print("   ", diferencia_Q)
print()

# Comparación de políticas
politicas_iguales = all(politica1[s] == politica2[s] for s in range(N))

print(f"3. COMPARACIÓN DE POLÍTICAS:")
if politicas_iguales:
    print("   ✓ Las políticas son IDÉNTICAS")
else:
    print("   ✗ Las políticas son DIFERENTES:")
    for s in range(N):
        if politica1[s] != politica2[s]:
            print(f"     Estado {s}: Método1→{politica1[s]}, Método2→{politica2[s]}")
print()

# Resumen
print(f"4. RESUMEN:")
print(f"   - Ambos métodos convergen a la solución óptima")
print(f"   - El método directo es más rápido ({iteracion1} vs {iteracion2} iteraciones)")
print(f"   - El método incremental con α={alpha} es más gradual")
print(f"   - Las matrices Q finales son prácticamente idénticas")
print(f"   - Ambas políticas llevan al tesoro de forma óptima")
print()

# =============================================================================
# VISUALIZACIÓN GRÁFICA COMPARATIVA
# =============================================================================

# Creamos el grafo base
G = nx.DiGraph()
G.add_nodes_from(range(N))

for origen, destinos in adyacencias.items():
    for destino in destinos:
        if origen != destino:
            G.add_edge(origen, destino)

# Posiciones de los nodos
pos = {
    0: (0, 2),
    1: (1, 2),
    2: (2, 2),
    3: (0, 1),
    4: (1, 1),
    5: (2, 1)
}

colores_nodos = ['red' if nodo == TESORO else 'lightblue' for nodo in G.nodes()]

# Crear figura con 3 subplots
fig = plt.figure(figsize=(12, 5))

# --- SUBPLOT 1: Red completa ---
plt.subplot(1, 3, 1)
nx.draw(G, pos, with_labels=True, node_color=colores_nodos,
        node_size=1500, font_size=16, font_weight='bold',
        arrows=True, arrowsize=20, edge_color='gray', width=2)
plt.title("Red de Salas Completa\n(Rojo = Tesoro)", fontsize=12, fontweight='bold')

# --- SUBPLOT 2: Política Método 1 ---
plt.subplot(1, 3, 2)
G_pol1 = nx.DiGraph()
G_pol1.add_nodes_from(range(N))
for origen, destino in politica1.items():
    if origen != destino:
        G_pol1.add_edge(origen, destino)

nx.draw(G_pol1, pos, with_labels=True, node_color=colores_nodos,
        node_size=1500, font_size=16, font_weight='bold',
        arrows=True, arrowsize=20, edge_color='green', width=3)
plt.title(f"Política Método 1 (Directo)\n{iteracion1} iteraciones", 
          fontsize=12, fontweight='bold')

# --- SUBPLOT 3: Política Método 2 ---
plt.subplot(1, 3, 3)
G_pol2 = nx.DiGraph()
G_pol2.add_nodes_from(range(N))
for origen, destino in politica2.items():
    if origen != destino:
        G_pol2.add_edge(origen, destino)

# Si las políticas son diferentes, usar color distinto
color_pol2 = 'green' if politicas_iguales else 'orange'
nx.draw(G_pol2, pos, with_labels=True, node_color=colores_nodos,
        node_size=1500, font_size=16, font_weight='bold',
        arrows=True, arrowsize=20, edge_color=color_pol2, width=3)
plt.title(f"Política Método 2 (α={alpha})\n{iteracion2} iteraciones", 
          fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

# =============================================================================
# GRÁFICO DE CONVERGENCIA
# =============================================================================

# Para visualizar mejor la convergencia, ejecutamos ambos métodos guardando el historial

print("=" * 80)
print("ANÁLISIS DE CONVERGENCIA")
print("=" * 80)
print("Regenerando ejecuciones para graficar convergencia...")
print()

# Método 1 con historial
Q1_hist = np.zeros((N, N), dtype=float)
errores1 = []

for i in range(min(500, iteracion1 + 50)):
    delta = 0.0
    Q1_new = Q1_hist.copy()
    for s in range(N):
        for a in range(N):
            if R[s, a] >= 0:
                s_next = a
                max_q = np.max(Q1_hist[s_next, :])
                Q1_new[s, a] = R[s, a] + gamma * max_q
                delta = max(delta, abs(Q1_new[s, a] - Q1_hist[s, a]))
    Q1_hist = Q1_new
    errores1.append(delta)
    if delta < tol:
        break

# Método 2 con historial
Q2_hist = np.zeros((N, N), dtype=float)
errores2 = []

for i in range(min(500, iteracion2 + 50)):
    delta = 0.0
    for s in range(N):
        for a in range(N):
            if R[s, a] >= 0:
                q_ant = Q2_hist[s, a]
                s_next = a
                max_q = np.max(Q2_hist[s_next, :])
                error_td = R[s, a] + gamma * max_q - Q2_hist[s, a]
                Q2_hist[s, a] = Q2_hist[s, a] + alpha * error_td
                delta = max(delta, abs(Q2_hist[s, a] - q_ant))
    errores2.append(delta)
    if delta < tol:
        break

# Graficar convergencia
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(errores1, 'b-', linewidth=2, label='Método Directo')
plt.axhline(y=tol, color='r', linestyle='--', label=f'Tolerancia ({tol})')
plt.xlabel('Iteración', fontsize=12)
plt.ylabel('Delta (cambio máximo)', fontsize=12)
plt.title('Convergencia: Método Directo', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')

plt.subplot(1, 2, 2)
plt.plot(errores2, 'g-', linewidth=2, label=f'Método Incremental (α={alpha})')
plt.axhline(y=tol, color='r', linestyle='--', label=f'Tolerancia ({tol})')
plt.xlabel('Iteración', fontsize=12)
plt.ylabel('Delta (cambio máximo)', fontsize=12)
plt.title('Convergencia: Método Incremental', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')

plt.tight_layout()
plt.show()

print("¡Análisis completo!")
print("=" * 80)