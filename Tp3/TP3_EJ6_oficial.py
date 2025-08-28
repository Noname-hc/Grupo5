"""
ALGORITMO GENÉTICO PARA PROBLEMA 0/1 KNAPSACK
==============================================

Este script implementa desde cero un algoritmo genético para resolver
el problema de la mochila 0/1 (0/1 knapsack problem).

Autor: Claude (Algoritmo Genético Didáctico)
Fecha: 2025

CONCEPTOS IMPLEMENTADOS:
- Individuo: Vector binario que representa qué items incluir
- Fitness: Función objetivo (precio total si es válido, penalizado si no)
- Selección: Selección por ruleta (roulette wheel)
- Cruce: One-point crossover
- Mutación: Bit-flip mutation
- Reparación: Por ratio precio/peso
- Elitismo: Conservar los k mejores individuos
- Criterio de parada: Número máximo de generaciones
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import time

# =============================================================================
# PARÁMETROS DEL PROBLEMA Y ALGORITMO GENÉTICO
# =============================================================================

# Datos del problema 0/1 Knapsack
CAPACIDAD_MAXIMA = 1000  # C: Capacidad máxima de la mochila en kg
PRECIOS = [100, 50, 115, 25, 200, 30, 40, 100, 100, 100]  # Precios de cada item
PESOS = [300, 200, 450, 145, 664, 90, 150, 355, 401, 395]  # Pesos de cada item
N_ITEMS = 10  # n: Número de items disponibles

# Parámetros del Algoritmo Genético
# Estos valores fueron elegidos basándose en buenas prácticas:
N_POBLACION = 50        # N: Tamaño de población (suficiente diversidad sin ser excesivo)
G_MAX = 100             # Número máximo de generaciones
PROB_CRUCE = 0.8        # pc: Probabilidad de cruce (alta para explorar bien)
PROB_MUTACION = 0.1     # pm: Probabilidad de mutación por bit (balance exploración/explotación)
ELITISMO_K = 1          # Número de individuos elite a preservar
SEMILLA = 42            # Seed para reproducibilidad

# =============================================================================
# FUNCIONES AUXILIARES Y DE EVALUACIÓN
# =============================================================================

def calcular_ratio_items() -> List[Tuple[int, float]]:
    """
    Calcula el ratio precio/peso para cada item y los ordena de mayor a menor.
    
    CONCEPTO: Esta función es clave para la REPARACIÓN por ratio.
    El ratio precio/peso indica la "eficiencia" de cada item.
    
    Returns:
        Lista de tuplas (índice_item, ratio) ordenada por ratio descendente
    """
    ratios = []
    for i in range(N_ITEMS):
        ratio = PRECIOS[i] / PESOS[i] if PESOS[i] > 0 else 0
        ratios.append((i, ratio))
    
    # Ordenar por ratio descendente (más eficientes primero)
    ratios.sort(key=lambda x: x[1], reverse=True)
    return ratios

def evaluar_individuo(individuo: List[int]) -> Tuple[int, int, bool]:
    """
    Evalúa un individuo calculando su precio total, peso total y validez.
    
    CONCEPTO: INDIVIDUO - Representación de una solución candidata
    Un individuo es un vector binario donde individuo[i] = 1 significa
    que el item i está incluido en la mochila.
    
    Args:
        individuo: Lista binaria de longitud N_ITEMS
        
    Returns:
        Tupla (precio_total, peso_total, es_valido)
    """
    precio_total = 0
    peso_total = 0
    
    # Sumar precios y pesos de los items incluidos (bit = 1)
    for i in range(N_ITEMS):
        if individuo[i] == 1:  # Si el item i está incluido
            precio_total += PRECIOS[i]
            peso_total += PESOS[i]
    
    # Un individuo es válido si no excede la capacidad máxima
    es_valido = peso_total <= CAPACIDAD_MAXIMA
    
    return precio_total, peso_total, es_valido

def calcular_fitness(individuo: List[int]) -> float:
    """
    Calcula el fitness de un individuo.
    
    CONCEPTO: FITNESS/IDONEIDAD - Medida de qué tan buena es una solución
    Para el knapsack: queremos maximizar el precio, pero solo si es válido.
    Si es inválido, aplicamos una penalización fuerte.
    
    Args:
        individuo: Vector binario representando la solución
        
    Returns:
        Valor de fitness (mayor es mejor)
    """
    precio, peso, es_valido = evaluar_individuo(individuo)
    
    if es_valido:
        # Si es válido, el fitness es directamente el precio
        return float(precio)
    else:
        # Si es inválido, aplicamos penalización severa
        # Penalización = precio - exceso_de_peso * factor_penalizacion
        exceso = peso - CAPACIDAD_MAXIMA
        factor_penalizacion = max(PRECIOS)  # Usar el precio máximo como factor
        return float(precio - exceso * factor_penalizacion)

def reparar_individuo(individuo: List[int]) -> List[int]:
    """
    Repara un individuo inválido eliminando items de menor ratio precio/peso.
    
    CONCEPTO: REPARACIÓN - Técnica para convertir soluciones inválidas en válidas
    Esta implementación usa reparación por ratio: elimina items incluidos
    empezando por los de menor eficiencia (ratio precio/peso) hasta que
    la solución sea válida.
    
    Args:
        individuo: Individuo posiblemente inválido
        
    Returns:
        Individuo reparado (garantizado válido)
    """
    individuo_reparado = individuo.copy()
    _, peso_actual, es_valido = evaluar_individuo(individuo_reparado)
    
    # Si ya es válido, no necesita reparación
    if es_valido:
        return individuo_reparado
    
    # Obtener ratios ordenados (de mayor a menor eficiencia)
    ratios_ordenados = calcular_ratio_items()
    
    # Crear lista de items incluidos ordenados por ratio ascendente (menos eficientes primero)
    items_incluidos = []
    for i in range(N_ITEMS):
        if individuo_reparado[i] == 1:
            # Encontrar el ratio de este item
            ratio = next(r for idx, r in ratios_ordenados if idx == i)
            items_incluidos.append((i, ratio))
    
    # Ordenar por ratio ascendente (eliminar menos eficientes primero)
    items_incluidos.sort(key=lambda x: x[1])
    
    # Eliminar items de menor ratio hasta que sea válido
    for item_idx, _ in items_incluidos:
        if peso_actual <= CAPACIDAD_MAXIMA:
            break
            
        # Eliminar este item (cambiar de 1 a 0)
        individuo_reparado[item_idx] = 0
        peso_actual -= PESOS[item_idx]
    
    return individuo_reparado

# =============================================================================
# OPERADORES GENÉTICOS
# =============================================================================

def inicializar_poblacion(tamano_poblacion: int) -> List[List[int]]:
    """
    Inicializa la población con individuos aleatorios válidos.
    
    CONCEPTO: INICIALIZACIÓN - Crear la población inicial del algoritmo genético
    Genera individuos binarios aleatorios y los repara para asegurar que todos
    sean válidos desde el inicio.
    
    Args:
        tamano_poblacion: Número de individuos a crear
        
    Returns:
        Lista de individuos (población inicial)
    """
    poblacion = []
    
    for _ in range(tamano_poblacion):
        # Crear individuo binario aleatorio
        individuo = [random.randint(0, 1) for _ in range(N_ITEMS)]
        
        # Reparar para asegurar que sea válido
        individuo_reparado = reparar_individuo(individuo)
        poblacion.append(individuo_reparado)
        #poblacion.append(individuo)  # Sin reparar para ver efecto de reparación luego

    return poblacion

def seleccion_ruleta(poblacion: List[List[int]], k: int) -> List[List[int]]:
    """
    Selecciona k individuos usando selección por ruleta.
    
    CONCEPTO: SELECCIÓN - Proceso de elegir individuos para reproducción
    La selección por ruleta da mayor probabilidad de selección a individuos
    con mejor fitness. Es proporcional al fitness: fitness más alto = mayor
    probabilidad de ser seleccionado.
    
    Args:
        poblacion: Población actual
        k: Número de individuos a seleccionar
        
    Returns:
        Lista de k individuos seleccionados
    """
    # Calcular fitness de todos los individuos
    fitness_values = [calcular_fitness(individuo) for individuo in poblacion]
    
    # Ajustar fitness para que todos sean positivos (necesario para ruleta)
    min_fitness = min(fitness_values)
    if min_fitness < 0:
        fitness_values = [f - min_fitness + 1 for f in fitness_values]
    
    # Calcular fitness total para la ruleta
    fitness_total = sum(fitness_values)
    
    seleccionados = []
    for _ in range(k):
        # Generar número aleatorio entre 0 y fitness_total
        r = random.uniform(0, fitness_total)
        
        # Encontrar el individuo correspondiente en la ruleta
        acumulado = 0
        for i, fitness in enumerate(fitness_values):
            acumulado += fitness
            if acumulado >= r:
                seleccionados.append(poblacion[i].copy())
                break
    
    return seleccionados

def cruce_one_point(padre1: List[int], padre2: List[int]) -> Tuple[List[int], List[int]]:
    """
    Realiza cruce de un punto entre dos padres.
    
    CONCEPTO: CRUCE - Operador que combina información de dos padres
    El cruce one-point elige un punto aleatorio y intercambia las partes
    de los cromosomas. Permite explorar nuevas combinaciones de genes.
    
    Args:
        padre1, padre2: Los dos individuos padres
        
    Returns:
        Tupla con los dos hijos generados
    """
    # Elegir punto de cruce aleatorio (no en los extremos)
    punto_cruce = random.randint(1, N_ITEMS - 1)
    
    # Crear hijos intercambiando segmentos
    hijo1 = padre1[:punto_cruce] + padre2[punto_cruce:]
    hijo2 = padre2[:punto_cruce] + padre1[punto_cruce:]
    
    return hijo1, hijo2

def mutacion_bit_flip(individuo: List[int], prob_mutacion: float) -> List[int]:
    """
    Aplica mutación bit a bit con cierta probabilidad.
    
    CONCEPTO: MUTACIÓN - Operador que introduce variabilidad aleatoria
    La mutación bit-flip cambia cada bit con probabilidad prob_mutacion.
    Ayuda a mantener diversidad y explorar nuevas regiones del espacio de búsqueda.
    
    Args:
        individuo: Individuo a mutar
        prob_mutacion: Probabilidad de mutar cada bit
        
    Returns:
        Individuo mutado
    """
    individuo_mutado = individuo.copy()
    
    # Para cada bit, decidir si mutarlo o no
    for i in range(N_ITEMS):
        if random.random() < prob_mutacion:
            # Flip del bit: 0 -> 1, 1 -> 0
            individuo_mutado[i] = 1 - individuo_mutado[i]
    
    return individuo_mutado

def aplicar_elitismo(poblacion_actual: List[List[int]], nueva_poblacion: List[List[int]], 
                    k_elite: int) -> List[List[int]]:
    """
    Aplica elitismo preservando los k mejores individuos.
    
    CONCEPTO: ELITISMO - Estrategia que garantiza que los mejores individuos
    no se pierdan entre generaciones. Los k individuos con mejor fitness
    pasan automáticamente a la siguiente generación.
    
    Args:
        poblacion_actual: Población de la generación actual
        nueva_poblacion: Nueva población generada
        k_elite: Número de individuos elite a preservar
        
    Returns:
        Nueva población con elitismo aplicado
    """
    # Calcular fitness de la población actual
    fitness_actuales = [(i, calcular_fitness(individuo)) 
                       for i, individuo in enumerate(poblacion_actual)]
    
    # Ordenar por fitness descendente (mejores primero)
    fitness_actuales.sort(key=lambda x: x[1], reverse=True)
    
    # Tomar los k mejores (elite)
    poblacion_con_elitismo = nueva_poblacion.copy()
    
    # Reemplazar los k peores de la nueva población con los k mejores de la actual
    for i in range(k_elite):
        idx_mejor = fitness_actuales[i][0]
        poblacion_con_elitismo[-(i+1)] = poblacion_actual[idx_mejor].copy()
    
    return poblacion_con_elitismo

# =============================================================================
# ALGORITMO GENÉTICO PRINCIPAL
# =============================================================================

def algoritmo_genetico(verbose: bool = True) -> Tuple[List[int], float, List[float], List[float]]:
    """
    Implementación principal del algoritmo genético.
    
    CONCEPTO: ALGORITMO GENÉTICO - Metaheurística evolutiva completa
    Combina todos los operadores: inicialización, evaluación, selección,
    cruce, mutación, reparación y elitismo en un ciclo evolutivo.
    
    Args:
        verbose: Si mostrar información durante la ejecución
        
    Returns:
        Tupla (mejor_individuo, mejor_fitness, historial_mejor_fitness, historial_promedio_fitness)
    """
    if verbose:
        print("🧬 INICIANDO ALGORITMO GENÉTICO")
        print("=" * 50)
        print(f"Población: {N_POBLACION}, Generaciones: {G_MAX}")
        print(f"Prob. Cruce: {PROB_CRUCE}, Prob. Mutación: {PROB_MUTACION}")
        print(f"Elitismo: {ELITISMO_K} individuos")
        print()
    
    # INICIALIZACIÓN: Crear población inicial
    poblacion = inicializar_poblacion(N_POBLACION)
    
    # Historial para graficar evolución
    historial_mejor_fitness = []
    historial_promedio_fitness = []
    mejor_individuo_global = None
    mejor_fitness_global = float('-inf')
    
    # CRITERIO DE PARADA: Bucle principal por G_MAX generaciones
    for generacion in range(G_MAX):
        
        # EVALUACIÓN: Calcular fitness de toda la población
        fitness_poblacion = []
        for individuo in poblacion:
            fitness = calcular_fitness(individuo)
            fitness_poblacion.append(fitness)
        
        # Encontrar el mejor de esta generación
        mejor_fitness_gen = max(fitness_poblacion)
        mejor_idx = fitness_poblacion.index(mejor_fitness_gen)
        mejor_individuo_gen = poblacion[mejor_idx]
        
        # Calcular promedio de fitness de la generación
        promedio_fitness_gen = sum(fitness_poblacion) / len(fitness_poblacion)
        
        # Actualizar el mejor global
        if mejor_fitness_gen > mejor_fitness_global:
            mejor_fitness_global = mejor_fitness_gen
            mejor_individuo_global = mejor_individuo_gen.copy()
        
        # Guardar para historial
        historial_mejor_fitness.append(mejor_fitness_gen)
        historial_promedio_fitness.append(promedio_fitness_gen)
        
        # Mostrar progreso (incluyendo la última generación siempre)
        if verbose and (generacion % 20 == 0 or generacion == G_MAX - 1):
            precio, peso, valido = evaluar_individuo(mejor_individuo_gen)
            print(f"Gen {generacion:3d}: Mejor={mejor_fitness_gen:6.1f}, "
                  f"Promedio={promedio_fitness_gen:6.1f}, "
                  f"Precio=${precio:4d}, Peso={peso:4d}kg, Válido={valido}")
        
        # Si no es la última generación, generar nueva población
        if generacion < G_MAX - 1:
            nueva_poblacion = []
            
            # Generar nueva población mediante selección, cruce y mutación
            while len(nueva_poblacion) < N_POBLACION:
                
                # SELECCIÓN: Seleccionar dos padres por ruleta
                padres = seleccion_ruleta(poblacion, 2)
                padre1, padre2 = padres[0], padres[1]
                
                # CRUCE: Aplicar cruce con probabilidad pc
                if random.random() < PROB_CRUCE:
                    hijo1, hijo2 = cruce_one_point(padre1, padre2)
                else:
                    # Si no hay cruce, los hijos son copias de los padres
                    hijo1, hijo2 = padre1.copy(), padre2.copy()
                
                # MUTACIÓN: Aplicar mutación a cada hijo
                hijo1_mutado = mutacion_bit_flip(hijo1, PROB_MUTACION)
                hijo2_mutado = mutacion_bit_flip(hijo2, PROB_MUTACION)
                
                # REPARACIÓN: Asegurar que los hijos sean válidos
                hijo1_reparado = reparar_individuo(hijo1_mutado)
                hijo2_reparado = reparar_individuo(hijo2_mutado)
                
                # Agregar hijos a la nueva población
                nueva_poblacion.extend([hijo1_reparado, hijo2_reparado])
                #nueva_poblacion.extend([hijo1_mutado, hijo2_mutado])
            
            # Ajustar tamaño si se excedió
            nueva_poblacion = nueva_poblacion[:N_POBLACION]
            
            # ELITISMO: Preservar los mejores individuos
            poblacion = aplicar_elitismo(poblacion, nueva_poblacion, ELITISMO_K)
    
    if verbose:
        print()
        print("✅ ALGORITMO GENÉTICO COMPLETADO")
        print()
    
    return mejor_individuo_global, mejor_fitness_global, historial_mejor_fitness, historial_promedio_fitness

# =============================================================================
# VERIFICACIÓN POR FUERZA BRUTA
# =============================================================================

def fuerza_bruta() -> Tuple[List[int], int, int]:
    """
    Encuentra la solución óptima por fuerza bruta.
    
    CONCEPTO: COMPARACIÓN CON FUERZA BRUTA - Verificación de calidad del GA
    Para n=10 items, hay 2^10 = 1024 combinaciones posibles. Es factible
    evaluar todas para encontrar el óptimo real y comparar con el GA.
    
    Returns:
        Tupla (mejor_solucion, mejor_precio, peso_solucion)
    """
    mejor_solucion = None
    mejor_precio = 0
    peso_mejor = 0
    
    # Evaluar todas las 2^n combinaciones posibles
    for i in range(2**N_ITEMS):
        # Convertir número binario a vector binario
        solucion = []
        temp = i
        for _ in range(N_ITEMS):
            solucion.append(temp % 2)
            temp //= 2
        
        # Evaluar esta solución
        precio, peso, es_valido = evaluar_individuo(solucion)
        
        # Si es válida y mejor que la actual, actualizarla
        if es_valido and precio > mejor_precio:
            mejor_precio = precio
            mejor_solucion = solucion.copy()
            peso_mejor = peso
    
    return mejor_solucion, mejor_precio, peso_mejor

# =============================================================================
# VISUALIZACIÓN Y ANÁLISIS DE RESULTADOS
# =============================================================================

def mostrar_solucion(titulo: str, solucion: List[int], precio: int, peso: int):
    """Muestra una solución de forma legible."""
    print(f"\n📋 {titulo}")
    print("-" * 40)
    print(f"Vector binario: {solucion}")
    print(f"Items incluidos:", end=" ")
    items_incluidos = [i for i in range(N_ITEMS) if solucion[i] == 1]
    if items_incluidos:
        print(f"{items_incluidos} (índices)")
    else:
        print("Ninguno")
    print(f"Precio total: ${precio}")
    print(f"Peso total: {peso} kg")
    print(f"Capacidad usada: {peso/CAPACIDAD_MAXIMA*100:.1f}%")

def graficar_evolucion(historial_mejor_fitness: List[float], historial_promedio_fitness: List[float]):
    """
    Crea gráfica de la evolución del fitness mostrando mejor y promedio.
    
    Muestra cómo evoluciona tanto el mejor fitness como el fitness promedio
    a lo largo de las generaciones, permitiendo observar la convergencia 
    del algoritmo genético y la evolución de toda la población.
    
    Args:
        historial_mejor_fitness: Lista con el mejor fitness de cada generación
        historial_promedio_fitness: Lista con el fitness promedio de cada generación
    """
    plt.figure(figsize=(12, 7))
    
    generaciones = list(range(len(historial_mejor_fitness)))
    
    # Graficar mejor fitness
    plt.plot(generaciones, historial_mejor_fitness, 'b-', linewidth=2.5, 
             marker='o', markersize=4, label='Mejor Fitness', alpha=0.8)
    
    # Graficar fitness promedio
    plt.plot(generaciones, historial_promedio_fitness, 'r--', linewidth=2, 
             marker='s', markersize=3, label='Fitness Promedio', alpha=0.7)
    
    # Configuración de la gráfica
    plt.title('Evolución del Fitness por Generación', fontsize=16, fontweight='bold')
    plt.xlabel('Generación', fontsize=12)
    plt.ylabel('Fitness (Precio)', fontsize=12)
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3, linestyle=':', linewidth=1)
    
    # Añadir información adicional
    mejor_final = historial_mejor_fitness[-1]
    promedio_final = historial_promedio_fitness[-1]
    plt.axhline(y=mejor_final, color='blue', linestyle=':', alpha=0.5)
    plt.axhline(y=promedio_final, color='red', linestyle=':', alpha=0.5)
    
    # Anotaciones con valores finales
    plt.annotate(f'Mejor Final: {mejor_final:.1f}', 
                xy=(len(generaciones)-1, mejor_final), 
                xytext=(len(generaciones)*0.7, mejor_final + (mejor_final - promedio_final)*0.3),
                arrowprops=dict(arrowstyle='->', color='blue', alpha=0.7),
                fontsize=10, color='blue')
    
    plt.annotate(f'Promedio Final: {promedio_final:.1f}', 
                xy=(len(generaciones)-1, promedio_final), 
                xytext=(len(generaciones)*0.7, promedio_final - (mejor_final - promedio_final)*0.3),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                fontsize=10, color='red')
    
    plt.tight_layout()
    plt.show()

def estadisticas_problema():
    """Muestra estadísticas del problema."""
    print("📊 ESTADÍSTICAS DEL PROBLEMA")
    print("=" * 50)
    print(f"Número de items: {N_ITEMS}")
    print(f"Capacidad máxima: {CAPACIDAD_MAXIMA} kg")
    print(f"Precio total si incluimos todo: ${sum(PRECIOS)}")
    print(f"Peso total si incluimos todo: {sum(PESOS)} kg")
    print(f"Espacio de búsqueda: 2^{N_ITEMS} = {2**N_ITEMS:,} combinaciones")
    
    # Mostrar ratios
    ratios = calcular_ratio_items()
    print(f"\nRatios precio/peso (ordenados):")
    for i, (idx, ratio) in enumerate(ratios):
        print(f"  {i+1}. Item {idx}: ${PRECIOS[idx]}/{PESOS[idx]}kg = {ratio:.3f}")

# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    """
    Función principal que ejecuta todo el experimento.
    
    FLUJO COMPLETO:
    1. Configuración y estadísticas del problema
    2. Ejecución del algoritmo genético
    3. Verificación por fuerza bruta
    4. Comparación de resultados
    5. Visualización de la evolución
    """
    print("🎒 PROBLEMA 0/1 KNAPSACK CON ALGORITMO GENÉTICO")
    print("=" * 60)
    
    # Configurar semilla para reproducibilidad
    random.seed(SEMILLA)
    np.random.seed(SEMILLA)
    
    # Mostrar estadísticas del problema
    estadisticas_problema()
    
    print(f"\n⚙️  PARÁMETROS DEL ALGORITMO GENÉTICO")
    print("-" * 40)
    print(f"Tamaño población: {N_POBLACION}")
    print(f"Máximo generaciones: {G_MAX}")
    print(f"Probabilidad cruce: {PROB_CRUCE}")
    print(f"Probabilidad mutación: {PROB_MUTACION}")
    print(f"Individuos elite: {ELITISMO_K}")
    print(f"Semilla aleatoria: {SEMILLA}")
    
    # Ejecutar algoritmo genético
    print(f"\n" + "🚀" * 20)
    start_time = time.time()
    mejor_ga, fitness_ga, historial_mejor, historial_promedio = algoritmo_genetico(verbose=True)
    tiempo_ga = time.time() - start_time
    
    # Ejecutar fuerza bruta para comparación
    print("🔍 EJECUTANDO VERIFICACIÓN POR FUERZA BRUTA...")
    start_time = time.time()
    mejor_fb, precio_fb, peso_fb = fuerza_bruta()
    tiempo_fb = time.time() - start_time
    print(f"✅ Fuerza bruta completada en {tiempo_fb:.3f} segundos")
    
    # Calcular datos de la solución del GA
    precio_ga, peso_ga, valido_ga = evaluar_individuo(mejor_ga)
    
    # Mostrar resultados
    print(f"\n" + "📈" * 20)
    print("COMPARACIÓN DE RESULTADOS")
    print("=" * 60)
    
    mostrar_solucion("SOLUCIÓN POR ALGORITMO GENÉTICO", mejor_ga, precio_ga, peso_ga)
    mostrar_solucion("SOLUCIÓN ÓPTIMA (FUERZA BRUTA)", mejor_fb, precio_fb, peso_fb)
    
    # Análisis de calidad
    print(f"\n🎯 ANÁLISIS DE CALIDAD")
    print("-" * 30)
    print(f"Precio GA: ${precio_ga}")
    print(f"Precio óptimo: ${precio_fb}")
    if precio_fb > 0:
        calidad = (precio_ga / precio_fb) * 100
        print(f"Calidad del GA: {calidad:.2f}% del óptimo")
        if calidad == 100:
            print("🏆 ¡El GA encontró la solución óptima!")
        elif calidad >= 95:
            print("🥈 Excelente resultado del GA")
        elif calidad >= 90:
            print("🥉 Buen resultado del GA")
        else:
            print("⚠️  El GA podría mejorar")
    
    print(f"\n⏱️  TIEMPOS DE EJECUCIÓN")
    print("-" * 25)
    print(f"Algoritmo genético: {tiempo_ga:.3f} segundos")
    print(f"Fuerza bruta: {tiempo_fb:.3f} segundos")
    
    # Análisis de estadísticas de evolución
    print(f"\n📊 ESTADÍSTICAS DE EVOLUCIÓN")
    print("-" * 35)
    print(f"Mejor fitness inicial: {historial_mejor[0]:.1f}")
    print(f"Mejor fitness final: {historial_mejor[-1]:.1f}")
    print(f"Mejora absoluta: {historial_mejor[-1] - historial_mejor[0]:.1f}")
    print(f"Promedio fitness inicial: {historial_promedio[0]:.1f}")
    print(f"Promedio fitness final: {historial_promedio[-1]:.1f}")
    print(f"Convergencia (mejor-promedio final): {historial_mejor[-1] - historial_promedio[-1]:.1f}")
    
    # Verificar que el GA siempre produce soluciones válidas
    print(f"\n✅ VALIDACIÓN")
    print("-" * 15)
    print(f"Solución GA válida: {valido_ga}")
    print(f"Peso GA ≤ Capacidad: {peso_ga} ≤ {CAPACIDAD_MAXIMA} = {peso_ga <= CAPACIDAD_MAXIMA}")
    
    print(f"\n🎉 EXPERIMENTO COMPLETADO")
    print("=" * 30)
    print("El código demostró todos los conceptos del algoritmo genético:")
    print("✓ Representación binaria de individuos")
    print("✓ Función de fitness con penalización")
    print("✓ Selección por ruleta")
    print("✓ Cruce de un punto")
    print("✓ Mutación bit a bit")
    print("✓ Reparación por ratio precio/peso")
    print("✓ Elitismo")
    print("✓ Criterio de parada por generaciones")
    print("✓ Comparación con solución exacta")
        
    # Graficar evolución
    print(f"\n📊 Mostrando gráfica de evolución...")
    graficar_evolucion(historial_mejor, historial_promedio)

# =============================================================================
# EJECUCIÓN DEL PROGRAMA
# =============================================================================

if __name__ == "__main__":
    main()