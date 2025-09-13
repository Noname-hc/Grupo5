import random
import matplotlib.pyplot as plt

def resolver_teoricamente():
    """Resuelve el problema usando el Teorema de Bayes"""
    print("=== RESOLUCIÓN TEÓRICA ===")
    print("\nDatos del problema:")
    print("- Máquina A: produce 30% de clavos, 2% defectuosos")
    print("- Máquina B: produce 70% de clavos, 3% defectuosos")
    print("- Pregunta: Si un clavo es defectuoso, ¿cuál es la probabilidad de que venga de cada máquina?")
    
    # Datos
    P_A = 0.30  # Probabilidad de que un clavo sea de máquina A
    P_B = 0.70  # Probabilidad de que un clavo sea de máquina B
    P_D_A = 0.02  # Probabilidad de defectuoso dado que es de A
    P_D_B = 0.03  # Probabilidad de defectuoso dado que es de B
    
    print(f"\nPaso 1: Calcular P(Defectuoso) - Probabilidad Total")
    P_D = P_D_A * P_A + P_D_B * P_B
    print(f"P(D) = P(D|A) × P(A) + P(D|B) × P(B)")
    print(f"P(D) = {P_D_A} × {P_A} + {P_D_B} × {P_B}")
    print(f"P(D) = {P_D_A * P_A} + {P_D_B * P_B} = {P_D}")
    
    print(f"\nPaso 2: Aplicar Teorema de Bayes")
    P_A_D = (P_D_A * P_A) / P_D  # P(A|D)
    P_B_D = (P_D_B * P_B) / P_D  # P(B|D)
    
    print(f"P(A|D) = P(D|A) × P(A) / P(D) = ({P_D_A} × {P_A}) / {P_D} = {P_A_D:.4f}")
    print(f"P(B|D) = P(D|B) × P(B) / P(D) = ({P_D_B} × {P_B}) / {P_D} = {P_B_D:.4f}")
    
    print(f"\n=== RESPUESTA ===")
    print(f"Probabilidad de que el clavo defectuoso sea de Máquina A: {P_A_D:.2%}")
    print(f"Probabilidad de que el clavo defectuoso sea de Máquina B: {P_B_D:.2%}")
    
    return P_A_D, P_B_D

def simular_produccion(num_simulaciones=100000):
    """Simula la producción de clavos y verifica el resultado teórico"""
    print(f"\n=== SIMULACIÓN CON {num_simulaciones:,} CLAVOS ===")
    
    clavos_defectuosos_A = 0
    clavos_defectuosos_B = 0
    total_defectuosos = 0
    
    # Contadores para verificar las proporciones
    total_A = 0
    total_B = 0
    defectuosos_A = 0
    defectuosos_B = 0
    
    for _ in range(num_simulaciones):
        # Decidir qué máquina produce el clavo
        if random.random() < 0.30:  # 30% máquina A
            maquina = 'A'
            total_A += 1
            # Verificar si es defectuoso (2% de probabilidad)
            if random.random() < 0.02:
                defectuosos_A += 1
                clavos_defectuosos_A += 1
                total_defectuosos += 1
        else:  # 70% máquina B
            maquina = 'B'
            total_B += 1
            # Verificar si es defectuoso (3% de probabilidad)
            if random.random() < 0.03:
                defectuosos_B += 1
                clavos_defectuosos_B += 1
                total_defectuosos += 1
    
    print(f"\nResultados de la simulación:")
    print(f"Total de clavos producidos: {num_simulaciones:,}")
    print(f"Clavos de máquina A: {total_A:,} ({total_A/num_simulaciones:.1%})")
    print(f"Clavos de máquina B: {total_B:,} ({total_B/num_simulaciones:.1%})")
    print(f"Total de clavos defectuosos: {total_defectuosos:,} ({total_defectuosos/num_simulaciones:.3%})")
    print(f"Defectuosos de máquina A: {defectuosos_A:,} ({defectuosos_A/total_A:.3%} de los de A)")
    print(f"Defectuosos de máquina B: {defectuosos_B:,} ({defectuosos_B/total_B:.3%} de los de B)")
    
    if total_defectuosos > 0:
        prob_A_simulada = clavos_defectuosos_A / total_defectuosos
        prob_B_simulada = clavos_defectuosos_B / total_defectuosos
        
        print(f"\n=== RESULTADOS DE LA SIMULACIÓN ===")
        print(f"De los clavos defectuosos:")
        print(f"Probabilidad de ser de máquina A: {prob_A_simulada:.2%}")
        print(f"Probabilidad de ser de máquina B: {prob_B_simulada:.2%}")
        
        return prob_A_simulada, prob_B_simulada
    else:
        print("No se encontraron clavos defectuosos en la simulación")
        return 0, 0

def crear_visualizacion(P_A_teorica, P_B_teorica, P_A_simulada, P_B_simulada):
    """Crea una visualización comparando resultados teóricos y simulados"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Gráfico de barras comparativo
    maquinas = ['Máquina A', 'Máquina B']
    teoricas = [P_A_teorica, P_B_teorica]
    simuladas = [P_A_simulada, P_B_simulada]
    
    x = range(len(maquinas))
    width = 0.35
    
    ax1.bar([i - width/2 for i in x], teoricas, width, label='Teórico', alpha=0.8, color='skyblue')
    ax1.bar([i + width/2 for i in x], simuladas, width, label='Simulado', alpha=0.8, color='lightcoral')
    
    ax1.set_xlabel('Máquina')
    ax1.set_ylabel('Probabilidad')
    ax1.set_title('Comparación: Teórico vs Simulado')
    ax1.set_xticks(x)
    ax1.set_xticklabels(maquinas)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Añadir valores en las barras
    for i, v in enumerate(teoricas):
        ax1.text(i - width/2, v + 0.01, f'{v:.2%}', ha='center', va='bottom')
    for i, v in enumerate(simuladas):
        ax1.text(i + width/2, v + 0.01, f'{v:.2%}', ha='center', va='bottom')
    
    # Gráfico circular
    labels = ['Máquina A', 'Máquina B']
    sizes = [P_A_teorica, P_B_teorica]
    colors = ['skyblue', 'lightcoral']
    explode = (0.05, 0.05)
    
    ax2.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.2f%%',
            shadow=True, startangle=90)
    ax2.set_title('Distribución de Clavos Defectuosos\n(Resultado Teórico)')
    
    plt.tight_layout()
    plt.show()

def main():
    """Función principal que ejecuta todo el análisis"""
    print("ANÁLISIS DE PROBABILIDAD: FÁBRICA DE CLAVOS")
    print("=" * 50)
    
    # Resolver teóricamente
    P_A_teorica, P_B_teorica = resolver_teoricamente()
    
    # Simular
    P_A_simulada, P_B_simulada = simular_produccion()
    
    # Comparar resultados
    print(f"\n=== COMPARACIÓN DE RESULTADOS ===")
    print(f"{'Método':<15} {'Máquina A':<12} {'Máquina B':<12} {'Diferencia A':<15} {'Diferencia B':<15}")
    print("-" * 70)
    print(f"{'Teórico':<15} {P_A_teorica:<12.4f} {P_B_teorica:<12.4f} {'-':<15} {'-':<15}")
    print(f"{'Simulado':<15} {P_A_simulada:<12.4f} {P_B_simulada:<12.4f} {abs(P_A_teorica-P_A_simulada):<15.4f} {abs(P_B_teorica-P_B_simulada):<15.4f}")
    
    # Crear visualización
    try:
        crear_visualizacion(P_A_teorica, P_B_teorica, P_A_simulada, P_B_simulada)
    except ImportError:
        print("\nNota: matplotlib no está disponible para crear gráficos")
    
    print(f"\n=== CONCLUSIÓN ===")
    print(f"La probabilidad de que un clavo defectuoso haya sido fabricado por la máquina A es: {P_A_teorica:.2%}")
    print(f"La probabilidad de que un clavo defectuoso haya sido fabricado por la máquina B es: {P_B_teorica:.2%}")

if __name__ == "__main__":
    main()