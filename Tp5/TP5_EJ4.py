import numpy as np
import pandas as pd

# Matriz identidad 4x4
I = np.eye(4)

# Matriz N (Porcentual)
N = np.array([[0, 95, 0, 0],   # Porcentual
              [7, 0, 90, 0],
              [0, 0, 0, 95],
              [0, 0, 7, 0]])

N = 0.01 * N

# Matriz A (Porcentual)
A = np.array([[5, 0],   # Porcentual
              [3, 0],
              [5, 0],
              [3, 90]])

A = 0.01 * A

# Tiempos en minutos
Tiempos = np.array([20, 5, 30, 7]).reshape(-1, 1)  # Columna

# Cálculo de probabilidades de absorción
Prob_Abs = np.linalg.inv(I - N) @ A
Prob_Abs = np.round(Prob_Abs, 2)  # Redondear a 2 decimales
Porcentual = Prob_Abs * 100

# --- Crear tabla con nombres ---
row_names = ['M1', 'I1', 'M2', 'I2']
col_names = ['D', 'B']

print('\nTabla de Probabilidades (%):')
porcentual_table = pd.DataFrame(Porcentual, 
                               columns=col_names, 
                               index=row_names)
print(porcentual_table)

# Cálculo de tiempos esperados
calculo_tiempo = np.linalg.inv(I - N) @ Tiempos

print('\nTabla de Tiempos Esperados (minutos):')
tiempo_tabla = pd.DataFrame(calculo_tiempo, 
                           columns=['Tiempo'], 
                           index=row_names)
print(tiempo_tabla)