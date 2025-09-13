import numpy as np

# Probabilidades base
P_M = 1e-5
P_E = 1e-3
P_ninguna = 1 - P_M - P_E

print("=== RED BAYESIANA - INFERENCIA POR ENUMERACIÓN ===")
print(f"Probabilidades base:")
print(f"P(M) = {P_M}")
print(f"P(E) = {P_E}")
print(f"P(ninguna) = {P_ninguna}")
print()

# Distribuciones condicionales de temperatura
# P(temperatura | estado_avería)
temp_probs = {
    'ninguna': {'TE': 0.17, 'TR': 0.05, 'TN': 0.78},
    'E': {'TE': 0.90, 'TR': 0.01, 'TN': 0.09},
    'M': {'TE': 0.10, 'TR': 0.40, 'TN': 0.50}
}

# Distribuciones condicionales del piloto
# P(p | temperatura)
P_p_given_temp = {
    'TE': 0.95,
    'TR': 0.99,
    'TN': 1e-6
}

print("=== 1. CÁLCULO DE MARGINALES DE TEMPERATURA ===")
print("P(temperatura) = Σ_averías P(temperatura|avería) * P(avería)")
print()

# Cálculo de marginales de temperatura
P_TE = (temp_probs['ninguna']['TE'] * P_ninguna + 
        temp_probs['E']['TE'] * P_E + 
        temp_probs['M']['TE'] * P_M)

P_TR = (temp_probs['ninguna']['TR'] * P_ninguna + 
        temp_probs['E']['TR'] * P_E + 
        temp_probs['M']['TR'] * P_M)

P_TN = (temp_probs['ninguna']['TN'] * P_ninguna + 
        temp_probs['E']['TN'] * P_E + 
        temp_probs['M']['TN'] * P_M)

print(f"P(TE) = {temp_probs['ninguna']['TE']} * {P_ninguna} + {temp_probs['E']['TE']} * {P_E} + {temp_probs['M']['TE']} * {P_M}")
print(f"P(TE) = {P_TE}")
print()
print(f"P(TR) = {temp_probs['ninguna']['TR']} * {P_ninguna} + {temp_probs['E']['TR']} * {P_E} + {temp_probs['M']['TR']} * {P_M}")
print(f"P(TR) = {P_TR}")
print()
print(f"P(TN) = {temp_probs['ninguna']['TN']} * {P_ninguna} + {temp_probs['E']['TN']} * {P_E} + {temp_probs['M']['TN']} * {P_M}")
print(f"P(TN) = {P_TN}")
print()

print("=== 2. CÁLCULO DE P(p) ===")
print("P(p) = Σ_temperaturas P(p|temperatura) * P(temperatura)")
print()

P_p = (P_p_given_temp['TE'] * P_TE + 
       P_p_given_temp['TR'] * P_TR + 
       P_p_given_temp['TN'] * P_TN)

print(f"P(p) = {P_p_given_temp['TE']} * {P_TE} + {P_p_given_temp['TR']} * {P_TR} + {P_p_given_temp['TN']} * {P_TN}")
print(f"P(p) = {P_p}")
print()

print("=== 3. CÁLCULO DE P(p,M) y P(p,¬M) ===")
print()

# P(p,M) = Σ_temperaturas P(p|temperatura) * P(temperatura|M) * P(M)
P_p_M = (P_p_given_temp['TE'] * temp_probs['M']['TE'] * P_M + 
         P_p_given_temp['TR'] * temp_probs['M']['TR'] * P_M + 
         P_p_given_temp['TN'] * temp_probs['M']['TN'] * P_M)

print(f"P(p,M) = Σ_temp P(p|temp) * P(temp|M) * P(M)")
print(f"P(p,M) = {P_p_given_temp['TE']} * {temp_probs['M']['TE']} * {P_M} + {P_p_given_temp['TR']} * {temp_probs['M']['TR']} * {P_M} + {P_p_given_temp['TN']} * {temp_probs['M']['TN']} * {P_M}")
print(f"P(p,M) = {P_p_M}")
print()

# P(p,¬M) = P(p,E) + P(p,ninguna)
# P(p,E)
P_p_E = (P_p_given_temp['TE'] * temp_probs['E']['TE'] * P_E + 
         P_p_given_temp['TR'] * temp_probs['E']['TR'] * P_E + 
         P_p_given_temp['TN'] * temp_probs['E']['TN'] * P_E)

# P(p,ninguna)
P_p_ninguna = (P_p_given_temp['TE'] * temp_probs['ninguna']['TE'] * P_ninguna + 
               P_p_given_temp['TR'] * temp_probs['ninguna']['TR'] * P_ninguna + 
               P_p_given_temp['TN'] * temp_probs['ninguna']['TN'] * P_ninguna)

P_p_not_M = P_p_E + P_p_ninguna

print(f"P(p,¬M) = P(p,E) + P(p,ninguna)")
print(f"P(p,E) = {P_p_E}")
print(f"P(p,ninguna) = {P_p_ninguna}")
print(f"P(p,¬M) = {P_p_not_M}")
print()

print("=== 4. CÁLCULO DEL FACTOR DE NORMALIZACIÓN α ===")
print()

alpha = 1 / (P_p_M + P_p_not_M)
print(f"α = 1 / (P(p,M) + P(p,¬M))")
print(f"α = 1 / ({P_p_M} + {P_p_not_M})")
print(f"α = 1 / {P_p_M + P_p_not_M}")
print(f"α = {alpha}")
print()

print("=== 5. CÁLCULO DE P(M|p) USANDO INFERENCIA POR ENUMERACIÓN ===")
print()

P_M_given_p = alpha * P_p_M
P_not_M_given_p = alpha * P_p_not_M

print(f"P(M|p) = α * P(p,M)")
print(f"P(M|p) = {alpha} * {P_p_M}")
print(f"P(M|p) = {P_M_given_p}")
print()
print(f"P(¬M|p) = α * P(p,¬M)")
print(f"P(¬M|p) = {alpha} * {P_p_not_M}")
print(f"P(¬M|p) = {P_not_M_given_p}")
print()

# Verificación
print(f"Verificación: P(M|p) + P(¬M|p) = {P_M_given_p + P_not_M_given_p}")
print()

print("=== 6. CÁLCULO DE P(M|p,TE) ===")
print()

# P(p,TE,M) = P(p|TE) * P(TE|M) * P(M)
P_p_TE_M = P_p_given_temp['TE'] * temp_probs['M']['TE'] * P_M

# P(p,TE,¬M) = P(p,TE,E) + P(p,TE,ninguna)
P_p_TE_E = P_p_given_temp['TE'] * temp_probs['E']['TE'] * P_E
P_p_TE_ninguna = P_p_given_temp['TE'] * temp_probs['ninguna']['TE'] * P_ninguna
P_p_TE_not_M = P_p_TE_E + P_p_TE_ninguna

print(f"P(p,TE,M) = P(p|TE) * P(TE|M) * P(M)")
print(f"P(p,TE,M) = {P_p_given_temp['TE']} * {temp_probs['M']['TE']} * {P_M}")
print(f"P(p,TE,M) = {P_p_TE_M}")
print()

print(f"P(p,TE,¬M) = P(p,TE,E) + P(p,TE,ninguna)")
print(f"P(p,TE,E) = {P_p_given_temp['TE']} * {temp_probs['E']['TE']} * {P_E} = {P_p_TE_E}")
print(f"P(p,TE,ninguna) = {P_p_given_temp['TE']} * {temp_probs['ninguna']['TE']} * {P_ninguna} = {P_p_TE_ninguna}")
print(f"P(p,TE,¬M) = {P_p_TE_not_M}")
print()

# Nuevo factor de normalización
alpha_new = 1 / (P_p_TE_M + P_p_TE_not_M)
print(f"Nuevo α = 1 / (P(p,TE,M) + P(p,TE,¬M))")
print(f"Nuevo α = 1 / ({P_p_TE_M} + {P_p_TE_not_M})")
print(f"Nuevo α = 1 / {P_p_TE_M + P_p_TE_not_M}")
print(f"Nuevo α = {alpha_new}")
print()

# P(M|p,TE)
P_M_given_p_TE = alpha_new * P_p_TE_M
P_not_M_given_p_TE = alpha_new * P_p_TE_not_M

print(f"P(M|p,TE) = α * P(p,TE,M)")
print(f"P(M|p,TE) = {alpha_new} * {P_p_TE_M}")
print(f"P(M|p,TE) = {P_M_given_p_TE}")
print()
print(f"P(¬M|p,TE) = α * P(p,TE,¬M)")
print(f"P(¬M|p,TE) = {alpha_new} * {P_p_TE_not_M}")
print(f"P(¬M|p,TE) = {P_not_M_given_p_TE}")
print()

# Verificación final
print(f"Verificación: P(M|p,TE) + P(¬M|p,TE) = {P_M_given_p_TE + P_not_M_given_p_TE}")
print()

print("=== RESUMEN DE RESULTADOS ===")
print(f"P(M|p) = {P_M_given_p:.8f}")
print(f"P(M|p,TE) = {P_M_given_p_TE:.8f}")
print()
print(f"La evidencia adicional TE {'aumenta' if P_M_given_p_TE > P_M_given_p else 'disminuye'} la probabilidad de avería mecánica")