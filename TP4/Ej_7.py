# ----------------------------
# Motor de inferencia - Encadenamiento hacia atrás
# ----------------------------

# Base de reglas: cada regla se representa como una tupla (premisas, conclusión)
reglas = [
    (["b", "c"], "a"),   # R1
    (["d", "e"], "b"),   # R2
    (["g", "e"], "b"),   # R3
    (["e"], "c"),        # R4
    ([], "d"),           # R5 (hecho base)
    ([], "e"),           # R6 (hecho base)
    (["a", "g"], "f"),   # R7
]

# Iniciamos un contador de iteraciones para finalmente saber en cuantas se logra demostrar (o no) la hipotesis
contador_iteraciones = 0

# Hechos iniciales (se infieren de reglas sin premisas)
# Básicamente si la lista de premisas de alguna de las reglas está vacía, entonces establece que esa conclusión es un hecho
hechos = set([conclusion for premisas, conclusion in reglas if len(premisas) == 0])

# ----------------------------
# Encadenamiento hacia atrás
# ----------------------------
def probar_objetivo(objetivo, visitados=None, nivel=0):
    global contador_iteraciones
    contador_iteraciones += 1

    if visitados is None:
        visitados = set()

    indent = "  " * nivel  # sangría visual

    print(f"{indent}Intentando demostrar: {objetivo}")

    if objetivo in hechos:
        print(f"{indent}✔ '{objetivo}' es un hecho conocido.")
        return True

    if objetivo in visitados:
        print(f"{indent}✘ Ciclo detectado con '{objetivo}', se descarta.")
        return False
    visitados.add(objetivo)

    for premisas, conclusion in reglas:
        if conclusion == objetivo:
            if premisas:
                print(f"{indent}Usando regla: {' ∧ '.join(premisas)} → {conclusion}")
            else:
                print(f"{indent}Usando regla: {conclusion} (hecho base)")

            # Intentar probar cada premisa recursivamente
            if all(probar_objetivo(p, visitados.copy(), nivel + 1) for p in premisas):
                hechos.add(objetivo)
                print(f"{indent}✔ '{objetivo}' demostrado.")
                return True
            else:
                print(f"{indent}✘ No se pudo demostrar '{objetivo}' con esta regla.")

    print(f"{indent}✘ No hay manera de demostrar '{objetivo}'.")
    return False

# ----------------------------
# Menú interactivo
# ----------------------------
def menu():
    while True:
        print("\n--- Motor de Inferencia (Encadenamiento hacia atrás) ---")
        print("1. Ver base de conocimientos")
        print("2. Demostrar una hipótesis")
        print("3. Salir")
        opcion = input("Seleccione una opción: ")

        if opcion == "1":
            print("\nBase de conocimientos:")
            for i, (premisas, conclusion) in enumerate(reglas, start=1):
                if premisas:
                    print(f"R{i}: {' ∧ '.join(premisas)} → {conclusion}")
                else:
                    print(f"R{i}: {conclusion} (hecho)")
        elif opcion == "2":
            h = input("Ingrese la hipótesis a demostrar: ").strip()
            global contador_iteraciones
            contador_iteraciones = 0  # reiniciamos antes de probar
            if probar_objetivo(h):
                print(f"La hipótesis '{h}' SE PUEDE demostrar.")
            else:
                print(f"La hipótesis '{h}' NO se puede demostrar.")
            
            print(f"Iteraciones realizadas: {contador_iteraciones}")        
        elif opcion == "3":
            print("Saliendo...")
            break
        else:
            print("Opción inválida. Intente de nuevo.")


# ----------------------------
# Programa principal
# ----------------------------
if __name__ == "__main__":
    menu()