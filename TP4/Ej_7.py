# ----------------------------
# Motor de inferencia - Encadenamiento hacia atrás
# ----------------------------

# Base de reglas: cada regla se representa como (premisas, conclusión)
reglas = [
    (["b", "c"], "a"),   # R1
    (["d", "e"], "b"),   # R2
    (["g", "e"], "b"),   # R3
    (["e"], "c"),        # R4
    ([], "d"),           # R5 (hecho base)
    ([], "e"),           # R6 (hecho base)
    (["a", "g"], "f"),   # R7
]

# Hechos iniciales (se infieren de reglas sin premisas)
hechos = set([conclusion for premisas, conclusion in reglas if len(premisas) == 0])


# ----------------------------
# Encadenamiento hacia atrás
# ----------------------------
def probar_objetivo(objetivo, visitados=None):
    if visitados is None:
        visitados = set()

    # Si ya lo tenemos como hecho, devolvemos True
    if objetivo in hechos:
        return True

    # Evitar bucles
    if objetivo in visitados:
        return False
    visitados.add(objetivo)

    # Buscar reglas cuya conclusión sea el objetivo
    for premisas, conclusion in reglas:
        if conclusion == objetivo:
            # Verificar si todas las premisas se pueden demostrar
            if all(probar_objetivo(p, visitados.copy()) for p in premisas):
                hechos.add(objetivo)  # lo agregamos a los hechos conocidos
                return True

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
            if probar_objetivo(h):
                print(f"La hipótesis '{h}' SE PUEDE demostrar.")
            else:
                print(f"La hipótesis '{h}' NO se puede demostrar.")
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