# Clase que representa una regla de la base de conocimiento
class Regla:
    def __init__(self, _condiciones, _conclusion):
        self.condiciones = _condiciones  # guarda la lista de condiciones (hechos necesarios)
        self.conclusion = _conclusion    # guarda el hecho que se deduce si se cumplen las condiciones


# Clase que representa el motor de inferencia
class MotorInferencia:
    def __init__(self, _reglas, _hechos_iniciales):
        self.reglas = _reglas                  # lista de objetos Regla
        self.hechos = list(_hechos_iniciales)  # copia los hechos iniciales en una lista interna
        self.sumador = 0

    def encadenamiento_adelante(self):
        nuevos = True              # bandera para controlar si en una pasada se añadieron hechos nuevos
        while nuevos:              # repetir mientras se estén añadiendo hechos nuevos
            self.sumador += 1
            nuevos = False        # asumimos que no habrá nuevos hechos hasta demostrar lo contrario

            # mostrar los hechos actuales por pantalla (útil para depurar / entender el proceso)
            print("\nHechos actuales:", self.hechos)

            # recorremos todas las reglas definidas en la base de conocimiento
            for regla in self.reglas:
                self.sumador += 1
                # 'all(...)' comprueba que TODAS las condiciones de la regla estén en los hechos
                # No se necesita un bucle adicional aquí, se puede eliminar el for i,cond...
                V_bool = []
                for cond in regla.condiciones: 
                    self.sumador += 1
                    V_bool.append(cond in self.hechos)
                    
                if(all(V_bool)):
                    # si la conclusión todavía no está entre los hechos, la añadimos
                    if regla.conclusion not in self.hechos:
                        self.hechos.append(regla.conclusion)                      # añade nuevo hecho
                        print(f"Regla aplicada: {regla.condiciones} -> {regla.conclusion}")
                        nuevos = True     # marcamos que sí hubo un cambio, para repetir el bucle

    def mostrar_hechos(self):
        print("\n=== Hechos finales ===")
        for hecho in self.hechos:
            print(" -", hecho)
        print(f"Iteraciones finales: {self.sumador}")


# --------------------
# Ejemplo de uso
# --------------------
if __name__ == "__main__":
    # Definir la base de conocimiento (reglas)
    r1 = Regla(["b", "c"], "a")
    r2 = Regla(["d", "e"], "b")
    r3 = Regla(["g", "e"], "b")
    r4 = Regla(["e"], "c")
    r5 = Regla(["a", "g"], "f")

    reglas = [r1, r2, r3, r4]

    # Hechos iniciales
    hechos_iniciales = ["d", "e"]

    # Crear motor y ejecutar
    motor = MotorInferencia(reglas, hechos_iniciales)
    motor.encadenamiento_adelante()
    motor.mostrar_hechos()
