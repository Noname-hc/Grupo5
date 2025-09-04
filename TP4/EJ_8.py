class MotorInferenciaResolucion:
    """
    Motor de inferencia basado en el método de resolución (prueba por contradicción).
    
    La resolución funciona convirtiendo toda la base de conocimiento a Forma Normal
    Conjuntiva (FNC), agregando la negación de la hipótesis, y aplicando sistemáticamente
    la regla de resolución hasta encontrar la cláusula vacía (contradicción) o hasta
    que no se puedan generar más cláusulas nuevas.
    """
    
    def __init__(self):
        """Inicializar el motor con reglas y hechos predeterminados en FNC"""
        
        # Base de conocimiento en FNC - cada elemento es una cláusula (conjunto de literales)
        # Un literal positivo es una proposición, uno negativo tiene el prefijo '¬'
        
        # Conversión de las reglas originales a FNC:
        # Regla: A → B equivale a ¬A ∨ B en FNC
        
        self.clausulas = [
            # Regla 1: b ∧ c → a  ===  ¬b ∨ ¬c ∨ a
            {'¬b', '¬c', 'a'},
            
            # Regla 2: d ∧ e → b  ===  ¬d ∨ ¬e ∨ b  
            {'¬d', '¬e', 'b'},
            
            # Regla 3: g ∧ e → b  ===  ¬g ∨ ¬e ∨ b
            {'¬g', '¬e', 'b'},
            
            # Regla 4: e → c  ===  ¬e ∨ c
            {'¬e', 'c'},
            
            # Regla 5: d (hecho)  ===  d
            {'d'},
            
            # Regla 6: e (hecho)  ===  e  
            {'e'},
            
            # Regla 7: a ∧ g → f  ===  ¬a ∨ ¬g ∨ f
            {'¬a', '¬g', 'f'}
        ]
        
        # Contador para generar nombres únicos de cláusulas derivadas
        self.contador_clausulas = len(self.clausulas)
        
    def es_literal_complementario(self, lit1, lit2):
        """
        Verificar si dos literales son complementarios (uno es la negación del otro)
        
        Args:
            lit1, lit2 (str): Los literales a comparar
            
        Returns:
            bool: True si son complementarios, False en caso contrario
            
        Ejemplos:
            'a' y '¬a' son complementarios
            '¬b' y 'b' son complementarios  
            'a' y 'b' no son complementarios
        """
        # Si lit1 es negativo y lit2 es positivo
        if lit1.startswith('¬') and not lit2.startswith('¬'):
            return lit1[1:] == lit2  # Comparar sin el símbolo ¬
            
        # Si lit1 es positivo y lit2 es negativo  
        elif not lit1.startswith('¬') and lit2.startswith('¬'):
            return lit1 == lit2[1:]  # Comparar sin el símbolo ¬
            
        # Si ambos son del mismo tipo (ambos positivos o ambos negativos)
        else:
            return False
            
    def aplicar_resolucion(self, clausula1, clausula2):
        """
        Aplicar la regla de resolución entre dos cláusulas
        
        La regla de resolución dice que si tenemos:
        - Cláusula 1: {L, A₁, A₂, ..., Aₙ}  
        - Cláusula 2: {¬L, B₁, B₂, ..., Bₘ}
        
        Entonces podemos derivar:
        - Resolvente: {A₁, A₂, ..., Aₙ, B₁, B₂, ..., Bₘ}
        
        Args:
            clausula1, clausula2 (set): Las cláusulas a resolver
            
        Returns:
            set o None: La cláusula resolvente, o None si no se puede resolver
        """
        # Buscar todos los pares de literales complementarios
        pares_complementarios = []
        
        for lit1 in clausula1:
            for lit2 in clausula2:
                if self.es_literal_complementario(lit1, lit2):
                    pares_complementarios.append((lit1, lit2))
                    
        # Solo podemos resolver si hay exactamente UN par complementario
        # (la resolución requiere exactamente un literal complementario)
        if len(pares_complementarios) != 1:
            return None
            
        lit1_comp, lit2_comp = pares_complementarios[0]
        
        # Crear la cláusula resolvente eliminando los literales complementarios
        resolvente = (clausula1 - {lit1_comp}) | (clausula2 - {lit2_comp})
        
        return resolvente
        
    def mostrar_clausula(self, clausula):
        """
        Formatear una cláusula para mostrarla de manera legible
        
        Args:
            clausula (set): La cláusula a formatear
            
        Returns:
            str: Representación legible de la cláusula
        """
        if not clausula:
            return "□"  # Cláusula vacía
        elif len(clausula) == 1:
            return list(clausula)[0]
        else:
            return " ∨ ".join(sorted(clausula, key=lambda x: (x.startswith('¬'), x)))
            
    def mostrar_base_conocimiento(self):
        """Mostrar la base de conocimiento actual en FNC"""
        print("\n" + "="*60)
        print("BASE DE CONOCIMIENTO EN FORMA NORMAL CONJUNTIVA (FNC)")
        print("="*60)
        
        print(f"\n📋 CLÁUSULAS ({len(self.clausulas)} total):")
        for i, clausula in enumerate(self.clausulas):
            clausula_str = self.mostrar_clausula(clausula)
            print(f"  C{i+1}: {clausula_str}")
            
        print(f"\n💡 Nota: Cada cláusula representa una disyunción de literales.")
        print(f"    El símbolo '¬' representa negación.")
        print(f"    La conjunción de todas las cláusulas forma la base de conocimiento.")
        
    def agregar_clausula(self, clausula):
        """
        Agregar una nueva cláusula a la base de conocimiento
        
        Args:
            clausula (set o str): La cláusula a agregar
        """
        if isinstance(clausula, str):
            # Parsear string simple (ej: "a ∨ ¬b ∨ c")
            literales = set()
            partes = clausula.replace('∨', ',').split(',')
            for parte in partes:
                literal = parte.strip()
                if literal:
                    literales.add(literal)
            clausula = literales
            
        self.clausulas.append(clausula)
        clausula_str = self.mostrar_clausula(clausula)
        print(f"✓ Cláusula '{clausula_str}' agregada")
        
    def quitar_clausula(self, indice):
        """
        Quitar una cláusula por su índice
        
        Args:
            indice (int): Posición de la cláusula (empezando desde 0)
        """
        if 0 <= indice < len(self.clausulas):
            clausula_removida = self.clausulas.pop(indice)
            clausula_str = self.mostrar_clausula(clausula_removida)
            print(f"✗ Cláusula '{clausula_str}' removida")
        else:
            print(f"⚠ Índice {indice} fuera de rango. Hay {len(self.clausulas)} cláusulas.")
            
    def demostrar_por_resolucion(self, hipotesis):
        """
        Demostrar una hipótesis usando el método de resolución
        
        PROCEDIMIENTO PASO A PASO:
        1. La base de conocimiento ya está en FNC
        2. Agregar la negación de la hipótesis como cláusula adicional
        3. Aplicar resolución sistemáticamente sobre pares de cláusulas
        4. Repetir hasta encontrar cláusula vacía (contradicción) o no poder generar más
        
        Args:
            hipotesis (str): La proposición que queremos demostrar
            
        Returns:
            bool: True si la hipótesis se puede demostrar, False en caso contrario
        """
        print(f"\n🎯 DEMOSTRACIÓN POR RESOLUCIÓN: '{hipotesis}'")
        print("="*70)
        
        # PASO 1: Mostrar base de conocimiento inicial en FNC
        print("PASO 1: Base de conocimiento en FNC")
        print("-" * 40)
        for i, clausula in enumerate(self.clausulas):
            clausula_str = self.mostrar_clausula(clausula)
            print(f"  C{i+1}: {clausula_str}")
            
        # PASO 2: Agregar la negación de la hipótesis
        print(f"\nPASO 2: Agregando la negación de la hipótesis '¬{hipotesis}'")
        print("-" * 40)
        
        # Crear la cláusula de negación de la hipótesis
        negacion_hipotesis = {f'¬{hipotesis}'}
        
        # Crear conjunto de trabajo con todas las cláusulas
        clausulas_trabajo = self.clausulas.copy()
        clausulas_trabajo.append(negacion_hipotesis)
        
        print(f"  C{len(clausulas_trabajo)}: {self.mostrar_clausula(negacion_hipotesis)}")
        
        # PASO 3: Aplicar resolución sistemáticamente
        print(f"\nPASO 3: Aplicando resolución sistemáticamente")
        print("-" * 40)
        
        iteracion = 1
        nuevas_clausulas = True
        clausulas_creadas_total = 0  # Contador de cláusulas creadas
        
        # Conjunto para rastrear todas las cláusulas generadas
        clausulas_generadas = set()
        for c in clausulas_trabajo:
            clausulas_generadas.add(frozenset(c))
            
        while nuevas_clausulas:
            print(f"\n--- Iteración {iteracion} ---")
            nuevas_clausulas = False
            clausulas_iteracion = []
            
            # Intentar resolver cada par de cláusulas
            for i in range(len(clausulas_trabajo)):
                for j in range(i + 1, len(clausulas_trabajo)):
                    clausula1 = clausulas_trabajo[i]
                    clausula2 = clausulas_trabajo[j]
                    
                    # Aplicar resolución
                    resolvente = self.aplicar_resolucion(clausula1, clausula2)
                    
                    if resolvente is not None:
                        # Verificar si es una cláusula nueva
                        resolvente_frozen = frozenset(resolvente)
                        
                        if resolvente_frozen not in clausulas_generadas:
                            clausulas_generadas.add(resolvente_frozen)
                            clausulas_iteracion.append(resolvente)
                            clausulas_creadas_total += 1  # Incrementar contador
                            
                            # Mostrar el paso de resolución
                            c1_str = self.mostrar_clausula(clausula1)
                            c2_str = self.mostrar_clausula(clausula2)
                            res_str = self.mostrar_clausula(resolvente)
                            
                            print(f"  Resolviendo C{i+1} y C{j+1}:")
                            print(f"    C{i+1}: {c1_str}")  
                            print(f"    C{j+1}: {c2_str}")
                            print(f"    ────────────────────")
                            print(f"    Resultado: {res_str}")
                            
                            # VERIFICAR SI ENCONTRAMOS LA CLÁUSULA VACÍA
                            if len(resolvente) == 0:
                                print(f"\n✅ ¡CLÁUSULA VACÍA ENCONTRADA!")
                                print(f"   Esto significa que tenemos una CONTRADICCIÓN.")
                                print(f"   Por lo tanto, la hipótesis '{hipotesis}' es VERDADERA.")
                                print(f"\n📊 ESTADÍSTICAS DEL PROCESO:")
                                print(f"   • Cláusulas creadas durante el proceso: {clausulas_creadas_total}")
                                print(f"   • Iteraciones necesarias: {iteracion}")
                                return True
                                
                            nuevas_clausulas = True
                            
            # Agregar las nuevas cláusulas al conjunto de trabajo
            clausulas_trabajo.extend(clausulas_iteracion)
            
            if clausulas_iteracion:
                print(f"\n  Cláusulas nuevas generadas en esta iteración: {len(clausulas_iteracion)}")
                for clausula in clausulas_iteracion:
                    print(f"    • {self.mostrar_clausula(clausula)}")
            else:
                print(f"  No se generaron cláusulas nuevas en esta iteración.")
                
            iteracion += 1
            
            # Límite de seguridad para evitar loops infinitos
            if iteracion > 20:
                print(f"\n⚠ Límite de iteraciones alcanzado (20). Deteniendo proceso.")
                break
                
        # PASO 4: Conclusión
        print(f"\nPASO 4: Conclusión")  
        print("-" * 40)
        print(f"❌ NO SE ENCONTRÓ LA CLÁUSULA VACÍA")
        print(f"   No se pudo derivar una contradicción.")
        print(f"   Por lo tanto, la hipótesis '{hipotesis}' NO puede ser demostrada")
        print(f"   con la base de conocimiento actual.")
        print(f"\n📊 ESTADÍSTICAS DEL PROCESO:")
        print(f"   • Cláusulas creadas durante el proceso: {clausulas_creadas_total}")
        print(f"   • Iteraciones realizadas: {iteracion - 1}")
        
        return False
        
    def demostrar_multiple(self, hipotesis_lista):
        """
        Demostrar múltiples hipótesis y mostrar un resumen
        
        Args:
            hipotesis_lista (list): Lista de proposiciones a demostrar
        """
        print(f"\n🎯 DEMOSTRACIÓN MÚLTIPLES HIPÓTESIS POR RESOLUCIÓN")
        print("="*60)
        
        resultados = {}
        
        for hip in hipotesis_lista:
            resultado = self.demostrar_por_resolucion(hip)
            resultados[hip] = resultado
            print("\n" + "="*70)  # Separador entre demostraciones
            
        # Mostrar resumen final
        print(f"\n📊 RESUMEN DE RESULTADOS:")
        print("="*40)
        for hip, resultado in resultados.items():
            estado = "✅ DEMOSTRABLE" if resultado else "❌ NO DEMOSTRABLE"
            print(f"  '{hip}': {estado}")
            
        return resultados

# ===============================================================================
# FUNCIÓN PRINCIPAL Y MENÚ INTERACTIVO
# ===============================================================================

def menu_interactivo():
    """
    Función principal que proporciona un menú interactivo para usar el motor
    """
    # Crear instancia del motor de inferencia por resolución
    motor = MotorInferenciaResolucion()
    
    print("🧠 MOTOR DE INFERENCIA POR RESOLUCIÓN")
    print("="*60)
    print("Este sistema demuestra proposiciones usando el método de resolución")
    print("trabajando con cláusulas en Forma Normal Conjuntiva (FNC).")
    print("Cargado con las reglas y hechos del enunciado convertidos a FNC.")
    
    while True:
        print(f"\n{'='*60}")
        print("MENÚ PRINCIPAL")
        print("="*60)
        print("1. Mostrar base de conocimiento (FNC)")
        print("2. Demostrar una hipótesis")
        print("3. Demostrar múltiples hipótesis")
        print("4. Agregar cláusula")
        print("5. Quitar cláusula") 
        print("6. Salir")
        
        try:
            opcion = input("\nSelecciona una opción (1-6): ").strip()
            
            if opcion == '1':
                motor.mostrar_base_conocimiento()
                
            elif opcion == '2':
                hip = input("Ingresa la hipótesis a demostrar: ").strip()
                if hip:
                    motor.demostrar_por_resolucion(hip)
                else:
                    print("⚠ Debes ingresar una hipótesis válida")
                    
            elif opcion == '3':
                print("Ingresa hipótesis separadas por comas (ej: a,f,x):")
                entrada = input().strip()
                if entrada:
                    hipotesis_lista = [h.strip() for h in entrada.split(',')]
                    motor.demostrar_multiple(hipotesis_lista)
                else:
                    print("⚠ Debes ingresar al menos una hipótesis")
                    
            elif opcion == '4':
                print("Ingresa la nueva cláusula.")
                print("Formato: literales separados por ∨ (ej: a ∨ ¬b ∨ c)")
                print("O ingresa literales separados por comas (ej: a,¬b,c)")
                clausula_str = input("Cláusula: ").strip()
                
                if clausula_str:
                    # Procesar la entrada
                    if '∨' in clausula_str:
                        literales = [lit.strip() for lit in clausula_str.split('∨')]
                    else:
                        literales = [lit.strip() for lit in clausula_str.split(',')]
                        
                    clausula = set(literales)
                    motor.agregar_clausula(clausula)
                else:
                    print("⚠ Debes ingresar una cláusula válida")
                    
            elif opcion == '5':
                motor.mostrar_base_conocimiento()
                try:
                    indice = int(input("Ingresa el índice de la cláusula a quitar (empezando desde 1): ")) - 1
                    motor.quitar_clausula(indice)
                except ValueError:
                    print("⚠ Debes ingresar un número válido")
                    
            elif opcion == '6':
                print("¡Hasta luego! 👋")
                break
                
            else:
                print("⚠ Opción no válida. Selecciona un número del 1 al 6.")
                
        except KeyboardInterrupt:
            print("\n\n¡Hasta luego! 👋")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

# ===============================================================================
# EJERCICIO 8 - DEMOSTRACIÓN AUTOMÁTICA
# ===============================================================================

def ejercicio_8():
    """
    Ejecutar el Ejercicio 8: demostración automática de la hipótesis 'a'
    """
    print("🚀 EJERCICIO 8 - DEMOSTRACIÓN POR RESOLUCIÓN")
    print("="*60)
    
    # Crear motor y mostrar configuración inicial
    motor = MotorInferenciaResolucion()
    motor.mostrar_base_conocimiento()
    
    # Demostrar la hipótesis 'a' como se solicita
    print(f"\n🧪 DEMOSTRANDO LA HIPÓTESIS PREDETERMINADA: 'a'")
    resultado = motor.demostrar_por_resolucion('a')
    
    print(f"\n📋 RESULTADO FINAL DEL EJERCICIO 8:")
    if resultado:
        print(f"✅ La hipótesis 'a' ha sido DEMOSTRADA usando resolución.")
    else:
        print(f"❌ La hipótesis 'a' NO pudo ser demostrada.")
        
    return motor, resultado

# ===============================================================================
# PUNTO DE ENTRADA
# ===============================================================================

if __name__ == "__main__":
    print("Selecciona el modo de ejecución:")
    print("1. Ejercicio 8 (demostrar 'a')")
    print("2. Menú interactivo")
    
    try:
        modo = input("Ingresa 1 o 2: ").strip()
        
        if modo == '1':
            ejercicio_8()
        elif modo == '2':
            menu_interactivo()
        else:
            print("Ejecutando Ejercicio 8 por defecto...")
            ejercicio_8()
            
    except KeyboardInterrupt:
        print("\n¡Programa terminado!")
    except Exception as e:
        print(f"Error: {e}")
        print("Ejecutando Ejercicio 8...")
        ejercicio_8()