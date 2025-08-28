import tkinter as tk
from tkinter import ttk
import math
import random
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Definición de la función objetivo a maximizar
def f(x):
    return math.sin(x) / (x + 0.1)

# Algoritmo de Hill Climbing (ascenso de colinas)
def HILL_CLIMBING(funcion, comienzo, paso = 0.01, error = 0.0001):
    x = comienzo                  # Punto inicial
    y = funcion(x)                # Valor de la función en el punto inicial
    i = 0                         # Contador de iteraciones
    Err = float('inf')            # Diferencia entre valores sucesivos (criterio de parada)

    # Bucle principal: continúa mientras la mejora sea mayor que el error y no se exceda el máximo de iteraciones
    while (Err > error and i < 10000):
        i += 1
        y_old = y                 # Guarda el valor anterior para calcular la mejora

        # Calcula los vecinos a la izquierda y derecha
        x_sig = x + paso
        y_sig = funcion(x_sig)

        x_ant = x - paso
        y_ant = funcion(x_ant)

        # Decide hacia dónde moverse: al vecino que tenga mayor valor de función
        if y_sig > y and y_sig >= y_ant:
            x, y = x_sig, y_sig
            Err = abs(y - y_old)  # Actualiza la mejora
        elif y_ant > y and y_ant > y_sig:
            x, y = x_ant, y_ant
            Err = abs(y - y_old)  # Actualiza la mejora
        else:
            # Si ningún vecino mejora, termina el bucle (máximo local encontrado)
            break

    return x, y                   # Devuelve el punto del máximo local y su valor

# Clase principal de la interfaz gráfica
class HillClimbGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Hill Climbing - Visualización')
        self.geometry('900x600')
        self.create_widgets()     # Crea los controles de la interfaz
        self.result_frame = ttk.Frame(self)
        self.result_frame.pack(pady=10)
        self.draw_function()      # Dibuja la función al iniciar la app

    # Crea los widgets de control (entrada de datos y botón)
    def create_widgets(self):
        control_frame = ttk.Frame(self)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)

        # Título de la sección de parámetros
        ttk.Label(control_frame, text='Parámetros', font=('Arial', 12, 'bold')).pack(pady=4)

        # Campo para ingresar el punto inicial
        self.start_var = tk.DoubleVar(value=random.uniform(-10, 10))
        ttk.Label(control_frame, text='Punto inicial (x)').pack(anchor='w')
        self.start_entry = ttk.Entry(control_frame, textvariable=self.start_var)
        self.start_entry.pack(fill=tk.X)

        # Botón para calcular el máximo
        self.calc_button = ttk.Button(control_frame, text="Calcular máximo", command=self.calcular_maximo)
        self.calc_button.pack(pady=10)

    # Dibuja la función y, si se pasa, el máximo encontrado
    def draw_function(self, max_x=None, max_y=None):
        # Si ya hay un gráfico, lo elimina antes de dibujar uno nuevo
        if hasattr(self, 'canvas'):
            self.canvas.get_tk_widget().destroy()

        # Crea la figura de matplotlib
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        x_vals = np.linspace(-10, 10, 400)
        y_vals = [f(x) for x in x_vals]
        ax.plot(x_vals, y_vals, label='f(x)')
        ax.set_title('Función f(x)')
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.legend()

        # Si se encontró un máximo, lo marca con una X roja
        if max_x is not None and max_y is not None:
            ax.plot(max_x, max_y, 'rx', markersize=12, label='Máximo')
            ax.legend()

        # Inserta el gráfico en la ventana de Tkinter
        self.canvas = FigureCanvasTkAgg(fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)

    # Ejecuta el algoritmo y muestra el resultado en la interfaz
    def calcular_maximo(self):
        comienzo = self.start_var.get()  # Obtiene el valor ingresado por el usuario
        X, Y = HILL_CLIMBING(f, comienzo)
        resultado = f"El máximo está en x = {X:.2f}, f(x) = {Y:.4f}"
        # Muestra el resultado como un nuevo label debajo del anterior
        ttk.Label(self.result_frame, text=resultado, font=('Arial', 12)).pack(anchor='w')
        # Redibuja la función mostrando el máximo encontrado
        self.draw_function(max_x=X, max_y=Y)

# Punto de entrada principal: inicia la aplicación gráfica
if __name__ == "__main__":
    app = HillClimbGUI()
    app.mainloop()