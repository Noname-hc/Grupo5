import tkinter as tk
from tkinter import ttk
import math
import random
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

def f(x):
    return math.sin(x) / (x + 0.1)

def HILL_CLIMBING(funcion, comienzo, paso = 0.01, error = 0.0001):
    x = comienzo
    y = funcion(x)
    i = 0
    Err = float('inf')

    while  (Err > error and i < 10000):
        i += 1
        y_old = y

        # vecinos
        x_sig = x + paso
        y_sig = funcion(x_sig)

        x_ant = x - paso
        y_ant = funcion(x_ant)

        # decidir movimiento
        if y_sig > y and y_sig >= y_ant:
            x, y = x_sig, y_sig
            Err = abs(y - y_old)

        elif y_ant > y and y_ant > y_sig:
            x, y = x_ant, y_ant
            Err = abs(y - y_old)
        else:
            # No hay mejora, el error es 0 (o la diferencia mínima)
            break

    return x, y

class HillClimbGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Hill Climbing - Visualización')
        self.geometry('900x600')
        self.create_widgets()
        self.result_frame = ttk.Frame(self)
        self.result_frame.pack(pady=10)
        self.draw_function()  # Dibuja la función al inicio

    def create_widgets(self):
        control_frame = ttk.Frame(self)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)

        ttk.Label(control_frame, text='Parámetros', font=('Arial', 12, 'bold')).pack(pady=4)

        self.start_var = tk.DoubleVar(value=random.uniform(-10, 10))
        ttk.Label(control_frame, text='Punto inicial (x)').pack(anchor='w')
        self.start_entry = ttk.Entry(control_frame, textvariable=self.start_var)
        self.start_entry.pack(fill=tk.X)

        self.calc_button = ttk.Button(control_frame, text="Calcular máximo", command=self.calcular_maximo)
        self.calc_button.pack(pady=10)

    def draw_function(self, max_x=None, max_y=None):
        # Si ya hay un canvas, lo destruimos para evitar superposiciones
        if hasattr(self, 'canvas'):
            self.canvas.get_tk_widget().destroy()

        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        x_vals = np.linspace(-10, 10, 400)
        y_vals = [f(x) for x in x_vals]
        ax.plot(x_vals, y_vals, label='f(x)')
        ax.set_title('Función f(x)')
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.legend()

        # Si se pasa el máximo, lo dibuja como una X roja
        if max_x is not None and max_y is not None:
            ax.plot(max_x, max_y, 'rx', markersize=12, label='Máximo')
            ax.legend()

        self.canvas = FigureCanvasTkAgg(fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)

    def calcular_maximo(self):
        comienzo = self.start_var.get()
        X, Y = HILL_CLIMBING(f, comienzo)
        resultado = f"El máximo está en x = {X:.2f}, f(x) = {Y:.4f}"
        # Mostrar el resultado como un nuevo label debajo del anterior
        ttk.Label(self.result_frame, text=resultado, font=('Arial', 12)).pack(anchor='w')
        # Redibujar la función mostrando el máximo
        self.draw_function(max_x=X, max_y=Y)

if __name__ == "__main__":
    app = HillClimbGUI()
    app.mainloop()

    #comienzo = random.uniform(-10, 10)
    #X, Y = HILL_CLIMBING(f, comienzo, 0.05, 0.1)
    #print(comienzo)
    #print(f"El máximo está en x = {X:.2f}, f(x) = {Y:.4f}")

