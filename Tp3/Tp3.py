import math
import random

def f(x):
    return math.sin(x) / (x + 0.1)

def HILL_CLIMBING(funcion, comienzo, paso = 0.05, error = 0.1):
    x = comienzo
    y = funcion(x)

    Continuar = True
    while (Continuar == True):
        x_sig= x + paso
        y_sig= funcion(x_sig)

        x_ant = x - paso
        y_ant = funcion(x_ant)

        if (y_sig > y):
            x = x_sig

            Err = abs(y_sig - y)
            break

        if (y_ant > y):
            x = x_ant

            Err = abs(y_ant - y)
            break

        if (Err <= error):
            Continuar = False
            break

    return x, y

if __name__ == "__main__":
    # Elegimos un punto inicial aleatorio en el intervalo
    start = random.uniform(-10, -6)
    max_x, max_f = HILL_CLIMBING(f, start)

    print(f"Comenzamos en x = {start:.2f}")
    print(f"Máximo aproximado en x = {max_x:.2f}, f(x) = {max_f:.2f}")