"""
A* Pathfinding con Pygame — versión MUY comentada
--------------------------------------------------
Objetivo: reescribir el programa con comentarios exhaustivos y explicar el uso de
- **clases** (en particular `Spot`),
- **funciones/definiciones** (`def`),
- y la lógica del algoritmo A*.

Notas de lectura rápida:
- Este archivo es **ejecutable** tal cual si tenés `pygame` instalado.
- Interacción:
    • Clic **izquierdo**: marcar `start`, `end` y obstáculos.
    • Clic **derecho**: borrar una celda.
    • **Barra espaciadora**: corre A*.
    • **C**: limpia el tablero.
- La heurística usada es **Manhattan**, acorde a un grid con 4 vecinos (sin diagonales).

Sobre clases en este programa
----------------------------
Se define una clase `Spot` que representa **cada celda** del grid. ¿Por qué usar una clase?
- **Encapsulación de estado**: en cada celda guardamos fila/columna, color, vecinos, etc.
- **Comportamiento asociado**: métodos como `make_barrier()`, `is_end()`, `draw()`, etc.,
  viven dentro de `Spot`, manteniendo ordenado el código y evitando estructuras paralelas.
- **Beneficio práctico**: A* trabaja con "nodos". Aquí, cada `Spot` es un nodo del grafo.

Sintaxis breve de clases en Python:
- `class Spot:` declara la clase.
- `def __init__(self, ...)`: **constructor**, se llama al crear la instancia: `Spot(...)`.
- `self` es la referencia al **objeto actual** (similar a `this` en otros lenguajes).
- Métodos como `def is_barrier(self):` son funciones ligadas a la clase (reciben `self`).
- Método especial `__lt__` define "menor que"; se implementa para que ciertas estructuras
  como `PriorityQueue` no fallen al comparar objetos (aunque acá devolvemos `False` y
  resolvemos los empates con un contador `count`).

Sobre funciones (definiciones)
------------------------------
- `def h(p1, p2)`: **heurística** (estimación restante). Devuelve distancia Manhattan.
- `def reconstruct_path(...)`: remarca visualmente el camino reconstruyendo con `came_from`.
- `def algorithm(...)`: implementación de **A***.
- `def make_grid(...)`, `def draw_grid(...)`, `def draw(...)`, `def get_clicked_pos(...)` y
  `def main(...)`: utilitarias para construir, dibujar y manejar la interacción.

"""

# =========================
# Imports y configuración
# =========================
import pygame  # Librería para gráficos 2D y manejo de eventos.
from queue import PriorityQueue  # Cola de prioridad (mínimo primero) para A*.

# NUEVO IMPORT: tkinter.messagebox para mostrar ventanas de diálogo cuando no hay solución
import tkinter.messagebox as messagebox
import tkinter as tk

# (Opcional) Tipado estático ligero; no es obligatorio para correr.
from typing import Dict, List, Optional, Tuple

# -------------------------
# Tamaño de la ventana.
# `WIDTH` define tanto el ancho como el alto (ventana cuadrada).
# -------------------------
WIDTH: int = 600
WIN = pygame.display.set_mode((WIDTH, WIDTH))  # Superficie principal donde dibujamos.
pygame.display.set_caption("A* Path Finding Algorithm")  # Título de la ventana.

# -------------------------
# Colores en formato RGB.
# Nota: Corregimos BLUE a (0, 0, 255). En el código original estaba igual que GREEN.
# -------------------------
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
GREY = (128, 128, 128)
TURQUOISE = (64, 224, 208)


# =========================
# Clase "Spot" (celda del grid)
# =========================
class Spot:
    """Representa una celda del grid (un nodo en el grafo implícito).

    A nivel de A*:
    - Cada `Spot` puede ser transitado o no (pared/obstáculo).
    - Conoce a sus vecinos (arriba, abajo, izquierda, derecha) que no sean paredes.
    - Tiene métodos para pintarse con diferentes colores según su estado en la búsqueda.
    """

    def __init__(self, row: int, col: int, width: int, total_rows: int) -> None:
        # Coordenadas discretas en el grid
        self.row = row
        self.col = col

        # Posición en píxeles (superficie Pygame). OJO: se usa un mapeo
        # consistente con `get_clicked_pos`. Aunque parezca "invertido",
        # el programa mantiene la correspondencia clic ↔ celda.
        self.x = row * width
        self.y = col * width

        # Color actual de la celda (estado visual/semántico)
        self.color = WHITE

        # Lista de vecinos accesibles (se calcula con `update_neighbors`)
        self.neighbors: List["Spot"] = []

        # Ancho de la celda en píxeles y cantidad total de filas del grid
        self.width = width
        self.total_rows = total_rows

    # ---------------
    # Consultas de estado (predicados)
    # ---------------
    def get_pos(self) -> Tuple[int, int]:
        return self.row, self.col

    def is_closed(self) -> bool:
        # Celda ya evaluada y "cerrada" por A*
        return self.color == RED

    def is_open(self) -> bool:
        # Celda descubierta pero aún por evaluar completamente
        return self.color == GREEN

    def is_barrier(self) -> bool:
        # Obstáculo (no transitable)
        return self.color == BLACK

    def is_start(self) -> bool:
        return self.color == ORANGE

    def is_end(self) -> bool:
        return self.color == TURQUOISE

    # ---------------
    # Mutadores de estado (cambian el color/rol de la celda)
    # ---------------
    def reset(self) -> None:
        self.color = WHITE

    def make_start(self) -> None:
        self.color = ORANGE

    def make_closed(self) -> None:
        self.color = RED

    def make_open(self) -> None:
        self.color = GREEN

    def make_barrier(self) -> None:
        self.color = BLACK

    def make_end(self) -> None:
        self.color = TURQUOISE

    def make_path(self) -> None:
        # Parte del camino final reconstruido
        self.color = PURPLE

    # ---------------
    # Render de la celda
    # ---------------
    def draw(self, win: pygame.Surface) -> None:
        # Dibuja un rectángulo lleno en su posición y color actual
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

    # ---------------
    # Vecindad (4-direcciones)
    # ---------------
    def update_neighbors(self, grid: List[List["Spot"]]) -> None:
        """Calcula y almacena los vecinos transitables.
        Solo se contemplan movimientos ortogonales (sin diagonales).
        """
        self.neighbors = []  # Siempre recomputamos desde cero

        # DOWN: fila + 1, misma columna (si no salgo del grid y no es pared)
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier():
            self.neighbors.append(grid[self.row + 1][self.col])

        # UP
        if self.row > 0 and not grid[self.row - 1][self.col].is_barrier():
            self.neighbors.append(grid[self.row - 1][self.col])

        # RIGHT
        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier():
            self.neighbors.append(grid[self.row][self.col + 1])

        # LEFT
        if self.col > 0 and not grid[self.row][self.col - 1].is_barrier():
            self.neighbors.append(grid[self.row][self.col - 1])

    # ---------------
    # Soporte para PriorityQueue
    # ---------------
    def __lt__(self, other: "Spot") -> bool:
        # No se usa realmente la comparación entre Spots; devolvemos False para
        # evitar errores cuando la cola trata de desempatar directamente objetos.
        return False


# =========================
# Heurística: distancia Manhattan
# =========================

def h(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    """Devuelve |x1 - x2| + |y1 - y2|.

    Al usar solo movimientos en cruz (arriba/abajo/izq/der), la Manhattan es:
    - Admisible (no sobrestima) y
    - Consistente, por lo que A* encuentra el camino óptimo.
    """
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)


# =========================
# Reconstrucción del camino final
# =========================

def reconstruct_path(came_from: Dict[Spot, Spot], current: Spot, draw_cb) -> None:
    """Pinta de morado el camino desde `current` hacia el inicio usando `came_from`.

    `came_from` es un mapa: hijo -> padre. Se recorre hacia atrás hasta el inicio.
    `draw_cb` es un callback para refrescar la pantalla en cada paso.
    """
    while current in came_from:
        current = came_from[current]  # Retrocedo un paso hacia el origen
        current.make_path()           # Pinto la celda como parte del camino
        draw_cb()                     # Refresco la vista (animación)


# =========================
# A* (A-star)
# =========================

def algorithm(draw_cb, grid: List[List[Spot]], start: Spot, end: Spot) -> bool:
    """Implementación de A*.

    Estructuras clave:
    - `open_set` (PriorityQueue): nodos frontera, ordenados por f(n) = g(n) + h(n).
    - `came_from`: para reconstruir el camino cuando se alcanza `end`.
    - `g_score`: costo real acumulado desde `start`.
    - `f_score`: prioridad = g + h; determina el orden de expansión.

    Devuelve True si se encontró un camino; False si no.
    """
    count = 0  # Contador creciente para desempatar (estabilidad en la cola de prioridad)

    open_set: PriorityQueue[Tuple[float, int, Spot]] = PriorityQueue()
    open_set.put((0, count, start))  # Insertamos el inicio con prioridad 0

    came_from: Dict[Spot, Spot] = {}  # Mapa para reconstrucción del camino

    # Inicializamos g y f con infinito para todas las celdas
    g_score: Dict[Spot, float] = {spot: float("inf") for row in grid for spot in row}
    g_score[start] = 0.0

    f_score: Dict[Spot, float] = {spot: float("inf") for row in grid for spot in row}
    f_score[start] = h(start.get_pos(), end.get_pos())

    # `open_set_hash` nos permite consulta O(1) para saber si un nodo ya está en open_set
    open_set_hash = {start}

    while not open_set.empty():
        # Procesamos eventos para permitir cerrar la app mientras corre el algoritmo
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        # Tomamos el nodo con menor f-score (desempate por `count` y luego por objeto)
        current: Spot = open_set.get()[2]
        open_set_hash.remove(current)

        # Caso objetivo: llegamos a la meta
        if current == end:
            reconstruct_path(came_from, end, draw_cb)
            end.make_end()  # Re-pintamos el final por si quedó morado
            return True

        # Relajación de aristas hacia cada vecino
        for neighbor in current.neighbors:
            tentative_g = g_score[current] + 1  # Costo uniforme = 1 por paso

            # Si encontramos un mejor camino hacia `neighbor`, actualizamos
            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + h(neighbor.get_pos(), end.get_pos())

                # Si el vecino aún no estaba en la frontera, lo insertamos
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()  # Visual: celda descubierta

        # Redibujamos el frame tras procesar `current`
        draw_cb()

        # Marcamos `current` como cerrado (procesado), salvo que sea el inicio
        if current != start:
            current.make_closed()

    # Si salimos del while sin retornar, no hay camino
    return False


# =========================
# Utilidades de construcción y render del grid
# =========================

def make_grid(rows: int, width: int) -> List[List[Spot]]:
    """Crea un grid `rows x rows` de `Spot`.

    `gap` es el tamaño de cada celda en píxeles (ancho de la ventana dividido en filas).
    """
    grid: List[List[Spot]] = []
    gap = width // rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            spot = Spot(i, j, gap, rows)
            grid[i].append(spot)
    return grid


def draw_grid(win: pygame.Surface, rows: int, width: int) -> None:
    """Dibuja la cuadrícula (líneas grises) sobre la ventana."""
    gap = width // rows
    for i in range(rows):
        # Líneas horizontales (y = i * gap)
        pygame.draw.line(win, GREY, (0, i * gap), (width, i * gap))
        for j in range(rows):
            # Líneas verticales (x = j * gap)
            pygame.draw.line(win, GREY, (j * gap, 0), (j * gap, width))


def draw(win: pygame.Surface, grid: List[List[Spot]], rows: int, width: int) -> None:
    """Limpia la pantalla y dibuja todas las celdas + la grilla."""
    win.fill(WHITE)  # Fondo blanco
    for row in grid:
        for spot in row:
            spot.draw(win)
    draw_grid(win, rows, width)
    pygame.display.update()  # Refresco de la ventana


def get_clicked_pos(pos: Tuple[int, int], rows: int, width: int) -> Tuple[int, int]:
    """Convierte coordenadas de píxel (x, y) del mouse a coordenadas de celda (row, col)."""
    gap = width // rows
    y, x = pos  # nota: se invierte para mantener consistencia con el mapeo usado en `Spot`
    row = y // gap
    col = x // gap
    return row, col


# =========================
# Bucle principal (eventos e interacción)
# =========================

def main(win: pygame.Surface, width: int) -> None:
    # Cantidad de filas/columnas del grid (grid cuadrado)
    ROWS = 33 
    grid = make_grid(ROWS, width)

    start: Optional[Spot] = None  # Celda de inicio (se elige con clic)
    end: Optional[Spot] = None    # Celda de destino

    run = True
    while run:
        # Redibujo constante (permite ver los cambios al interactuar)
        draw(win, grid, ROWS, width)

        # Manejo de eventos (teclado, mouse, cerrar ventana, etc.)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False  # Sale del bucle y cierra la app

            # ---------------
            # Clic izquierdo: setear inicio, fin, o pared
            # ---------------
            if pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                spot = grid[row][col]

                if not start and spot != end:
                    start = spot
                    start.make_start()
                elif not end and spot != start:
                    end = spot
                    end.make_end()
                elif spot != end and spot != start:
                    spot.make_barrier()

            # ---------------
            # Clic derecho: limpiar celda
            # ---------------
            elif pygame.mouse.get_pressed()[2]:
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                spot = grid[row][col]
                spot.reset()
                if spot == start:
                    start = None
                elif spot == end:
                    end = None

            # ---------------
            # Teclas
            # ---------------
            if event.type == pygame.KEYDOWN:
                # Barra espaciadora: ejecutar A* si hay inicio y fin
                if event.key == pygame.K_SPACE and start and end:
                    # Antes de correr A*, cada celda debe saber quiénes son sus vecinos
                    for row in grid:
                        for spot in row:
                            spot.update_neighbors(grid)

                    # MODIFICACIÓN: Ahora capturamos el resultado del algoritmo A*
                    # `algorithm` devuelve True si encontró un camino, False si no
                    found_path = algorithm(lambda: draw(win, grid, ROWS, width), grid, start, end)
                    
                    # NUEVA FUNCIONALIDAD: Mostrar mensaje si no se encontró solución
                    if not found_path:
                        # Creamos una ventana invisible de tkinter para mostrar el messagebox
                        # Esto es necesario porque tkinter necesita una ventana raíz
                        root = tk.Tk()
                        root.withdraw()  # Ocultamos la ventana principal de tkinter
                        
                        # Mostramos el mensaje de error usando messagebox.showerror
                        # Primer parámetro: título de la ventana de diálogo
                        # Segundo parámetro: mensaje que se mostrará al usuario
                        messagebox.showerror("Sin solución", "No se encontró una ruta desde el inicio hasta el destino.\n\nVerifica que no haya barreras bloqueando completamente el camino.")
                        
                        # Destruimos la ventana raíz de tkinter ya que no la necesitamos más
                        # Esto libera los recursos y evita problemas de memoria
                        root.destroy()

                # Tecla C: limpiar todo y reiniciar grid
                if event.key == pygame.K_c:
                    start = None
                    end = None
                    grid = make_grid(ROWS, width)

    pygame.quit()


# Punto de entrada: ejecuta el bucle principal si corrés este archivo directamente
main(WIN, WIDTH)