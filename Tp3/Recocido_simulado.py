"""
Ta-Te-Ti con Recocido Simulado (Simulated Annealing)

Estructura general del programa:
- Lógica del juego: representación del tablero, comprobación de ganador y empate.
- Funciones de evaluación: heurísticas que valoran estados intermedios (dos en línea, uno en línea).
- Algoritmo de recocido simulado: temperatura controlada por el nivel de dificultad.
- Interfaz con Pygame: tablero grande, animaciones de colocación, selector de dificultad,
  botón de reinicio y panel de estado.
"""

import pygame
import sys
import random
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum

# =========================================================
#                        CONFIGURACION
# =========================================================
# Tamaños de ventana y tablero: aumentados para evitar solapamiento
WIDTH = 600                 # anchura de la ventana
BOARD_SIZE = 600            # tablero cuadrado de 600x600 (3x3 celdas grandes)
PANEL_H = 200               # panel inferior para controles y textos
HEIGHT = BOARD_SIZE + PANEL_H

# Colores y estilos (RGB)
BG_COLOR = (245, 245, 245)
GRID_COLOR = (80, 80, 80)
X_COLOR = (200, 30, 30)
O_COLOR = (40, 70, 210)
WIN_LINE_COLOR = (20, 160, 80)
TEXT_COLOR = (30, 30, 30)
HOVER_COLOR = (230, 230, 230)
LINE_W = 8

# Colores para selector de dificultad
EASY_COLOR = (90, 200, 90)      # Verde para Fácil
MEDIUM_COLOR = (255, 165, 0)    # Naranja para Medio  
HARD_COLOR = (200, 50, 50)      # Rojo para Difícil
SELECTED_COLOR = (255, 255, 255)
DISABLED_COLOR = (180, 180, 180)

# Calculados 
CELL = BOARD_SIZE // 3   # tamaño de cada celda
FPS = 60
ANIM_PLACEMENT_MS = 200  # duración (ms) de la animación de aparición X/O
AI_DELAY_MS = 500        # delay entre jugada del jugador y la respuesta de la IA (ms)

# Pygame init -> inicia la libreria de la interfaz grafica
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))   # modelado de la ventana  
pygame.display.set_caption("Ta-Te-Ti • Recocido Simulado")
clock = pygame.time.Clock()

# Fuentes: ajustar tamaños para que encajen en el panel sin solaparse
font_large = pygame.font.SysFont(None, 96)
font_mid = pygame.font.SysFont(None, 36)
font_small = pygame.font.SysFont(None, 24)
font_tiny = pygame.font.SysFont(None, 20)

# =========================================================
#                    DIFICULTADES
# =========================================================

class Difficulty(Enum):
    """Niveles de dificultad con sus respectivos valores de temperatura inicial"""
    EASY = ("Fácil", 4.5, EASY_COLOR)      # T0 alto = más exploración = IA más débil
    MEDIUM = ("Medio", 2.0, MEDIUM_COLOR)   # T0 intermedio = equilibrio
    HARD = ("Difícil", 0.3, HARD_COLOR)    # T0 bajo = más inteligente = IA más fuerte
    
    def __init__(self, label: str, temperature: float, color: tuple): # Constructor 
        # Asignamos los atributos de los objetos
        self.label = label
        self.temperature = temperature
        self.color = color

# =========================================================
#                    LÓGICA DEL JUEGO
# =========================================================
# Definición de las líneas ganadoras como trios de índices
WIN_LINES = [
    (0,1,2), (3,4,5), (6,7,8),
    (0,3,6), (1,4,7), (2,5,8),
    (0,4,8), (2,4,6)
]

def check_winner(board: List[str]) -> Optional[Tuple[str, Tuple[int,int,int]]]:
    """
    Comprueba si hay un ganador en el tablero.
    Devuelve una tupla ("X" o "O", trio_de_indices) si existe ganador, o None si no.
    """
    for trio in WIN_LINES:
        a, b, c = trio
        if board[a] != '' and board[a] == board[b] == board[c]:
            return board[a], trio
    return None

def is_draw(board: List[str]) -> bool:
    """Devuelve True si no hay espacios vacíos (empate si no hay ganador)."""
    return all(cell != '' for cell in board)

def evaluate_board_detailed(board: List[str]) -> int:
    """
    Función heurística mejorada con valores escalados apropiadamente
    para trabajar mejor con diferentes temperaturas.
    """
    win = check_winner(board)
    if win:
        return 1000 if win[0] == 'O' else -1000

    # Inicializamos el puntaje en 0, valores positivos favorecen a 'O' y negativos favorecen a 'X'
    score = 0
    
    # Evaluar cada línea ganadora
    for a, b, c in WIN_LINES:
        line = [board[a], board[b], board[c]]
        # Contamos cuantas celdas están vacías, cuantas tienen 'O' y cuantas tienen 'X' en una misma línea
        # En función de esto planteamos los condicionales
        empty_count = line.count('')
        o_count = line.count('O')
        x_count = line.count('X')
        
        # Dos de O y un hueco (oportunidad de victoria)
        if o_count == 2 and empty_count == 1:
            score += 60
        # Dos de X y un hueco (amenaza crítica)
        elif x_count == 2 and empty_count == 1:
            score -= 50
        # Una de O y dos huecos (desarrollo)
        elif o_count == 1 and empty_count == 2:
            score += 8
        # Una de X y dos huecos
        elif x_count == 1 and empty_count == 2:
            score -= 5
    
    # Bonus por posiciones estratégicas
    # Centro vale más
    if board[4] == 'O':
        score += 15
    elif board[4] == 'X':
        score -= 10
    
    # Esquinas valen algo
    corners = [0, 2, 6, 8]
    for corner in corners:
        if board[corner] == 'O':
            score += 7
        elif board[corner] == 'X':
            score -= 5
    
    return score

# =========================================================
#            RECOCIDO SIMULADO (ALGORITMO IA)
# =========================================================

def simulated_annealing_move_experimental(board: List[str], T0: float,
                                        alpha: float = 0.90, T_min: float = 0.01,
                                        max_iter: int = 1000) -> Optional[int]:
    """
    Versión experimental que enfatiza el efecto de la temperatura.
    Esta versión elimina casi completamente las heurísticas predefinidas
    para que la temperatura tenga control total.
    """
    empties = [i for i, v in enumerate(board) if v == '']
    if not empties:
        return None
    
    # Solo aplicar heurísticas críticas a temperatura muy baja
    if T0 < 1.0:
        # Victoria inmediata
        for i in empties:
            test = board.copy()
            test[i] = 'O'
            if check_winner(test) and check_winner(test)[0] == 'O':
                return i
        
        # Bloqueo crítico
        for i in empties:
            test = board.copy()
            test[i] = 'X'
            if check_winner(test) and check_winner(test)[0] == 'X':
                if random.random() < 0.9:  # 90% probabilidad de bloquear
                    return i
    
    # Recocido puro
    current_move = random.choice(empties)
    current_score = evaluate_board_detailed(board[:])
    
    # Temperatura escalada según T0
    T = T0 * 20.0
    
    for _ in range(max_iter):
        if T < T_min:
            break
            
        neighbor_move = random.choice(empties)
        neighbor_board = board.copy()
        neighbor_board[neighbor_move] = 'O'
        neighbor_score = evaluate_board_detailed(neighbor_board)
        
        delta = neighbor_score - current_score
        
        if delta >= 0 or random.random() < math.exp(min(500, delta / max(0.01, T))):
            current_move = neighbor_move
            current_score = neighbor_score
        
        T *= alpha
    
    return current_move

# =========================================================
#                 COMPONENTES DE INTERFAZ (UI)
# =========================================================

@dataclass
class DifficultySelector:
    """
    Selector de dificultades con 3 botones horizontales.
    Solo permite selección cuando game_started es False.
    """
    x: int
    y: int
    button_width: int
    button_height: int
    spacing: int
    selected: Difficulty
    enabled: bool = True  # Se desactiva cuando empieza la partida

    def get_button_rect(self, difficulty: Difficulty) -> pygame.Rect:
        """Calcula el rectángulo de un botón de dificultad específico"""
        difficulties = list(Difficulty)
        index = difficulties.index(difficulty)
        btn_x = self.x + index * (self.button_width + self.spacing)
        return pygame.Rect(btn_x, self.y, self.button_width, self.button_height)

    def handle_event(self, event) -> bool:
        """
        Maneja clics en los botones de dificultad.
        Retorna True si hubo un cambio de selección.
        """
        if not self.enabled:
            return False
            
        if event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = event.pos
            for difficulty in Difficulty:
                rect = self.get_button_rect(difficulty)
                if rect.collidepoint(mx, my):
                    if self.selected != difficulty:
                        self.selected = difficulty
                        return True
        return False

    def draw(self, surf):
        """Dibuja los tres botones de dificultad con el estilo apropiado"""
        # Etiqueta descriptiva
        label_color = TEXT_COLOR if self.enabled else DISABLED_COLOR
        label = font_small.render("Dificultad:", True, label_color)
        surf.blit(label, (self.x, self.y - 28))
        
        # Dibujar cada botón
        for difficulty in Difficulty:
            rect = self.get_button_rect(difficulty)
            
            # Color de fondo del botón
            if not self.enabled:
                bg_color = DISABLED_COLOR
                text_color = (120, 120, 120)
                border_color = (150, 150, 150)
            elif self.selected == difficulty:
                bg_color = difficulty.color
                text_color = SELECTED_COLOR
                border_color = TEXT_COLOR
            else:
                bg_color = (250, 250, 250)
                text_color = difficulty.color
                border_color = difficulty.color
            
            # Dibujar botón
            pygame.draw.rect(surf, bg_color, rect, border_radius=8)
            pygame.draw.rect(surf, border_color, rect, 2, border_radius=8)
            
            # Texto del botón
            text = font_small.render(difficulty.label, True, text_color)
            text_x = rect.centerx - text.get_width() // 2
            text_y = rect.centery - text.get_height() // 2
            surf.blit(text, (text_x, text_y))
        
        # Mostrar temperatura seleccionada (solo si está habilitado)
        if self.enabled:
            temp_info = font_tiny.render(f"T0 = {self.selected.temperature}", True, (100, 100, 100))
            surf.blit(temp_info, (self.x + 280, self.y + self.button_height // 2 - temp_info.get_height() // 2))

@dataclass
class Button:
    """Botón simple con rectángulo y texto; se usa para reiniciar el juego."""
    rect: pygame.Rect
    text: str

    def draw(self, surf):
        pygame.draw.rect(surf, (250,250,250), self.rect, border_radius=10)
        pygame.draw.rect(surf, GRID_COLOR, self.rect, 2, border_radius=10)
        label = font_small.render(self.text, True, TEXT_COLOR)
        surf.blit(label, (self.rect.centerx - label.get_width()//2,
                          self.rect.centery - label.get_height()//2))

    def clicked(self, pos) -> bool:
        return self.rect.collidepoint(pos)

# =========================================================
#                DIBUJO Y ANIMACIONES
# =========================================================

def cell_center(idx: int) -> Tuple[int, int]:
    """Devuelve coordenadas (x,y) del centro de la celda idx."""
    col = idx % 3
    row = idx // 3
    x = col * CELL + CELL // 2
    y = row * CELL + CELL // 2
    return x, y

def draw_grid(highlight_hover: bool = True):
    """Dibuja el fondo y la cuadrícula del tablero. Si highlight_hover, resalta la celda bajo el cursor."""
    screen.fill(BG_COLOR)
    mx, my = pygame.mouse.get_pos()
    # Resaltar celda solo si el cursor está sobre el tablero
    if highlight_hover and my < BOARD_SIZE:
        c = mx // CELL
        r = my // CELL
        if 0 <= c < 3 and 0 <= r < 3:
            pygame.draw.rect(screen, HOVER_COLOR,
                             pygame.Rect(c*CELL + 4, r*CELL + 4, CELL - 8, CELL - 8), border_radius=10)

    # dibujar líneas de la cuadricula
    pygame.draw.line(screen, GRID_COLOR, (CELL, 0), (CELL, BOARD_SIZE), LINE_W)
    pygame.draw.line(screen, GRID_COLOR, (2*CELL, 0), (2*CELL, BOARD_SIZE), LINE_W)
    pygame.draw.line(screen, GRID_COLOR, (0, CELL), (BOARD_SIZE, CELL), LINE_W)
    pygame.draw.line(screen, GRID_COLOR, (0, 2*CELL), (BOARD_SIZE, 2*CELL), LINE_W)

def draw_piece_X(center: Tuple[int,int], scale: float):
    """Dibuja una X con 'scale' entre 0 y 1; la X aparece con una animación de trazo creciente."""
    cx, cy = center
    s = int((CELL//2 - 28) * scale)
    if s <= 0:
        return
    thickness = max(6, CELL // 20)
    pygame.draw.line(screen, X_COLOR, (cx - s, cy - s), (cx + s, cy + s), thickness)
    pygame.draw.line(screen, X_COLOR, (cx - s, cy + s), (cx + s, cy - s), thickness)

def draw_piece_O(center: Tuple[int,int], scale: float):
    """Dibuja una O con 'scale' entre 0 y 1; la O aparece con radio creciente."""
    cx, cy = center
    r = int((CELL//2 - 28) * scale)
    if r <= 0:
        return
    thickness = max(6, CELL // 20)
    pygame.draw.circle(screen, O_COLOR, (cx, cy), r, thickness)

def draw_board_pieces(board: List[str], appear_time: List[Optional[int]], now_ms: int):
    """
    Dibuja todas las piezas del tablero aplicando la animación de aparición según el timestamp guardado
    en 'appear_time' para cada celda. Esto crea un efecto visual "pop"/trazo cuando se colocan las fichas.
    """
    for i, cell in enumerate(board):
        if cell == '':
            continue
        center = cell_center(i)
        t0 = appear_time[i] if appear_time[i] is not None else now_ms - ANIM_PLACEMENT_MS
        progress = max(0.0, min(1.0, (now_ms - t0) / ANIM_PLACEMENT_MS))
        if cell == 'X':
            draw_piece_X(center, progress)
        else:
            draw_piece_O(center, progress)

def draw_win_line(trio: Tuple[int,int,int], now_ms: int, start_ms: int):
    """Dibuja animación de la línea ganadora desde el centro de la primera celda a la tercera."""
    a, b, c = trio
    p1 = cell_center(a)
    p3 = cell_center(c)
    progress = max(0.0, min(1.0, (now_ms - start_ms) / 450.0))
    x = p1[0] + (p3[0] - p1[0]) * progress
    y = p1[1] + (p3[1] - p1[1]) * progress
    pygame.draw.line(screen, WIN_LINE_COLOR, p1, (int(x), int(y)), CELL // 16)

def is_board_empty(board: List[str]) -> bool:
    """Verifica si el tablero está completamente vacío"""
    return all(cell == '' for cell in board)

# =========================================================
#                        BUCLE PRINCIPAL
# =========================================================

def main():
    """
    Bucle principal del juego que gestiona estados, eventos, temporización de la IA y dibujado.
    """
    # Estado del juego
    board = [''] * 9
    appear_time: List[Optional[int]] = [None] * 9
    game_over = False
    winner_info: Optional[Tuple[str, Tuple[int,int,int]]] = None
    win_anim_start: Optional[int] = None
    game_started = False  # Nueva variable para controlar si la partida ha comenzado

    player_turn = True
    ai_scheduled_at: Optional[int] = None

    # UI: selector de dificultad y botón reiniciar
    difficulty_selector = DifficultySelector(
        x=24, 
        y=BOARD_SIZE + 48, 
        button_width=80, 
        button_height=32, 
        spacing=12,
        selected=Difficulty.MEDIUM,  # Dificultad por defecto
        enabled=True
    )
    
    btn_reset = Button(pygame.Rect(WIDTH - 148, HEIGHT - 80, 124, 44), "Nueva Partida")

    running = True
    while running:
        now = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Manejo del selector de dificultad (solo antes de empezar)
            difficulty_selector.handle_event(event)

            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                
                # Reiniciar si se pulsa el botón
                if btn_reset.clicked((mx, my)):
                    # Reset completo del estado del juego
                    board[:] = [''] * 9
                    appear_time[:] = [None] * 9
                    game_over = False
                    winner_info = None
                    win_anim_start = None
                    player_turn = True
                    ai_scheduled_at = None
                    game_started = False  # Habilitar selector de dificultad nuevamente
                    difficulty_selector.enabled = True
                    continue

                # Click en tablero: solo si es el turno del jugador y no ha terminado el juego
                if my < BOARD_SIZE and player_turn and not game_over:
                    col = mx // CELL
                    row = my // CELL
                    idx = row * 3 + col
                    if 0 <= idx < 9 and board[idx] == '':
                        # Primera jugada: deshabilitar selector de dificultad
                        if not game_started:
                            game_started = True
                            difficulty_selector.enabled = False
                        
                        # Colocar 'X' y registrar tiempo para animación
                        board[idx] = 'X'
                        appear_time[idx] = now

                        # Validar ganador tras la jugada del usuario
                        w = check_winner(board)
                        if w:
                            game_over = True
                            winner_info = w
                            win_anim_start = now
                        elif is_draw(board):
                            game_over = True
                            winner_info = None
                        else:
                            # Programar jugada de la IA tras un pequeño delay
                            player_turn = False
                            ai_scheduled_at = now + AI_DELAY_MS

        # EJECUCIÓN DE LA IA: cuando el tiempo programado llegue y no haya acabado el juego
        if not player_turn and not game_over and ai_scheduled_at is not None and now >= ai_scheduled_at:
            # Usar la temperatura de la dificultad seleccionada
            T0 = difficulty_selector.selected.temperature
            move = simulated_annealing_move_experimental(board, T0=T0)
            
            if move is not None:
                board[move] = 'O'
                appear_time[move] = now

            # Validar ganador/empate inmediatamente tras la jugada de la IA
            w = check_winner(board)
            if w:
                game_over = True
                winner_info = w
                win_anim_start = now
            elif is_draw(board):
                game_over = True
                winner_info = None

            # Si no terminó, devolver turno al jugador
            if not game_over:
                player_turn = True
            ai_scheduled_at = None

        # ------------------ DIBUJADO ------------------
        draw_grid()
        draw_board_pieces(board, appear_time, now)

        # Si hubo ganador, dibujar línea animada
        if game_over and winner_info and win_anim_start is not None:
            _, trio = winner_info
            draw_win_line(trio, now, win_anim_start)

        # Panel inferior: fondo y separador
        pygame.draw.rect(screen, (252,252,252), (0, BOARD_SIZE, WIDTH, PANEL_H))
        pygame.draw.line(screen, (210,210,210), (0, BOARD_SIZE), (WIDTH, BOARD_SIZE), 2)

        # Mensaje de estado
        status_msg = ''
        if game_over:
            if winner_info is None:
                status_msg = "¡Empate!"
            else:
                who, _ = winner_info
                status_msg = "¡Ganaste!" if who == 'X' else "La computadora gana."
        else:
            if not game_started:
                status_msg = "Selecciona dificultad y haz tu primer movimiento (X)"
            else:
                status_msg = "Tu turno (X)" if player_turn else "Pensando..."

        status_label = font_small.render(status_msg, True, TEXT_COLOR)
        screen.blit(status_label, (20, BOARD_SIZE + 90))

        # Dibujar selector de dificultad
        difficulty_selector.draw(screen)

        # Botón reiniciar
        btn_reset.draw(screen)

        # Información sobre dificultades
        if not game_started:
            help_text = font_tiny.render("Fácil: comete errores • Medio: equilibrada • Difícil: casi perfecta", True, (100,100,100))
            screen.blit(help_text, (24, BOARD_SIZE + 144))
        else:
            current_diff = difficulty_selector.selected.label
            help_text = font_tiny.render(f"Jugando en dificultad: {current_diff} (T0={difficulty_selector.selected.temperature})", True, (100,100,100))
            screen.blit(help_text, (24, BOARD_SIZE + 144))

        # Actualizar pantalla
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()