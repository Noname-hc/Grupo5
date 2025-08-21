#!/usr/bin/env python3
"""
Visualizador integrado - A* con coste unitario
DFS, Greedy Best-First y A* (esta versión de A* trata todas las transiciones con coste 1).
- Ejecutar: python3 TP2_5.py
- En empate, se desempata por orden alfabético del nodo.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import heapq

# --------- CONFIGURACIÓN DEL TABLERO ----------
positions = {
    'A': (4,3), 'B': (5,3),
    'C': (4,2), 'D': (5,2), 'E': (6,2),
    'G': (1,1), 'I': (2,1), 'W': (3,1), 'K': (4,1), 'M': (5,1), 'N': (6,1),
    'P': (1,0), 'Q': (2,0), 'R': (3,0), 'T': (4,0), 'F': (5,0)
}

walls = {
    ('C','D'), ('D','C'),
    ('D','E'), ('E','D'),
    ('T','F'), ('F','T'),
    ('W','R'), ('R','W')
}

dirs = [(1,0),(-1,0),(0,1),(0,-1)]
pos_to_node = {pos: n for n,pos in positions.items()}

# construir adyacencias ortogonales
adj = {n: [] for n in positions}
for n,p in positions.items():
    x,y = p
    for dx,dy in dirs:
        neigh_pos = (x+dx, y+dy)
        if neigh_pos in pos_to_node:
            adj[n].append(pos_to_node[neigh_pos])
for a,b in list(walls):
    if b in adj.get(a,[]): adj[a].remove(b)
for n in adj: adj[n] = sorted(adj[n])  # vecinos siempre en orden alfabético

def step_cost(to_node):
    return 30 if to_node == 'W' else 1

def manhattan(a, b='F'):
    x1,y1 = positions[a]; x2,y2 = positions[b]
    return abs(x1-x2) + abs(y1-y2)

def reconstruct(came_from, goal):
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = came_from.get(cur)
    return list(reversed(path))

# ---------------- Algoritmos ----------------

def dfs_events(start='I', goal='F'):
    stack = [start]
    came_from = {start: None}
    visited = set()
    yield ('stack_snapshot', list(stack))
    while stack:
        node = stack.pop()
        if node in visited:
            yield ('stack_snapshot', list(stack))
            continue
        yield ('visit', node)
        visited.add(node)
        if node == goal:
            path = reconstruct(came_from, goal)
            yield ('path', path)
            return
        neighbors = adj[node]  # ya ordenados alfabéticamente
        for nb in reversed(neighbors):  # mantener comportamiento de stack (último entra primero)
            if nb not in visited and nb not in stack:
                came_from[nb] = node
                stack.append(nb)
        yield ('stack_snapshot', list(stack))
    yield ('failed', None)

def greedy_events(start='I', goal='F'):
    open_heap = [(manhattan(start), start)]  # (heurística, nombre) → asegura desempate alfabético
    open_set = {start}
    came_from = {start: None}
    visited = set()
    yield ('frontier_snapshot', sorted(list(open_set), key=lambda n: (manhattan(n), n)))
    while open_heap:
        h, node = heapq.heappop(open_heap)
        if node not in open_set:
            continue
        open_set.remove(node)
        if node in visited:
            yield ('frontier_snapshot', sorted(list(open_set), key=lambda n: (manhattan(n), n)))
            continue
        yield ('visit', node)
        visited.add(node)
        if node == goal:
            path = reconstruct(came_from, goal)
            yield ('path', path)
            return
        # Greedy: solo heurística, no costo, desempate alfabético
        for nb in adj[node]:
            if nb in visited or nb in open_set:
                continue
            came_from[nb] = node
            heapq.heappush(open_heap, (manhattan(nb), nb))  # (heurística, nombre)
            open_set.add(nb)
            yield ('push', nb, node)
        # Ordenar el heap para asegurar desempate alfabético en caso de heurística igual
        open_heap.sort()
        yield ('frontier_snapshot', sorted(list(open_set), key=lambda n: (manhattan(n), n)))
    yield ('failed', None)

def astar_unitcost_events(start='I', goal='F'):
    counter = 0
    open_heap = []
    gscore = {n: float('inf') for n in positions}
    fscore = {n: float('inf') for n in positions}
    gscore[start] = 0
    fscore[start] = manhattan(start)
    heapq.heappush(open_heap, (fscore[start], start, gscore[start], counter))  # incluye nombre en la tupla
    frontier_f = {start: (fscore[start], gscore[start])}
    came_from = {start: None}
    closed = set()
    yield ('frontier_snapshot', sorted(list(frontier_f.keys()), key=lambda n: (frontier_f.get(n,(float('inf'),float('inf'))), n)))
    while open_heap:
        f, node, g, _ = heapq.heappop(open_heap)
        cur = frontier_f.get(node)
        if cur is None or (abs(cur[0]-f) > 1e-9 or cur[1] != g):
            continue
        frontier_f.pop(node, None)
        if node in closed:
            yield ('frontier_snapshot', sorted(list(frontier_f.keys()), key=lambda n: (frontier_f.get(n,(float('inf'),float('inf'))), n)))
            continue
        yield ('visit', node)
        closed.add(node)
        if node == goal:
            path = reconstruct(came_from, goal)
            yield ('path', path)
            return
        for nb in adj[node]:
            cost = step_cost(nb)
            tentative_g = gscore[node] + cost
            if tentative_g < gscore.get(nb, float('inf')):
                came_from[nb] = node
                gscore[nb] = tentative_g
                fnb = tentative_g + manhattan(nb)
                fscore[nb] = fnb
                if nb not in closed:
                    counter += 1
                    heapq.heappush(open_heap, (fnb, nb, tentative_g, counter))
                    frontier_f[nb] = (fnb, tentative_g)
                    yield ('push', nb, node)
        yield ('frontier_snapshot', sorted(list(frontier_f.keys()), key=lambda n: (frontier_f.get(n,(float('inf'),float('inf'))), n)))
    yield ('failed', None)

# ----------------- INTERFAZ GRÁFICA --------------------

CELL = 70
PAD = 20
xs = [p[0] for p in positions.values()]
ys = [p[1] for p in positions.values()]
minx, maxx = min(xs), max(xs)
miny, maxy = min(ys), max(ys)
W = (maxx - minx + 1) * CELL + PAD*2
H = (maxy - miny + 1) * CELL + PAD*2

class SearchGui(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("TP2 - Visualizador (A* unit-cost)")
        self.resizable(False, False)
        self.canvas = tk.Canvas(self, width=W, height=H, bg='white')
        self.canvas.grid(row=0, column=0, columnspan=6)
        # controls
        self.algo_var = tk.StringVar(value='DFS')
        algo_menu = ttk.OptionMenu(self, self.algo_var, 'DFS', 'DFS', 'Greedy', 'A*')
        algo_menu.grid(row=1, column=0, padx=5, pady=6)
        self.run_btn = ttk.Button(self, text="Run", command=self.start_search)
        self.run_btn.grid(row=1, column=1, sticky='ew', padx=5)
        self.step_btn = ttk.Button(self, text="Step", command=self.step_once)
        self.step_btn.grid(row=1, column=2, sticky='ew', padx=5)
        self.reset_btn = ttk.Button(self, text="Reset", command=self.reset_board)
        self.reset_btn.grid(row=1, column=3, sticky='ew', padx=5)
        self.speed_label = ttk.Label(self, text="Velocidad (ms):")
        self.speed_label.grid(row=1, column=4, sticky='e')
        self.speed_var = tk.IntVar(value=300)
        self.speed_spin = ttk.Spinbox(self, from_=50, to=2000, increment=50, textvariable=self.speed_var, width=6)
        self.speed_spin.grid(row=1, column=5, sticky='w', padx=5)
        # info labels
        self.info_var = tk.StringVar(value='')
        self.info_label = ttk.Label(self, textvariable=self.info_var)
        self.info_label.grid(row=2, column=0, columnspan=6, sticky='w', padx=8)
        # internals
        self.rects = {}
        self.texts = {}
        self.edge_rects = []
        self.draw_board()
        self._running = False
        self._after_id = None
        self._events = None
        self._visited_nodes = set()
        self._frontier_nodes = []
        self._expanded = 0
        self._current_path = None

    def grid_to_canvas(self, pos):
        x,y = pos
        cx = PAD + (x - minx) * CELL
        cy = PAD + (maxy - y) * CELL
        return cx, cy

    def draw_board(self):
        self.canvas.delete('all')
        self.rects.clear(); self.texts.clear(); self.edge_rects.clear()
        for node, pos in positions.items():
            cx, cy = self.grid_to_canvas(pos)
            r = self.canvas.create_rectangle(cx, cy, cx+CELL, cy+CELL, fill='white', outline='black', width=2)
            t = self.canvas.create_text(cx+CELL/2, cy+CELL/2, text=node, font=('Helvetica', 14, 'bold'))
            self.rects[node]=r; self.texts[node]=t
        if 'I' in self.rects: self.canvas.itemconfigure(self.rects['I'], fill='#fff275')
        if 'F' in self.rects: self.canvas.itemconfigure(self.rects['F'], fill='#fff275')
        if 'W' in self.rects: self.canvas.itemconfigure(self.rects['W'], fill='#ffd8a6')
        for (a,b) in [('C','D'),('D','E'),('T','F'),('W','R')]:
            if a in positions and b in positions:
                ax,ay = positions[a]; bx,by = positions[b]
                cax, cay = self.grid_to_canvas((ax,ay)); cbx, cby = self.grid_to_canvas((bx,by))
                cx = (cax + cbx)/2 + CELL/2; cy = (cay + cby)/2 + CELL/2
                if ax == bx:
                    x1 = cx - CELL*0.4; x2 = cx + CELL*0.4; y1 = cy - CELL*0.05; y2 = cy + CELL*0.05
                else:
                    x1 = cx - CELL*0.05; x2 = cx + CELL*0.05; y1 = cy - CELL*0.4; y2 = cy + CELL*0.4
                edge = self.canvas.create_rectangle(x1, y1, x2, y2, fill='red', outline='red')
                self.edge_rects.append(edge)
        self.canvas.create_rectangle(10, H-30, 420, H-10, fill='white', outline='black')
        self.canvas.create_text(215, H-20, text='Leyenda: amarillo=start/goal, naranja=W, azul=visitado, celeste=frontera, naranja oscuro=cima', font=('Helvetica', 9))

    def reset_board(self):
        if self._after_id:
            self.after_cancel(self._after_id); self._after_id = None
        self._running = False
        self._events = None
        self._visited_nodes.clear()
        self._frontier_nodes = []
        self._expanded = 0
        self._current_path = None
        self.info_var.set('')
        self.draw_board()

    def start_search(self):
        if self._running: return
        self.reset_board()
        algo = self.algo_var.get()
        if algo == 'DFS':
            self._events = dfs_events('I','F')
        elif algo == 'Greedy':
            self._events = greedy_events('I','F')
        else:  # A* -> unit cost variant
            self._events = astar_unitcost_events('I','F')
        self._running = True
        self._after_id = self.after(1, self._animate_step)

    def step_once(self):
        if self._running:
            return
        if self._events is None:
            algo = self.algo_var.get()
            if algo == 'DFS':
                self._events = dfs_events('I','F')
            elif algo == 'Greedy':
                self._events = greedy_events('I','F')
            else:
                self._events = astar_unitcost_events('I','F')
        try:
            ev = next(self._events)
        except StopIteration:
            self._events = None
            return
        self._process_event(ev)

    def _animate_step(self):
        try:
            ev = next(self._events)
        except StopIteration:
            self._running = False
            self._events = None
            return
        self._process_event(ev)
        if self._running:
            delay = self.speed_var.get()
            self._after_id = self.after(delay, self._animate_step)

    def _clear_frontier_colors(self):
        for n in self._frontier_nodes:
            if n not in self._visited_nodes and n not in ('I','F'):
                self.canvas.itemconfigure(self.rects[n], fill='white')
        self._frontier_nodes = []

    def _process_event(self, ev):
        typ = ev[0]
        if typ == 'visit':
            node = ev[1]
            self._expanded += 1
            self._visited_nodes.add(node)
            if node not in ('I','F'):
                self.canvas.itemconfigure(self.rects[node], fill='#9ad3de')
            self.info_var.set(f'Expanded: {self._expanded}')
        elif typ == 'push':
            nb = ev[1]; parent = ev[2]
            if nb not in ('I','F') and nb not in self._visited_nodes:
                self.canvas.itemconfigure(self.rects[nb], fill='#cce7ff')
        elif typ == 'stack_snapshot':
            stack = ev[1]
            self._clear_frontier_colors()
            for n in stack:
                if n not in self._visited_nodes and n not in ('I','F'):
                    self.canvas.itemconfigure(self.rects[n], fill='#cce7ff')
            self._frontier_nodes = list(stack)
            if stack:
                top = stack[-1]
                if top not in ('I','F'):
                    self.canvas.itemconfigure(self.rects[top], fill='#ffb86b')
        elif typ == 'frontier_snapshot':
            frontier = ev[1]
            self._clear_frontier_colors()
            for n in frontier:
                if n not in self._visited_nodes and n not in ('I','F'):
                    self.canvas.itemconfigure(self.rects[n], fill='#cce7ff')
            self._frontier_nodes = list(frontier)
            if frontier:
                top = frontier[0]
                if top not in ('I','F'):
                    self.canvas.itemconfigure(self.rects[top], fill='#ffb86b')
        elif typ == 'path':
            path = ev[1]
            self._current_path = path
            for node in path:
                if node in ('I','F'): continue
                self.canvas.itemconfigure(self.rects[node], fill='#9ee29e')
            for i in range(len(path)-1):
                a = path[i]; b = path[i+1]
                ax,ay = positions[a]; bx,by = positions[b]
                cax, cay = self.grid_to_canvas((ax,ay)); cbx, cby = self.grid_to_canvas((bx,by))
                x1 = cax + CELL/2; y1 = cay + CELL/2
                x2 = cbx + CELL/2; y2 = cby + CELL/2
                self.canvas.create_line(x1,y1,x2,y2, width=4)
            # calcular costo del camino: A* usa coste unitario aquí, otras búsquedas mantienen step_cost
            algo = self.algo_var.get()
            if algo == 'A*':
                cost = len(path) - 1
            else:
                cost = 0
                for s in path[1:]:
                    cost += step_cost(s)
            self.info_var.set(f'Expanded: {self._expanded}   Cost: {cost}   Path: {"-".join(path)}')
            messagebox.showinfo("Resultado", f"Camino encontrado. Cost: {cost}. Expanded: {self._expanded}")
            self._running = False
        elif typ == 'failed':
            messagebox.showinfo("Resultado", "No se encontró camino")
            self._running = False

if __name__ == '__main__':
    app = SearchGui()
    app.mainloop()
