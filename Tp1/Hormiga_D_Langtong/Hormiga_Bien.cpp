#include <iostream>
#include <vector>
#include <thread>
#include <chrono>

using namespace std;
bool endgame=false;

enum Direccion { ARRIBA, DERECHA, ABAJO, IZQUIERDA };

int main() {
    const int FILAS = 40;
    const int COLUMNAS = 80;
    vector<vector<bool>> grid(FILAS, vector<bool>(COLUMNAS, false)); // false=blanco, true=negro

    int x = COLUMNAS / 2;
    int y = FILAS / 2;
    Direccion dir = ARRIBA;

    
    while (endgame == false) {
        // Mostrar grilla
        system("clear"); // en Linux/Mac
        for (int i = 0; i < FILAS; i++) {
            for (int j = 0; j < COLUMNAS; j++) {
                if (i == y && j == x)
                    cout << 'A'; // Hormiga
                else
                    cout << (grid[i][j] ? '#' : '.'); // negro o blanco
            }
            cout << '\n';
        }

        // Reglas de la hormiga
        if (!grid[y][x]) { // blanco
            dir = (Direccion)((dir + 1) % 4); // gira derecha
            grid[y][x] = true; // pinta negro
        } else { // negro
            dir = (Direccion)((dir + 3) % 4); // gira izquierda
            grid[y][x] = false; // pinta blanco
        }

        // Avanza (con borde que envuelve)
        switch (dir) {
            case ARRIBA:    y = (y - 1 + FILAS) % FILAS; break;
            case DERECHA:   x = (x + 1) % COLUMNAS; break;
            case ABAJO:     y = (y + 1) % FILAS; break;
            case IZQUIERDA: x = (x - 1 + COLUMNAS) % COLUMNAS; break;
        }

        if(x == 0){
            endgame=true;
            return 0;
        }
        if(y == 0){
            endgame=true;
            return 0;
        }
        if(x == COLUMNAS){
            endgame=true;
            return 0;
        }
        if(y == FILAS){
            endgame=true;
            return 0;
        }
        this_thread::sleep_for(chrono::milliseconds(5)); // más lento para ver el cambio
        
    }
    
    return 0;
}