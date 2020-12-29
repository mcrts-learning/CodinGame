#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

struct position
{
    int x;
    int y;
};
typedef struct position position;

int main()
{
    int N;
    scanf("%d", &N);
    bool matrix[N][N];

    int pointer = 0;
    position lights[N*N];
    int L;
    position pos;
    scanf("%d", &L);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            char cell[4];
            scanf("%s", cell);
            if (strcmp(cell, "C") == 0) {
                pos = (position) {j, i};
                lights[pointer] = pos;
                pointer++;
            }
        }
    }
    for (int i = 0; i < pointer; i++) {
        pos = lights[i];
        for (int y = 0; y < N; y++) {
            for (int x = 0; x < N; x++) {
                int dx = abs(pos.x - x);
                int dy = abs(pos.y - y);
                if (dx < L && dy < L) {
                    matrix[y][x] = true;
                }
            }
        }
    }

    int counter = 0;
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            if (!matrix[y][x]) {
                counter++;
            }
        }
    }

    fprintf(stdout, "%d\n", counter);
    return 0;
}
