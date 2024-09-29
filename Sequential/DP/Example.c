#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>

#define N 4
#define MAX 1000000

int dist[N + 1][N + 1] = {
    {0, 0, 0, 0, 0},
    {0, 0, 10, 15, 20},
    {0, 10, 0, 25, 25},
    {0, 15, 25, 0, 30},
    {0, 20, 25, 30, 0},
};

int memo[N + 1][1 << (N + 1)];

int min(int a, int b) {
    return (a < b) ? a : b;
}

void generateGraph() {
    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= N; j++) {
            if (i != j) {
                dist[i][j] = rand() % 100 + 1;  
            } else {
                dist[i][j] = 0; 
            }
        }
    }
}

int tsp(int i, int mask) {
    if (mask == ((1 << i) | 3))
        return dist[1][i];
    if (memo[i][mask] != 0)
        return memo[i][mask];
    int res = MAX;
    for (int j = 1; j <= N; j++)
        if ((mask & (1 << j)) && j != i && j != 1)
            res = min(res, tsp(j, mask & (~(1 << i))) + dist[j][i]);
    return memo[i][mask] = res;
}

int main() {
    srand(time(NULL));
    // generateGraph();
    for (int i = 0; i <= N; i++) {
        for (int j = 0; j < (1 << (N + 1)); j++) {
            memo[i][j] = 0;
        }
    }

    clock_t start = clock();

    int ans = MAX;
    for (int i = 1; i <= N; i++)
        ans = min(ans, tsp(i, (1 << (N + 1)) - 1) + dist[i][1]);

    clock_t end = clock();
    double exec_time = (double)(end - start) / CLOCKS_PER_SEC;

    printf("For N = %d, the cost of most efficient tour = %d\n", N, ans);
    printf("Execution time: %f seconds\n", exec_time);

    return 0;
}