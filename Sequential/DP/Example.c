#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#define MAX 20 
#define INF INT_MAX

int graph[MAX][MAX]; 
int dp[1 << MAX][MAX]; 

int tspDP(int mask, int pos, int n) {

    if (mask == (1 << n) - 1) {
        return graph[pos][0]; 
    }

    if (dp[mask][pos] != -1) {
        return dp[mask][pos];
    }

    int min_cost = INF; 

    for (int city = 0; city < n; city++) {
        if ((mask & (1 << city)) == 0) {
            int new_cost = graph[pos][city] + tspDP(mask | (1 << city), city, n);
            min_cost = (new_cost < min_cost) ? new_cost : min_cost; 
        }
    }

    return dp[mask][pos] = min_cost;
}

int main() {
    int n = 4; 
    int min_cost;

    int example_graph[4][4] = { { 0, 10, 15, 20 },
                                  { 10, 0, 35, 25 },
                                  { 15, 35, 0, 30 },
                                  { 20, 25, 30, 0 } };

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            graph[i][j] = example_graph[i][j];
        }
    }

    for (int i = 0; i < (1 << n); i++) {
        for (int j = 0; j < n; j++) {
            dp[i][j] = -1;
        }
    }

    min_cost = tspDP(1, 0, n);
    printf("Minimum Cost = %d\n", min_cost);

    return 0;
}
