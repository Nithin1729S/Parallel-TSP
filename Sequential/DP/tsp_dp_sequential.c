#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>

#define MAX_V 25
#define MAX 1000000

int dist[MAX_V + 1][MAX_V + 1];

int min(int a, int b) {
    return (a < b) ? a : b;
}

void generateGraph(int n) {
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= n; j++) {
            if (i != j) {
                dist[i][j] = rand() % 100 + 1;  
            } else {
                dist[i][j] = 0; 
            }
        }
    }
}

void saveExecutionTime(int n, double exec_time) {
    FILE *fptr;
    fptr = fopen("tsp_dp_execution_times.csv", "a");
    if (fptr == NULL) {
        printf("Error opening file!");
        exit(1);
    }
    fprintf(fptr, "%d,%f\n", n, exec_time);
    fclose(fptr);
}

// Dynamic Programming function for TSP
int fun(int i, int mask, int n, int **memo) {
    if (mask == ((1 << i) | 3)) 
        return dist[1][i];

    if (memo[i][mask] != 0) 
        return memo[i][mask];
    //each recusive call can be parallized by sharing tasks ( eg. taskgroup )
    int res = MAX;
    for (int j = 1; j <= n; j++) {
        if ((mask & (1 << j)) && j != i && j != 1) {
            res = min(res, fun(j, mask & (~(1 << i)), n, memo) + dist[j][i]);
        }
    }
    return memo[i][mask] = res;
}

int main() {
    srand(time(NULL)); 
    FILE *fptr = fopen("tsp_dp_execution_times.csv", "w");
    fprintf(fptr, "Nodes,Execution_Time\n");
    fclose(fptr);

    for (int n = 4; n <= MAX_V; n++) {
        generateGraph(n);
        //can paralllize the task of initializing dp array
        int **memo = (int **)malloc((n + 1) * sizeof(int *));
        for (int i = 0; i <= n; i++) {
            memo[i] = (int *)malloc((1 << (n + 1)) * sizeof(int));
            for (int j = 0; j < (1 << (n + 1)); j++) {
                memo[i][j] = 0;  
            }
        }

        clock_t start = clock();
        int ans = MAX;
        for (int i = 1; i <= n; i++) {
            ans = min(ans, fun(i, (1 << (n + 1)) - 1, n, memo) + dist[i][1]);
        }
        clock_t end = clock();
        double exec_time = (double)(end - start) / CLOCKS_PER_SEC;

        printf("Nodes: %d, Execution Time: %f seconds, Min Cost: %d\n", n, exec_time, ans);
        saveExecutionTime(n, exec_time);

        for (int i = 0; i <= n; i++) {
            free(memo[i]);
        }
        free(memo);
    }

    return 0;
}
