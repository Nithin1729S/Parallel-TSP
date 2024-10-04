#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>

#define MAX_V 40 
#define MAX 1000000

int dist[MAX_V + 1][MAX_V + 1] = {
    {0, 0, 0, 0, 0, 0, 0},
    {0, 0, 10, 20, 30, 40, 50},
    {0, 10, 0, 15, 25, 35, 45},
    {0, 20, 15, 0, 22, 30, 40},
    {0, 30, 25, 22, 0, 18, 28},
    {0, 40, 35, 30, 18, 0, 12},
    {0, 50, 45, 40, 28, 12, 0},
};


int min(int a, int b) {
    return (a < b) ? a : b;
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
int fun(int i, int mask, int n, int **memo) 
{
    if (mask == ((1 << i) | 3)) 
        return dist[1][i];
            
    if (memo[i][mask] != 0) 
        return memo[i][mask];
    
    int res = MAX;

    #pragma omp parallel for reduction(min:res) schedule(dynamic)
    for (int j = 1; j <= n; j++) {
        if ((mask & (1 << j)) && j != i && j != 1) 
        {
            int newMask = mask & (~(1 << i));
            int result = fun(j, newMask, n, memo);
            res = min(res, result + dist[j][i]);
        }
    }
        
    #pragma omp critical
    {
        memo[i][mask] = res;
    }

    return res;
}


int main() {
    FILE *fptr = fopen("tsp_dp_execution_times.csv", "w");
    fprintf(fptr, "Nodes,Execution_Time\n");
    fclose(fptr);

    int n = 4; 
    int **memo = (int **)malloc((n + 1) * sizeof(int *));
    for (int i = 0; i <= n; i++) {
        memo[i] = (int *)malloc((1 << (n + 1)) * sizeof(int));
        for (int j = 0; j < (1 << (n + 1)); j++) {
            memo[i][j] = 0;  
        }
    }

    clock_t start = clock();
    int ans = MAX;

    #pragma omp parallel for reduction(min:ans) schedule(dynamic)
    for (int i = 1; i <= n; i++) 
    {
        int mask = (1 << (n + 1)) - 1;
        int result = fun(i, mask, n, memo);
        int total = result + dist[i][1];
        ans = min(ans, total);
    }

    clock_t end = clock();
    double exec_time = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Minimum Cost = %d\n",ans);
    saveExecutionTime(n, exec_time);

    for (int i = 0; i <= n; i++) {
        free(memo[i]);
    }
    free(memo);

    return 0;
}
