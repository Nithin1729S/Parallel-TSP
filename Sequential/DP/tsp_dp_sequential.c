//Time Complexity: O(N^2 * 2^N )
//Space Complexity: O(N * 2^N)

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>

#define MAX_V 19 
#define MAX 1000000

int dist[MAX_V + 1][MAX_V + 1];
int memo[MAX_V + 1][1 << (MAX_V + 1)];

int min(int a, int b) {
    return (a < b) ? a : b;
}

//Nested loop can be collapsed using collapse(2)
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

//can parallize this using tasks
int fun(int i, int mask, int n)
{
    if (mask == ((1 << i) | 3))
        return dist[1][i];
    if (memo[i][mask] != 0)
        return memo[i][mask];
    int res = MAX;
    for (int j = 1; j <= n; j++)
        if ((mask & (1 << j)) && j != i && j != 1)
            res = min(res, fun(j, mask & (~(1 << i)), n) + dist[j][i]);
    return memo[i][mask] = res;
}

int main()
{
    srand(time(NULL)); 
    FILE *fptr = fopen("tsp_dp_execution_times.csv", "w");
    fprintf(fptr, "Nodes,Execution_Time\n");
    fclose(fptr);
    for (int n = 4; n <= MAX_V; n++) {
        generateGraph(n);
        //can use parrallel for region for this loop
        for (int i = 0; i <= n; i++) {
            for (int j = 0; j < (1 << (n + 1)); j++) {
                memo[i][j] = 0;
            }
        }

        clock_t start = clock();

        //can use parallel reduction clause here 
        int ans = MAX;
        for (int i = 1; i <= n; i++)
            ans = min(ans, fun(i, (1 << (n + 1)) - 1, n) + dist[i][1]);

        clock_t end = clock();
        double exec_time = (double)(end - start) / CLOCKS_PER_SEC;
    
        printf("Nodes: %d, Execution Time: %f seconds, Min Cost: %d\n", n, exec_time, ans);

        saveExecutionTime(n, exec_time);
    }

    return 0;
}