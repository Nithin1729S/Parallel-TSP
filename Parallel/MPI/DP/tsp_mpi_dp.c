#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>

#define MAX_V 25
#define MAX 1000000

int dist[MAX_V + 1][MAX_V + 1];

// Helper function to find minimum of two values
int min(int a, int b) {
    return (a < b) ? a : b;
}

// Function to generate random distances between nodes in the graph
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

// Function to save execution time in a CSV file
void saveExecutionTime(int n, double exec_time) {
    FILE *fptr;
    fptr = fopen("mpi_dp_execution_times.csv", "a");
    if (fptr == NULL) {
        printf("Error opening file!");
        exit(1);
    }
    fprintf(fptr, "%d,%f\n", n, exec_time);
    fclose(fptr);
}

// Dynamic Programming function for TSP with MPI
int fun(int i, int mask, int n, int **memo) {
    if (mask == ((1 << i) | 3)) 
        return dist[1][i];

    if (memo[i][mask] != 0) 
        return memo[i][mask];

    int res = MAX;
    for (int j = 1; j <= n; j++) {
        if ((mask & (1 << j)) && j != i && j != 1) {
            res = min(res, fun(j, mask & (~(1 << i)), n, memo) + dist[j][i]);
        }
    }
    return memo[i][mask] = res;
}

int main(int argc, char** argv) {
    int rank, size;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    srand(time(NULL) + rank);  // Seed with rank for different random values

    // Master process sets up CSV file
    if (rank == 0) {
        FILE *fptr = fopen("tsp_dp_execution_times.csv", "w");
        fprintf(fptr, "Nodes,Execution_Time\n");
        fclose(fptr);
    }

    // Loop over number of nodes (from 4 to MAX_V)
    for (int n = 4; n <= MAX_V; n++) {
        if (rank == 0) generateGraph(n);  // Generate graph only on master

        // Broadcast the distance matrix to all processes
        MPI_Bcast(dist, (MAX_V + 1) * (MAX_V + 1), MPI_INT, 0, MPI_COMM_WORLD);

        // Initialize the memoization table
        int **memo = (int **)malloc((n + 1) * sizeof(int *));
        for (int i = 0; i <= n; i++) {
            memo[i] = (int *)malloc((1 << (n + 1)) * sizeof(int));
            for (int j = 0; j < (1 << (n + 1)); j++) {
                memo[i][j] = 0;
            }
        }

        clock_t start = clock();
        int local_min_cost = MAX;
        
        // Distribute tasks to each process based on the starting city
        for (int i = rank + 1; i <= n; i += size) {
            int cost = fun(i, (1 << (n + 1)) - 1, n, memo) + dist[i][1];
            local_min_cost = min(local_min_cost, cost);
        }

        // Reduce local minimums to find the global minimum cost
        int global_min_cost;
        MPI_Reduce(&local_min_cost, &global_min_cost, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

        // Master process calculates and records execution time
        if (rank == 0) {
            clock_t end = clock();
            double exec_time = (double)(end - start) / CLOCKS_PER_SEC;

            printf("Nodes: %d, Execution Time: %f seconds, Min Cost: %d\n", n, exec_time, global_min_cost);
            saveExecutionTime(n, exec_time);
        }

        // Free allocated memory for memoization table
        for (int i = 0; i <= n; i++) {
            free(memo[i]);
        }
        free(memo);
    }

    MPI_Finalize();  // Finalize MPI
    return 0;
}
