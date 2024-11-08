#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>

#define V 100

void generateGraph(int graph[][V], int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i != j) {
                graph[i][j] = rand() % V + 1;
            } else {
                graph[i][j] = 0;
            }
        }
    }
}

int calculatePathCost(int graph[][V], int path[], int n) {
    int total_cost = 0;
    for (int i = 0; i < n - 1; i++) {
        total_cost += graph[path[i]][path[i+1]];
    }
    total_cost += graph[path[n-1]][path[0]];
    return total_cost;
}

void swap(int *x, int *y) {
    int temp = *x;
    *x = *y;
    *y = temp;
}

int next_permutation(int *first, int *last) {
    if (first == last) return 0;
    int *i = first;
    ++i;
    if (i == last) return 0;
    i = last;
    --i;

    while (1) {
        int *ii = i;
        --i;
        if (*i < *ii) {
            int *j = last;
            while (!(*i < *--j));
            swap(i, j);
            ++ii;
            j = last;
            while (ii < --j) {
                swap(ii++, j);
            }
            return 1;
        }
        if (i == first) return 0;
    }
}

void tspBruteForceMPI(int graph[][V], int n, int *min_cost, int rank, int size) {
    int path[V];
    for (int i = 0; i < n; i++) {
        path[i] = i;
    }

    int local_min_cost = INT_MAX;
    int count = 0;

    // Skip permutations to split work across MPI processes
    do {
        if (count % size == rank) {  // Only process specific permutations
            int current_cost = calculatePathCost(graph, path, n);
            if (current_cost < local_min_cost) {
                local_min_cost = current_cost;
            }
        }
        count++;
    } while (next_permutation(path, path + n));

    // Use MPI_Reduce to find the minimum cost across all processes
    MPI_Reduce(&local_min_cost, min_cost, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
}

void saveExecutionTime(int n, double exec_time) {
    FILE *fptr;
    fptr = fopen("mpi_bruteforce_execution_times.csv", "a");
    if (fptr == NULL) {
        printf("Error opening file!");
        exit(1);
    }
    fprintf(fptr, "%d,%f\n", n, exec_time);
    fclose(fptr);
}

int main(int argc, char **argv) {
    int graph[V][V];
    int min_cost, n;
    int rank, size;

    // Initialize MPI environment
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    srand(time(NULL) + rank);

    if (rank == 0) {
        FILE *fptr = fopen("tsp_bruteforce_execution_times.csv", "w");
        fprintf(fptr, "Nodes,Execution_Time\n");
        fclose(fptr);
    }

    for (n = 3; n <= 19; n++) {
        if (rank == 0) {
            generateGraph(graph, n);
        }

        // Broadcast the graph to all processes
        MPI_Bcast(graph, V * V, MPI_INT, 0, MPI_COMM_WORLD);

        // Measure execution time for TSP brute-force using MPI
        double start_time = MPI_Wtime();
        tspBruteForceMPI(graph, n, &min_cost, rank, size);
        double end_time = MPI_Wtime();
        double exec_time = end_time - start_time;

        if (rank == 0) {
            saveExecutionTime(n, exec_time);
            printf("Nodes: %d, Execution Time: %f seconds, Min Cost: %d\n", n, exec_time, min_cost);
        }
    }

    MPI_Finalize();
    return 0;
}
