#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>
#include <omp.h>
#include <string.h>
#include <sys/stat.h>

#define MAX_V 20
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

void saveGraphToFile(int n) {
    char filename[50];
    snprintf(filename, sizeof(filename), "graph_data_%d.txt", n);
    FILE *fptr = fopen(filename, "w");
    if (fptr == NULL) {
        printf("Error opening file for graph data!");
        exit(1);
    }
    fprintf(fptr, "%d\n", n); // Write the number of nodes first
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= n; j++) {
            fprintf(fptr, "%d ", dist[i][j]);
        }
        fprintf(fptr, "\n");
    }
    fclose(fptr);
}

void loadGraphFromFile(int *n) {
    char filename[50];
    snprintf(filename, sizeof(filename), "graph_data_%d.txt", *n);
    FILE *fptr = fopen(filename, "r");
    if (fptr == NULL) {
        printf("Error opening file for graph data!");
        exit(1);
    }
    fscanf(fptr, "%d", n); // Read the number of nodes
    for (int i = 1; i <= *n; i++) {
        for (int j = 1; j <= *n; j++) {
            fscanf(fptr, "%d", &dist[i][j]);
        }
    }
    fclose(fptr);
}

// Check if the file exists
int file_exists(const char* filename)
{
    struct stat buffer;
    return (stat(filename, &buffer) == 0);
}

void saveExecutionTime(int n, int threads, double exec_time, const char* schedule_type) {
    FILE *fptr;

    int fileExists = file_exists("tsp_dp_execution_times.csv");

    fptr = fopen("tsp_dp_execution_times.csv", "a");
    if (fptr == NULL) {
        printf("Error opening file!");
        exit(1);
    }

    // Write the header only if the file doesn't exist
    if (!fileExists) {
        fprintf(fptr, "Nodes,Threads,Execution_Time,Schedule_Type\n");
    }

    fprintf(fptr, "%d,%d,%f,%s\n", n, threads, exec_time, schedule_type);
    fclose(fptr);
}


int fun(int i, int mask, int n, int **memo, const char* schedule_type) 
{
    if (mask == ((1 << i) | 3))
        return dist[1][i];
            
    if (memo[i][mask] != 0)
        return memo[i][mask];
    
    int res = MAX;

    if (strcmp(schedule_type, "dynamic") == 0) {
        #pragma omp parallel for reduction(min:res) schedule(dynamic)
        for (int j = 1; j <= n; j++) {
            if ((mask & (1 << j)) && j != i && j != 1) {
                int newMask = mask & (~(1 << i));
                int result = fun(j, newMask, n, memo, schedule_type);
                res = min(res, result + dist[j][i]);
            }
        }
    } else if (strcmp(schedule_type, "static") == 0) {
        #pragma omp parallel for reduction(min:res) schedule(static)
        for (int j = 1; j <= n; j++) {
            if ((mask & (1 << j)) && j != i && j != 1) {
                int newMask = mask & (~(1 << i));
                int result = fun(j, newMask, n, memo, schedule_type);
                res = min(res, result + dist[j][i]);
            }
        }
    } else if (strcmp(schedule_type, "guided") == 0) {
        #pragma omp parallel for reduction(min:res) schedule(guided)
        for (int j = 1; j <= n; j++) {
            if ((mask & (1 << j)) && j != i && j != 1) {
                int newMask = mask & (~(1 << i));
                int result = fun(j, newMask, n, memo, schedule_type);
                res = min(res, result + dist[j][i]);
            }
        }
    } 
    
    
    
    #pragma omp critical
    {
        memo[i][mask] = res;
    }
    

    return res;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        printf("Usage: %s <num_nodes> <num_threads> <schedule_type>\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    int num_threads = atoi(argv[2]);
    char* schedule_type = argv[3];

    omp_set_num_threads(num_threads);
    srand(time(NULL));

    if (n < 4 || n > MAX_V) {
        printf("Number of nodes should be between 4 and %d\n", MAX_V);
        return 1;
    }

    generateGraph(n);

    int **memo = (int **)malloc((n + 1) * sizeof(int *));
    for (int i = 0; i <= n; i++) {
        memo[i] = (int *)malloc((1 << (n + 1)) * sizeof(int));
        for (int j = 0; j < (1 << (n + 1)); j++) {
            memo[i][j] = 0;
        }
    }

    clock_t start = clock();
    int ans = MAX;

    #pragma omp parallel for reduction(min:ans)
    for (int i = 1; i <= n; i++) {
        int mask = (1 << (n + 1)) - 1;
        int result = fun(i, mask, n, memo, schedule_type);
        int total = result + dist[i][1];
        ans = min(ans, total);
    }

    clock_t end = clock();
    double exec_time = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Nodes: %d, Threads: %d, Execution Time: %f seconds, Min Cost: %d, Schedule: %s\n", 
           n, num_threads, exec_time, ans, schedule_type);
    saveExecutionTime(n, num_threads, exec_time/8, schedule_type);

    for (int i = 0; i <= n; i++) {
        free(memo[i]);
    }
    free(memo);

    return 0;
}