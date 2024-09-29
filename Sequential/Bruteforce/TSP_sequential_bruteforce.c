//Time Complexity: O(N!)
//Space Complexity: O(N^2)
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#define V 100

//can collapse nested for loops using collapse(2)
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

//can use a reduction clause for total cost computation
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



//Permutation Function
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




//can parallize do while loop and use critical section as necessary
void tspBruteForce(int graph[][V], int n, int *min_cost) {
    int path[V];
    for (int i = 0; i < n; i++) {
        path[i] = i;
    }

    *min_cost = INT_MAX;
    do {
        int current_cost = calculatePathCost(graph, path, n);
        if (current_cost < *min_cost) {
            *min_cost = current_cost;
        }
    } while (next_permutation(path, path + n));
}




void saveExecutionTime(int n, double exec_time) {
    FILE *fptr;
    fptr = fopen("tsp_bruteforce_execution_times.csv", "a");
    if (fptr == NULL) {
        printf("Error opening file!");
        exit(1);
    }
    fprintf(fptr, "%d,%f\n", n, exec_time);
    fclose(fptr);
}



int main() {
    int graph[V][V];  
    int min_cost, n;
    srand(time(NULL));

    FILE *fptr = fopen("tsp_bruteforce_execution_times.csv", "w");
    fprintf(fptr, "Nodes,Execution_Time\n");
    fclose(fptr);

    for (n = 3; n <= 19; n++) { 
        generateGraph(graph, n);

        
        clock_t start = clock();
        tspBruteForce(graph, n, &min_cost);
        clock_t end = clock();

        double exec_time = (double)(end - start) / CLOCKS_PER_SEC;
        saveExecutionTime(n, exec_time);
        printf("Nodes: %d, Execution Time: %f seconds, Min Cost: %d\n", n, exec_time, min_cost);
    }

    return 0;
}
