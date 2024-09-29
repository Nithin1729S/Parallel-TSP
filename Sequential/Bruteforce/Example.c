#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>


int calculatePathCost(int graph[][100], int path[], int n) {
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


void tspBruteForce(int graph[][100], int n, int *min_cost) {
    int path[100];
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


int main() {
    int min_cost, n=4;
    int graph[100][100] = { { 0, 10, 15, 20 },
                       { 10, 0, 35, 25 },
                       { 15, 35, 0, 30 },
                       { 20, 25, 30, 0 } };
    tspBruteForce(graph, n, &min_cost);
    printf("Minimum Cost = %d\n",min_cost);
    return 0;
}
