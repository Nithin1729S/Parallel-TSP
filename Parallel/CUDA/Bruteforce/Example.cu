#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#define NODES 4
#define BLOCK_SIZE 256

__device__ int d_factorial(int n) {
    int result = 1;
    for (int i = 2; i <= n; ++i) {
        result *= i;
    }
    return result;
}

int h_factorial(int n) {
    int result = 1;
    for (int i = 2; i <= n; ++i) {
        result *= i;
    }
    return result;
}

__global__ void tspKernel(int *d_graph, int *d_result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_permutations = d_factorial(NODES - 1);

    if (tid < total_permutations) {
        int path[NODES];
        for (int i = 0; i < NODES; ++i) {
            path[i] = i;
        }

        // Generate permutation
        int temp = tid;
        for (int i = 1; i < NODES - 1; ++i) {
            int j = temp % (NODES - i) + i;
            int swap = path[i];
            path[i] = path[j];
            path[j] = swap;
            temp /= (NODES - i);
        }

        // Calculate path length
        int length = 0;
        for (int i = 0; i < NODES - 1; ++i) {
            length += d_graph[path[i] * NODES + path[i + 1]];
        }
        length += d_graph[path[NODES - 1] * NODES + path[0]];

        atomicMin(d_result, length);
    }
}

double getTime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main() {
    int graph[NODES][NODES] = {
        { 0, 10, 15, 20 },
        { 10, 0, 35, 25 },
        { 15, 35, 0, 30 },
        { 20, 25, 30, 0 }
    };

    int *d_graph;
    cudaMalloc(&d_graph, NODES * NODES * sizeof(int));
    cudaMemcpy(d_graph, graph, NODES * NODES * sizeof(int), cudaMemcpyHostToDevice);

    int *d_result;
    cudaMalloc(&d_result, sizeof(int));

    // Set initial result to a large value
    int initial_result = 1000000;
    cudaMemcpy(d_result, &initial_result, sizeof(int), cudaMemcpyHostToDevice);

    int total_permutations = h_factorial(NODES - 1);
    int grid_size = (total_permutations + BLOCK_SIZE - 1) / BLOCK_SIZE;

    double start_time = getTime();

    // Launch kernel
    tspKernel<<<grid_size, BLOCK_SIZE>>>(d_graph, d_result);

    // Synchronize and get result
    cudaDeviceSynchronize();
    int result;
    cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    double end_time = getTime();
    double execution_time = end_time - start_time;

    printf("Nodes: %d, Shortest Path: %d, Execution Time: %.6f seconds\n", NODES, result, execution_time);

    // Clean up
    cudaFree(d_graph);
    cudaFree(d_result);

    return 0;
}