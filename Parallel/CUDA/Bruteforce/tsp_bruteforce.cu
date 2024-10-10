#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <sys/time.h>

#define MAX_NODES 12  // Maximum number of nodes
#define BLOCK_SIZE 256

__device__ int d_factorial(int n) {
    int result = 1;
    for (int i = 2; i <= n; ++i) {
        result *= i;
    }
    return result;
}

// Host version of factorial
int h_factorial(int n) {
    int result = 1;
    for (int i = 2; i <= n; ++i) {
        result *= i;
    }
    return result;
}

__global__ void tspKernel(int *d_adjacency, int *d_result, int nodes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_permutations = d_factorial(nodes - 1);

    if (tid < total_permutations) {
        int path[MAX_NODES];
        for (int i = 0; i < nodes; ++i) {
            path[i] = i;
        }

        // Generate permutation
        int temp = tid;
        for (int i = 1; i < nodes - 1; ++i) {
            int j = temp % (nodes - i) + i;
            int swap = path[i];
            path[i] = path[j];
            path[j] = swap;
            temp /= (nodes - i);
        }

        // Calculate path length
        int length = 0;
        for (int i = 0; i < nodes - 1; ++i) {
            length += d_adjacency[path[i] * nodes + path[i + 1]];
        }
        length += d_adjacency[path[nodes - 1] * nodes + path[0]];

        atomicMin(d_result, length);
    }
}

__global__ void initRNG(curandState *state, unsigned long seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, tid, 0, &state[tid]);
}

__global__ void generateAdjacencyMatrix(int *d_adjacency, int nodes, curandState *state) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < nodes * nodes) {
        int row = tid / nodes;
        int col = tid % nodes;
        if (row != col) {
            d_adjacency[tid] = curand(&state[tid]) % 100 + 1;  // Random distance between 1 and 100
        } else {
            d_adjacency[tid] = 0;  // Distance to self is 0
        }
    }
}

double getTime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main() {
    for (int nodes = 3; nodes <= MAX_NODES; ++nodes) {
        int *h_adjacency = (int *)malloc(nodes * nodes * sizeof(int));
        int *d_adjacency;
        cudaMalloc(&d_adjacency, nodes * nodes * sizeof(int));

        int *d_result;
        cudaMalloc(&d_result, sizeof(int));

        curandState *d_state;
        cudaMalloc(&d_state, nodes * nodes * sizeof(curandState));

        // Initialize RNG
        initRNG<<<(nodes * nodes + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_state, time(NULL));

        // Generate adjacency matrix
        generateAdjacencyMatrix<<<(nodes * nodes + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_adjacency, nodes, d_state);

        // Copy adjacency matrix to host for verification (optional)
        cudaMemcpy(h_adjacency, d_adjacency, nodes * nodes * sizeof(int), cudaMemcpyDeviceToHost);

        // Set initial result to a large value
        int initial_result = 1000000;
        cudaMemcpy(d_result, &initial_result, sizeof(int), cudaMemcpyHostToDevice);

        int total_permutations = h_factorial(nodes - 1);  // Use host version of factorial
        int grid_size = (total_permutations + BLOCK_SIZE - 1) / BLOCK_SIZE;

        double start_time = getTime();

        // Launch kernel
        tspKernel<<<grid_size, BLOCK_SIZE>>>(d_adjacency, d_result, nodes);

        // Synchronize and get result
        cudaDeviceSynchronize();
        int result;
        cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

        double end_time = getTime();
        double execution_time = end_time - start_time;

        printf("Nodes: %d, Shortest Path: %d, Execution Time: %.6f seconds\n", nodes, result, execution_time);

        // Clean up
        free(h_adjacency);
        cudaFree(d_adjacency);
        cudaFree(d_result);
        cudaFree(d_state);
    }

    return 0;
}
