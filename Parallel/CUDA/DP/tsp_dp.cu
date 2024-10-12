#include <iostream>
#include <vector>
#include <limits>
#include <chrono>
#include <random>
#include <cuda_runtime.h>

#define MAX 20  // Maximum number of nodes
#define THREADS_PER_BLOCK 256
#define GPU_MAX_NODES 9  // Maximum nodes for GPU computation

// Error checking macro for CUDA calls
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Function to generate a random adjacency matrix
void generateRandomAdjacencyMatrix(int nodes, int maxWeight, std::vector<std::vector<int>>& adj) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, maxWeight);

    adj.resize(nodes, std::vector<int>(nodes));
    for (int i = 0; i < nodes; ++i) {
        for (int j = 0; j < nodes; ++j) {
            if (i == j) {
                adj[i][j] = 0;  // Diagonal elements are zero (no self-loop)
            } else {
                adj[i][j] = dis(gen);
            }
        }
    }
}

// CPU implementation of TSP using dynamic programming
long long tspCPU(int mask, int pos, const std::vector<std::vector<int>>& adj, std::vector<std::vector<long long>>& dp) {
    int n = adj.size();
    if (mask == (1 << n) - 1) {
        return adj[pos][0]; // Return to the starting city
    }

    if (dp[mask][pos] != -1) {
        return dp[mask][pos];
    }

    long long ans = std::numeric_limits<long long>::max();
    for (int city = 0; city < n; city++) {
        if ((mask & (1 << city)) == 0) {
            long long newAns = adj[pos][city] + tspCPU(mask | (1 << city), city, adj, dp);
            ans = std::min(ans, newAns);
        }
    }

    return dp[mask][pos] = ans;
}

// Device function for TSP using dynamic programming
__device__ long long tspDP(int mask, int pos, const int *adj, long long *dp, int n) {
    if (mask == (1 << n) - 1) {
        return adj[pos * n + 0]; // Return to the starting city
    }

    int index = mask * n + pos; // Create a unique index for dp
    if (dp[index] != -1) {
        return dp[index];
    }

    long long ans = LLONG_MAX;
    for (int city = 0; city < n; city++) {
        if ((mask & (1 << city)) == 0) {
            long long newAns = (long long)adj[pos * n + city] + tspDP(mask | (1 << city), city, adj, dp, n);
            ans = min(ans, newAns);
        }
    }

    return dp[index] = ans;
}

// Kernel to initialize the DP table
__global__ void initDP(long long *dp, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dp[idx] = -1;
    }
}

// Kernel to launch the TSP DP calculation
__global__ void tspLauncher(int *adj, long long *dp, int n, long long *result) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        result[0] = tspDP(1, 0, adj, dp, n); // Start TSP from node 0 with mask 1
    }
}

int main() {
    int maxWeight = 100;

    for (int nodes = 4; nodes <= MAX; nodes++) {
        std::vector<std::vector<int>> adj;
        generateRandomAdjacencyMatrix(nodes, maxWeight, adj);

        auto start = std::chrono::high_resolution_clock::now();
        long long result;

        if (nodes <= GPU_MAX_NODES) {
            // GPU implementation
            int *d_adj;
            long long *d_dp, *d_result;
            long long *h_result = new long long[1];

            gpuErrchk(cudaMalloc((void**)&d_adj, nodes * nodes * sizeof(int)));
            gpuErrchk(cudaMalloc((void**)&d_dp, (1LL << nodes) * nodes * sizeof(long long)));
            gpuErrchk(cudaMalloc((void**)&d_result, sizeof(long long)));

            // Copy adjacency matrix to device
            std::vector<int> flat_adj(nodes * nodes);
            for (int i = 0; i < nodes; ++i) {
                for (int j = 0; j < nodes; ++j) {
                    flat_adj[i * nodes + j] = adj[i][j];
                }
            }
            gpuErrchk(cudaMemcpy(d_adj, flat_adj.data(), nodes * nodes * sizeof(int), cudaMemcpyHostToDevice));

            // Initialize DP table
            int dpSize = (1LL << nodes) * nodes;
            int blocks = (dpSize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            initDP<<<blocks, THREADS_PER_BLOCK>>>(d_dp, dpSize);
            gpuErrchk(cudaPeekAtLastError());
            gpuErrchk(cudaDeviceSynchronize());

            // Launch the TSP kernel
            tspLauncher<<<1, 1>>>(d_adj, d_dp, nodes, d_result);
            gpuErrchk(cudaPeekAtLastError());
            gpuErrchk(cudaDeviceSynchronize());

            // Copy result back to host
            gpuErrchk(cudaMemcpy(h_result, d_result, sizeof(long long), cudaMemcpyDeviceToHost));

            result = h_result[0];

            // Free device memory
            gpuErrchk(cudaFree(d_adj));
            gpuErrchk(cudaFree(d_dp));
            gpuErrchk(cudaFree(d_result));
            delete[] h_result;
        } else {
            // CPU implementation
            std::vector<std::vector<long long>> dp(1 << nodes, std::vector<long long>(nodes, -1));
            result = tspCPU(1, 0, adj, dp);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;

        std::cout << "Nodes: " << nodes << " Time: " << duration.count() << " seconds. Cost: " << result << "\n";
    }

    return 0;
}