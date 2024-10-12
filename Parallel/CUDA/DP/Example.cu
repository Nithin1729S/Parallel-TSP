#include <iostream>
#include <vector>
#include <limits.h>
#include <chrono>

#define MAX_V 4  // Maximum number of nodes

__device__ const int dist[MAX_V][MAX_V] = {
    { 0, 10, 15, 20 },
    { 10, 0, 35, 25 },
    { 15, 35, 0, 30 },
    { 20, 25, 30, 0 }
};

// Kernel to compute TSP
__device__ int tspKernel(int mask, int pos, int *dp, int n) {
    // Base case: if all cities have been visited
    if (mask == ((1 << n) - 1)) {
        return dist[pos][0]; // Return to the starting city (city 0)
    }

    int index = mask * n + pos; // Create a unique index for dp
    if (dp[index] != -1) {
        return dp[index]; // Return cached result if available
    }

    int ans = INT_MAX; // Initialize answer to maximum
    for (int city = 0; city < n; city++) {
        // If the city has not been visited
        if ((mask & (1 << city)) == 0) {
            int newAns = dist[pos][city] + tspKernel(mask | (1 << city), city, dp, n);
            ans = min(ans, newAns); // Update answer
        }
    }

    dp[index] = ans; // Cache the result
    return ans; // Return computed result
}

// Kernel wrapper for launching TSP
__global__ void tspLauncher(int *dp, int n, int *result) {
    result[0] = tspKernel(1, 0, dp, n); // Start TSP with node 0 and mask 1
}

int main() {
    int n = MAX_V;  // Number of cities

    // Initialize DP table in host memory
    int *dp = new int[(1 << n) * n];
    std::fill(dp, dp + (1 << n) * n, -1);

    int *d_dp, *d_result;
    int h_result;

    // Allocate device memory
    cudaMalloc((void**)&d_dp, (1 << n) * n * sizeof(int));
    cudaMalloc((void**)&d_result, sizeof(int));

    // Copy dp to device
    cudaMemcpy(d_dp, dp, (1 << n) * n * sizeof(int), cudaMemcpyHostToDevice);

    // Measure execution time
    auto start = std::chrono::high_resolution_clock::now();

    // Launch the TSP kernel
    tspLauncher<<<1, 1>>>(d_dp, n, d_result);
    cudaDeviceSynchronize(); // Wait for the kernel to finish

    // Copy result back to host
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << "Nodes: " << n << " Time: " << duration.count() << " seconds. Min Cost: " << h_result << "\n";

    // Free device memory
    cudaFree(d_dp);
    cudaFree(d_result);

    // Free host memory
    delete[] dp;

    return 0;
}
