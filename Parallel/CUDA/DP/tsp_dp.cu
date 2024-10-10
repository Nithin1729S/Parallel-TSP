#include <iostream>
#include <vector>
#include <limits.h>
#include <chrono>
#include <random>

#define MAX 20  // maximum number of nodes

std::vector<std::vector<int>> generateRandomAdjacencyMatrix(int nodes, int maxWeight) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, maxWeight);

    std::vector<std::vector<int>> matrix(nodes, std::vector<int>(nodes));
    for (int i = 0; i < nodes; ++i) {
        for (int j = 0; j < nodes; ++j) {
            if (i == j) {
                matrix[i][j] = 0;  // Diagonal elements are zero (no self-loop)
            } else {
                matrix[i][j] = dis(gen);
            }
        }
    }
    return matrix;
}

// The main dynamic programming function for solving TSP
int tspDP(int mask, int pos, const std::vector<std::vector<int>>& adj, std::vector<std::vector<int>>& dp, int n) {
    if (mask == (1 << n) - 1) {
        return adj[pos][0]; // Return to the starting city
    }

    if (dp[mask][pos] != -1) {
        return dp[mask][pos];
    }

    int ans = INT_MAX;
    for (int city = 0; city < n; city++) {
        if ((mask & (1 << city)) == 0) {
            int newAns = adj[pos][city] + tspDP(mask | (1 << city), city, adj, dp, n);
            ans = std::min(ans, newAns);
        }
    }

    return dp[mask][pos] = ans;
}

int main() {
    int maxWeight = 100;

    for (int nodes = 4; nodes <= MAX; nodes++) {
        // Generate random adjacency matrix on CPU
        auto adj = generateRandomAdjacencyMatrix(nodes, maxWeight);

        // Debug: Print the random matrix
        //std::cout << "Adjacency Matrix (" << nodes << "x" << nodes << "):\n";
        int zeroCount = 0;
        //for (int i = 0; i < nodes; i++) {
           // for (int j = 0; j < nodes; j++) {
              //  std::cout << adj[i][j] << " ";
               // if (adj[i][j] == 0) zeroCount++;
           // }
           // std::cout << "\n";
        //}
        //std::cout << "Number of zeros: " << zeroCount << " (expected: " << nodes << ")\n\n";

        // Initialize DP table
        std::vector<std::vector<int>> dp(1 << nodes, std::vector<int>(nodes, -1));

        // Measure execution time
        auto start = std::chrono::high_resolution_clock::now();

        // Start TSP from node 0 and mask 1 (visited node 0)
        int result = tspDP(1, 0, adj, dp, nodes);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;

        std::cout << "Nodes: " << nodes << " Time: " << duration.count() << " seconds. Cost: " << result << "\n";
    }

    return 0;
}





