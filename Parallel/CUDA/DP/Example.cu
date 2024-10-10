#include <iostream>
#include <vector>
#include <limits.h>
#include <chrono>
#include <bitset>

#define MAX_V 4  // maximum number of nodes

const int dist[MAX_V][MAX_V] = {
    { 0, 10, 15, 20 },
    { 10, 0, 35, 25 },
    { 15, 35, 0, 30 },
    { 20, 25, 30, 0 }
};

// The main dynamic programming function for solving TSP
int tspDP(int mask, int pos, std::vector<std::vector<int>>& dp, int n) {
    if (mask == ((1 << n) - 1)) {
        return dist[pos][0]; // Return to the starting city (city 0)
    }

    if (dp[mask][pos] != -1) {
        return dp[mask][pos];
    }

    int ans = INT_MAX;
    for (int city = 0; city < n; city++) {
        if ((mask & (1 << city)) == 0) {
            int newAns = dist[pos][city] + tspDP(mask | (1 << city), city, dp, n);
            ans = std::min(ans, newAns);
        }
    }

    return dp[mask][pos] = ans;
}

int main() {
    int n = MAX_V;  // number of cities

    // Initialize DP table
    std::vector<std::vector<int>> dp(1 << n, std::vector<int>(n, -1));

    // Measure execution time
    auto start = std::chrono::high_resolution_clock::now();

    // Start TSP from node 0 and mask 1 (visited node 0)
    int result = tspDP(1, 0, dp, n);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << "Nodes: " << n << " Time: " << duration.count() << " seconds. Min Cost: " << result << "\n";

    // Print the optimal tour
    int mask = 1, pos = 0;
    std::cout << "Optimal Tour: 0";
    for (int i = 0; i < n - 1; i++) {
        int nextCity = -1;
        for (int city = 0; city < n; city++) {
            if ((mask & (1 << city)) == 0) {
                if (nextCity == -1 || dist[pos][city] + dp[mask | (1 << city)][city] <
                                      dist[pos][nextCity] + dp[mask | (1 << nextCity)][nextCity]) {
                    nextCity = city;
                }
            }
        }
        std::cout << " -> " << nextCity;
        mask |= (1 << nextCity);
        pos = nextCity;
    }
    std::cout << " -> 0\n";

    return 0;
}
