import numpy as np
import matplotlib.pyplot as plt

# Existing data for nodes and corresponding execution times and shortest paths
nodes = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
shortest_paths = np.array([89, 212, 133, 121, 154, 156, 210, 113, 233, 125, 154, 156, 167, 211, 226])
execution_times = np.array([
    0.000039,0.000021, 0.000020, 0.000023 , 0.000022,
   0.000043, 0.000046, 0.000162 , 0.001543 , 0.018341,
   0.225360,0.403661 ,0.539362, 0.661198, 0.703940
])

# Function to extrapolate execution times and shortest paths
def extrapolate_tsp(nodes, shortest_paths, execution_times, new_nodes):
    # Estimate execution time growth
    estimated_times = []
    estimated_paths = []
    
    for n in new_nodes:
        # Estimate shortest path (Assuming a small increase)
        estimated_path = shortest_paths[-1] + (n - nodes[-1]) * 5  # Rough estimate
        estimated_paths.append(estimated_path)

        # Estimate execution time based on last execution time
        last_time = execution_times[-1]
        # Using a rough doubling time for each additional node
        estimated_time = last_time * (2 ** (n - nodes[-1]))
        estimated_times.append(estimated_time)
    
    return estimated_paths, estimated_times

# Nodes to extrapolate
new_nodes = np.arange(18, 31)

# Extrapolate
estimated_paths, estimated_times = extrapolate_tsp(nodes, shortest_paths, execution_times, new_nodes)

# Combine existing and new data for plotting
all_nodes = np.concatenate((nodes, new_nodes))
all_shortest_paths = np.concatenate((shortest_paths, estimated_paths))
all_execution_times = np.concatenate((execution_times, estimated_times))

# Display results
for n, path, time in zip(new_nodes, estimated_paths, estimated_times):
    print(f"Nodes: {n}, Estimated Shortest Path: {path}, Estimated Execution Time: {time:.6f} seconds")

# Plotting the results

plt.plot(all_nodes, all_execution_times, marker='o', color='orange')
plt.title('Estimated Execution Time vs. Nodes')
plt.xlabel('Number of Nodes')
plt.ylabel('Execution Time (seconds)')
plt.yscale('log')  # Log scale for better visualization
plt.grid()

plt.tight_layout()
plt.savefig("extrapolate_bruteforce.png")
