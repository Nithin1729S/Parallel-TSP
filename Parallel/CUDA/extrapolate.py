import numpy as np
import matplotlib.pyplot as plt

# Existing data for nodes and corresponding execution times and shortest paths
nodes = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,18,19,20,21,22,23,24,25,26,27])
execution_times = np.array( [3.9e-05, 2.1e-05, 2e-05, 2.3e-05, 2.2e-05, 4.3e-05, 4.6e-05, 0.000162, 0.001543, 0.018341, 0.22536, 0.403661, 0.539362, 0.661198, 0.70394, 1.40788, 2.81576, 5.63152, 11.26304, 22.52608, 45.05216, 90.10432, 180.20864, 360.41728, 720.83456])

# Function to extrapolate execution times and shortest paths
def extrapolate_tsp(nodes, execution_times, new_nodes):
    # Estimate execution time growth
    estimated_times = []
   
    
    for n in new_nodes:
        # Estimate shortest path (Assuming a small increase)
        

        # Estimate execution time based on last execution time
        last_time = execution_times[-1]
        # Using a rough doubling time for each additional node
        estimated_time = last_time * (2 ** (n - nodes[-1]))
        estimated_times.append(estimated_time)
    
    return estimated_times

# Nodes to extrapolate
new_nodes = np.arange(28, 31)

# Extrapolate
estimated_times = extrapolate_tsp(nodes,execution_times, new_nodes)

# Combine existing and new data for plotting
all_nodes = np.concatenate((nodes, new_nodes))
all_execution_times = np.concatenate((execution_times, estimated_times))

# Display results
for n,time in zip(new_nodes, estimated_times):
    print(f"Nodes: {n}, Estimated Execution Time: {time:.6f} seconds")


