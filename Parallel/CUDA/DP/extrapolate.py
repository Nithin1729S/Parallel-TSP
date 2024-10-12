import numpy as np
import matplotlib.pyplot as plt

# Existing data for nodes and corresponding execution times and shortest paths
nodes = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
execution_times = np.array([8e-06, 1.4e-05, 3.1e-05, 8e-05, 0.000203, 0.000424, 0.000959, 0.002202, 0.004654, 0.008088, 0.021587, 0.073654, 0.181658, 0.583788, 1.531165, 3.549083, 7.900504])

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
new_nodes = np.arange(18, 31)

# Extrapolate
estimated_times = extrapolate_tsp(nodes,execution_times, new_nodes)

# Combine existing and new data for plotting
all_nodes = np.concatenate((nodes, new_nodes))
all_execution_times = np.concatenate((execution_times, estimated_times))

# Display results
for n,time in zip(new_nodes, estimated_times):
    print(f"Nodes: {n}, Estimated Execution Time: {time:.6f} seconds")


