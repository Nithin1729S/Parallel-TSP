import pandas as pd
import matplotlib.pyplot as plt

# Load data from CSV file
data = pd.read_csv('tsp_cuda_dp.csv')

# Extracting nodes and execution times
nodes = data['Nodes']
execution_time_cuda = data['Execution_Time_CUDA']
execution_time_serial = data['Execution_Time_Serial']

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(nodes, execution_time_cuda, label='CUDA Execution Time', marker='o')
plt.plot(nodes, execution_time_serial, label='Serial Execution Time', marker='o')

# Adding titles and labels
plt.title('CUDA_DP vs Serial DP Execution Time Comparison')
plt.xlabel('Number of Nodes')
plt.ylabel('Execution Time (seconds)')
plt.xticks(nodes)  # Show all nodes on the x-axis
plt.legend()
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.savefig("CUDA_DP_vs_Serial_DP.png")
